# AGENTS.md — BigRedGym (Q2)

BigRedGym is the **Q2** Python robotics / reinforcement-learning codebase
(descended from `legged_gym`/pkGym; the installed package is named `q2` in
`pyproject.toml`, not `BigRedGym`). Task and learning logic is separated from
interchangeable physics backends: MuJoCo CPU, MuJoCo Warp (CUDA), and optional
licensed VSim.

## Read first — durable rules live elsewhere

Do not re-derive these; read the authoritative source:

- `genAI_skills/AGENTS.md` — repository-wide Q2 developer guide (architecture
  contracts, configs/learning rules, style, validation).
- `genAI_skills/README_MUJOCO.md` — setup/CLI, plus its "limitations and
  validation scope" for evidence boundaries.
- `README.md` (install / train / play) and `README_DEPLOY.md` (Go2 hardware).
- `tests/TEST_CATALOG.md` — what each test protects and which oracle it uses.

`genAI_skills/AGENTS.md` is auto-loaded only while reading files under
`genAI_skills/`, so when working in `gym/` or `learning/` you must open it
yourself.

## Repository layout — two nested git repos

- Outer `BigRedGym/` is the product repo: `gym/`, `learning/`, `scripts/`,
  `go2_deploy/`, `resources/`, `tests/`, `genAI_skills/`.
- `BigRedGym/.opencode/` is a **separate git repo** (opencode config, agents,
  skills). The outer repo ignores it via `.gitignore`. Target the owning repo
  (`git -C .opencode ...` from the root, or `cd .opencode && git ...`) and never
  commit `.opencode` changes to BigRedGym.
- Q2 skills are registered by `.opencode/opencode.json` `skills.paths`, which
  points at `genAI_skills/.agents/skills`.
- Never commit `logs/`, `thirdparty/vlearn`, `thirdparty/unitree_sdk2_python/`,
  `.venv`, licenses, or any file > 100 KB (enforced by pre-commit and CI).

## Build

Dependencies are uv-managed; `pyproject.toml` + `uv.lock` are the only source of
truth (no pip/requirements files). Use `--frozen` so a missing optional
wheel/checkout never triggers re-resolution. Run all commands from the root.

```bash
uv sync --frozen                       # create/refresh the locked .venv
uv run --frozen python -m pytest -q    # default gate: tests/unit_tests (excludes warp/vsim/unitree)
uv run --frozen python -m pytest gym -q        # colocated gym tests (outside default testpaths)
uv run --frozen python -m pytest learning -q   # colocated learning tests
uv run --frozen python -m pytest -q -m warp    # MuJoCo Warp, CUDA only
uv run --frozen python -m pytest tests/unit_tests/<file>.py -q   # narrow while iterating
uv run --frozen ruff format <touched.py> && uv run --frozen ruff check .
```

Train / play (a viewer opens by default; add `--headless` to disable it):

```bash
uv run --frozen scripts/train.py --task=go2trot --device=cuda:0 --headless --max_iterations=300
uv run --frozen scripts/play.py  --task=go2trot
```

Optional extras install with `uv sync --frozen --extra <name>`: `gpu` (MuJoCo
Warp), `vsim` (licensed, CUDA, machine-local; also pass `--env-file .env.vsim`),
`unitree_sdk` (Go2 deploy; first run `scripts/fetch_unitree_sdk.py`). CI
(`.github/workflows/unit_tests.yml`) runs the default gate, both colocated
suites, Ruff, and a package build on Ubuntu/Python 3.11; it cannot establish
Warp/VSim/hardware results.

## This machine — CPU-only go2trot training

No GPU here: use `--device cpu` (the default) and the MuJoCo CPU backend. That
backend builds one MuJoCo world per environment, and `go2trot` turns on
link-mass domain randomization, so each env gets a private model copy: RAM grows
~0.9 MB/env beyond a ~1.5 GB baseline. Always pass `--headless`. Without it each
copy also keeps the visual skybox/ground textures (~5 MB/env), which at the
default `num_envs = 4096` needs ~20 GB and gets OOM-killed. Measured headless
peaks (iteration 1): 1024 → 2.3 GiB, 4096 → 5.0 GiB, 5120 → 5.9 GiB. So cap
`--num_envs` at 4096 (≤6 GiB with headroom); 5120 is the ~6 GiB ceiling.

Guard the process with an 8 GiB cgroup. `ulimit -v 8000000` does **not** work —
it caps address space, and MuJoCo reserves a large virtual arena and aborts with
`mujoco.FatalError: Could not allocate memory`. Cap resident memory instead:

```bash
systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0 -- \
  uv run --frozen scripts/train.py --task=go2trot --device=cpu --headless \
  --num_envs=4096 --max_iterations=300 --run_name=<unique-name>
```

Always pass a unique `--run_name` — whoever launches the run (the user or
another agent) suggests it, e.g. `cpu-6g-trot-<topic>-<date>`; it is appended to
the timestamped log dir so concurrent runs do not collide.

Runs land in `logs/go2trot/<MonDD_HH-MM-SS>_<run_name>/` (`model_<it>.pt`,
`vitals.jsonl`); the dir appears only once the env builds and the first
checkpoint/log is written. With WandB unconfigured, `vitals.jsonl` is the only
metrics record.

Plot progress from a run's `vitals.jsonl` (latest run by default, using the same
selection as `play.py`):

```bash
uv run --frozen scripts/plot_vitals.py --experiment_name go2trot
uv run --frozen scripts/plot_vitals.py --run logs/go2trot/<run> --out progress.png
```

The 2x2 figure shows return, reward terms, algorithm losses, and actor metrics.

## Skills — load the matching one

General (`.opencode/skills/general/`):

| Task | Skill |
|---|---|
| Organize notes under `./research/` | `research-notes` |
| Plan/result journal entries | `dev-journal` |
| Bootstrap or sync this file | `agents-bootstrap` |

Q2 repo skills (registered from `genAI_skills/.agents/skills/`):

| Change or question | Skill |
|---|---|
| uv/venv, CUDA/Warp, macOS, VSim setup | `q2-development-environment` |
| Backend contract, state/reset, contacts, routing | `q2-backend-development` |
| Robots, assets, configs, tasks, registration | `q2-task-authoring` |
| Observations, rewards, PPO/storage/normalization | `q2-rl-development` |
| Training, resume, playback, evaluation | `q2-train-and-evaluate` |
| Test selection, regressions, diagnosis | `q2-testing-and-debugging` |

## Agents

- `plan` / `scaffold` — read-only planning; produces a written plan for approval.
- `execute` — build-mode executor. Reads §Docs / §Build / §Code style below and
  runs a plan through `edit` / `trial` / `discussion` phases; phase changes
  happen only on explicit user request.
- `librarian` — delegate deep source exploration (`/librarian <question>` or a
  `librarian` subagent) to save context; it keeps a private KB under
  `.opencode/workspace/library/`.

## Docs — execute-agent contract

- `mode`: centralized (all docs live under the `.opencode/` workspace tree; no
  per-package routing).
- `plans_active`: `.opencode/workspace/plans/`
- `plans_archive`: `.opencode/workspace/plans/archive/`
- `journal_path`: `.opencode/workspace/journal.md`
- `todos_path`: `.opencode/workspace/todos.md`

## Code style

`genAI_skills/AGENTS.md` §"Style and Change Discipline" is authoritative. The
rules most likely to change an edit:

- Python 3.11, four spaces, double quotes, Ruff 88-column format.
- Prefer explicit failures over silent fallback: no broad `try/except`, no
  default-valued config reads that mask a missing required field, no auto
  backend substitution.
- Read attributes directly (`cfg.seed`), not `getattr(cfg, "seed", None)`.
- Preserve historical public names (e.g. `MuJocoCPUBackend`) unless doing an
  explicit API migration. No new wildcard imports or `eval`; use
  `TODO(Scope):` for deferred work.
- Comment non-obvious engine semantics and tensor aliasing/liveness, not
  obvious Python.

## Notes

- Quit and restart opencode after changing `opencode.json`, an agent, a skill,
  or this file — config is read once at startup.
- Dirty worktrees are normal here; preserve unrelated changes and keep each
  change focused and separately reviewable.
