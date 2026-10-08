"""Plot training progress from a run's ``vitals.jsonl``.

``vitals.jsonl`` is the local per-iteration JSON sink written by
``learning/utils/logger/Logger.py``. It holds the same scalars that WandB
receives, so it is the plotting source when WandB is disabled.

Usage:
    uv run scripts/plot_vitals.py --experiment_name go2trot
    uv run scripts/plot_vitals.py --run logs/go2trot/Oct08_02-44-49_height-training
    uv run scripts/plot_vitals.py --run logs/go2trot/<run> --out progress.png
"""

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd

from gym import GYM_ROOT_DIR
from gym.utils.helpers import select_run


def get_args(argv=None):
    parser = argparse.ArgumentParser(description="Plot a run's vitals.jsonl")
    parser.add_argument(
        "--run",
        type=str,
        default=None,
        help="Run dir containing vitals.jsonl "
        "(default: latest under logs/<experiment_name>/)",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="go2trot",
        help="Experiment dir under logs/ used when --run is omitted",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output image path (default: <run>/vitals.png)",
    )
    return parser.parse_args(argv)


def resolve_run(args):
    if args.run is not None:
        run_dir = args.run
    else:
        root = os.path.join(GYM_ROOT_DIR, "logs", args.experiment_name)
        run_dir = select_run(root, -1)
    vitals_path = os.path.join(run_dir, "vitals.jsonl")
    if not os.path.isfile(vitals_path):
        raise FileNotFoundError(f"No vitals.jsonl in {run_dir}")
    return run_dir, vitals_path


def _columns(df, prefix):
    return sorted(c for c in df.columns if c.startswith(prefix))


def _legend(ax):
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=7, ncol=2, loc="best")


def plot_vitals(df, title, out_path):
    df = df.sort_values("iteration")
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle(title)

    # Overall return — the headline "is it learning" curve.
    ax = axes[0, 0]
    if "rewards/total_rewards" in df.columns:
        ax.plot(df["iteration"], df["rewards/total_rewards"], color="black")
    else:
        ax.plot(df["iteration"], df.filter(regex=r"^rewards/").sum(axis=1))
    ax.set_title("Return")

    # Individual reward terms.
    ax = axes[0, 1]
    for col in _columns(df, "rewards/"):
        if col == "rewards/total_rewards":
            continue
        ax.plot(df["iteration"], df[col], label=col.removeprefix("rewards/"))
    ax.set_title("Reward terms")

    # Optimizer/algorithm diagnostics.
    ax = axes[1, 0]
    for col in _columns(df, "algorithm/"):
        ax.plot(df["iteration"], df[col], label=col.removeprefix("algorithm/"))
    ax.set_title("Algorithm")

    # Policy outputs (entropy, action std, ...).
    ax = axes[1, 1]
    for col in _columns(df, "actor/"):
        ax.plot(df["iteration"], df[col], label=col.removeprefix("actor/"))
    ax.set_title("Actor")

    for ax in axes.flat:
        ax.set_xlabel("iteration")
        ax.grid(alpha=0.3)
        _legend(ax)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main(argv=None):
    args = get_args(argv)
    run_dir, vitals_path = resolve_run(args)
    df = pd.read_json(vitals_path, lines=True)
    if df.empty:
        raise ValueError(f"vitals.jsonl has no rows: {vitals_path}")
    title = os.path.basename(os.path.normpath(run_dir))
    out_path = args.out or os.path.join(run_dir, "vitals.png")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    print(f"Plotting {len(df)} iterations from {vitals_path}")
    plot_vitals(df, title, out_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
