"""Plot base height against the height command from a play.py CSV log.

Usage:
    uv run python assignments/assignment_04/plot_height.py [csv_path]

The figure is written back next to the source CSV, using the same file stem
(e.g. data/height_02.csv -> data/height_02.png and data/height_02.pdf).
"""

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd

# Time window to plot, in seconds.  START_TIME defaults to the beginning of the
# recording; END_TIME=None falls back to the end of the data ("t_end").
START_TIME = 0.0
END_TIME = None


def parse_args(argv=None):
    here = os.path.dirname(os.path.abspath(__file__))
    default_csv = os.path.join(here, "data", "height_02.csv")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv_path",
        nargs="?",
        default=default_csv,
        help="CSV written by scripts/play.py (default: data/height_02.csv)",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    df = pd.read_csv(args.csv_path)

    end_time = df["time"].max() if END_TIME is None else END_TIME
    window = df[(df["time"] >= START_TIME) & (df["time"] <= end_time)]

    fig, ax = plt.subplots()
    ax.plot(window["time"], window["base_height"], label="base height")
    ax.plot(window["time"], window["height_command"], label="height command")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("height [m]")
    ax.set_title("height tracking")
    ax.legend()
    fig.tight_layout()

    out_dir = os.path.dirname(os.path.abspath(args.csv_path))
    stem = os.path.splitext(os.path.basename(args.csv_path))[0]
    for ext in ("png", "pdf"):
        out_path = os.path.join(out_dir, f"{stem}.{ext}")
        fig.savefig(out_path)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
