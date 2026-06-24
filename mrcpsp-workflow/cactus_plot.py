"""Cactus plot: instances solved vs. time limit.

Each input file contains whitespace-separated finish times (seconds). For each
file, a step line is drawn: at time t the y-value is the number of finish
times <= t. The legend label is the file's basename.

Usage:
    python3 cactus_plot.py [options] FILE [FILE ...]

Options:
    -o, --output PATH    Save plot to file (PNG/PDF/SVG by extension).
                         If omitted, opens an interactive window.
    --xlog               Log scale on x-axis.
    --ylog               Log scale on y-axis.
    --title TEXT         Plot title.
    --xlabel TEXT        Override x-axis label.
    --ylabel TEXT        Override y-axis label.
    --xmax FLOAT         Clip x-axis at this value.
    --no-stem            Do not draw the leading (0, 0) -> (t_min, 0) segment.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def read_times(path: Path) -> list[float]:
    tokens = path.read_text().split()
    return [float(t) for t in tokens]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("-o", "--output", type=Path, default=None)
    ap.add_argument("--xlog", action="store_true")
    ap.add_argument("--ylog", action="store_true")
    ap.add_argument("--title", default=None)
    ap.add_argument("--xlabel", default="Time (seconds)")
    ap.add_argument("--ylabel", default="Instances solved")
    ap.add_argument("--xmax", type=float, default=None)
    ap.add_argument("--no-stem", action="store_true")
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(8, 5))

    for path in args.files:
        if not path.exists():
            print(f"warning: {path} not found, skipping", file=sys.stderr)
            continue
        times = sorted(read_times(path))
        if not times:
            print(f"warning: {path} contains no values, skipping", file=sys.stderr)
            continue
        if args.no_stem:
            xs = list(times)
            ys = list(range(1, len(times) + 1))
        else:
            xs = [0.0] + times
            ys = list(range(0, len(times) + 1))
        ax.step(xs, ys, where="post", label=path.name)

    ax.set_xlabel(args.xlabel)
    ax.set_ylabel(args.ylabel)
    if args.xlog:
        ax.set_xscale("log")
    if args.ylog:
        ax.set_yscale("log")
    if args.xmax is not None:
        ax.set_xlim(right=args.xmax)
    if args.title:
        ax.set_title(args.title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if args.output:
        fig.savefig(args.output, bbox_inches="tight", dpi=120)
        print(f"wrote {args.output}", file=sys.stderr)
    else:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
