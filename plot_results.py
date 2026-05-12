"""
Usage:
    python plot_results.py                          # reads lisr_logs/training.csv
    python plot_results.py path/to/training.csv
    python plot_results.py run1.csv run2.csv        # overlay multiple runs
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SMOOTH_WINDOW = 20   # rolling-average window (generations); set to 1 to disable


def _smooth(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1, center=True).mean()


def plot(csv_paths: list[str]):
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, path in enumerate(csv_paths):
        df     = pd.read_csv(path)
        frames = df["frames"] / 1e6
        color  = colors[i % len(colors)]
        label  = Path(path).parent.name or Path(path).stem

        eval_s = _smooth(df["eval_return"], SMOOTH_WINDOW)
        ax.plot(frames, eval_s, color=color, linewidth=2, label=label)
        ax.plot(frames, df["eval_return"], color=color, linewidth=0.5, alpha=0.3)

    ax.set_title("LISR — Average Return (best learner, deterministic, 5 episodes)")
    ax.set_xlabel("Frames (×10⁶)")
    ax.set_ylabel("Average Return")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = Path(csv_paths[0]).parent / "return_plot.png"
    plt.savefig(out, dpi=150)
    print(f"Saved → {out}")
    plt.show()


if __name__ == "__main__":
    paths = sys.argv[1:] or ["lisr_logs/training.csv"]
    plot(paths)
