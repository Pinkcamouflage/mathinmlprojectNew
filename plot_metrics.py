"""
Plot key training metrics from a LISR training.csv:
  - best_ea_fitness       (best evolutionary actor, extrinsic return)
  - best_learner_fitness  (best SR learner, extrinsic return)
  - eval_return           (deterministic eval of the champion learner)

all against environment frames.

Usage:
    python plot_metrics.py                          # reads bestRun/training.csv
    python plot_metrics.py path/to/training.csv
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

SMOOTH_WINDOW = 20  # rolling-average window (generations); set to 1 to disable

METRICS = {
    "best_ea_fitness":      "Best EA actor",
    "best_learner_fitness": "Best SR learner",
    "eval_return":          "Eval return (champion)",
}


def _smooth(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1, center=True).mean()


def plot(csv_path: str):
    df     = pd.read_csv(csv_path)
    frames = df["frames"] / 1e6
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, (col, label) in enumerate(METRICS.items()):
        if col not in df.columns:
            print(f"  (skipping missing column: {col})")
            continue
        color = colors[i % len(colors)]
        ax.plot(frames, _smooth(df[col], SMOOTH_WINDOW), color=color, linewidth=2, label=label)
        ax.plot(frames, df[col], color=color, linewidth=0.5, alpha=0.25)

    ax.set_title(f"LISR training metrics  ({Path(csv_path).parent.name or Path(csv_path).stem})")
    ax.set_xlabel("Frames (×10⁶)")
    ax.set_ylabel("Extrinsic episodic return")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = Path(csv_path).parent / "metrics_plot.png"
    plt.savefig(out, dpi=150)
    print(f"Saved → {out}")
    plt.show()


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "bestRun/training.csv"
    plot(path)
