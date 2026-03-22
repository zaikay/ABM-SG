#!/usr/bin/env python3
"""
Plot multi-seed mean adoption trajectory with percentile band per scenario.
"""

import argparse
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


SCENARIO_ORDER = [
    "deterministic_rational",
    "herding",
    "loss_aversion",
    "all_biases",
]

SCENARIO_TITLE = {
    "deterministic_rational": "Deterministic rational",
    "herding": "Herding",
    "loss_aversion": "Loss aversion",
    "all_biases": "All biases combined",
}


def infer_n_label(*paths):
    for p in paths:
        m = re.search(r"N(\d+)", str(p))
        if m:
            return f"N{m.group(1)}"
    return "N100"


def plot_one_scenario(df, scenario, out_dir: Path, n_label: str, low_label: str, high_label: str):
    sdf = df[df["scenario"] == scenario].copy()
    if sdf.empty:
        print(f"[WARN] No rows for scenario '{scenario}', skipping plot.")
        return

    if "time" not in sdf.columns:
        raise ValueError("Input CSV must contain 'time' column.")
    if "mean" not in sdf.columns:
        raise ValueError("Input CSV must contain 'mean' column.")
    if low_label not in sdf.columns or high_label not in sdf.columns:
        raise ValueError(f"Input CSV must contain '{low_label}' and '{high_label}' columns.")

    # Ensure numeric sorting/plotting.
    sdf["time_num"] = pd.to_numeric(sdf["time"], errors="coerce")
    if sdf["time_num"].isna().any():
        raise ValueError(f"Scenario '{scenario}' has non-numeric values in 'time' column.")
    sdf = sdf.sort_values("time_num")

    x = sdf["time_num"].to_numpy()
    y_mean = pd.to_numeric(sdf["mean"], errors="coerce").to_numpy()
    y_low = pd.to_numeric(sdf[low_label], errors="coerce").to_numpy()
    y_high = pd.to_numeric(sdf[high_label], errors="coerce").to_numpy()

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(x, y_mean, linewidth=2.0, label="Mean adoption")
    ax.fill_between(x, y_low, y_high, alpha=0.2, label=f"Band ({low_label}-{high_label})")

    ax.set_ylim(0, 100)
    ax.set_xlabel("Year")
    ax.set_ylabel("Adoption rate (%)")
    ax.set_title(f"Multi-seed robustness ({n_label}): {SCENARIO_TITLE.get(scenario, scenario)}")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    out_base = out_dir / f"multiseed_band_{scenario}_{n_label}"
    fig.tight_layout()
    fig.savefig(f"{out_base}.png", dpi=300)
    fig.savefig(f"{out_base}.pdf", dpi=300)
    plt.close(fig)
    print(f"[OK] Wrote: {out_base}.png/.pdf")


def main():
    parser = argparse.ArgumentParser(description="Plot multiseed adoption mean ± percentile band.")
    parser.add_argument("--in-csv", required=True, type=str)
    parser.add_argument("--out-dir", required=True, type=str)
    parser.add_argument("--low-label", default="p_low", type=str)
    parser.add_argument("--high-label", default="p_high", type=str)
    args = parser.parse_args()

    in_csv = Path(args.in_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {in_csv}")

    df = pd.read_csv(in_csv)
    if df.empty:
        raise ValueError(f"Input CSV is empty: {in_csv}")

    n_label = infer_n_label(str(in_csv), str(out_dir))

    for scenario in SCENARIO_ORDER:
        plot_one_scenario(
            df=df,
            scenario=scenario,
            out_dir=out_dir,
            n_label=n_label,
            low_label=args.low_label,
            high_label=args.high_label,
        )


if __name__ == "__main__":
    main()
