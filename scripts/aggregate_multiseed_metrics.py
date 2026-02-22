#!/usr/bin/env python3
"""
Aggregate key metrics across multi-seed runs.

Expected per-seed folder layout:
  <results_dir>/seed_0001/data/metrics/*.csv
"""

import argparse
import glob
import math
import os
import re

import numpy as np
import pandas as pd

SCENARIOS = [
    "deterministic_rational",
    "herding",
    "loss_aversion",
    "all_biases",
]

METRICS = ["T50", "T90", "final_adoption", "cumulative_gap"]


def parse_seed_spec(seed_spec):
    """Parse seed specification like '1,2,3' or '1-20'."""
    if not seed_spec:
        return []

    seeds = []
    for part in seed_spec.split(","):
        token = part.strip()
        if not token:
            continue

        range_match = re.match(r"^(\d+)\s*-\s*(\d+)$", token)
        if range_match:
            start = int(range_match.group(1))
            end = int(range_match.group(2))
            if end < start:
                raise ValueError(f"Invalid seed range '{token}': end < start")
            seeds.extend(range(start, end + 1))
        else:
            seeds.append(int(token))

    # Preserve order, deduplicate.
    unique = []
    seen = set()
    for seed in seeds:
        if seed not in seen:
            unique.append(seed)
            seen.add(seed)
    return unique


def find_single_file(seed_dir, pattern):
    matches = glob.glob(os.path.join(seed_dir, "**", pattern), recursive=True)
    if not matches:
        return None
    matches.sort(key=lambda p: (p.count(os.sep), p))
    return matches[0]


def discover_seed_dirs(results_dir, seeds=None):
    if seeds:
        seed_dirs = []
        missing = []
        for seed in seeds:
            path = os.path.join(results_dir, f"seed_{seed:04d}")
            if os.path.isdir(path):
                seed_dirs.append((seed, path))
            else:
                missing.append(path)
        if missing:
            print("Warning: missing seed directories:")
            for path in missing:
                print(f"  - {path}")
        return seed_dirs

    discovered = []
    for path in sorted(glob.glob(os.path.join(results_dir, "seed_*"))):
        if not os.path.isdir(path):
            continue
        m = re.search(r"seed_(\d+)", os.path.basename(path))
        if not m:
            continue
        discovered.append((int(m.group(1)), path))
    return discovered


def extract_seed_metrics(seed, seed_dir):
    critical_mass_file = find_single_file(seed_dir, "critical_mass_timing*.csv")
    summary_file = find_single_file(seed_dir, "scenario_comparison_summary*.csv")
    area_file = find_single_file(seed_dir, "area_analysis*.csv")
    adoption_ts_file = find_single_file(seed_dir, "adoption_time_series*.csv")

    missing = [p for p in [critical_mass_file, summary_file, area_file, adoption_ts_file] if p is None]
    if missing:
        raise FileNotFoundError(f"Required metric files not found under {seed_dir}")

    critical_mass = pd.read_csv(critical_mass_file)
    summary = pd.read_csv(summary_file)
    area = pd.read_csv(area_file)
    adoption_ts = pd.read_csv(adoption_ts_file)

    rows = []
    for scenario in SCENARIOS:
        row = {"seed": seed, "scenario": scenario}

        t50 = critical_mass[
            (critical_mass["scenario"] == scenario) & (critical_mass["threshold_pct"] == 50)
        ]
        t90 = critical_mass[
            (critical_mass["scenario"] == scenario) & (critical_mass["threshold_pct"] == 90)
        ]
        row["T50"] = float(t50.iloc[0]["threshold_year"]) if not t50.empty else np.nan
        row["T90"] = float(t90.iloc[0]["threshold_year"]) if not t90.empty else np.nan

        s = summary[summary["scenario"] == scenario]
        if not s.empty:
            row["final_adoption"] = float(s.iloc[0]["final_adoption_rate_pct"])
        else:
            # Fallback to adoption_time_series final point.
            col = f"{scenario}_adoption_rate_pct"
            row["final_adoption"] = float(adoption_ts.iloc[-1][col]) if col in adoption_ts.columns else np.nan

        a = area[area["scenario"] == scenario]
        row["cumulative_gap"] = float(a.iloc[0]["cumulative_adoption_gap"]) if not a.empty else np.nan

        rows.append(row)

    return rows


def summarize_metric(values):
    values = pd.Series(values).dropna()
    n = len(values)
    if n == 0:
        return {
            "n": 0, "mean": np.nan, "sd": np.nan, "min": np.nan, "max": np.nan,
            "ci95_lower": np.nan, "ci95_upper": np.nan
        }

    mean = float(values.mean())
    sd = float(values.std(ddof=1)) if n > 1 else 0.0
    ci = 1.96 * sd / math.sqrt(n) if n > 1 else 0.0
    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "min": float(values.min()),
        "max": float(values.max()),
        "ci95_lower": mean - ci,
        "ci95_upper": mean + ci,
    }


def build_summary(seed_metrics_df):
    rows = []
    for scenario in SCENARIOS:
        subset = seed_metrics_df[seed_metrics_df["scenario"] == scenario]
        for metric in METRICS:
            stats = summarize_metric(subset[metric])
            rows.append({
                "scenario": scenario,
                "metric": metric,
                **stats,
            })
    return pd.DataFrame(rows)


def infer_n_label(results_dir):
    m = re.search(r"N(\d+)", os.path.basename(os.path.normpath(results_dir)))
    if m:
        return f"N{m.group(1)}"
    return "Nunknown"


def main():
    parser = argparse.ArgumentParser(description="Aggregate multi-seed ABM metrics.")
    parser.add_argument("--results-dir", type=str, required=True,
                        help="Base directory containing seed_* folders (e.g., results/N100/).")
    parser.add_argument("--seeds", type=str, default=None,
                        help="Optional seed list/range (e.g., 1,2,3 or 1-20).")
    args = parser.parse_args()

    seeds = parse_seed_spec(args.seeds) if args.seeds else []
    seed_dirs = discover_seed_dirs(args.results_dir, seeds=seeds if seeds else None)

    if not seed_dirs:
        raise FileNotFoundError(f"No seed directories found in {args.results_dir}")

    per_seed_rows = []
    for seed, seed_dir in seed_dirs:
        per_seed_rows.extend(extract_seed_metrics(seed, seed_dir))

    seed_metrics_df = pd.DataFrame(per_seed_rows)
    summary_df = build_summary(seed_metrics_df)

    n_label = infer_n_label(args.results_dir)
    out_file = os.path.join(args.results_dir, f"multiseed_summary_{n_label}.csv")
    summary_df.to_csv(out_file, index=False)

    print(f"Processed seeds: {[seed for seed, _ in seed_dirs]}")
    print(f"Saved summary: {out_file}")


if __name__ == "__main__":
    main()
