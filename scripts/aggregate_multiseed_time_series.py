#!/usr/bin/env python3
"""
Aggregate per-seed adoption time series into mean/percentile band data.
"""

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd


SCENARIOS = {
    "deterministic_rational": [
        "deterministic_rational_adoption_rate_pct",
        "deterministic_rational_adoption_rate",
        "deterministic_rational_adoptionrate",
    ],
    "herding": [
        "herding_adoption_rate_pct",
        "herding_adoption_rate",
        "herding_adoptionrate",
    ],
    "loss_aversion": [
        "loss_aversion_adoption_rate_pct",
        "loss_aversion_adoption_rate",
        "loss_aversion_adoptionrate",
    ],
    "all_biases": [
        "all_biases_adoption_rate_pct",
        "all_biases_adoption_rate",
        "all_biases_adoptionrate",
    ],
}


def norm(s: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in str(s))


def discover_seed_dirs(results_dir: Path):
    return sorted(
        [p for p in results_dir.iterdir() if p.is_dir() and p.name.startswith("seed_")],
        key=lambda p: p.name,
    )


def find_first_matching_file(seed_dir: Path, pattern: str):
    matches = sorted(glob.glob(str(seed_dir / "**" / pattern), recursive=True))
    return Path(matches[0]) if matches else None


def detect_time_column(df: pd.DataFrame):
    cols = list(df.columns)
    cols_norm = {c: norm(c) for c in cols}

    # Priority: DecimalYear, else year, else year_range, else first column.
    for target in ("decimalyear", "year", "year_range"):
        for c in cols:
            if cols_norm[c] == target:
                return c
    return cols[0]


def detect_scenario_column(df: pd.DataFrame, scenario_key: str):
    cols = list(df.columns)
    cols_norm = {c: norm(c) for c in cols}

    # 1) Prefer exact known names.
    exact_candidates = SCENARIOS[scenario_key]
    for c in cols:
        if cols_norm[c] in exact_candidates:
            return c

    # 2) Fallback substring matching.
    def has(cn, *tokens):
        return all(t in cn for t in tokens)

    ranked = []
    for c in cols:
        cn = cols_norm[c]
        if scenario_key == "deterministic_rational":
            # Strong preference for deterministic+rational+adoption+rate;
            # then fallback to rational+adoption+rate.
            if has(cn, "deterministic", "rational", "adoption", "rate"):
                ranked.append((0, c))
            elif has(cn, "rational", "adoption", "rate"):
                ranked.append((1, c))
        elif scenario_key == "herding":
            if has(cn, "herding", "adoption", "rate"):
                ranked.append((0, c))
        elif scenario_key == "loss_aversion":
            if has(cn, "loss", "aversion", "adoption", "rate"):
                ranked.append((0, c))
        elif scenario_key == "all_biases":
            if has(cn, "all", "biases", "adoption", "rate"):
                ranked.append((0, c))

    if not ranked:
        return None
    ranked.sort(key=lambda x: (x[0], x[1]))
    return ranked[0][1]


def percentile(a, q):
    if len(a) == 0:
        return np.nan
    return float(np.percentile(a, q))


def aggregate_scenario(time_series_rows, scenario_name, low_q, high_q):
    """
    time_series_rows: list of tuples (seed, df_with_time_and_value_cols)
    """
    if not time_series_rows:
        return pd.DataFrame()

    # Build inner-joined table on time across seeds that have this scenario.
    merged = None
    seed_cols = []
    for seed, df in time_series_rows:
        df2 = df.copy()
        vcol = f"value_seed_{seed}"
        df2 = df2.rename(columns={"value": vcol})
        seed_cols.append(vcol)
        if merged is None:
            merged = df2[["time", vcol]]
        else:
            merged = merged.merge(df2[["time", vcol]], on="time", how="inner")

    if merged is None or merged.empty:
        return pd.DataFrame()

    values = merged[seed_cols].to_numpy(dtype=float)

    out = pd.DataFrame(
        {
            "time": merged["time"],
            "scenario": scenario_name,
            "mean": np.nanmean(values, axis=1),
            "median": np.nanmedian(values, axis=1),
            "p_low": [percentile(row[~np.isnan(row)], low_q) for row in values],
            "p_high": [percentile(row[~np.isnan(row)], high_q) for row in values],
            "p25": [percentile(row[~np.isnan(row)], 25) for row in values],
            "p75": [percentile(row[~np.isnan(row)], 75) for row in values],
            "n": np.sum(~np.isnan(values), axis=1).astype(int),
        }
    )

    # Add convenient alternate time axes when time is numeric decimal years.
    time_numeric = pd.to_numeric(out["time"], errors="coerce")
    if not time_numeric.isna().all():
        # DecimalYear convention in this project: 1.0, 1+1/12, ...
        step = np.rint((time_numeric - 1.0) * 12.0).astype("Int64")
        out["step"] = step
        out["year"] = ((step // 12) + 1).astype("Int64")
        out["month"] = ((step % 12) + 1).astype("Int64")
    else:
        out["step"] = pd.Series([pd.NA] * len(out), dtype="Int64")
        out["year"] = pd.Series([pd.NA] * len(out), dtype="Int64")
        out["month"] = pd.Series([pd.NA] * len(out), dtype="Int64")

    return out


def main():
    parser = argparse.ArgumentParser(description="Aggregate multiseed adoption time series.")
    parser.add_argument("--results-dir", required=True, type=str)
    parser.add_argument("--pattern", required=True, type=str)
    parser.add_argument("--low", type=float, default=5.0)
    parser.add_argument("--high", type=float, default=95.0)
    parser.add_argument("--out-csv", required=True, type=str)
    args = parser.parse_args()

    if args.low >= args.high:
        raise ValueError("--low must be less than --high")

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        raise FileNotFoundError(f"Results dir not found: {results_dir}")

    seed_dirs = discover_seed_dirs(results_dir)
    if not seed_dirs:
        raise FileNotFoundError(f"No seed_* directories found in {results_dir}")

    # scenario -> list[(seed, df(time,value))]
    scenario_data = {k: [] for k in SCENARIOS.keys()}
    total_seed_count = 0

    for seed_dir in seed_dirs:
        seed_str = seed_dir.name.replace("seed_", "")
        try:
            seed = int(seed_str)
        except ValueError:
            print(f"[WARN] Skipping invalid seed directory name: {seed_dir.name}")
            continue

        ts_file = find_first_matching_file(seed_dir, args.pattern)
        if ts_file is None:
            print(f"[WARN] No file matching '{args.pattern}' under {seed_dir}; skipping seed.")
            continue

        df = pd.read_csv(ts_file)
        if df.empty:
            print(f"[WARN] Empty time series file for seed {seed}: {ts_file}; skipping seed.")
            continue

        total_seed_count += 1

        time_col = detect_time_column(df)
        time_series = pd.to_numeric(df[time_col], errors="coerce")
        if time_series.isna().all():
            # Keep original values if non-numeric time axis (e.g., year_range string)
            time_vals = df[time_col].astype(str)
        else:
            time_vals = time_series

        for scenario in SCENARIOS.keys():
            scol = detect_scenario_column(df, scenario)
            if scol is None:
                print(f"[WARN] Seed {seed}: scenario column not found for '{scenario}' in {ts_file.name}")
                continue
            vals = pd.to_numeric(df[scol], errors="coerce")
            sdf = pd.DataFrame({"time": time_vals, "value": vals}).dropna(subset=["time", "value"])
            scenario_data[scenario].append((seed, sdf))

    if total_seed_count == 0:
        raise RuntimeError("No valid seed time series files were found.")

    outputs = []
    for scenario, rows in scenario_data.items():
        if len(rows) == 0:
            print(f"[WARN] No usable data for scenario '{scenario}'.")
            continue
        if len(rows) < 3:
            print(f"[WARN] Scenario '{scenario}' uses only {len(rows)} seed(s) (<3).")

        out = aggregate_scenario(rows, scenario, args.low, args.high)
        if out.empty:
            print(f"[WARN] Scenario '{scenario}' produced empty output after time alignment.")
            continue

        min_n = int(out["n"].min())
        if min_n < 3:
            print(f"[WARN] Scenario '{scenario}' has n<3 at some timepoints (min n={min_n}).")
        outputs.append(out)

    if not outputs:
        raise RuntimeError("No scenario output could be generated.")

    final = pd.concat(outputs, ignore_index=True)
    final = final.sort_values(["scenario", "time"]).reset_index(drop=True)

    # Preferred column order with multiple x-axis options.
    ordered_cols = [
        "time", "step", "year", "month", "scenario",
        "mean", "median", "p_low", "p_high", "p25", "p75", "n"
    ]
    final = final[[c for c in ordered_cols if c in final.columns]]

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(out_csv, index=False)
    print(f"[OK] Wrote aggregated band data: {out_csv}")
    print(f"[OK] Rows: {len(final)}")


if __name__ == "__main__":
    main()
