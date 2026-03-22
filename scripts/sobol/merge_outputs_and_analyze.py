#!/usr/bin/env python
"""Merge chunked Sobol outputs and compute final indices."""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd

try:
    from SALib.analyze.sobol import analyze as sobol_analyze
except ImportError as exc:
    raise ImportError("SALib is required. Install with `pip install SALib`.") from exc


def _build_indices_dataframe(problem: Dict[str, object], si: Dict[str, np.ndarray]) -> pd.DataFrame:
    data: Dict[str, object] = {"parameter": list(problem["names"])}  # type: ignore[index]
    for key in ["S1", "ST", "S1_conf", "ST_conf"]:
        if key in si:
            data[key] = si[key]
    return pd.DataFrame(data)


def _analyze(problem: Dict[str, object], y: np.ndarray) -> pd.DataFrame:
    si = sobol_analyze(problem, y, calc_second_order=False, print_to_console=False)
    return _build_indices_dataframe(problem, si)


def _print_top_st(indices_df: pd.DataFrame, label: str, top_n: int = 5) -> None:
    if "ST" not in indices_df.columns:
        return
    top = indices_df.sort_values("ST", ascending=False).head(top_n)
    print(f"\nTop {top_n} ST parameters for {label}:")
    for _, row in top.iterrows():
        print(f"  {row['parameter']}: ST={float(row['ST']):.6f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge chunk outputs and run Sobol analysis")
    parser.add_argument("--outdir", required=True, type=str, help="Run directory containing chunk CSVs")
    parser.add_argument(
        "--pattern",
        type=str,
        default="outputs_chunk_*.csv",
        help="Glob pattern for chunk files (relative to outdir)",
    )
    parser.add_argument("--horizon", type=int, default=None, help="Override horizon for right-censor fallback")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    problem_path = os.path.join(args.outdir, "problem.json")
    if not os.path.exists(problem_path):
        raise FileNotFoundError(f"Missing problem.json in {args.outdir}")

    with open(problem_path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    problem = payload["problem"]
    expected_rows = int(problem["num_vars"])  # type: ignore[index]
    expected_rows = int(payload["n_base"]) * (expected_rows + 2)

    horizon = int(args.horizon if args.horizon is not None else payload.get("horizon", 240))
    censor_value = float(horizon + 1)

    pattern = os.path.join(args.outdir, args.pattern)
    files: List[str] = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")

    frames = [pd.read_csv(path) for path in files]
    merged = pd.concat(frames, ignore_index=True)
    if "sample_id" not in merged.columns:
        raise RuntimeError("Merged outputs do not contain sample_id")

    merged["sample_id"] = merged["sample_id"].astype(int)
    merged = merged.sort_values("sample_id").drop_duplicates(subset=["sample_id"], keep="last")

    merged_path = os.path.join(args.outdir, "outputs.csv")
    merged.to_csv(merged_path, index=False)
    print(f"Merged outputs written: {merged_path} (rows={len(merged)})")

    if len(merged) != expected_rows:
        raise RuntimeError(
            f"Cannot analyze yet: merged rows={len(merged)} expected={expected_rows}. Complete all chunks first."
        )

    y_t50 = pd.to_numeric(merged["T50_mean"], errors="coerce").to_numpy(dtype=float)
    y_t90 = pd.to_numeric(merged["T90_mean"], errors="coerce").to_numpy(dtype=float)
    y_t50 = np.where(np.isnan(y_t50), censor_value, y_t50)
    y_t90 = np.where(np.isnan(y_t90), censor_value, y_t90)

    idx_t50 = _analyze(problem, y_t50)
    idx_t90 = _analyze(problem, y_t90)
    t50_path = os.path.join(args.outdir, "indices_T50.csv")
    t90_path = os.path.join(args.outdir, "indices_T90.csv")
    idx_t50.to_csv(t50_path, index=False)
    idx_t90.to_csv(t90_path, index=False)
    _print_top_st(idx_t50, "T50")
    _print_top_st(idx_t90, "T90")
    print(f"Indices written: {t50_path}, {t90_path}")


if __name__ == "__main__":
    main()
