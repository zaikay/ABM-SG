#!/usr/bin/env python
"""Generate final Sobol figures from full-scale index CSV files."""

from __future__ import annotations

import argparse
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ["parameter", "S1", "ST", "S1_conf", "ST_conf"]


def _validate_columns(df: pd.DataFrame, label: str) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _load_indices(path_t50: str, path_t90: str) -> tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    t50 = pd.read_csv(path_t50)
    t90 = pd.read_csv(path_t90)
    _validate_columns(t50, "indices_T50.csv")
    _validate_columns(t90, "indices_T90.csv")

    order = t50.sort_values("ST", ascending=False)["parameter"].tolist()
    t90_params = set(t90["parameter"].tolist())
    missing_in_t90 = [p for p in order if p not in t90_params]
    if missing_in_t90:
        raise ValueError(f"T90 file missing parameters found in T50: {missing_in_t90}")
    return t50, t90, order


def _prepare(df: pd.DataFrame, order: List[str]) -> pd.DataFrame:
    data = df.set_index("parameter").loc[order].reset_index()
    data["interaction"] = (data["ST"] - data["S1"]).clip(lower=0.0)
    return data


def _plot_stacked(df: pd.DataFrame, title: str, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    y = np.arange(len(df))
    ax.barh(y, df["S1"], label="Direct effect (S1)")
    ax.barh(y, df["interaction"], left=df["S1"], label="Interaction contribution (ST - S1)")
    ax.set_yticks(y)
    ax.set_yticklabels(df["parameter"])
    ax.invert_yaxis()
    ax.set_xlabel("Sobol index value")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def _plot_scatter(df: pd.DataFrame, title: str, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    x = df["S1"].to_numpy(dtype=float)
    y = df["ST"].to_numpy(dtype=float)
    labels = df["parameter"].tolist()

    ax.scatter(x, y)
    for xi, yi, label in zip(x, y, labels):
        ax.annotate(label, (xi, yi), xytext=(4, 3), textcoords="offset points")

    max_st = float(np.nanmax(y))
    lim_hi = max(0.05, max_st * 1.08)
    ax.plot([0.0, lim_hi], [0.0, lim_hi], linestyle="--")
    ax.set_xlim(0.0, lim_hi)
    ax.set_ylim(0.0, lim_hi)
    ax.set_xlabel("S1")
    ax.set_ylabel("ST")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def _save_summary_table(t50: pd.DataFrame, t90: pd.DataFrame, order: List[str], out_path: str) -> None:
    a = t50.set_index("parameter").loc[order]
    b = t90.set_index("parameter").loc[order]
    out = pd.DataFrame(
        {
            "parameter": order,
            "S1_T50": a["S1"].to_numpy(dtype=float),
            "ST_T50": a["ST"].to_numpy(dtype=float),
            "S1_T90": b["S1"].to_numpy(dtype=float),
            "ST_T90": b["ST"].to_numpy(dtype=float),
        }
    )
    out["Interaction_T50"] = out["ST_T50"] - out["S1_T50"]
    out["Interaction_T90"] = out["ST_T90"] - out["S1_T90"]
    out.to_csv(out_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate final Sobol figures and summary table")
    parser.add_argument("--indices_t50", type=str, default="results/sobol/full_N512/indices_T50.csv")
    parser.add_argument("--indices_t90", type=str, default="results/sobol/full_N512/indices_T90.csv")
    parser.add_argument("--outdir", type=str, default="results/sobol/final_figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    t50, t90, order = _load_indices(args.indices_t50, args.indices_t90)
    t50o = _prepare(t50, order)
    t90o = _prepare(t90, order)

    _plot_stacked(t50o, "Sobol Variance Decomposition (T50)", os.path.join(args.outdir, "sobol_stacked_T50.png"))
    _plot_stacked(t90o, "Sobol Variance Decomposition (T90)", os.path.join(args.outdir, "sobol_stacked_T90.png"))

    _plot_scatter(
        t50o,
        "Sobol First-Order vs Total-Effect Indices (T50)",
        os.path.join(args.outdir, "sobol_scatter_T50.png"),
    )
    _plot_scatter(
        t90o,
        "Sobol First-Order vs Total-Effect Indices (T90)",
        os.path.join(args.outdir, "sobol_scatter_T90.png"),
    )

    _save_summary_table(t50, t90, order, os.path.join(args.outdir, "sobol_summary_table.csv"))
    print(f"Saved figures and summary table to: {args.outdir}")


if __name__ == "__main__":
    main()

