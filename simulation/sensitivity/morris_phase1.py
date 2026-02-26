"""Phase-1 Morris screening with CRN, replications, and resume checkpoints."""

from __future__ import annotations

import json
import os
import random
import time
import traceback
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from simulation.models.multi_experiment_model import MultiExperimentModel
from simulation.utils.config_loader import create_multi_experiment_config

try:
    from SALib.sample.morris import sample as morris_sample
    from SALib.analyze.morris import analyze as morris_analyze
except ImportError as exc:
    raise ImportError(
        "SALib is required for Morris phase-1 sensitivity. Install with `pip install SALib`."
    ) from exc


PARAMETER_NAMES: List[str] = [
    "C_fixed",
    "C_variable",
    "r_learn",
    "Y",
    "P_f0",
    "g_f",
    "alpha_feed",
    "discount_rate",
    "beta_elasticity",
    "E_base",
]

PARAMETER_BOUNDS: List[List[float]] = [
    [3500.0, 8000.0],
    [1400.0, 2800.0],
    [0.02, 0.08],
    [15.0, 25.0],
    [0.10, 0.25],
    [0.00, 0.05],
    [0.30, 1.00],
    [0.03, 0.07],
    [0.2, 0.7],
    [10.0, 25.0],
]

PROBLEM: Dict[str, object] = {
    "num_vars": len(PARAMETER_NAMES),
    "names": PARAMETER_NAMES,
    "bounds": PARAMETER_BOUNDS,
}

DEFAULT_OUTPUT_DIR = "outputs"
POINTS_CSV = "morris_phase1_points.csv"
ERROR_LOG = "morris_phase1_errors.log"
SAMPLE_NPY = "morris_phase1_sample.npy"
PROBLEM_JSON = "morris_phase1_problem.json"
T50_OUT = "morris_phase1_T50_indices.csv"
T90_OUT = "morris_phase1_T90_indices.csv"


def _point_row_columns() -> List[str]:
    return [
        "point_id",
        "seed_list",
        *PARAMETER_NAMES,
        "T50_mean",
        "T90_mean",
        "T50_sd",
        "T90_sd",
        "runtime_seconds",
        "timestamp",
    ]


def _ensure_output_dir(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)


def _save_static_problem(output_dir: str) -> None:
    problem_path = os.path.join(output_dir, PROBLEM_JSON)
    if not os.path.exists(problem_path):
        with open(problem_path, "w", encoding="utf-8") as fh:
            json.dump(PROBLEM, fh, indent=2)


def _load_or_create_sample(
    output_dir: str,
    r: int,
    p: int,
    grid_jump: int,
    sample_seed: int,
) -> np.ndarray:
    sample_path = os.path.join(output_dir, SAMPLE_NPY)
    if os.path.exists(sample_path):
        return np.load(sample_path)

    try:
        x = morris_sample(
            PROBLEM,
            N=r,
            num_levels=p,
            grid_jump=grid_jump,
            seed=sample_seed,
        )
    except TypeError:
        try:
            # SALib variants without grid_jump in sample()
            x = morris_sample(
                PROBLEM,
                N=r,
                num_levels=p,
                seed=sample_seed,
            )
        except TypeError:
            # Older variants without seed in sample()
            np.random.seed(sample_seed)
            x = morris_sample(
                PROBLEM,
                N=r,
                num_levels=p,
            )

    np.save(sample_path, x)
    return x


def _load_points_df(points_path: str) -> pd.DataFrame:
    if not os.path.exists(points_path):
        return pd.DataFrame(columns=_point_row_columns())

    df = pd.read_csv(points_path)
    if df.empty:
        return pd.DataFrame(columns=_point_row_columns())

    if "point_id" not in df.columns:
        raise ValueError(f"Checkpoint file is missing 'point_id': {points_path}")

    df["point_id"] = df["point_id"].astype(int)
    deduped = df.drop_duplicates(subset=["point_id"], keep="last").sort_values("point_id")

    if len(deduped) != len(df):
        deduped.to_csv(points_path, index=False)

    return deduped


def _append_point_row(points_path: str, row: Dict[str, object]) -> None:
    row_df = pd.DataFrame([row], columns=_point_row_columns())
    header = not os.path.exists(points_path)
    row_df.to_csv(points_path, mode="a", header=header, index=False)


def _log_error(output_dir: str, point_id: int, seed: int, exc: BaseException) -> None:
    error_path = os.path.join(output_dir, ERROR_LOG)
    now = datetime.utcnow().isoformat(timespec="seconds")
    with open(error_path, "a", encoding="utf-8") as fh:
        fh.write(f"[{now}] point_id={point_id} seed={seed}\n")
        fh.write("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
        fh.write("\n")


def _build_overrides(params: Dict[str, float], seed: int, n_agents: int) -> Dict[str, object]:
    return {
        "num_households": n_agents,
        "consumption_params": {
            "income_elasticity": float(params["beta_elasticity"]),
            "base_consumption": float(params["E_base"]),
        },
        "solar_params": {
            "fixed_cost": float(params["C_fixed"]),
            "variable_cost_per_kw": float(params["C_variable"]),
            "annual_cost_reduction": float(params["r_learn"]),
            "lifetime_years": int(round(params["Y"])),
            "discount_rate": float(params["discount_rate"]),
        },
        "grid_params": {
            "initial_fossil_price": float(params["P_f0"]),
            "fossil_annual_increase": float(params["g_f"]),
            "feed_in_factor": float(params["alpha_feed"]),
        },
        "run_settings": {"random_seed": int(seed)},
    }


def _apply_overrides(base_config: Dict[str, object], overrides: Dict[str, object]) -> Dict[str, object]:
    merged = json.loads(json.dumps(base_config))

    def _merge(dst: Dict[str, object], src: Dict[str, object]) -> None:
        for key, value in src.items():
            if isinstance(value, dict) and isinstance(dst.get(key), dict):
                _merge(dst[key], value)  # type: ignore[index]
            else:
                dst[key] = value

    _merge(merged, overrides)
    return merged


def _extract_threshold_time(
    system_df: pd.DataFrame, threshold: float, adoption_col: str = "rational_AdoptionRate"
) -> float:
    if adoption_col not in system_df.columns:
        return np.nan
    series = pd.to_numeric(system_df[adoption_col], errors="coerce").to_numpy(dtype=float)
    hit_idx = np.where(series >= threshold)[0]
    if len(hit_idx) == 0:
        return np.nan
    return float(int(hit_idx[0]))


def _run_single_seed(params: Dict[str, float], seed: int, n_agents: int) -> Tuple[float, float, int]:
    random.seed(seed)
    np.random.seed(seed)

    base_config = create_multi_experiment_config()
    overrides = _build_overrides(params=params, seed=seed, n_agents=n_agents)
    config = _apply_overrides(base_config, overrides)

    model = MultiExperimentModel(config)
    steps = int(config.get("steps", 240))
    model.run(steps=steps)

    system_df = model.data_collector.get_system_metrics_dataframe()
    t50 = _extract_threshold_time(system_df, threshold=0.5)
    t90 = _extract_threshold_time(system_df, threshold=0.9)
    return t50, t90, steps


def _save_indices_csv(problem: Dict[str, object], si: Dict[str, np.ndarray], out_path: str) -> None:
    data = {"parameter": problem["names"]}  # type: ignore[index]
    for key in ["mu", "mu_star", "sigma", "mu_star_conf"]:
        if key in si:
            data[key] = si[key]
    pd.DataFrame(data).to_csv(out_path, index=False)


def _analyze_morris(
    output_dir: str,
    x: np.ndarray,
    p: int,
    grid_jump: int,
    horizon_cap: int,
) -> None:
    points_path = os.path.join(output_dir, POINTS_CSV)
    df = _load_points_df(points_path).sort_values("point_id")
    expected_points = x.shape[0]

    if len(df) != expected_points:
        raise RuntimeError(
            f"Cannot analyze Morris indices: completed points={len(df)} but expected={expected_points}."
        )

    y_t50 = pd.to_numeric(df["T50_mean"], errors="coerce").to_numpy(dtype=float)
    y_t90 = pd.to_numeric(df["T90_mean"], errors="coerce").to_numpy(dtype=float)

    y_t50 = np.where(np.isnan(y_t50), float(horizon_cap), y_t50)
    y_t90 = np.where(np.isnan(y_t90), float(horizon_cap), y_t90)

    try:
        si_t50 = morris_analyze(
            PROBLEM,
            x,
            y_t50,
            conf_level=0.95,
            print_to_console=False,
            num_levels=p,
            grid_jump=grid_jump,
        )
        si_t90 = morris_analyze(
            PROBLEM,
            x,
            y_t90,
            conf_level=0.95,
            print_to_console=False,
            num_levels=p,
            grid_jump=grid_jump,
        )
    except TypeError:
        si_t50 = morris_analyze(
            PROBLEM,
            x,
            y_t50,
            conf_level=0.95,
            print_to_console=False,
            num_levels=p,
        )
        si_t90 = morris_analyze(
            PROBLEM,
            x,
            y_t90,
            conf_level=0.95,
            print_to_console=False,
            num_levels=p,
        )

    _save_indices_csv(PROBLEM, si_t50, os.path.join(output_dir, T50_OUT))
    _save_indices_csv(PROBLEM, si_t90, os.path.join(output_dir, T90_OUT))


def run_morris_phase1(
    output_dir: str = DEFAULT_OUTPUT_DIR,
    sample_seed: int = 12345,
    seeds: List[int] | None = None,
    p: int = 4,
    grid_jump: int = 2,
    r: int = 9,
    n_agents: int = 100,
) -> None:
    """Run/resume Morris phase-1 sensitivity and export SALib indices."""
    if seeds is None:
        seeds = [101, 202, 303]

    _ensure_output_dir(output_dir)
    _save_static_problem(output_dir)

    x = _load_or_create_sample(
        output_dir=output_dir, r=r, p=p, grid_jump=grid_jump, sample_seed=sample_seed
    )

    points_path = os.path.join(output_dir, POINTS_CSV)
    existing_df = _load_points_df(points_path)
    completed_ids = set(existing_df["point_id"].astype(int).tolist()) if not existing_df.empty else set()

    total_points = x.shape[0]
    horizon_steps = 240

    for point_id in range(total_points):
        if point_id in completed_ids:
            continue

        point_start = time.time()
        point_values = [float(v) for v in x[point_id]]
        params = dict(zip(PARAMETER_NAMES, point_values))

        print(f"[Morris] point {point_id + 1}/{total_points}")

        t50_runs: List[float] = []
        t90_runs: List[float] = []

        for seed in seeds:
            try:
                t50, t90, steps = _run_single_seed(params=params, seed=seed, n_agents=n_agents)
                t50_runs.append(t50)
                t90_runs.append(t90)
                horizon_steps = max(horizon_steps, int(steps))
            except Exception as exc:
                _log_error(output_dir=output_dir, point_id=point_id, seed=seed, exc=exc)

        row = {
            "point_id": int(point_id),
            "seed_list": json.dumps(seeds),
            **{k: float(params[k]) for k in PARAMETER_NAMES},
            "T50_mean": float(np.nanmean(t50_runs)) if len(t50_runs) else np.nan,
            "T90_mean": float(np.nanmean(t90_runs)) if len(t90_runs) else np.nan,
            "T50_sd": float(np.nanstd(t50_runs, ddof=1)) if len(t50_runs) > 1 else np.nan,
            "T90_sd": float(np.nanstd(t90_runs, ddof=1)) if len(t90_runs) > 1 else np.nan,
            "runtime_seconds": round(time.time() - point_start, 3),
            "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
        }

        _append_point_row(points_path, row)
        completed_ids.add(point_id)

    _analyze_morris(
        output_dir=output_dir,
        x=x,
        p=p,
        grid_jump=grid_jump,
        horizon_cap=horizon_steps + 1,
    )


if __name__ == "__main__":
    run_morris_phase1()
