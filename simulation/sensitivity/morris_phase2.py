"""Phase-2 Morris screening for behavioral parameters with CRN and resume."""

from __future__ import annotations

import copy
import json
import os
import random
import time
import traceback
from contextlib import contextmanager
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from simulation.models.multi_experiment_model import MultiExperimentModel
from simulation.utils.config_loader import create_multi_experiment_config
from simulation.utils import parameters as sim_params

try:
    from SALib.sample.morris import sample as morris_sample
    from SALib.analyze.morris import analyze as morris_analyze
except ImportError as exc:
    raise ImportError(
        "SALib is required for Morris phase-2 sensitivity. Install with `pip install SALib`."
    ) from exc


PARAMETER_NAMES: List[str] = [
    "lambda_0",
    "beta_pb_lo",
    "beta_pb_hi",
    "gamma_SQ",
    "mu_omega",
    "phi",
    "w_spatial",
]

PARAMETER_BOUNDS: List[List[float]] = [
    [1.5, 3.0],
    [0.55, 0.70],
    [0.70, 0.85],
    [0.3, 0.9],
    [0.05, 0.20],
    [0.02, 0.20],
    [0.0, 1.0],
]

PROBLEM: Dict[str, object] = {
    "num_vars": len(PARAMETER_NAMES),
    "names": PARAMETER_NAMES,
    "bounds": PARAMETER_BOUNDS,
}

DEFAULT_OUTPUT_DIR = "outputs"
POINTS_CSV = "morris_phase2_points.csv"
ERROR_LOG = "morris_phase2_errors.log"
SAMPLE_NPY = "morris_phase2_sample.npy"
PROBLEM_JSON = "morris_phase2_problem.json"
T50_OUT = "morris_phase2_T50_indices.csv"
T90_OUT = "morris_phase2_T90_indices.csv"
TARGET_ADOPTION_COL = "all_biases_AdoptionRate"


def _point_row_columns() -> List[str]:
    return [
        "point_id",
        "seed_list",
        *PARAMETER_NAMES,
        "w_class",
        "beta_bounds_swapped",
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
            x = morris_sample(
                PROBLEM,
                N=r,
                num_levels=p,
                seed=sample_seed,
            )
        except TypeError:
            np.random.seed(sample_seed)
            x = morris_sample(PROBLEM, N=r, num_levels=p)

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


def _normalize_params(params: Dict[str, float]) -> Tuple[Dict[str, float], bool]:
    normalized = dict(params)
    swapped = False

    if normalized["beta_pb_hi"] < normalized["beta_pb_lo"]:
        normalized["beta_pb_lo"], normalized["beta_pb_hi"] = (
            normalized["beta_pb_hi"],
            normalized["beta_pb_lo"],
        )
        swapped = True

    normalized["w_class"] = 1.0 - float(normalized["w_spatial"])
    return normalized, swapped


@contextmanager
def _temporary_bias_override(params: Dict[str, float]):
    original = copy.deepcopy(sim_params.BEHAVIORAL_BIASES)
    try:
        sim_params.BEHAVIORAL_BIASES["loss_aversion"]["parameters"]["baseline_coefficient"] = float(
            params["lambda_0"]
        )
        sim_params.BEHAVIORAL_BIASES["present_bias"]["parameters"]["beta_min"] = float(
            params["beta_pb_lo"]
        )
        sim_params.BEHAVIORAL_BIASES["present_bias"]["parameters"]["beta_max"] = float(
            params["beta_pb_hi"]
        )
        sim_params.BEHAVIORAL_BIASES["status_quo"]["parameters"]["baseline_strength"] = float(
            params["gamma_SQ"]
        )
        sim_params.BEHAVIORAL_BIASES["optimism_bias"]["parameters"]["base_optimism"] = float(
            params["mu_omega"]
        )
        # Keep truncation bounds fixed as requested.
        sim_params.BEHAVIORAL_BIASES["optimism_bias"]["parameters"]["effect_variation_min"] = 0.05
        sim_params.BEHAVIORAL_BIASES["optimism_bias"]["parameters"]["effect_variation_max"] = 0.20
        sim_params.BEHAVIORAL_BIASES["herding"]["parameters"]["target_effect_per_neighbor"] = float(
            params["phi"]
        )
        sim_params.BEHAVIORAL_BIASES["herding"]["parameters"]["spatial_weight"] = float(
            params["w_spatial"]
        )
        sim_params.BEHAVIORAL_BIASES["herding"]["parameters"]["class_weight"] = float(params["w_class"])
        yield
    finally:
        sim_params.BEHAVIORAL_BIASES.clear()
        sim_params.BEHAVIORAL_BIASES.update(original)


def _build_overrides(params: Dict[str, float], seed: int, n_agents: int) -> Dict[str, object]:
    return {
        "num_households": n_agents,
        "run_settings": {"random_seed": int(seed)},
        # Keep a config-level record, even though current core reads from BEHAVIORAL_BIASES globals.
        "behavioral_params": {
            "loss_aversion": {"baseline_coefficient": float(params["lambda_0"])},
            "present_bias": {
                "beta_min": float(params["beta_pb_lo"]),
                "beta_max": float(params["beta_pb_hi"]),
            },
            "status_quo": {"baseline_strength": float(params["gamma_SQ"])},
            "optimism_bias": {
                "base_optimism": float(params["mu_omega"]),
                "effect_variation_min": 0.05,
                "effect_variation_max": 0.20,
            },
            "herding": {
                "target_effect_per_neighbor": float(params["phi"]),
                "spatial_weight": float(params["w_spatial"]),
                "class_weight": float(params["w_class"]),
            },
        },
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
    system_df: pd.DataFrame, threshold: float, adoption_col: str = TARGET_ADOPTION_COL
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

    with _temporary_bias_override(params):
        model = MultiExperimentModel(config)
        steps = int(config.get("steps", 240))
        model.run(steps=steps)

    system_df = model.data_collector.get_system_metrics_dataframe()
    t50 = _extract_threshold_time(system_df, threshold=0.5, adoption_col=TARGET_ADOPTION_COL)
    t90 = _extract_threshold_time(system_df, threshold=0.9, adoption_col=TARGET_ADOPTION_COL)
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
            PROBLEM, x, y_t50, conf_level=0.95, print_to_console=False, num_levels=p
        )
        si_t90 = morris_analyze(
            PROBLEM, x, y_t90, conf_level=0.95, print_to_console=False, num_levels=p
        )

    _save_indices_csv(PROBLEM, si_t50, os.path.join(output_dir, T50_OUT))
    _save_indices_csv(PROBLEM, si_t90, os.path.join(output_dir, T90_OUT))


def run_morris_phase2(
    output_dir: str = DEFAULT_OUTPUT_DIR,
    sample_seed: int = 12345,
    seeds: List[int] | None = None,
    p: int = 4,
    grid_jump: int = 2,
    r: int = 9,
    n_agents: int = 100,
) -> None:
    """Run/resume Morris phase-2 sensitivity and export SALib indices."""
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
        raw_params = dict(zip(PARAMETER_NAMES, [float(v) for v in x[point_id]]))
        params, swapped = _normalize_params(raw_params)

        if swapped:
            print(
                f"[Morris-Phase2] point {point_id + 1}/{total_points} "
                "swapped beta bounds to enforce beta_pb_lo <= beta_pb_hi"
            )
        else:
            print(f"[Morris-Phase2] point {point_id + 1}/{total_points}")

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
            "w_class": float(params["w_class"]),
            "beta_bounds_swapped": int(swapped),
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
    run_morris_phase2()
