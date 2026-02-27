#!/usr/bin/env python
"""Joint Sobol runner (pilot + chunk-ready) for ABM smart-grid manuscript metrics.

Smoke test:
python scripts/sobol/run_sobol_joint.py --outdir results/sobol/smoke --n_base 8 --seeds 11 --jobs 2 --smoke

Pilot run:
python scripts/sobol/run_sobol_joint.py --outdir results/sobol/pilot --n_base 64 --seeds 11 22 33 --jobs 6 --batch_size 16 --resume
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
import time
import traceback
from contextlib import contextmanager
from datetime import datetime
from multiprocessing import Pool
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from simulation.models.multi_experiment_model import MultiExperimentModel
from simulation.utils.config_loader import create_multi_experiment_config
from simulation.utils import parameters as sim_params

try:
    try:
        from SALib.sample.saltelli import sample as saltelli_sample
    except ImportError:
        from SALib.sample import sobol as sobol_sample_module

        def saltelli_sample(problem: Dict[str, object], n_base: int, calc_second_order: bool) -> np.ndarray:
            return sobol_sample_module.sample(problem, n_base, calc_second_order=calc_second_order)

    from SALib.analyze.sobol import analyze as sobol_analyze
except ImportError as exc:
    raise ImportError("SALib is required. Install with `pip install SALib`.") from exc


PARAMETER_NAMES: List[str] = [
    "P_f0",
    "g_f",
    "C_variable",
    "r_learn",
    "E_base",
    "phi",
    "w_spatial",
    "lambda0",
    "delta_PB",
    "gamma_SQ",
]

PARAMETER_BOUNDS: List[List[float]] = [
    [0.10, 0.25],
    [0.00, 0.05],
    [1400.0, 2800.0],
    [0.02, 0.08],
    [10.0, 25.0],
    [0.05, 0.15],
    [0.0, 1.0],
    [1.5, 3.0],
    [0.0, 0.15],
    [0.3, 0.9],
]

PROBLEM: Dict[str, object] = {
    "num_vars": len(PARAMETER_NAMES),
    "names": PARAMETER_NAMES,
    "bounds": PARAMETER_BOUNDS,
}

DEFAULT_ADOPTION_COL = "all_biases_AdoptionRate"
DEFAULT_SEEDS = [11, 22, 33]
FIXED_BETA_PB_LO = 0.625
OUTPUT_COLUMNS = [
    "sample_id",
    *PARAMETER_NAMES,
    "w_class",
    "beta_PB_lo",
    "beta_PB_hi",
    "seed_list",
    "seed_T50_json",
    "seed_T90_json",
    "T50_mean",
    "T90_mean",
    "T50_sd",
    "T90_sd",
    "runtime_seconds",
    "timestamp",
]

WORKER_CONTEXT: Dict[str, object] = {}


def _as_json_list(values: Sequence[float]) -> str:
    return json.dumps([None if (isinstance(v, float) and math.isnan(v)) else float(v) for v in values])


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _deep_merge(dst: Dict[str, object], src: Dict[str, object]) -> None:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _deep_merge(dst[key], value)  # type: ignore[index]
        else:
            dst[key] = value


def _extract_threshold_time(
    system_df: pd.DataFrame, threshold: float, adoption_col: str = DEFAULT_ADOPTION_COL
) -> float:
    if adoption_col not in system_df.columns:
        return np.nan
    series = pd.to_numeric(system_df[adoption_col], errors="coerce").to_numpy(dtype=float)
    hit_idx = np.where(series >= threshold)[0]
    if len(hit_idx) == 0:
        return np.nan
    return float(int(hit_idx[0]))


@contextmanager
def _temporary_bias_override(params: Dict[str, float]):
    original = copy.deepcopy(sim_params.BEHAVIORAL_BIASES)
    try:
        sim_params.BEHAVIORAL_BIASES["loss_aversion"]["parameters"]["baseline_coefficient"] = float(
            params["lambda0"]
        )
        sim_params.BEHAVIORAL_BIASES["present_bias"]["parameters"]["beta_min"] = float(
            params["beta_PB_lo"]
        )
        sim_params.BEHAVIORAL_BIASES["present_bias"]["parameters"]["beta_max"] = float(
            params["beta_PB_hi"]
        )
        sim_params.BEHAVIORAL_BIASES["status_quo"]["parameters"]["baseline_strength"] = float(
            params["gamma_SQ"]
        )
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


def _build_overrides(params: Dict[str, float], seed: int, n_agents: int, horizon: int) -> Dict[str, object]:
    return {
        "num_households": int(n_agents),
        "steps": int(horizon),
        "consumption_params": {
            "base_consumption": float(params["E_base"]),
        },
        "solar_params": {
            "variable_cost_per_kw": float(params["C_variable"]),
            "annual_cost_reduction": float(params["r_learn"]),
        },
        "grid_params": {
            "initial_fossil_price": float(params["P_f0"]),
            "fossil_annual_increase": float(params["g_f"]),
        },
        "run_settings": {"random_seed": int(seed)},
    }


def _run_single_seed(
    params: Dict[str, float], seed: int, n_agents: int, horizon: int, adoption_col: str
) -> Tuple[float, float, int]:
    random.seed(seed)
    np.random.seed(seed)

    base_config = create_multi_experiment_config()
    merged = json.loads(json.dumps(base_config))
    _deep_merge(merged, _build_overrides(params=params, seed=seed, n_agents=n_agents, horizon=horizon))

    with _temporary_bias_override(params):
        model = MultiExperimentModel(merged)
        steps = int(merged.get("steps", horizon))
        model.run(steps=steps)

    system_df = model.data_collector.get_system_metrics_dataframe()
    t50 = _extract_threshold_time(system_df, threshold=0.5, adoption_col=adoption_col)
    t90 = _extract_threshold_time(system_df, threshold=0.9, adoption_col=adoption_col)
    return t50, t90, steps


def _transform_row(raw_row: Sequence[float]) -> Dict[str, float]:
    params = dict(zip(PARAMETER_NAMES, [float(v) for v in raw_row]))
    params["w_class"] = 1.0 - float(params["w_spatial"])
    params["beta_PB_lo"] = FIXED_BETA_PB_LO
    params["beta_PB_hi"] = min(FIXED_BETA_PB_LO + float(params["delta_PB"]), 0.85)
    return params


def _row_dict(sample_id: int, raw_row: Sequence[float]) -> Dict[str, float]:
    params = _transform_row(raw_row)
    return {
        "sample_id": int(sample_id),
        **{name: float(params[name]) for name in PARAMETER_NAMES},
        "w_class": float(params["w_class"]),
        "beta_PB_lo": float(params["beta_PB_lo"]),
        "beta_PB_hi": float(params["beta_PB_hi"]),
    }


def _init_worker(context: Dict[str, object]) -> None:
    global WORKER_CONTEXT
    WORKER_CONTEXT = context


def _evaluate_one_sample(task: Tuple[int, Sequence[float]]) -> Dict[str, object]:
    sample_id, raw_row = task
    seeds = WORKER_CONTEXT["seeds"]  # type: ignore[index]
    n_agents = int(WORKER_CONTEXT["agents"])  # type: ignore[index]
    horizon = int(WORKER_CONTEXT["horizon"])  # type: ignore[index]
    adoption_col = str(WORKER_CONTEXT["adoption_col"])  # type: ignore[index]
    error_log_path = str(WORKER_CONTEXT["error_log_path"])  # type: ignore[index]
    censor_value = float(WORKER_CONTEXT["censor_value"])  # type: ignore[index]

    point_start = time.time()
    params = _transform_row(raw_row)

    t50_runs: List[float] = []
    t90_runs: List[float] = []

    for seed in seeds:
        try:
            t50, t90, _ = _run_single_seed(
                params=params,
                seed=int(seed),
                n_agents=n_agents,
                horizon=horizon,
                adoption_col=adoption_col,
            )
            if np.isnan(t50):
                t50 = censor_value
            if np.isnan(t90):
                t90 = censor_value
            t50_runs.append(float(t50))
            t90_runs.append(float(t90))
        except Exception as exc:  # pragma: no cover - robustness path
            now = datetime.utcnow().isoformat(timespec="seconds")
            with open(error_log_path, "a", encoding="utf-8") as fh:
                fh.write(f"[{now}] sample_id={sample_id} seed={seed}\n")
                fh.write("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
                fh.write("\n")
            t50_runs.append(np.nan)
            t90_runs.append(np.nan)

    t50 = np.array(t50_runs, dtype=float)
    t90 = np.array(t90_runs, dtype=float)

    row = _row_dict(sample_id=sample_id, raw_row=raw_row)
    row.update(
        {
            "seed_list": json.dumps([int(s) for s in seeds]),
            "seed_T50_json": _as_json_list(t50_runs),
            "seed_T90_json": _as_json_list(t90_runs),
            "T50_mean": float(np.nanmean(t50)) if np.isfinite(t50).any() else np.nan,
            "T90_mean": float(np.nanmean(t90)) if np.isfinite(t90).any() else np.nan,
            "T50_sd": float(np.nanstd(t50, ddof=1)) if np.isfinite(t50).sum() > 1 else np.nan,
            "T90_sd": float(np.nanstd(t90, ddof=1)) if np.isfinite(t90).sum() > 1 else np.nan,
            "runtime_seconds": round(time.time() - point_start, 3),
            "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
        }
    )
    return row


def _write_samples_csv(samples_path: str, x: np.ndarray) -> None:
    rows = [_row_dict(sample_id=i, raw_row=x[i]) for i in range(x.shape[0])]
    pd.DataFrame(rows).to_csv(samples_path, index=False)


def _save_problem_json(problem_path: str, payload: Dict[str, object]) -> None:
    if os.path.exists(problem_path):
        with open(problem_path, "r", encoding="utf-8") as fh:
            existing = json.load(fh)
        if existing != payload:
            raise RuntimeError(
                "Existing problem.json does not match current run settings. Use a new outdir or remove old artifacts."
            )
        return
    with open(problem_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def _load_or_create_samples(args: argparse.Namespace, samples_path: str) -> np.ndarray:
    if os.path.exists(samples_path):
        df = pd.read_csv(samples_path)
        missing = [c for c in PARAMETER_NAMES if c not in df.columns]
        if missing:
            raise RuntimeError(f"samples.csv missing expected columns: {missing}")
        return df[PARAMETER_NAMES].to_numpy(dtype=float)

    try:
        x = saltelli_sample(PROBLEM, args.n_base, calc_second_order=False)
    except TypeError:
        np.random.seed(args.seed_sampling)
        x = saltelli_sample(PROBLEM, args.n_base, calc_second_order=False)
    _write_samples_csv(samples_path, x)
    return x


def _load_completed_sample_ids(outputs_path: str) -> set[int]:
    if not os.path.exists(outputs_path):
        return set()
    df = pd.read_csv(outputs_path)
    if "sample_id" not in df.columns:
        raise RuntimeError(f"{outputs_path} is missing sample_id")
    ids = df["sample_id"].dropna().astype(int).tolist()
    return set(ids)


def _append_rows(outputs_path: str, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    header = not os.path.exists(outputs_path)
    df.to_csv(outputs_path, mode="a", header=header, index=False)


def _build_indices_dataframe(problem: Dict[str, object], si: Dict[str, np.ndarray]) -> pd.DataFrame:
    names = list(problem["names"])  # type: ignore[index]
    data: Dict[str, object] = {"parameter": names}
    for key in ["S1", "ST", "S1_conf", "ST_conf"]:
        if key in si:
            data[key] = si[key]
    return pd.DataFrame(data)


def _analyze_and_write(problem: Dict[str, object], outputs_path: str, metric: str, out_path: str, horizon: int) -> pd.DataFrame:
    df = pd.read_csv(outputs_path).sort_values("sample_id").drop_duplicates(subset=["sample_id"], keep="last")
    y = pd.to_numeric(df[metric], errors="coerce").to_numpy(dtype=float)
    y = np.where(np.isnan(y), float(horizon + 1), y)
    si = sobol_analyze(problem, y, calc_second_order=False, print_to_console=False)
    idx_df = _build_indices_dataframe(problem, si)
    idx_df.to_csv(out_path, index=False)
    return idx_df


def _print_top_st(indices_df: pd.DataFrame, label: str, top_n: int = 5) -> None:
    if "ST" not in indices_df.columns:
        return
    top = indices_df.sort_values("ST", ascending=False).head(top_n)
    print(f"\nTop {top_n} ST parameters for {label}:")
    for _, row in top.iterrows():
        print(f"  {row['parameter']}: ST={float(row['ST']):.6f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Joint Sobol sensitivity runner with resume and chunk mode")
    parser.add_argument("--n_base", type=int, default=32, help="Sobol base sample size (N)")
    parser.add_argument("--agents", type=int, default=100, help="Number of households per run")
    parser.add_argument("--horizon", type=int, default=240, help="Simulation horizon in months")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS, help="CRN seeds list")
    parser.add_argument("--jobs", type=int, default=4, help="Parallel worker processes")
    parser.add_argument("--batch_size", type=int, default=16, help="Flush interval for checkpoint writes")
    parser.add_argument("--outdir", type=str, required=True, help="Output directory")
    parser.add_argument("--resume", action="store_true", help="Resume from existing outputs.csv")
    parser.add_argument("--smoke", action="store_true", help="Run first 5 samples with first seed, skip analysis")
    parser.add_argument("--calc_second_order", action="store_true", help="Keep compatibility flag")
    parser.add_argument("--chunk_id", type=int, default=0, help="Chunk id for distributed execution")
    parser.add_argument("--num_chunks", type=int, default=1, help="Total chunks for distributed execution")
    parser.add_argument("--seed_sampling", type=int, default=12345, help="Sampling RNG seed")
    parser.add_argument(
        "--adoption_col",
        type=str,
        default=DEFAULT_ADOPTION_COL,
        help="System-metrics adoption column used for T50/T90 extraction",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.num_chunks < 1:
        raise ValueError("--num_chunks must be >= 1")
    if args.chunk_id < 0 or args.chunk_id >= args.num_chunks:
        raise ValueError("--chunk_id must satisfy 0 <= chunk_id < num_chunks")
    if args.n_base < 1:
        raise ValueError("--n_base must be >= 1")
    if args.jobs < 1:
        raise ValueError("--jobs must be >= 1")
    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1")
    if args.calc_second_order:
        print("--calc_second_order is currently ignored; running first-order + total-order only.")

    _ensure_dir(args.outdir)
    problem_path = os.path.join(args.outdir, "problem.json")
    samples_path = os.path.join(args.outdir, "samples.csv")
    error_log_path = os.path.join(args.outdir, "errors.log")

    if args.num_chunks == 1:
        outputs_filename = "outputs.csv"
    else:
        outputs_filename = f"outputs_chunk_{args.chunk_id}.csv"
    outputs_path = os.path.join(args.outdir, outputs_filename)
    if os.path.exists(outputs_path) and not args.resume:
        raise RuntimeError(f"{outputs_filename} already exists. Re-run with --resume to continue safely.")

    if args.smoke:
        seeds = [int(args.seeds[0])]
        print(
            "Smoke mode active: evaluating first 5 Sobol rows with 1 seed. "
            "Command template: --n_base 8 --seeds 11 --jobs 2 --smoke"
        )
    else:
        seeds = [int(s) for s in args.seeds]

    problem_payload = {
        "problem": PROBLEM,
        "k": len(PARAMETER_NAMES),
        "n_base": int(args.n_base),
        "agents": int(args.agents),
        "horizon": int(args.horizon),
        "seeds": seeds,
        "seed_sampling": int(args.seed_sampling),
        "calc_second_order": False,
        "transforms": {
            "w_class": "1 - w_spatial",
            "beta_PB_lo": FIXED_BETA_PB_LO,
            "beta_PB_hi": "min(beta_PB_lo + delta_PB, 0.85)",
        },
        "adoption_col": str(args.adoption_col),
    }
    _save_problem_json(problem_path, problem_payload)

    np.random.seed(args.seed_sampling)
    x = _load_or_create_samples(args, samples_path)

    total_samples = x.shape[0]
    all_ids = np.arange(total_samples, dtype=int)
    selected_ids = [int(i) for i in all_ids if int(i) % args.num_chunks == args.chunk_id]
    if args.smoke:
        selected_ids = selected_ids[:5]

    completed_ids = _load_completed_sample_ids(outputs_path) if args.resume else set()
    pending_ids = [sample_id for sample_id in selected_ids if sample_id not in completed_ids]

    print(f"Total Sobol rows: {total_samples}")
    print(f"Chunk {args.chunk_id}/{args.num_chunks} rows selected: {len(selected_ids)}")
    print(f"Pending rows: {len(pending_ids)}")
    if args.num_chunks == 1 and not args.smoke:
        print("Pilot command template: --n_base 64 --seeds 11 22 33 --jobs 6 --batch_size 16 --resume")

    if not pending_ids:
        print("No pending samples to run.")
    else:
        tasks = [(sample_id, x[sample_id]) for sample_id in pending_ids]
        worker_context = {
            "seeds": seeds,
            "agents": int(args.agents),
            "horizon": int(args.horizon),
            "adoption_col": str(args.adoption_col),
            "error_log_path": error_log_path,
            "censor_value": float(args.horizon + 1),
        }

        buffer: List[Dict[str, object]] = []
        with Pool(processes=args.jobs, initializer=_init_worker, initargs=(worker_context,)) as pool:
            iterator = pool.imap_unordered(_evaluate_one_sample, tasks, chunksize=1)
            for row in tqdm(iterator, total=len(tasks), desc="Sobol samples"):
                buffer.append(row)
                if len(buffer) >= args.batch_size:
                    _append_rows(outputs_path, buffer)
                    buffer = []
            _append_rows(outputs_path, buffer)

    if args.smoke:
        print(f"Smoke run complete. Outputs written to {outputs_path}")
        return

    if args.num_chunks > 1:
        print(
            "Chunk run complete. Indices are skipped in chunk mode; run merge utility after all chunks finish."
        )
        return

    completed_ids = _load_completed_sample_ids(outputs_path)
    if len(completed_ids) != total_samples:
        raise RuntimeError(
            f"Cannot analyze yet: completed rows={len(completed_ids)} expected={total_samples}. Continue with --resume."
        )

    t50_path = os.path.join(args.outdir, "indices_T50.csv")
    t90_path = os.path.join(args.outdir, "indices_T90.csv")
    t50_idx = _analyze_and_write(PROBLEM, outputs_path, "T50_mean", t50_path, horizon=args.horizon)
    t90_idx = _analyze_and_write(PROBLEM, outputs_path, "T90_mean", t90_path, horizon=args.horizon)
    _print_top_st(t50_idx, "T50")
    _print_top_st(t90_idx, "T90")

    print(f"Analysis complete: {t50_path}, {t90_path}")


if __name__ == "__main__":
    main()
