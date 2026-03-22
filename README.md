# ABM-SG

Unified multilayer agent-based simulation framework for prosumer renewable-energy adoption in a smart-grid setting.

The codebase supports:

- physically grounded prosumer energy-flow simulation
- rational baseline simulation
- behavioral scenario comparison
- spatial and network-aware diffusion analysis
- robustness testing across population sizes and random seeds
- global sensitivity analysis using Morris screening and Sobol decomposition

## What This Framework Does

The framework simulates household-level adoption of distributed renewable energy technologies over a multi-year horizon. In the current implementation, photovoltaic prosumer adoption is the representative use case. The framework combines:

- a temporal layer coordinating multiple simulation resolutions
- a prosumer energy layer implementing hourly bidirectional energy flows and net-billing economics
- an agent population layer representing heterogeneous households embedded in a social network
- a decision layer based on NPV-style adoption logic with modular cognitive-bias extensions
- comparative multi-scenario execution to study how each bias changes adoption dynamics

The main source code lives under `simulation/`. Most users will interact with the repository through the root runner scripts and the utilities under `scripts/`.

## Framework Orientation

This repository is a diffusion model designed as a unified simulation laboratory where:

- energy flows shape household economics
- household economics shape adoption decisions
- adoption decisions propagate through the social network
- changing adoption levels feed back into aggregate system behavior


## Repository Guide

- `simulation/agents/`
  Household agents, behavioral logic, provider agents, and evaluation triggers.
- `simulation/environment/`
  Prosumer energy-system, grid, weather, and system-metric components.
- `simulation/models/`
  Main simulation models:
  `rational_model.py` for the rational baseline and `multi_experiment_model.py` for scenario comparison.
- `simulation/data/`
  Data collection, comparative metrics, visualization, and analysis utilities.
- `simulation/sensitivity/`
  Morris phase-1 and phase-2 sensitivity workflows.
- `simulation/utils/`
  Central parameters and configuration factories.
- `scripts/`
  Post-processing scripts for multi-seed aggregation and figure creation.
- `scripts/sobol/`
  Sobol sampling, chunk merging, and final sensitivity-figure generation.
- `run_multi_experiment.py`
  Main multi-scenario runner for full-scale experiments.
- `runner.py`
   Entry point for Morris screening.

## Installation

Python 3.11+ is recommended.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Key dependencies include `Mesa`, `numpy`, `pandas`, `networkx`, `matplotlib`, `seaborn`, `scikit-learn`, and `SALib`.

## Architecture Overview

The framework implements four interacting layers:

- `Temporal Layer`
  Coordinates the resolutions used by the other layers and governs when information is exchanged.
- `Prosumer Energy Layer`
  Models hourly electricity flows, local generation, grid exchanges, and net-billing economics.
- `Agent Population Layer`
  Represents heterogeneous households, a central provider, and the graph-based social structure.
- `Decision Layer`
  Implements the rational NPV baseline and the behavioral extensions used for scenario comparison.

This layered structure is implemented across:

- `simulation/environment/`
- `simulation/agents/`
- `simulation/network/`
- `simulation/models/`
- `simulation/utils/parameters.py`
- `simulation/utils/config_loader.py`

## Quick Start

### 1. Run a small test configuration

```bash
python run_multi_experiment.py --config testing --output results
```

This is the fastest way to confirm the framework is installed correctly.

### 2. Run the main multi-scenario model

```bash
python run_multi_experiment.py --output results
```

This is the main entry point for the unified behavioral framework.

Useful options from `run_multi_experiment.py`:

- `--config`
  Configuration type. Use `multi_experiment` or `testing`.
- `--N`
  Override the number of households.
- `--seeds`
  Run multi-seed experiments with a list or range like `1,2,3` or `1-20`.
- `--output`
  Base output directory.

Example with a smaller population:

```bash
python run_multi_experiment.py --N 100 --output results
```

## Main Use Cases

### Rational baseline validation

Use:

```bash
python run_experiment1.py
```

Main code path:

- [`run_experiment1.py`](run_experiment1.py)
- [`simulation/models/rational_model.py`](simulation/models/rational_model.py)

### Behavioral scenario comparison

Use:

```bash
python run_multi_experiment.py --output results
```

Main code path:

- [`run_multi_experiment.py`](run_multi_experiment.py)
- [`simulation/models/multi_experiment_model.py`](simulation/models/multi_experiment_model.py)
- [`simulation/agents/bias_manager.py`](simulation/agents/bias_manager.py)
- [`simulation/utils/parameters.py`](simulation/utils/parameters.py)

### Population-scale robustness

Run the same experiment at several population sizes, for example:

```bash
python run_multi_experiment.py --N 100 --output results
python run_multi_experiment.py --N 500 --output results
python run_multi_experiment.py --N 1000 --output results
python run_multi_experiment.py --N 5000 --output results
```

Main code path:

- [`run_multi_experiment.py`](run_multi_experiment.py)

### Multi-seed stochastic robustness

Run:

```bash
python run_multi_experiment.py --N 100 --seeds 1-20 --output results
```

In multi-seed mode, the runner creates a directory structure like:

```text
results/
  N100/
    seed_0001/
    seed_0002/
    ...
```

Main code path:

- [`run_multi_experiment.py`](run_multi_experiment.py)

Aggregate the seed-level metric outputs:

```bash
python scripts/aggregate_multiseed_metrics.py --results-dir results/N100 --seeds 1-20
```

Script:

- [`scripts/aggregate_multiseed_metrics.py`](scripts/aggregate_multiseed_metrics.py)

Aggregate adoption trajectories into percentile-band data:

```bash
python scripts/aggregate_multiseed_time_series.py ^
  --results-dir results/N100 ^
  --pattern adoption_time_series*.csv ^
  --low 5 --high 95 ^
  --out-csv results/N100/multiseed_adoption_band_data_N100.csv
```

Script:

- [`scripts/aggregate_multiseed_time_series.py`](scripts/aggregate_multiseed_time_series.py)

Plot the trajectory bands:

```bash
python scripts/plot_multiseed_adoption_band.py ^
  --in-csv results/N100/multiseed_adoption_band_data_N100.csv ^
  --out-dir results/N100
```

Script:

- [`scripts/plot_multiseed_adoption_band.py`](scripts/plot_multiseed_adoption_band.py)

### Morris sensitivity screening

Convenience runner:

```bash
python runner.py --output-dir outputs
python runner.py --morris-phase2 --output-dir outputs
```

Scripts:

- [`runner.py`](runner.py)
- [`simulation/sensitivity/morris_phase1.py`](simulation/sensitivity/morris_phase1.py)
- [`simulation/sensitivity/morris_phase2.py`](simulation/sensitivity/morris_phase2.py)

Outputs are written under `outputs/` and include:

- `morris_phase1_points.csv`
- `morris_phase1_T50_indices.csv`
- `morris_phase1_T90_indices.csv`
- `morris_phase2_points.csv`
- `morris_phase2_T50_indices.csv`
- `morris_phase2_T90_indices.csv`

### Sobol variance decomposition

Run a pilot or full Sobol experiment:

```bash
python scripts/sobol/run_sobol_joint.py --outdir results/sobol/pilot --n_base 64 --seeds 11 22 33 --jobs 4 --resume
```

For a quick smoke test:

```bash
python scripts/sobol/run_sobol_joint.py --outdir results/sobol/smoke --n_base 8 --seeds 11 --jobs 2 --smoke
```

Scripts:

- [`scripts/sobol/run_sobol_joint.py`](scripts/sobol/run_sobol_joint.py)
- [`scripts/sobol/merge_outputs_and_analyze.py`](scripts/sobol/merge_outputs_and_analyze.py)
- [`scripts/sobol/generate_final_sobol_figures.py`](scripts/sobol/generate_final_sobol_figures.py)

Merge chunked Sobol outputs and compute indices:

```bash
python scripts/sobol/merge_outputs_and_analyze.py --outdir results/sobol/pilot
```

Generate final figures from the computed indices:

```bash
python scripts/sobol/generate_final_sobol_figures.py ^
  --indices_t50 results/sobol/pilot/indices_T50.csv ^
  --indices_t90 results/sobol/pilot/indices_T90.csv ^
  --outdir results/sobol/final_figures
```

## Typical Output Layout

After a standard multi-scenario run, outputs are usually organized under:

```text
results/
  multi_experiment_<timestamp>/
    data/
    visualizations/
    detailed_data/
    validation/
    logs/
```

For multi-seed robustness runs:

```text
results/
  N100/
    seed_0001/
    seed_0002/
    ...
```

For Morris screening:

```text
outputs/
  morris_phase1_points.csv
  morris_phase1_T50_indices.csv
  morris_phase1_T90_indices.csv
  morris_phase2_points.csv
  morris_phase2_T50_indices.csv
  morris_phase2_T90_indices.csv
```

For Sobol analysis:

```text
results/
  sobol/
    <run_name>/
      problem.json
      samples.csv
      outputs.csv
      indices_T50.csv
      indices_T90.csv
```


## Citation

If you use this repository, please cite the associated manuscript (X) and describe the specific workflow you used, for example:

- rational baseline validation
- multi-scenario behavioral comparison
- multi-seed robustness analysis
- Morris screening
- Sobol decomposition
