# /run_experiment1.py V5
"""
Runner script for the rational experiment.
Clean architecture with proper configuration centralization.
"""
import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm

from simulation.utils.config_loader import create_rational_experiment
from simulation.models.rational_model import RationalModel
from simulation.data.metrics import MetricsAnalyzer
from simulation.data.visualizers import SimulationVisualizer

def run_single_simulation(config, run_id=0, validate=True):
    """
    Run a single simulation with the given configuration.
    
    Args:
        config: Simulation configuration
        run_id: Run identifier
        validate: Enable validation checks
        
    Returns:
        tuple: (model, data_collector)
    """
    # Update random seed for this run
    config["run_settings"]["random_seed"] = 42 + run_id
    
    # Create and run model
    model = RationalModel(config)

    # Enable/disable validator
    if hasattr(model, 'validator'):
        model.validator.enabled = validate
    
    # Run for 12 initial months to build history before making adoption decisions
    print("Running initial 12 months to build consumption history...")
    for i in range(12):
        model.step()
    
    # Then run the main simulation
    print(f"Running main simulation {run_id+1}...")
    progress_bar = tqdm(total=config["steps"] - 12)
    
    for i in range(12, config["steps"]):
        model.step()
        progress_bar.update(1)
    
    progress_bar.close()

    # Run validation if enabled
    if validate and hasattr(model, 'validator') and model.validator.enabled:
        print("\nRunning validation checks...")
        model.validator.print_validation_results()
        validation_dir = f"results/{config['experiment_name']}/run_{run_id}/validation"
        model.validator.create_validation_report(output_dir=validation_dir)

    # Export detailed tracker data
    detailed_output_dir = f"results/{config['experiment_name']}/run_{run_id}/detailed_data"
    model.detailed_tracker.export_data(output_dir=detailed_output_dir)
    
    return model, model.data_collector

def run_experiment(num_runs=30, output_dir="results", validate=True):
    """
    Run the rational experiment with multiple runs.
    
    Args:
        num_runs: Number of simulation runs
        output_dir: Directory to save results
        validate: Enable validation checks
    """
    # Create base configuration (ALL config comes from config_loader.py - CLEAN!)
    config = create_rational_experiment().get_copy()
    
    # Update configuration for experiment (minimal additions only)
    config.update({
        "experiment_name": f"rational_experiment_{time.strftime('%Y%m%d_%H%M%S')}",
        "run_settings": {
            **config.get("run_settings", {}),  # Preserve existing settings
            "data_collection_interval": 1
        }
    })
    
    # Create output directories
    experiment_dir = os.path.join(output_dir, config["experiment_name"])
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Save experiment configuration
    pd.Series(config).to_csv(os.path.join(experiment_dir, "experiment_config.csv"))
    
    # Print unified metrics status (clean config reading)
    print("\n=== UNIFIED METRICS CONFIGURATION ===")
    unified_config = config.get("unified_metrics", {})
    if unified_config.get("enable_unified_metrics", False):
        print("✅ Unified metrics engine: ENABLED")
        print(f"✅ Granularity level: {unified_config.get('metrics_granularity', 'basic')}")
        print(f"✅ Peak load method: {unified_config.get('peak_load_method', 'estimated')}")
        print(f"✅ Credit utilization tracking: {unified_config.get('track_credit_utilization', False)}")
        print(f"✅ Seasonal stress analysis: {unified_config.get('track_seasonal_stress', False)}")
    else:
        print("❌ Unified metrics engine: DISABLED (using basic metrics only)")
    
    # Run simulations
    all_model_data = []
    
    for run_id in range(num_runs):
        print(f"\n{'='*50}")
        print(f"STARTING RUN {run_id + 1}/{num_runs}")
        print(f"{'='*50}")
        
        # Create run directory
        run_dir = os.path.join(experiment_dir, f"run_{run_id}")
        os.makedirs(run_dir, exist_ok=True)
        
        # Run simulation
        model, data_collector = run_single_simulation(config, run_id, validate=validate)
        
        # Save data
        data_collector.save_data(run_dir)
        
        # Collect model data for aggregation
        model_data = data_collector.get_model_data()
        model_data["run_id"] = run_id
        all_model_data.append(model_data)
        
        # Create visualizations for this run (FIXED: moved visualizer creation BEFORE usage)
        print("Creating visualizations for this run...")
        metrics = MetricsAnalyzer(model_data, data_collector.get_agent_data())
        visualizer = SimulationVisualizer(metrics)
        
        # Create standard visualizations
        visualizer.create_all_visualizations(run_dir)
        
        
        print(f"📈 Run {run_id + 1} visualizations created in {run_dir}")
        print(f"📊 Macro-level analysis available in {run_dir}/macro_analysis")
        
        # Print run summary with unified metrics (if available)
        print_run_summary(model, data_collector, unified_config)
    
    # Combine and analyze all runs
    if all_model_data:
        print(f"\n{'='*50}")
        print("CREATING AGGREGATE ANALYSIS")
        print(f"{'='*50}")
        
        combined_data = pd.concat(all_model_data)
        combined_data.to_csv(os.path.join(experiment_dir, "combined_model_data.csv"))
        
        # Create aggregate metrics and visualizations
        create_aggregate_analysis(combined_data, experiment_dir, unified_config)
    
    print(f"\n✅ Experiment completed successfully!")
    print(f"📁 Results saved to: {experiment_dir}")
    print(f"📊 {num_runs} simulation runs completed")
    
    return experiment_dir

def print_run_summary(model, data_collector, unified_config):
    """
    Print summary of a single run including unified metrics.
    
    Args:
        model: The completed model
        data_collector: Data collector instance
        unified_config: Unified metrics configuration
    """
    # Get basic statistics
    summary = data_collector.get_summary_statistics()
    print(f"\n📊 Run Summary:")
    print(f"   Total households: {summary.get('total_households', 'N/A')}")
    print(f"   Final adoption rate: {summary.get('adoption_rate', 0):.1%}")
    
    # Show unified metrics if available
    if unified_config.get("enable_unified_metrics", False):
        central_provider = model.central_provider
        
        if hasattr(central_provider, 'monthly_peak_load'):
            print(f"   Final peak load: {central_provider.monthly_peak_load:.1f} kW")
        
        if hasattr(central_provider, 'overall_credit_utilization'):
            print(f"   Credit utilization: {central_provider.overall_credit_utilization:.1%}")
        
        if hasattr(central_provider, 'avg_stress_index'):
            print(f"   Grid stress index: {central_provider.avg_stress_index:.2f}")

def create_aggregate_analysis(combined_data, output_dir, unified_config):
    """
    Create aggregate analysis of multiple simulation runs.
    
    Args:
        combined_data: Combined model data from all runs
        output_dir: Directory to save analysis
        unified_config: Unified metrics configuration
    """
    # Calculate statistics for adoption rate across runs
    adoption_stats = combined_data.groupby(["Year", "run_id"])["AdoptionRate"].last().reset_index()
    adoption_summary = adoption_stats.groupby("Year").agg({
        "AdoptionRate": ["mean", "std", "min", "max"]
    }).reset_index()
    
    # Save summary data
    adoption_summary.to_csv(os.path.join(output_dir, "adoption_summary.csv"))
    
    # Create aggregate adoption curve plot
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.figure(figsize=(12, 8))
    
    # Plot mean with confidence interval
    sns.lineplot(
        x="Year", 
        y="AdoptionRate", 
        data=adoption_stats, 
        ci=95, 
        estimator="mean"
    )
    
    plt.title("Aggregate Adoption Rate (Mean with 95% CI)")
    plt.xlabel("Year")
    plt.ylabel("Adoption Rate")
    plt.ylim(0, 1)
    plt.grid(True)
    
    plt.savefig(os.path.join(output_dir, "aggregate_adoption.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    # Create final adoption distribution by run
    final_year = adoption_stats["Year"].max()
    final_adoption = adoption_stats[adoption_stats["Year"] == final_year]
    
    plt.figure(figsize=(10, 6))
    sns.histplot(final_adoption["AdoptionRate"], kde=True, bins=10)
    
    plt.title(f"Distribution of Final Adoption Rates (Year {final_year})")
    plt.xlabel("Final Adoption Rate")
    plt.ylabel("Frequency")
    plt.grid(True)
    
    plt.savefig(os.path.join(output_dir, "final_adoption_distribution.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    # Create aggregate unified metrics analysis (if enabled)
    if unified_config.get("enable_unified_metrics", False):
        create_aggregate_unified_metrics_analysis(combined_data, output_dir)

def create_aggregate_unified_metrics_analysis(combined_data, output_dir):
    """
    Create aggregate analysis for unified metrics across runs.
    
    Args:
        combined_data: Combined model data from all runs
        output_dir: Directory to save analysis
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Peak load analysis (if available)
    if "MonthlyPeakLoad" in combined_data.columns:
        peak_stats = combined_data.groupby(["Year", "run_id"])["MonthlyPeakLoad"].last().reset_index()
        
        plt.figure(figsize=(12, 6))
        sns.lineplot(x="Year", y="MonthlyPeakLoad", data=peak_stats, ci=95, estimator="mean")
        plt.title("Aggregate Peak Load Evolution (Mean with 95% CI)")
        plt.xlabel("Year")
        plt.ylabel("Peak Load (kW)")
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "aggregate_peak_load.png"), dpi=300, bbox_inches="tight")
        plt.close()
    
    # Credit utilization analysis (if available)
    if "CreditUtilizationRate" in combined_data.columns:
        credit_stats = combined_data.groupby(["Year", "run_id"])["CreditUtilizationRate"].last().reset_index()
        
        plt.figure(figsize=(12, 6))
        sns.lineplot(x="Year", y="CreditUtilizationRate", data=credit_stats, ci=95, estimator="mean")
        plt.title("Aggregate Credit Utilization Rate (Mean with 95% CI)")
        plt.xlabel("Year")
        plt.ylabel("Credit Utilization Rate")
        plt.ylim(0, 1)
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "aggregate_credit_utilization.png"), dpi=300, bbox_inches="tight")
        plt.close()
    
    print("📊 Aggregate unified metrics analysis completed")

if __name__ == "__main__":
    # Run the experiment
    # For testing: use num_runs=1, for full experiment: use num_runs=30
    experiment_dir = run_experiment(num_runs=1, validate=True)
    
    print(f"\n🎉 Experiment completed successfully!")
    print(f"📁 All results available in: {experiment_dir}")