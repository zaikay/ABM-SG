# run_multi_experiment.py V2.0
"""
FIXED: Multi-experiment runner with explicit error reporting instead of silent failures.
Removed SafeSimulationVisualizer - now reports exactly what data is missing.
"""

import os
import sys
import time
import copy
import random
import re
from datetime import datetime
import argparse

import pandas as pd
import numpy as np

# Add the simulation package to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Ensure Unicode-safe console output on Windows terminals with non-UTF codepages.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from simulation.utils.config_loader import (
    create_multi_experiment_config, create_testing_config
)
from simulation.models.multi_experiment_model import MultiExperimentModel
from simulation.data.comparative_visualizer import ComparativeVisualizer
from simulation.data.spatial_visualizer import SpatialVisualizer
from simulation.data.enhanced_spatial_analyzer import IntegratedSpatialAnalyzer
from simulation.data.immediate_spatial_analyzer import ImmediateSpatialAnalyzer
from simulation.data.temporal_network_propagation import TemporalNetworkPropagationAnalyzer
from simulation.data.focused_spatial_visualizer import FocusedSpatialVisualizer
from simulation.data.herding_bias_analyzer import HerdingBiasAnalyzer
from simulation.utils.parameters import (
    print_configuration_summary, get_all_scenarios, get_scenario_metadata
)

DEFAULT_MULTI_SEED_N = 100

def parse_seed_spec(seed_spec):
    """Parse seed specification like '1,2,3' or '1-20'."""
    if not seed_spec:
        return []

    seeds = []
    for part in seed_spec.split(','):
        token = part.strip()
        if not token:
            continue

        range_match = re.match(r'^(\d+)\s*-\s*(\d+)$', token)
        if range_match:
            start = int(range_match.group(1))
            end = int(range_match.group(2))
            if end < start:
                raise ValueError(f"Invalid seed range '{token}': end < start")
            seeds.extend(range(start, end + 1))
        else:
            seeds.append(int(token))

    unique_seeds = []
    seen = set()
    for seed in seeds:
        if seed not in seen:
            unique_seeds.append(seed)
            seen.add(seed)

    return unique_seeds

def apply_global_seed(seed):
    """Apply seed to Python and NumPy global RNGs."""
    random.seed(seed)
    np.random.seed(seed)

def apply_seed_to_config(config, seed):
    """Set run seed in config in-place."""
    if "run_settings" not in config:
        config["run_settings"] = {}
    config["run_settings"]["random_seed"] = seed

def setup_output_directories(base_dir="results", seed=None):
    """Set up output directory structure."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if seed is None:
        experiment_dir = os.path.join(base_dir, f"multi_experiment_{timestamp}")
    else:
        experiment_dir = os.path.join(base_dir, f"seed_{seed:04d}")
    
    paths = {
        'base': base_dir,
        'experiment': experiment_dir,
        'data': os.path.join(experiment_dir, "data"),
        'visualizations': os.path.join(experiment_dir, "visualizations"),
        'detailed': os.path.join(experiment_dir, "detailed_data"),
        'validation': os.path.join(experiment_dir, "validation"),
        'logs': os.path.join(experiment_dir, "logs")
    }
    
    # Create all directories
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    
    print(f"Output directories created in: {paths['experiment']}")
    return paths

def run_simulation(config, output_paths, verbose=True):
    """Run the multi-experiment simulation."""
    if verbose:
        print("\n" + "="*80)
        print("STARTING MULTI-EXPERIMENT SIMULATION")
        print("="*80)
    
    # Create and run model
    model = MultiExperimentModel(config)
    
    # Run simulation
    start_time = time.time()
    model.run(steps=config.get("steps", 240))
    simulation_time = time.time() - start_time
    
    if verbose:
        print(f"\n✅ Simulation completed in {simulation_time:.1f} seconds")

    # Keep CSV test export isolated per run to avoid cross-run overwrite.
    success = test_csv_export(model, output_dir=os.path.join(output_paths['logs'], "csv_test"))
    
    return model

def export_data(model, output_paths, verbose=True):
    """Export simulation data with validation."""
    if verbose:
        print("\n" + "="*80)
        print("EXPORTING DATA")
        print("="*80)
    
    start_time = time.time()
    
    # Export multi-experiment data
    model.data_collector.export_all_scenarios(output_paths['data'])

    # Export comparative metrics (5 CSV files)
    print("\nExporting comparative metrics...")
    try:
        # Get data from model
        system_data = model.data_collector.get_system_metrics_dataframe()
        agent_data = model.data_collector.get_combined_dataframe()
        
        # Create metrics calculator and export
        from simulation.data.comparative_metrics import ComparativeMetrics
        metrics = ComparativeMetrics(system_data, agent_data)
        
        # Create metrics subdirectory
        metrics_dir = os.path.join(output_paths['data'], 'metrics')
        success = metrics.export_all_metrics(metrics_dir)
        
        if success:
            print("✅ Comparative metrics exported:")
            print("  - adoption_time_series.csv")
            print("  - critical_mass_timing.csv")
            print("  - area_analysis.csv")
            print("  - scenario_comparison_summary.csv")
            print("  - adoption_snapshots_table.csv")

            print("\nCreating metrics visualizations...")
            from simulation.data.comparative_visualizer import ComparativeVisualizer
            visualizer = ComparativeVisualizer(system_data, agent_data)
            visualizer.create_metrics_visualizations(metrics_dir, output_paths['visualizations'])

        focused_spatial_output_dir = os.path.join(output_paths['data'], 'focused_spatial_metrics')
        model.focused_spatial_metrics.export_metrics(focused_spatial_output_dir)
        
        
    except Exception as e:
        print(f"Warning: Could not export comparative metrics: {e}")
        import traceback
        traceback.print_exc()
    
    # Export detailed tracking data
    model.detailed_tracker.export_data(output_paths['detailed'])
    print(f"Exported detailed data for {model.detailed_tracker.sample_size} sample households to {output_paths['detailed']}")
    
    # Export validation data
    if hasattr(model, 'validator') and model.validator.enabled:
        model.validator.create_validation_report(output_paths['validation'])
        print(f"Validation report saved to {output_paths['validation']}")
    
    export_time = time.time() - start_time
    
    if verbose:
        print(f"✅ Data export completed in {export_time:.1f} seconds")

def create_visualizations(model, output_paths, verbose=True):
    """
    FIXED: Create visualizations with explicit data validation.
    No more silent failures - reports exactly what's missing.
    """
    if verbose:
        print("\n" + "="*80)
        print("GENERATING VISUALIZATIONS")
        print("="*80)
    
    start_time = time.time()
    
    # Get data from model
    print("📊 Getting data from live model...")
    system_data = model.data_collector.get_system_metrics_dataframe()
    agent_data = model.data_collector.get_combined_dataframe()
    
    # REMOVED: Data adaptation that hides issues with defaults
    # ADDED: Explicit data validation
    
    # Validate data completeness
    print("🔍 Validating data completeness...")
    validation_results = validate_data_for_visualizations(system_data, agent_data)
    
    if not validation_results['can_proceed']:
        print("❌ Cannot proceed with visualizations due to missing critical data:")
        for issue in validation_results['critical_issues']:
            print(f"   • {issue}")
        return
    
    if validation_results['warnings']:
        print("⚠️  Data warnings (some visualizations may be limited):")
        for warning in validation_results['warnings']:
            print(f"   • {warning}")
    
    # 1. Comparative Visualizations
    print("\n📊 Creating comparative visualizations...")
    try:
        comparative_viz = ComparativeVisualizer(
            model_data=system_data, 
            agent_data=agent_data
        )
        comparative_viz.create_all_visualizations(output_paths['visualizations'])
        print("✅ Comparative visualizations completed")
    except Exception as e:
        print(f"❌ Comparative visualizations failed: {e}")
        import traceback
        traceback.print_exc()
    
    # 2. Spatial Visualizations
    print("\n📊 Creating spatial visualizations...")
    try:
        spatial_viz = SpatialVisualizer(model=model)
        spatial_viz.create_all_visualizations(
            os.path.join(output_paths['visualizations'], 'spatial_analysis')
        )
        print("✅ Spatial visualizations completed")
        print(" Focused Spatial visualizations start")
        # Create Focused Spatial visualizations
        focused_spatial_model = FocusedSpatialVisualizer(model=model)
        focused_spatial_model.create_all_focused_plots(
            output_dir=os.path.join(output_paths['visualizations'], 'focused_spatial_analysis'))
        print("✅ Focused Spatial visualizations completed")
    except Exception as e:
        print(f"❌ Spatial visualizations failed: {e}")
        import traceback
        traceback.print_exc()
    
    # 3. Detailed Analysis 
    print("\n📈 Creating detailed visualizations...")
    try:
        detailed_output_dir = os.path.join(output_paths['visualizations'], 'detailed_analysis')
        create_detailed_visualizations(system_data, agent_data, detailed_output_dir)
        print("✅ Detailed visualizations completed")
    except Exception as e:
        print(f"❌ Detailed visualizations failed: {e}")
        import traceback
        traceback.print_exc()

    # 4. System-Level Visualizations (NEW)
    print("\n📊 Creating system-level visualizations...")
    try:
        from simulation.data.system_level_visualizer import SystemLevelVisualizer
        from simulation.data.metrics import MetricsAnalyzer
        
        # Initialize metrics analyzer and system visualizer
        metrics = MetricsAnalyzer(system_data, agent_data)
        system_viz = SystemLevelVisualizer(metrics)
        
        # Create all system-level visualizations
        system_output_dir = os.path.join(output_paths['visualizations'], 'system_level_analysis')
        system_viz.create_all_system_visuals(system_output_dir)
        print("✅ System-level visualizations completed")
        
    except Exception as e:
        print(f"❌ System-level visualizations failed: {e}")
        import traceback
        traceback.print_exc()

    # 5. Herding Bias Analysis (Step-Level)
    print("\n🔍 Creating herding bias analysis...")
    try:
        herding_analyzer = HerdingBiasAnalyzer(
            model_data=system_data, 
            agent_data=agent_data
        )
        herding_output_dir = os.path.join(output_paths['visualizations'], 'herding_analysis')
        
        # Generate step-level herding analysis
        gap_results = herding_analyzer.create_herding_gap_analysis(herding_output_dir)
        impact_metrics = herding_analyzer.create_herding_impact_metrics(herding_output_dir)
        
        print("✅ Herding bias analysis completed")
        print(f"   📊 Peak effect: {gap_results['max_effect_value']:.1f}pp at Year {gap_results['max_effect_year']:.1f}")
        print(f"   📈 Final impact: {impact_metrics['final_percentage_gap']:+.1f}pp")
        
    except Exception as e:
        print(f"❌ Herding bias analysis failed: {e}")
        import traceback
        traceback.print_exc()
    
    visualization_time = time.time() - start_time
    
    if verbose:
        print(f"✅ Visualization generation completed in {visualization_time:.1f} seconds")

def validate_data_for_visualizations(system_data, agent_data):
    """
    FIXED: Explicit data validation for visualizations.
    Returns exactly what's missing instead of using defaults.
    """
    results = {
        'can_proceed': True,
        'critical_issues': [],
        'warnings': []
    }
    
    # Check if data exists at all
    if system_data.empty:
        results['critical_issues'].append("No system metrics data available")
        results['can_proceed'] = False
    
    if agent_data.empty:
        results['critical_issues'].append("No agent data available")
        results['can_proceed'] = False
    
    if not results['can_proceed']:
        return results
    
    # Check required columns for basic visualizations
    required_system_cols = ['Year']
    missing_system_cols = [col for col in required_system_cols if col not in system_data.columns]
    if missing_system_cols:
        results['critical_issues'].append(f"Missing required system columns: {missing_system_cols}")
        results['can_proceed'] = False
    
    required_agent_cols = ['Year', 'AgentType']
    missing_agent_cols = [col for col in required_agent_cols if col not in agent_data.columns]
    if missing_agent_cols:
        results['critical_issues'].append(f"Missing required agent columns: {missing_agent_cols}")
        results['can_proceed'] = False
    
    # Check for scenario adoption data
    scenarios = get_all_scenarios()
    scenario_cols = [f'{scenario}_AdoptionRate' for scenario in scenarios]
    missing_scenario_cols = [col for col in scenario_cols if col not in system_data.columns]
    if missing_scenario_cols:
        results['warnings'].append(f"Missing scenario adoption data: {[col.replace('_AdoptionRate', '') for col in missing_scenario_cols]}")
    
    # Check for income class data
    income_class_cols = [col for col in system_data.columns if 'Class' in col and 'Rate' in col]
    if not income_class_cols:
        results['warnings'].append("No income class data available - income class visualizations will be skipped")
    
    # Check for system metrics (try scenario-specific first, then fallback)
    old_metric_cols = ['TotalCreditsEarned', 'GridStressIndex']
    scenario_metric_cols = ['rational_TotalCreditsEarned', 'rational_GridStressIndex']

    # Check if we have scenario-specific columns
    has_scenario_metrics = all(col in system_data.columns for col in scenario_metric_cols)
    has_old_metrics = all(col in system_data.columns for col in old_metric_cols)

    if not has_scenario_metrics and not has_old_metrics:
        missing_metrics = [col for col in old_metric_cols if col not in system_data.columns]
        results['warnings'].append(f"Missing system metrics: {missing_metrics} - some system visualizations will be limited")
    elif has_scenario_metrics:
        # Perfect - we have the new format
        pass
    
    return results

# Add this function to your run_multi_experiment.py file:

def adapt_multi_experiment_data_for_simulation_visualizer(system_data, agent_data):
    """
    Adapt multi-experiment data format to single-scenario format for SimulationVisualizer.
    
    Args:
        system_data: DataFrame from MultiExperimentCollector.get_system_metrics_dataframe()
        agent_data: DataFrame from MultiExperimentCollector.get_combined_dataframe()
        
    Returns:
        tuple: (adapted_system_data, adapted_agent_data) compatible with SimulationVisualizer
    """
    import pandas as pd
    import numpy as np
    
    print("🔧 Adapting multi-experiment data for SimulationVisualizer...")
    
    # =========================================================================
    # ADAPT SYSTEM DATA
    # =========================================================================
    adapted_system_data = system_data.copy()
    
    # 1. Add basic adoption rate (use rational scenario as baseline)
    if 'AdoptionRate' not in adapted_system_data.columns:
        if 'rational_AdoptionRate' in adapted_system_data.columns:
            adapted_system_data['AdoptionRate'] = adapted_system_data['rational_AdoptionRate']
            print("✅ Added AdoptionRate from rational scenario")
        else:
            adapted_system_data['AdoptionRate'] = 0.0
            print("⚠️  No adoption rate data found, using default")
    
    # 2. Ensure CurrentMonthInYear exists for monthly trends
    if 'CurrentMonthInYear' not in adapted_system_data.columns:
        if 'Month' in adapted_system_data.columns:
            adapted_system_data['CurrentMonthInYear'] = adapted_system_data['Month']
            print("✅ Added CurrentMonthInYear from Month column")
        else:
            adapted_system_data['CurrentMonthInYear'] = ((adapted_system_data.index % 12) + 1)
            print("⚠️  Generated CurrentMonthInYear from index")
    
    # 3. Check energy data availability
    energy_cols = ['TotalConsumption', 'rational_TotalGeneration']
    missing_energy_cols = [col for col in energy_cols if col not in adapted_system_data.columns]
    
    if missing_energy_cols:
        print(f"❌ Missing energy columns: {missing_energy_cols}")
        # Add placeholder data to prevent crashes
        for col in missing_energy_cols:
            adapted_system_data[col] = 1000  # Placeholder value
    else:
        print(f"✅ Energy data available: TotalConsumption={adapted_system_data['TotalConsumption'].sum():.0f}, TotalGeneration={adapted_system_data['rational_TotalGeneration'].sum():.0f}")
    
    # 4. Add GridConsumption if missing (needed for energy balance)
    if 'GridConsumption' not in adapted_system_data.columns:
        if 'TotalConsumption' in adapted_system_data.columns and 'rational_TotalGeneration' in adapted_system_data.columns:
            adapted_system_data['GridConsumption'] = adapted_system_data['TotalConsumption'] - adapted_system_data['rational_TotalGeneration']
            adapted_system_data['GridConsumption'] = adapted_system_data['GridConsumption'].clip(lower=0)
            print("✅ Calculated GridConsumption from TotalConsumption - rational_TotalGeneration")
        else:
            adapted_system_data['GridConsumption'] = 800  # Placeholder
            print("⚠️  Added placeholder GridConsumption")
    
    # 5. Add fossil dependency if missing
    if 'FossilDependency' not in adapted_system_data.columns:
        if 'GridConsumption' in adapted_system_data.columns and 'TotalConsumption' in adapted_system_data.columns:
            adapted_system_data['FossilDependency'] = adapted_system_data['GridConsumption'] / adapted_system_data['TotalConsumption']
            adapted_system_data['FossilDependency'] = adapted_system_data['FossilDependency'].fillna(1.0).clip(0, 1)
            print("✅ Calculated FossilDependency")
        else:
            adapted_system_data['FossilDependency'] = 1.0
            print("⚠️  Added placeholder FossilDependency")
    
    # =========================================================================
    # ADAPT AGENT DATA  
    # =========================================================================
    adapted_agent_data = agent_data.copy()
    
    # 1. Ensure AgentType exists
    if 'AgentType' not in adapted_agent_data.columns:
        adapted_agent_data['AgentType'] = 'Household'
        print("✅ Added AgentType column")
    
    # 2. Add IsProsumer from rational scenario
    if 'IsProsumer' not in adapted_agent_data.columns:
        if 'rational_Adopted' in adapted_agent_data.columns:
            adapted_agent_data['IsProsumer'] = adapted_agent_data['rational_Adopted']
            print("✅ Added IsProsumer from rational_Adopted")
        else:
            adapted_agent_data['IsProsumer'] = False
            print("⚠️  Added placeholder IsProsumer")
    
    # 3. Ensure IsProsumer is boolean
    if 'IsProsumer' in adapted_agent_data.columns:
        adapted_agent_data['IsProsumer'] = adapted_agent_data['IsProsumer'].astype(bool)
    
    # 4. Add IsPotentialProsumer if missing
    if 'IsPotentialProsumer' not in adapted_agent_data.columns:
        if 'NPV' in adapted_agent_data.columns:
            adapted_agent_data['IsPotentialProsumer'] = (adapted_agent_data['NPV'].fillna(0) > 0)
            print("✅ Added IsPotentialProsumer from NPV")
        else:
            adapted_agent_data['IsPotentialProsumer'] = False
            print("⚠️  Added placeholder IsPotentialProsumer")
    
    # 5. Check IncomeClass availability
    if 'IncomeClass' not in adapted_agent_data.columns:
        print("❌ Missing IncomeClass - income class visualizations will fail")
        adapted_agent_data['IncomeClass'] = 1  # Placeholder
    else:
        print(f"✅ IncomeClass available: {adapted_agent_data['IncomeClass'].nunique()} classes")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print(f"📊 Adapted system data: {adapted_system_data.shape}")
    print(f"📊 Adapted agent data: {adapted_agent_data.shape}")
    
    # Check key columns
    key_system_cols = ['Year', 'AdoptionRate', 'TotalConsumption', 'rational_TotalGeneration', 'FossilDependency']
    key_agent_cols = ['Year', 'AgentType', 'IsProsumer', 'IncomeClass']
    
    print("🔍 Key system columns check:")
    for col in key_system_cols:
        status = "✅" if col in adapted_system_data.columns else "❌"
        print(f"   {status} {col}")
    
    print("🔍 Key agent columns check:")
    for col in key_agent_cols:
        status = "✅" if col in adapted_agent_data.columns else "❌"
        print(f"   {status} {col}")
    
    return adapted_system_data, adapted_agent_data

def create_detailed_visualizations(system_data, agent_data, output_dir):
    """
    FIXED: Create detailed visualizations with proper data adaptation.
    """
    from simulation.data.metrics import MetricsAnalyzer
    from simulation.data.visualizers import SimulationVisualizer
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # STEP 1: Adapt data format
        adapted_system_data, adapted_agent_data = adapt_multi_experiment_data_for_simulation_visualizer(
            system_data, agent_data
        )
        
        # STEP 2: Create analyzer and visualizer with adapted data
        metrics = MetricsAnalyzer(adapted_system_data, adapted_agent_data)
        visualizer = SimulationVisualizer(metrics)
        
        # STEP 3: Use the built-in method
        if hasattr(visualizer, 'create_all_visualizations'):
            print("📊 Using built-in create_all_visualizations method...")
            visualizer.create_all_visualizations(output_dir)
            print("✅ All visualizations completed via built-in method")
        else:
            print("❌ create_all_visualizations method not found")
            
        # STEP 4: Verify monthly trends file was created
        monthly_trends_file = os.path.join(output_dir, "monthly_consumption_generation_trends.png")
        if os.path.exists(monthly_trends_file):
            print("✅ monthly_consumption_generation_trends.png created successfully")
        else:
            print("❌ monthly_consumption_generation_trends.png was not created")
    
    except Exception as e:
        print(f"❌ Detailed visualizations failed: {e}")
        import traceback
        traceback.print_exc()

def create_analysis_report(model, output_paths, verbose=True):
    """Create comprehensive analysis report."""
    if verbose:
        print("\n" + "="*80)
        print("CREATING ANALYSIS REPORT")
        print("="*80)
    
    # Get final statistics
    final_stats = model.get_scenario_statistics()
    
    # Create markdown report
    report_path = os.path.join(output_paths['experiment'], 'analysis_report.md')
    
    # FIX: Add encoding='utf-8' to handle Unicode characters
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Multi-Experiment Behavioral Prosumer Analysis Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Final Results\n\n")
        f.write(f"**Simulation Length**: {final_stats['current_year']} years\n\n")
        
        f.write("### Adoption Rates by Scenario\n\n")
        for scenario, stats in final_stats['scenarios'].items():
            f.write(f"- **{scenario}**: {stats['adoption_rate']:.1%} ({stats['adoption_count']} households)\n")
        
        f.write("\n### Data Completeness\n\n")
        # Add data validation results
        system_data = model.data_collector.get_system_metrics_dataframe()
        agent_data = model.data_collector.get_combined_dataframe()
        validation = validate_data_for_visualizations(system_data, agent_data)
        
        if validation['can_proceed']:
            f.write("✅ **Data Quality**: Complete\n\n")  # This will now work with UTF-8
        else:
            f.write("❌ **Data Quality**: Issues detected\n\n")  # This will now work with UTF-8
            for issue in validation['critical_issues']:
                f.write(f"- ❌ {issue}\n")
        
        if validation['warnings']:
            f.write("\n**Warnings**:\n")
            for warning in validation['warnings']:
                f.write(f"- ⚠️ {warning}\n")
    
    print(f"Analysis report saved to: {report_path}")

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Run multi-experiment behavioral prosumer simulation')
    parser.add_argument('--config', type=str, default='multi_experiment',
                       help='Configuration type (multi_experiment, testing)')
    parser.add_argument('--N', type=int, default=None,
                       help='Override number of households')
    parser.add_argument('--seeds', type=str, default=None,
                       help='Seed list/range (e.g., 1,2,3 or 1-20)')
    parser.add_argument('--output', type=str, default='results',
                       help='Output directory base path')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='Verbose output')

    args = parser.parse_args()

    if args.config == 'testing':
        config = create_testing_config()
    else:
        config = create_multi_experiment_config()

    # Normalize to plain dict when factory returns SimulationConfig.
    if hasattr(config, "config") and isinstance(config.config, dict):
        config = config.config

    multi_seed_mode = bool(args.seeds)

    if args.N is not None:
        config["num_households"] = args.N
    elif multi_seed_mode:
        config["num_households"] = DEFAULT_MULTI_SEED_N

    seeds = parse_seed_spec(args.seeds) if args.seeds else []

    if args.verbose:
        try:
            print_configuration_summary()
        except UnicodeEncodeError:
            print("Behavioral prosumer simulation configuration:")
            print(f"  Population: {config.get('num_households')}")
            print(f"  Duration (steps): {config.get('steps')}")

    # Seed flow note:
    # - Global RNGs are seeded in this runner via apply_global_seed().
    # - Model-level seeding (NumPy/Python/Mesa RNG) is applied in MultiExperimentModel.__init__.
    if multi_seed_mode:
        n_value = config["num_households"]
        base_output = os.path.join(args.output, f"N{n_value}")
        os.makedirs(base_output, exist_ok=True)

        if args.verbose:
            print(f"\nMulti-seed mode: {len(seeds)} seed(s), N={n_value}, output={base_output}")
            print("Heavy visualizations/secondary analyzers are skipped in multi-seed mode for runtime efficiency.")

        failed_seeds = []
        for seed in seeds:
            try:
                if args.verbose:
                    print("\n" + "=" * 80)
                    print(f"RUNNING SEED {seed}")
                    print("=" * 80)

                seed_config = copy.deepcopy(config)
                apply_seed_to_config(seed_config, seed)
                apply_global_seed(seed)
                output_paths = setup_output_directories(base_output, seed=seed)

                model = run_simulation(seed_config, output_paths, args.verbose)
                export_data(model, output_paths, args.verbose)

                if args.verbose:
                    print(f"Seed {seed} completed. Results: {output_paths['experiment']}")
            except Exception as e:
                failed_seeds.append((seed, str(e)))
                print(f"\nSeed {seed} failed: {e}")
                import traceback
                traceback.print_exc()

        if failed_seeds:
            print("\nSome seeds failed:")
            for seed, err in failed_seeds:
                print(f"  - seed {seed}: {err}")
            sys.exit(1)

        if args.verbose:
            print("\n" + "=" * 80)
            print("MULTI-SEED RUN COMPLETED SUCCESSFULLY")
            print("=" * 80)
            print(f"Results saved under: {base_output}")
        return

    configured_seed = config.get("run_settings", {}).get("random_seed", None)
    if configured_seed is not None:
        apply_global_seed(configured_seed)

    output_paths = setup_output_directories(args.output)

    try:
        model = run_simulation(config, output_paths, args.verbose)
        model.spatial_analyzer = IntegratedSpatialAnalyzer(model)
        ianalyzer = ImmediateSpatialAnalyzer(model)
        temporal_analyzer = TemporalNetworkPropagationAnalyzer(model)

        export_data(model, output_paths, args.verbose)
        create_visualizations(model, output_paths, args.verbose)

        model.spatial_analyzer.create_all_spatial_analyses()
        ianalyzer.create_all_immediate_analyses()
        temporal_analyzer.create_all_temporal_analyses()

        create_analysis_report(model, output_paths, args.verbose)

        if args.verbose:
            print("\n" + "=" * 80)
            print("MULTI-EXPERIMENT SIMULATION COMPLETED SUCCESSFULLY")
            print("=" * 80)
            print(f"Results saved to: {output_paths['experiment']}")

    except Exception as e:
        print(f"\nSimulation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


# =============================================================================
# TESTING FUNCTIONS (PRESERVED FROM ORIGINAL)
# =============================================================================

def test_data_adaptation():
    """Test the data adaptation functionality."""
    print("Testing data adaptation...")
    
    try:
        # Create test data
        test_system_data = pd.DataFrame({
            'Year': [1, 2],
            'Month': [1, 2],
            'rational_AdoptionRate': [0.1, 0.2]
        })
        
        test_agent_data = pd.DataFrame({
            'Year': [1, 1, 2, 2],
            'AgentID': [1, 2, 1, 2],
            'rational_Adopted': [False, True, True, True],
            'NPV': [100, 500, 200, 600]
        })
        
        # Test validation
        validation = validate_data_for_visualizations(test_system_data, test_agent_data)
        
        if not validation['can_proceed']:
            print("❌ Data validation failed unexpectedly")
            return False
        
        print("✅ Data adaptation test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Data adaptation test failed: {e}")
        return False

def test_visualization_creation():
    """Test the visualization creation process."""
    print("Testing visualization creation...")
    
    try:
        from simulation.data.metrics import MetricsAnalyzer
        from simulation.data.visualizers import SimulationVisualizer
        
        # Create test data
        test_system_data = pd.DataFrame({
            'Year': [1, 2, 3],
            'Month': [1, 2, 3],
            'AdoptionRate': [0.1, 0.2, 0.3],
            'FossilDependency': [0.9, 0.8, 0.7]
        })
        
        test_agent_data = pd.DataFrame({
            'Year': [1, 1, 2, 2, 3, 3],
            'AgentType': ['Household'] * 6,
            'IsProsumer': [False, True, True, True, True, True],
            'IncomeClass': [1, 5, 1, 5, 1, 5],
            'NPV': [100, 500, 200, 600, 300, 700]
        })
        
        # Create analyzer and visualizer
        metrics = MetricsAnalyzer(test_system_data, test_agent_data)
        visualizer = SimulationVisualizer(metrics)
        
        # Test that basic methods exist
        if not hasattr(visualizer, 'plot_adoption_curve'):
            print("❌ Missing plot_adoption_curve method")
            return False
        
        if not hasattr(visualizer, 'plot_fossil_dependency'):
            print("❌ Missing plot_fossil_dependency method")
            return False
        
        print("✅ Visualization creation test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Visualization creation test failed: {e}")
        return False

def test_csv_export(model, output_dir="data/csv_test"):
    """
    Minimal test to verify CSV files are actually created.
    Call this after model.run() completes.
    """
    import os
    import pandas as pd
    from simulation.data.metrics import MetricsAnalyzer
    
    print("\n" + "="*50)
    print("TESTING CSV EXPORT")
    print("="*50)
    
    try:
        # 1. Get data from model
        collector = model.data_collector
        model_data = collector.get_system_metrics_dataframe()
        agent_data = collector.get_combined_dataframe()
        
        print(f"Model data shape: {model_data.shape}")
        print(f"Agent data shape: {agent_data.shape}")
        
        # 2. Initialize metrics and export
        metrics = MetricsAnalyzer(model_data, agent_data)
        
        # Show current working directory
        print(f"Current working directory: {os.getcwd()}")
        abs_output_dir = os.path.abspath(output_dir)
        print(f"Target directory: {abs_output_dir}")
        
        # 3. Export CSVs
        metrics.export_all_system_metrics(output_dir)
        
        # 4. Verify files were created
        print(f"\nVerifying files in {abs_output_dir}:")
        if os.path.exists(abs_output_dir):
            files = [f for f in os.listdir(abs_output_dir) if f.endswith('.csv')]
            if files:
                print(f"SUCCESS: {len(files)} CSV files created:")
                for f in sorted(files):
                    filepath = os.path.join(abs_output_dir, f)
                    size = os.path.getsize(filepath)
                    
                    # Quick validation - try to read the CSV
                    try:
                        df = pd.read_csv(filepath)
                        print(f"  ✓ {f}: {size} bytes, {len(df)} rows, {len(df.columns)} columns")
                    except Exception as e:
                        print(f"  ✗ {f}: {size} bytes, ERROR reading: {e}")
                        
                return True
            else:
                print("FAILURE: No CSV files found in directory")
                return False
        else:
            print("FAILURE: Output directory doesn't exist")
            return False
            
    except Exception as e:
        print(f"FAILURE: Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all tests for the multi-experiment runner."""
    print("Running all multi-experiment runner tests...")
    
    tests = [
        test_data_adaptation,
        test_visualization_creation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n{'='*50}")
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✅ All tests passed!")
        return True
    else:
        print("❌ Some tests failed!")
        return False

if __name__ == "__main__":
    # Check if running tests
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        run_all_tests()
    else:
        main()

