# run_phase2_visualizations.py V1.0 - PHASE 2 INTEGRATION
"""
Phase 2 integration script for running all system-level and individual bias analyses.
Demonstrates how to use the enhanced visualizers together.
"""

import os
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from simulation.data.enhanced_system_visualizer import EnhancedSystemVisualizer
from simulation.data.individual_bias_analyzer import IndividualBiasAnalyzer
from simulation.data.multi_experiment_collector import MultiExperimentCollector
from simulation.models.multi_experiment_model import MultiExperimentModel
from simulation.utils.config_loader import create_multi_experiment_config

def run_phase2_analysis(config_path=None, output_base_dir="results/phase2"):
    """
    Run complete Phase 2 analysis pipeline.
    
    Args:
        config_path: Path to configuration file (optional)
        output_base_dir: Base directory for all outputs
    """
    print("🚀 Starting Phase 2 Multi-Scenario Analysis Pipeline")
    print("=" * 60)
    
    # Create output directories
    os.makedirs(output_base_dir, exist_ok=True)
    system_output_dir = f"{output_base_dir}/system_analysis"
    individual_output_dir = f"{output_base_dir}/individual_analysis"
    data_output_dir = f"{output_base_dir}/exported_data"
    
    os.makedirs(system_output_dir, exist_ok=True)
    os.makedirs(individual_output_dir, exist_ok=True)
    os.makedirs(data_output_dir, exist_ok=True)
    
    # Step 1: Create configuration
    print("📋 Step 1: Creating simulation configuration...")
    if config_path:
        # Load from file if provided
        config = load_config_from_file(config_path)
    else:
        # Use default multi-experiment configuration
        config = create_multi_experiment_config()
    
    config_dict = config.get_copy()
    print(f"   ✅ Configuration created with {len(config_dict.get('scenarios', []))} scenarios")
    
    # Step 2: Run simulation
    print("🔬 Step 2: Running multi-scenario simulation...")
    model = MultiExperimentModel(config_dict)
    
    # Run for specified number of steps (default: 24 months)
    total_steps = config_dict.get('run_settings', {}).get('total_steps', 24)
    print(f"   Running simulation for {total_steps} steps...")
    
    model.run(steps=total_steps)
    print(f"   ✅ Simulation completed")
    
    # Step 3: Export data
    print("💾 Step 3: Exporting simulation data...")
    data_collector = model.data_collector
    data_collector.export_all_scenarios(data_output_dir)
    print(f"   ✅ Data exported to {data_output_dir}")
    
    # Step 4: System-level analysis
    print("📊 Step 4: Running system-level analysis...")
    system_visualizer = EnhancedSystemVisualizer(data_collector, config_dict)
    system_visualizer.plot_all_system_analyses(system_output_dir)
    print(f"   ✅ System analysis completed in {system_output_dir}")
    
    # Step 5: Individual bias analysis
    print("🧠 Step 5: Running individual bias analysis...")
    individual_analyzer = IndividualBiasAnalyzer(data_collector, config_dict)
    individual_analyzer.plot_all_individual_analyses(individual_output_dir)
    print(f"   ✅ Individual analysis completed in {individual_output_dir}")
    
    # Step 6: Generate summary report
    print("📑 Step 6: Generating analysis summary...")
    generate_phase2_summary(data_collector, output_base_dir)
    print(f"   ✅ Summary report generated")
    
    print("=" * 60)
    print("🎉 Phase 2 Analysis Pipeline Completed!")
    print(f"📁 All results saved to: {output_base_dir}")
    print("\nGenerated outputs:")
    print(f"  • System analysis plots: {system_output_dir}")
    print(f"  • Individual bias plots: {individual_output_dir}")
    print(f"  • Raw data exports: {data_output_dir}")
    print(f"  • Summary report: {output_base_dir}/phase2_summary.txt")
    
    return data_collector, system_visualizer, individual_analyzer

def generate_phase2_summary(data_collector, output_dir):
    """
    Generate a text summary of the Phase 2 analysis results.
    
    Args:
        data_collector: MultiExperimentCollector instance
        output_dir: Output directory for summary file
    """
    summary_path = f"{output_dir}/phase2_summary.txt"
    
    # Get data summary
    data_summary = data_collector.get_data_summary()
    adoption_summary = data_collector.get_adoption_summary()
    
    with open(summary_path, 'w') as f:
        f.write("PHASE 2 MULTI-SCENARIO BEHAVIORAL ANALYSIS SUMMARY\n")
        f.write("=" * 55 + "\n\n")
        
        # Data collection summary
        f.write("DATA COLLECTION SUMMARY\n")
        f.write("-" * 25 + "\n")
        f.write(f"Scenarios tracked: {data_summary['scenarios_tracked']}\n")
        f.write(f"Scenario names: {', '.join(data_summary['scenario_names'])}\n")
        f.write(f"Combined records: {data_summary['combined_records']}\n")
        f.write(f"System metrics records: {data_summary['system_metrics_records']}\n")
        f.write(f"Bias effects records: {data_summary['bias_effects_records']}\n\n")
        
        # Records per scenario
        f.write("RECORDS PER SCENARIO\n")
        f.write("-" * 20 + "\n")
        for scenario, count in data_summary['data_records_by_scenario'].items():
            f.write(f"  {scenario}: {count} records\n")
        f.write("\n")
        
        # Adoption summary
        if 'scenario_adoption_rates' in adoption_summary:
            f.write("FINAL ADOPTION RATES\n")
            f.write("-" * 20 + "\n")
            for scenario, rate in adoption_summary['scenario_adoption_rates'].items():
                f.write(f"  {scenario}: {rate:.2%}\n")
            f.write("\n")
        
        # Generated visualizations
        f.write("GENERATED VISUALIZATIONS\n")
        f.write("-" * 24 + "\n")
        f.write("System-Level Analysis (Enhanced):\n")
        f.write("  • Energy Balance Evolution (Multi-Scenario)\n")
        f.write("  • Credit System Utilization (Multi-Scenario)\n")
        f.write("  • Energy Cost Distribution Evolution\n")
        f.write("  • Seasonal Grid Stress (Multi-Scenario)\n\n")
        
        f.write("Individual Bias Analysis (New):\n")
        f.write("  • Individual Bias Time Series (2x2 Grid)\n")
        f.write("  • Decision Process Analysis (2x3 Grid)\n")
        f.write("  • Adoption Velocity by Class (2x2 Grid)\n\n")
        
        # Key insights placeholder
        f.write("KEY INSIGHTS (for manual review)\n")
        f.write("-" * 32 + "\n")
        f.write("1. Review system-level plots for cross-scenario differences\n")
        f.write("2. Examine individual bias effects for validation\n")
        f.write("3. Analyze adoption velocity patterns by income class\n")
        f.write("4. Validate decision process mechanisms\n\n")
        
        f.write("Analysis completed successfully!\n")

def test_phase2_implementation():
    """
    Test the Phase 2 implementation with a small simulation.
    """
    print("🧪 Testing Phase 2 Implementation...")
    
    try:
        # Create minimal test configuration
        config = create_multi_experiment_config()
        config_dict = config.get_copy()
        
        # Reduce simulation size for testing
        config_dict['household_count'] = 10
        config_dict['run_settings']['total_steps'] = 6
        
        # Run test analysis
        test_output_dir = "results/phase2_test"
        data_collector, system_viz, individual_viz = run_phase2_analysis(
            config_path=None, 
            output_base_dir=test_output_dir
        )
        
        # Verify outputs exist
        expected_files = [
            f"{test_output_dir}/system_analysis/energy_balance_evolution_multi_scenario.png",
            f"{test_output_dir}/system_analysis/credit_system_utilization_multi_scenario.png",
            f"{test_output_dir}/individual_analysis/individual_bias_time_series.png",
            f"{test_output_dir}/individual_analysis/decision_process_analysis.png",
            f"{test_output_dir}/phase2_summary.txt"
        ]
        
        missing_files = []
        for file_path in expected_files:
            if not os.path.exists(file_path):
                missing_files.append(file_path)
        
        if missing_files:
            print("❌ Test failed - missing files:")
            for file_path in missing_files:
                print(f"   • {file_path}")
            return False
        else:
            print("✅ All Phase 2 tests passed!")
            print(f"   Test results available in: {test_output_dir}")
            return True
            
    except Exception as e:
        print(f"❌ Phase 2 test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def load_config_from_file(config_path):
    """
    Load configuration from file (placeholder for custom configurations).
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        SimulationConfig: Loaded configuration
    """
    # Placeholder - implement custom config loading if needed
    print(f"   Loading configuration from {config_path}...")
    # For now, return default config
    return create_multi_experiment_config()

if __name__ == "__main__":
    """
    Main execution - supports both testing and full analysis.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Phase 2 Multi-Scenario Analysis')
    parser.add_argument('--test', action='store_true', 
                       help='Run test implementation with small simulation')
    parser.add_argument('--config', type=str, 
                       help='Path to configuration file')
    parser.add_argument('--output', type=str, default='results/phase2',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    if args.test:
        # Run test implementation
        success = test_phase2_implementation()
        sys.exit(0 if success else 1)
    else:
        # Run full analysis
        try:
            run_phase2_analysis(
                config_path=args.config,
                output_base_dir=args.output
            )
        except KeyboardInterrupt:
            print("\n🛑 Analysis interrupted by user")
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)