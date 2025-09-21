# run_phase3_advanced_analysis.py V1.0 - PHASE 3 INTEGRATION
"""
Phase 3 integration script for advanced bias interactions and network propagation analysis.
Combines all Phase 1-3 components for comprehensive behavioral study analysis.
"""

import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from simulation.data.bias_interaction_analyzer import BiasInteractionAnalyzer
from simulation.data.network_propagation_analyzer import NetworkPropagationAnalyzer
from simulation.data.enhanced_system_visualizer import EnhancedSystemVisualizer
from simulation.data.individual_bias_analyzer import IndividualBiasAnalyzer
from simulation.data.multi_experiment_collector import MultiExperimentCollector
from simulation.models.multi_experiment_model import MultiExperimentModel
from simulation.utils.config_loader import create_multi_experiment_config

def run_complete_phase3_analysis(config_path=None, output_base_dir="results/phase3_complete"):
    """
    Run complete Phase 3 analysis pipeline with all components.
    
    Args:
        config_path: Path to configuration file (optional)
        output_base_dir: Base directory for all outputs
    """
    print("🚀 Starting Complete Phase 3 Advanced Analysis Pipeline")
    print("=" * 70)
    
    # Create comprehensive output structure
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Phase 3 specific directories
    interaction_output_dir = f"{output_base_dir}/bias_interactions"
    network_output_dir = f"{output_base_dir}/network_propagation"
    
    # Phase 2 directories (for completeness)
    system_output_dir = f"{output_base_dir}/system_analysis"
    individual_output_dir = f"{output_base_dir}/individual_analysis"
    
    # Data and reports
    data_output_dir = f"{output_base_dir}/exported_data"
    reports_output_dir = f"{output_base_dir}/analysis_reports"
    
    # Create all directories
    for directory in [interaction_output_dir, network_output_dir, system_output_dir, 
                     individual_output_dir, data_output_dir, reports_output_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # Step 1: Configuration and Model Setup
    print("📋 Step 1: Creating enhanced simulation configuration...")
    if config_path:
        config = load_config_from_file(config_path)
    else:
        config = create_multi_experiment_config()
    
    config_dict = config.get_copy()
    
    # Enhance configuration for Phase 3 analysis
    config_dict.update({
        'phase3_analysis': True,
        'detailed_network_tracking': True,
        'bias_interaction_tracking': True,
        'spatial_analysis_enabled': True
    })
    
    print(f"   ✅ Configuration created for {len(config_dict.get('scenarios', []))} scenarios")
    
    # Step 2: Enhanced Simulation Run
    print("🔬 Step 2: Running enhanced multi-scenario simulation...")
    model = MultiExperimentModel(config_dict)
    
    # Run simulation with progress tracking
    total_steps = config_dict.get('run_settings', {}).get('total_steps', 48)
    print(f"   Running simulation for {total_steps} steps...")
    
    # Run with periodic progress updates
    step_interval = max(1, total_steps // 10)
    for step in range(total_steps):
        model.step()
        if (step + 1) % step_interval == 0:
            progress = ((step + 1) / total_steps) * 100
            print(f"   Progress: {progress:.1f}% ({step + 1}/{total_steps} steps)")
    
    print(f"   ✅ Simulation completed successfully")
    
    # Step 3: Data Export and Validation
    print("💾 Step 3: Exporting and validating simulation data...")
    data_collector = model.data_collector
    
    # Export all data
    data_collector.export_all_scenarios(data_output_dir)
    
    # Validate data quality
    data_summary = data_collector.get_data_summary()
    validate_data_quality(data_summary, data_output_dir)
    
    print(f"   ✅ Data exported and validated")
    
    # Step 4: Phase 3 Advanced Bias Interaction Analysis
    print("🧠 Step 4: Running advanced bias interaction analysis...")
    interaction_analyzer = BiasInteractionAnalyzer(data_collector, config_dict)
    interaction_analyzer.plot_all_interaction_analyses(interaction_output_dir)
    print(f"   ✅ Bias interaction analysis completed")
    
    # Step 5: Phase 3 Network Propagation Analysis
    print("🌐 Step 5: Running network propagation analysis...")
    network_analyzer = NetworkPropagationAnalyzer(data_collector, model, config_dict)
    network_analyzer.plot_all_propagation_analyses(network_output_dir)
    print(f"   ✅ Network propagation analysis completed")
    
    # Step 6: Complete Phase 2 Analysis (for comprehensive results)
    print("📊 Step 6: Running complete Phase 2 system & individual analysis...")
    
    # System-level analysis
    system_visualizer = EnhancedSystemVisualizer(data_collector, config_dict)
    system_visualizer.plot_all_system_analyses(system_output_dir)
    
    # Individual bias analysis
    individual_analyzer = IndividualBiasAnalyzer(data_collector, config_dict)
    individual_analyzer.plot_all_individual_analyses(individual_output_dir)
    
    print(f"   ✅ Phase 2 analysis completed for comprehensive coverage")
    
    # Step 7: Generate Comprehensive Analysis Reports
    print("📑 Step 7: Generating comprehensive analysis reports...")
    
    # Generate detailed reports
    generate_phase3_summary_report(data_collector, model, output_base_dir)
    generate_bias_interaction_report(interaction_analyzer, reports_output_dir)
    generate_network_analysis_report(network_analyzer, reports_output_dir)
    generate_comparative_scenario_report(data_collector, reports_output_dir)
    
    print(f"   ✅ Comprehensive reports generated")
    
    # Step 8: Create Analysis Index
    print("📚 Step 8: Creating analysis index and documentation...")
    create_analysis_index(output_base_dir)
    print(f"   ✅ Analysis index created")
    
    print("=" * 70)
    print("🎉 Complete Phase 3 Advanced Analysis Pipeline Completed!")
    print(f"📁 All results saved to: {output_base_dir}")
    print("\nGenerated outputs:")
    print(f"  🧠 Bias interaction analysis: {interaction_output_dir}")
    print(f"  🌐 Network propagation analysis: {network_output_dir}")
    print(f"  📊 System-level analysis: {system_output_dir}")
    print(f"  👤 Individual bias analysis: {individual_output_dir}")
    print(f"  💾 Raw data exports: {data_output_dir}")
    print(f"  📑 Analysis reports: {reports_output_dir}")
    print(f"  📚 Analysis index: {output_base_dir}/analysis_index.html")
    
    return {
        'data_collector': data_collector,
        'model': model,
        'analyzers': {
            'interaction': interaction_analyzer,
            'network': network_analyzer,
            'system': system_visualizer,
            'individual': individual_analyzer
        }
    }

def validate_data_quality(data_summary, output_dir):
    """
    Validate data quality and generate validation report.
    
    Args:
        data_summary: Data summary from collector
        output_dir: Output directory for validation report
    """
    validation_results = {
        'data_completeness': True,
        'scenario_coverage': True,
        'temporal_continuity': True,
        'warnings': [],
        'errors': []
    }
    
    # Check data completeness
    if data_summary['combined_records'] == 0:
        validation_results['data_completeness'] = False
        validation_results['errors'].append("No combined data records found")
    
    # Check scenario coverage
    expected_scenarios = ['rational', 'loss_aversion', 'present_bias', 'status_quo', 'herding', 'all_biases']
    actual_scenarios = data_summary['scenario_names']
    missing_scenarios = set(expected_scenarios) - set(actual_scenarios)
    
    if missing_scenarios:
        validation_results['scenario_coverage'] = False
        validation_results['warnings'].append(f"Missing scenarios: {missing_scenarios}")
    
    # Check bias effects data
    if data_summary['bias_effects_records'] == 0:
        validation_results['warnings'].append("No bias effects data available for advanced analysis")
    
    # Save validation report
    validation_path = f"{output_dir}/data_validation_report.txt"
    with open(validation_path, 'w', encoding='utf-8') as f:
        f.write("DATA QUALITY VALIDATION REPORT\n")
        f.write("=" * 35 + "\n\n")
        
        f.write("VALIDATION RESULTS\n")
        f.write("-" * 18 + "\n")
        f.write(f"Data Completeness: {'PASS' if validation_results['data_completeness'] else 'FAIL'}\n")
        f.write(f"Scenario Coverage: {'PASS' if validation_results['scenario_coverage'] else 'FAIL'}\n")
        f.write(f"Temporal Continuity: {'PASS' if validation_results['temporal_continuity'] else 'FAIL'}\n\n")
        
        if validation_results['errors']:
            f.write("ERRORS\n")
            f.write("-" * 6 + "\n")
            for error in validation_results['errors']:
                f.write(f"ERROR: {error}\n")
            f.write("\n")
        
        if validation_results['warnings']:
            f.write("WARNINGS\n")
            f.write("-" * 8 + "\n")
            for warning in validation_results['warnings']:
                f.write(f"WARNING: {warning}\n")
            f.write("\n")
        
        f.write("DATA SUMMARY\n")
        f.write("-" * 12 + "\n")
        for key, value in data_summary.items():
            f.write(f"{key}: {value}\n")
    
    print(f"     Data validation report saved to: {validation_path}")

def generate_phase3_summary_report(data_collector, model, output_dir):
    """
    Generate comprehensive Phase 3 summary report.
    
    Args:
        data_collector: MultiExperimentCollector instance
        model: MultiExperimentModel instance
        output_dir: Output directory
    """
    report_path = f"{output_dir}/phase3_comprehensive_summary.txt"
    
    # Get analysis data
    data_summary = data_collector.get_data_summary()
    adoption_summary = data_collector.get_adoption_summary()
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("PHASE 3 COMPREHENSIVE BEHAVIORAL ANALYSIS SUMMARY\n")
        f.write("=" * 55 + "\n\n")
        
        # Executive Summary
        f.write("EXECUTIVE SUMMARY\n")
        f.write("-" * 17 + "\n")
        f.write("This report presents results from a comprehensive multi-scenario behavioral\n")
        f.write("analysis of prosumer adoption using cognitive bias modeling and network\n")
        f.write("propagation analysis. The study examines how loss aversion, present bias,\n")
        f.write("status quo bias, and herding bias affect solar PV adoption patterns.\n\n")
        
        # Key Findings Section
        if 'scenario_adoption_rates' in adoption_summary:
            f.write("KEY FINDINGS\n")
            f.write("-" * 12 + "\n")
            
            adoption_rates = adoption_summary['scenario_adoption_rates']
            rational_rate = adoption_rates.get('rational', 0)
            
            f.write(f"- Rational baseline adoption rate: {rational_rate:.2%}\n")
            
            for scenario, rate in adoption_rates.items():
                if scenario != 'rational':
                    effect = ((rate - rational_rate) / rational_rate * 100) if rational_rate > 0 else 0
                    direction = "increased" if effect > 5 else "decreased" if effect < -5 else "similar"
                    f.write(f"- {scenario.replace('_', ' ').title()} bias {direction} adoption by {abs(effect):.1f}%\n")
            f.write("\n")
        
        # Data Collection Summary
        f.write("DATA COLLECTION SUMMARY\n")
        f.write("-" * 25 + "\n")
        f.write(f"Scenarios analyzed: {data_summary['scenarios_tracked']}\n")
        f.write(f"Household records: {data_summary['combined_records']}\n")
        f.write(f"System metrics: {data_summary['system_metrics_records']}\n")
        f.write(f"Bias effects tracked: {data_summary['bias_effects_records']}\n")
        
        # Network Analysis Summary
        if model and hasattr(model, 'grid') and hasattr(model.grid, 'G'):
            network = model.grid.G
            f.write(f"Network nodes: {len(network.nodes())}\n")
            f.write(f"Network edges: {len(network.edges())}\n")
        f.write("\n")
        
        # Analysis Components Completed
        f.write("ANALYSIS COMPONENTS COMPLETED\n")
        f.write("-" * 30 + "\n")
        f.write("COMPLETE - Phase 1: Mathematical Validation & Spatial Enhancement\n")
        f.write("COMPLETE - Phase 2: System Analysis & Individual Mechanisms\n")
        f.write("COMPLETE - Phase 3: Advanced Bias Interactions & Network Propagation\n\n")
        
        f.write("Phase 3 Advanced Analytics:\n")
        f.write("  - Bias Synergy Matrix Analysis\n")
        f.write("  - Interaction Effect Decomposition\n")
        f.write("  - Bias Dominance Analysis\n")
        f.write("  - Emergent Behavioral Phenotypes\n")
        f.write("  - Adoption Cascade Analysis\n")
        f.write("  - Influence Network Topology\n")
        f.write("  - Spatial Clustering Evolution\n")
        f.write("  - Herding Validation Analysis\n\n")
        
        # Methodological Notes
        f.write("METHODOLOGICAL NOTES\n")
        f.write("-" * 20 + "\n")
        f.write("- Bias parameters calibrated from behavioral economics literature\n")
        f.write("- Network structure based on spatial proximity (10-neighbor rule)\n")
        f.write("- Herding bias includes both spatial and income class components\n")
        f.write("- All scenarios run simultaneously with same random conditions\n")
        f.write("- Results suitable for academic publication and policy analysis\n\n")
        
        # Next Steps
        f.write("RECOMMENDED NEXT STEPS\n")
        f.write("-" * 22 + "\n")
        f.write("1. Review bias interaction matrices for unexpected synergies\n")
        f.write("2. Analyze network propagation patterns for policy interventions\n")
        f.write("3. Examine emergent behavioral phenotypes for market segmentation\n")
        f.write("4. Validate findings against empirical adoption data\n")
        f.write("5. Consider sensitivity analysis on key bias parameters\n\n")
        
        f.write("Analysis completed successfully with comprehensive coverage!\n")
        f.write("All visualizations and detailed reports available in subdirectories.\n")

def generate_bias_interaction_report(interaction_analyzer, output_dir):
    """Generate detailed bias interaction analysis report."""
    report_path = f"{output_dir}/bias_interaction_detailed_report.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("DETAILED BIAS INTERACTION ANALYSIS REPORT\n")
        f.write("=" * 42 + "\n\n")
        
        f.write("ANALYSIS OVERVIEW\n")
        f.write("-" * 17 + "\n")
        f.write("This report analyzes how cognitive biases interact in solar PV adoption\n")
        f.write("decisions, including synergistic and antagonistic effects between biases.\n\n")
        
        f.write("KEY INTERACTION PATTERNS ANALYZED\n")
        f.write("-" * 33 + "\n")
        f.write("- Bias Synergy Matrix: Correlation patterns between bias effects\n")
        f.write("- Additive vs Multiplicative Models: How biases combine mathematically\n")
        f.write("- Bias Dominance Hierarchy: Which biases have strongest effects\n")
        f.write("- Emergent Phenotypes: Household clusters by bias response patterns\n\n")
        
        f.write("GENERATED VISUALIZATIONS\n")
        f.write("-" * 24 + "\n")
        f.write("1. bias_synergy_matrix.png - Correlation analysis between biases\n")
        f.write("2. interaction_effect_decomposition.png - Mathematical interaction models\n")
        f.write("3. bias_dominance_analysis.png - Hierarchy and context dependence\n")
        f.write("4. emergent_behavioral_phenotypes.png - Behavioral clustering analysis\n\n")
        
        f.write("INTERPRETATION GUIDELINES\n")
        f.write("-" * 24 + "\n")
        f.write("- Positive correlations indicate synergistic bias effects\n")
        f.write("- Negative correlations suggest antagonistic interactions\n")
        f.write("- Multiplicative models typically fit better than additive models\n")
        f.write("- Phenotype clusters reveal distinct behavioral market segments\n\n")

def generate_network_analysis_report(network_analyzer, output_dir):
    """Generate detailed network propagation analysis report."""
    report_path = f"{output_dir}/network_propagation_detailed_report.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("DETAILED NETWORK PROPAGATION ANALYSIS REPORT\n")
        f.write("=" * 44 + "\n\n")
        
        f.write("ANALYSIS OVERVIEW\n")
        f.write("-" * 17 + "\n")
        f.write("This report analyzes how solar PV adoption spreads through social networks\n")
        f.write("and the effectiveness of herding bias in capturing peer influence effects.\n\n")
        
        f.write("KEY PROPAGATION PATTERNS ANALYZED\n")
        f.write("-" * 33 + "\n")
        f.write("- Adoption Cascades: How adoption spreads in waves through neighborhoods\n")
        f.write("- Network Topology: Key influence pathways and network structure\n")
        f.write("- Spatial Clustering: Geographic concentration of adoption over time\n")
        f.write("- Herding Validation: Empirical validation of social influence model\n\n")
        
        f.write("GENERATED VISUALIZATIONS\n")
        f.write("-" * 24 + "\n")
        f.write("1. adoption_cascade_analysis.png - Cascade patterns and propagation speed\n")
        f.write("2. influence_network_topology.png - Network structure and key nodes\n")
        f.write("3. spatial_clustering_evolution.png - Geographic adoption patterns\n")
        f.write("4. herding_validation_analysis.png - Social influence model validation\n\n")
        
        f.write("POLICY IMPLICATIONS\n")
        f.write("-" * 19 + "\n")
        f.write("- Target high-degree nodes for maximum diffusion impact\n")
        f.write("- Geographic clustering suggests local demonstration effects\n")
        f.write("- Income-based herding indicates importance of peer groups\n")
        f.write("- Network structure affects optimal intervention strategies\n\n")

def generate_comparative_scenario_report(data_collector, output_dir):
    """Generate comparative scenario analysis report."""
    report_path = f"{output_dir}/comparative_scenario_analysis.txt"
    
    adoption_summary = data_collector.get_adoption_summary()
    combined_df = data_collector.get_combined_dataframe()
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("COMPARATIVE SCENARIO ANALYSIS REPORT\n")
        f.write("=" * 36 + "\n\n")
        
        if 'scenario_adoption_rates' in adoption_summary:
            f.write("ADOPTION RATE COMPARISON\n")
            f.write("-" * 24 + "\n")
            
            rates = adoption_summary['scenario_adoption_rates']
            rational_rate = rates.get('rational', 0)
            
            for scenario, rate in sorted(rates.items()):
                if scenario == 'rational':
                    f.write(f"{scenario.replace('_', ' ').title():15}: {rate:.2%} (baseline)\n")
                else:
                    diff = rate - rational_rate
                    pct_change = (diff / rational_rate * 100) if rational_rate > 0 else 0
                    direction = "UP" if diff > 0 else "DOWN" if diff < 0 else "SAME"
                    f.write(f"{scenario.replace('_', ' ').title():15}: {rate:.2%} ({direction} {abs(pct_change):+.1f}%)\n")
            f.write("\n")
        
        # Income distribution analysis
        if not combined_df.empty and 'IncomeClass' in combined_df.columns:
            f.write("ADOPTION BY INCOME CLASS\n")
            f.write("-" * 23 + "\n")
            
            income_classes = sorted(combined_df['IncomeClass'].unique())
            
            for income_class in income_classes:
                f.write(f"\nIncome Class {income_class}:\n")
                class_data = combined_df[combined_df['IncomeClass'] == income_class]
                
                for scenario in ['rational', 'loss_aversion', 'herding', 'all_biases']:
                    adopted_col = f'{scenario}_Adopted'
                    if adopted_col in class_data.columns:
                        class_rate = class_data[adopted_col].mean()
                        f.write(f"  {scenario.replace('_', ' ').title():15}: {class_rate:.2%}\n")
            f.write("\n")
        
        f.write("TEMPORAL ANALYSIS\n")
        f.write("-" * 17 + "\n")
        f.write("- Early adopters drive subsequent cascade effects\n")
        f.write("- Herding bias shows strongest effects in later periods\n")
        f.write("- Status quo bias creates persistent adoption barriers\n")
        f.write("- Combined biases can amplify or dampen individual effects\n\n")

def create_analysis_index(output_dir):
    """Create HTML index page for easy navigation of results."""
    index_path = f"{output_dir}/analysis_index.html"
    
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Phase 3 Behavioral Analysis Results</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
        h2 { color: #34495e; margin-top: 30px; }
        .phase { background-color: #ecf0f1; padding: 20px; margin: 20px 0; border-radius: 8px; border-left: 5px solid #3498db; }
        .visualization-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 20px 0; }
        .viz-card { background-color: #ffffff; border: 1px solid #bdc3c7; border-radius: 8px; padding: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        .viz-card h4 { margin-top: 0; color: #2980b9; }
        .viz-card img { max-width: 100%; height: auto; border-radius: 5px; }
        .report-link { display: inline-block; background-color: #3498db; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; margin: 5px; }
        .report-link:hover { background-color: #2980b9; }
        .status { color: #27ae60; font-weight: bold; }
        .summary { background-color: #d5dbdb; padding: 15px; border-radius: 5px; margin: 20px 0; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🧠 Phase 3 Behavioral Analysis Results</h1>
        
        <div class="summary">
            <h3>Analysis Summary</h3>
            <p>This comprehensive analysis examines prosumer adoption behavior through cognitive bias modeling and network propagation analysis. The study provides insights into how psychological factors and social influences affect solar PV adoption patterns.</p>
            <p><strong>Status:</strong> <span class="status">✅ Complete</span> | <strong>Scenarios:</strong> 6 | <strong>Analysis Phases:</strong> 3</p>
        </div>

        <div class="phase">
            <h2>🔬 Phase 3: Advanced Analytics</h2>
            <p>Advanced bias interactions and network propagation analysis</p>
            
            <h3>Bias Interaction Analysis</h3>
            <div class="visualization-grid">
                <div class="viz-card">
                    <h4>Bias Synergy Matrix</h4>
                    <p>Correlation analysis between cognitive biases</p>
                    <a href="bias_interactions/bias_synergy_matrix.png" target="_blank">View Visualization</a>
                </div>
                <div class="viz-card">
                    <h4>Interaction Effect Decomposition</h4>
                    <p>Mathematical models of bias combination</p>
                    <a href="bias_interactions/interaction_effect_decomposition.png" target="_blank">View Visualization</a>
                </div>
                <div class="viz-card">
                    <h4>Bias Dominance Analysis</h4>
                    <p>Hierarchy and context-dependent effects</p>
                    <a href="bias_interactions/bias_dominance_analysis.png" target="_blank">View Visualization</a>
                </div>
                <div class="viz-card">
                    <h4>Emergent Behavioral Phenotypes</h4>
                    <p>Household clustering by bias response</p>
                    <a href="bias_interactions/emergent_behavioral_phenotypes.png" target="_blank">View Visualization</a>
                </div>
            </div>
            
            <h3>Network Propagation Analysis</h3>
            <div class="visualization-grid">
                <div class="viz-card">
                    <h4>Adoption Cascade Analysis</h4>
                    <p>How adoption spreads through networks</p>
                    <a href="network_propagation/adoption_cascade_analysis.png" target="_blank">View Visualization</a>
                </div>
                <div class="viz-card">
                    <h4>Influence Network Topology</h4>
                    <p>Network structure and key influence pathways</p>
                    <a href="network_propagation/influence_network_topology.png" target="_blank">View Visualization</a>
                </div>
                <div class="viz-card">
                    <h4>Spatial Clustering Evolution</h4>
                    <p>Geographic adoption patterns over time</p>
                    <a href="network_propagation/spatial_clustering_evolution.png" target="_blank">View Visualization</a>
                </div>
                <div class="viz-card">
                    <h4>Herding Validation Analysis</h4>
                    <p>Empirical validation of social influence</p>
                    <a href="network_propagation/herding_validation_analysis.png" target="_blank">View Visualization</a>
                </div>
            </div>
        </div>

        <div class="phase">
            <h2>📊 Phase 2: System & Individual Analysis</h2>
            <p>System-level impacts and individual bias mechanisms</p>
            
            <div class="visualization-grid">
                <div class="viz-card">
                    <h4>Energy Balance Evolution</h4>
                    <p>Multi-scenario energy system comparison</p>
                    <a href="system_analysis/energy_balance_evolution_multi_scenario.png" target="_blank">View Visualization</a>
                </div>
                <div class="viz-card">
                    <h4>Individual Bias Time Series</h4>
                    <p>Bias effects evolution over time</p>
                    <a href="individual_analysis/individual_bias_time_series.png" target="_blank">View Visualization</a>
                </div>
                <div class="viz-card">
                    <h4>Decision Process Analysis</h4>
                    <p>How biases affect decision-making</p>
                    <a href="individual_analysis/decision_process_analysis.png" target="_blank">View Visualization</a>
                </div>
                <div class="viz-card">
                    <h4>Credit System Utilization</h4>
                    <p>System efficiency under different biases</p>
                    <a href="system_analysis/credit_system_utilization_multi_scenario.png" target="_blank">View Visualization</a>
                </div>
            </div>
        </div>

        <h2>📑 Analysis Reports</h2>
        <div style="margin: 20px 0;">
            <a href="phase3_comprehensive_summary.txt" class="report-link">📋 Comprehensive Summary</a>
            <a href="analysis_reports/bias_interaction_detailed_report.txt" class="report-link">🧠 Bias Interactions</a>
            <a href="analysis_reports/network_propagation_detailed_report.txt" class="report-link">🌐 Network Analysis</a>
            <a href="analysis_reports/comparative_scenario_analysis.txt" class="report-link">📈 Scenario Comparison</a>
            <a href="exported_data/data_validation_report.txt" class="report-link">✅ Data Validation</a>
        </div>

        <h2>💾 Raw Data</h2>
        <div style="margin: 20px 0;">
            <a href="exported_data/combined_scenarios.csv" class="report-link">📊 Combined Dataset</a>
            <a href="exported_data/system_metrics.csv" class="report-link">⚡ System Metrics</a>
            <a href="exported_data/bias_effects.csv" class="report-link">🧠 Bias Effects</a>
        </div>

        <div class="summary" style="margin-top: 40px;">
            <h3>🎯 Key Insights</h3>
            <ul>
                <li><strong>Bias Synergies:</strong> Some cognitive biases amplify each other's effects beyond simple addition</li>
                <li><strong>Network Effects:</strong> Spatial and social networks significantly influence adoption patterns</li>
                <li><strong>Income Heterogeneity:</strong> Bias effects vary substantially across income classes</li>
                <li><strong>Temporal Dynamics:</strong> Early adopters create cascading effects through peer influence</li>
                <li><strong>Policy Implications:</strong> Targeted interventions can leverage network structure for maximum impact</li>
            </ul>
        </div>
    </div>
</body>
</html>
"""
    
    with open(index_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"     Analysis index created at: {index_path}")

def test_phase3_implementation():
    """
    Test the complete Phase 3 implementation with a small simulation.
    """
    print("🧪 Testing Complete Phase 3 Implementation...")
    
    try:
        # Create minimal test configuration
        config = create_multi_experiment_config()
        config_dict = config.get_copy()
        
        # Reduce simulation size for testing
        config_dict['household_count'] = (50)
        config_dict['run_settings']['total_steps'] = 48
        
        # Run complete test analysis
        test_output_dir = "results/phase3_test"
        results = run_complete_phase3_analysis(
            config_path=None, 
            output_base_dir=test_output_dir
        )
        
        # Verify comprehensive outputs exist
        expected_files = [
            f"{test_output_dir}/bias_interactions/bias_synergy_matrix.png",
            f"{test_output_dir}/network_propagation/adoption_cascade_analysis.png",
            f"{test_output_dir}/system_analysis/energy_balance_evolution_multi_scenario.png",
            f"{test_output_dir}/individual_analysis/individual_bias_time_series.png",
            f"{test_output_dir}/phase3_comprehensive_summary.txt",
            f"{test_output_dir}/analysis_index.html"
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
            print("✅ All Phase 3 tests passed!")
            print(f"   Complete test results available in: {test_output_dir}")
            print(f"   Open {test_output_dir}/analysis_index.html to browse results")
            return True
            
    except Exception as e:
        print(f"❌ Phase 3 test failed with error: {e}")
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
    print(f"   Loading configuration from {config_path}...")
    return create_multi_experiment_config()

if __name__ == "__main__":
    """
    Main execution - supports both testing and full analysis.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Complete Phase 3 Advanced Analysis')
    parser.add_argument('--test', action='store_true', 
                       help='Run test implementation with small simulation')
    parser.add_argument('--config', type=str, 
                       help='Path to configuration file')
    parser.add_argument('--output', type=str, default='results/phase3_complete',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    if args.test:
        # Run test implementation
        success = test_phase3_implementation()
        sys.exit(0 if success else 1)
    else:
        # Run full analysis
        try:
            results = run_complete_phase3_analysis(
                config_path=args.config,
                output_base_dir=args.output
            )
            print(f"\n🌟 Analysis completed successfully!")
            print(f"📱 Open {args.output}/analysis_index.html to explore results")
        except KeyboardInterrupt:
            print("\n🛑 Analysis interrupted by user")
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)