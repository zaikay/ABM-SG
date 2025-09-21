#!/usr/bin/env python3
"""
Debug script to analyze exported data and identify missing columns for visualizations.
Run this script in your results folder to diagnose the missing visualization issue.
"""

import os
import pandas as pd
import glob

def analyze_exported_data(results_dir):
    """
    Analyze the exported data to identify what's missing for visualizations.
    
    Args:
        results_dir: Path to your results directory
    """
    print("🔍 DEBUGGING EXPORTED DATA")
    print("=" * 50)
    
    # Find the most recent experiment folder
    experiment_folders = glob.glob(os.path.join(results_dir, "multi_experiment_*"))
    if not experiment_folders:
        print("❌ No multi_experiment folders found!")
        return
    
    latest_folder = max(experiment_folders, key=os.path.getctime)
    data_folder = os.path.join(latest_folder, "data")
    
    print(f"📁 Analyzing: {data_folder}")
    print()
    
    # Check for required files
    required_files = [
        "combined_scenarios.csv",
        "system_metrics.csv",
        "bias_effects.csv"
    ]
    
    print("📋 FILE EXISTENCE CHECK:")
    for file in required_files:
        file_path = os.path.join(data_folder, file)
        exists = "✅" if os.path.exists(file_path) else "❌"
        print(f"  {exists} {file}")
        
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                print(f"      Shape: {df.shape}")
                if len(df) > 0:
                    print(f"      Columns: {len(df.columns)}")
                else:
                    print("      ⚠️  Empty file!")
            except Exception as e:
                print(f"      ❌ Error reading: {e}")
    
    print()
    
    # Analyze system_metrics.csv in detail
    system_metrics_path = os.path.join(data_folder, "system_metrics.csv")
    if os.path.exists(system_metrics_path):
        print("🔬 SYSTEM METRICS ANALYSIS:")
        try:
            system_df = pd.read_csv(system_metrics_path)
            print(f"  Shape: {system_df.shape}")
            print(f"  Years: {system_df['Year'].min() if 'Year' in system_df.columns else 'No Year column'} to {system_df['Year'].max() if 'Year' in system_df.columns else ''}")
            
            # Check for adoption rate columns
            adoption_cols = [col for col in system_df.columns if 'AdoptionRate' in col]
            print(f"  📊 Adoption Rate Columns ({len(adoption_cols)}):")
            for col in adoption_cols:
                final_rate = system_df[col].iloc[-1] if len(system_df) > 0 else 0
                print(f"    • {col}: {final_rate:.1%}")
            
            # Check for income class columns
            class_cols = [col for col in system_df.columns if 'Class' in col and 'Rate' in col]
            print(f"  👥 Income Class Columns ({len(class_cols)}):")
            for col in sorted(class_cols)[:10]:  # Show first 10
                print(f"    • {col}")
            if len(class_cols) > 10:
                print(f"    ... and {len(class_cols) - 10} more")
            
            # Check for system impact columns
            system_cols = [col for col in system_df.columns if any(keyword in col for keyword in 
                          ['TotalGeneration', 'SystemCostSavings', 'PeakLoad', 'FossilDependency'])]
            print(f"  ⚡ System Impact Columns ({len(system_cols)}):")
            for col in sorted(system_cols):
                print(f"    • {col}")
            
            # Check for rational scenarios specifically
            rational_cols = [col for col in system_df.columns if 'rational' in col.lower()]
            print(f"  🤖 Rational Scenario Columns ({len(rational_cols)}):")
            for col in rational_cols:
                print(f"    • {col}")
                
        except Exception as e:
            print(f"  ❌ Error analyzing system_metrics.csv: {e}")
    
    print()
    
    # Analyze combined_scenarios.csv
    combined_path = os.path.join(data_folder, "combined_scenarios.csv")
    if os.path.exists(combined_path):
        print("🔬 COMBINED SCENARIOS ANALYSIS:")
        try:
            combined_df = pd.read_csv(combined_path)
            print(f"  Shape: {combined_df.shape}")
            
            # Check scenarios
            if 'Scenario' in combined_df.columns:
                scenarios = combined_df['Scenario'].unique()
                print(f"  📈 Scenarios ({len(scenarios)}):")
                for scenario in scenarios:
                    count = len(combined_df[combined_df['Scenario'] == scenario])
                    print(f"    • {scenario}: {count} records")
            
            # Check key columns
            key_cols = ['Year', 'Step', 'IsProsumer', 'IncomeClass', 'Scenario']
            missing_cols = [col for col in key_cols if col not in combined_df.columns]
            if missing_cols:
                print(f"  ❌ Missing key columns: {missing_cols}")
            else:
                print(f"  ✅ All key columns present")
                
        except Exception as e:
            print(f"  ❌ Error analyzing combined_scenarios.csv: {e}")
    
    print()
    
    # Check for individual scenario files
    print("📁 INDIVIDUAL SCENARIO FILES:")
    scenario_files = glob.glob(os.path.join(data_folder, "*_data.csv"))
    if scenario_files:
        print(f"  Found {len(scenario_files)} scenario files:")
        for file_path in sorted(scenario_files):
            file_name = os.path.basename(file_path)
            try:
                df = pd.read_csv(file_path)
                print(f"    • {file_name}: {df.shape}")
            except Exception as e:
                print(f"    • {file_name}: Error - {e}")
    else:
        print("  ❌ No individual scenario files found")
    
    print()
    
    # DIAGNOSIS AND RECOMMENDATIONS
    print("💡 DIAGNOSIS AND RECOMMENDATIONS:")
    print("=" * 50)
    
    missing_visualizations = []
    
    # Check what's needed for missing visualizations
    if os.path.exists(system_metrics_path):
        system_df = pd.read_csv(system_metrics_path)
        
        # Check for income_class_analysis.png requirements
        class_cols = [col for col in system_df.columns if 'Class' in col and 'Rate' in col]
        if len(class_cols) == 0:
            missing_visualizations.append("income_class_analysis.png")
            print("❌ MISSING: Income class data")
            print("   📋 Required: Columns like 'rational_Class1_Rate', 'rational_Class2_Rate', etc.")
            print("   🔧 Fix: Enable income class tracking in data collection")
        
        # Check for system_impact_comparison.png requirements
        system_cols = [col for col in system_df.columns if 'TotalGeneration' in col]
        if len(system_cols) == 0:
            missing_visualizations.append("system_impact_comparison.png")
            print("❌ MISSING: System impact data")
            print("   📋 Required: Columns like 'rational_TotalGeneration', 'rational_SystemCostSavings'")
            print("   🔧 Fix: Enable system metrics tracking in central provider")
        
        # Check for rational_comparison.png requirements
        rational_scenarios = ['deterministic_rational', 'rational']
        rational_cols = [f'{scenario}_AdoptionRate' for scenario in rational_scenarios]
        missing_rational = [col for col in rational_cols if col not in system_df.columns]
        if missing_rational:
            missing_visualizations.append("rational_comparison.png")
            print("❌ MISSING: Rational scenario comparison data")
            print(f"   📋 Required: {missing_rational}")
            print("   🔧 Fix: Ensure both deterministic_rational and rational scenarios are included")
    
    # Check for monthly consumption/generation trends
    print()
    print("📊 MONTHLY CONSUMPTION/GENERATION TRENDS:")
    monthly_cols = ['TotalConsumption', 'TotalGeneration', 'CurrentMonthInYear']
    if os.path.exists(system_metrics_path):
        system_df = pd.read_csv(system_metrics_path)
        missing_monthly = [col for col in monthly_cols if col not in system_df.columns]
        if missing_monthly:
            print(f"❌ MISSING: Monthly trend data - {missing_monthly}")
            print("   📋 Required: Monthly consumption, generation, and month tracking")
            print("   🔧 Fix: This requires single-scenario Visualizer (not ComparativeVisualizer)")
        else:
            print("✅ Monthly data available - but no single-scenario visualizer in pipeline")
            print("   🔧 Fix: Add single-scenario Visualizer to run_multi_experiment.py")
    
    print()
    if missing_visualizations:
        print(f"🚨 SUMMARY: {len(missing_visualizations)} visualization(s) missing due to data issues")
    else:
        print("✅ SUMMARY: Data seems complete - issue might be in visualization pipeline")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        results_dir = "results"
    
    if not os.path.exists(results_dir):
        print(f"❌ Results directory not found: {results_dir}")
        print("Usage: python debug_exported_data.py [results_directory]")
        sys.exit(1)
    
    analyze_exported_data(results_dir)