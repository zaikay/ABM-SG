# data/system_level_visualizer.py
"""
Complete system-level visualization suite for behavioral prosumer adoption analysis.
Implements 5 core system-level visualizations with clean metrics layer separation.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from ..utils.parameters import get_scenario_colors, get_all_scenarios

class SystemLevelVisualizer:
    """
    Complete system-level visualization suite.
    Uses MetricsAnalyzer for all data processing (clean separation).
    """
    
    def __init__(self, metrics_analyzer):
        """Initialize with metrics analyzer instance"""
        self.metrics = metrics_analyzer
        self.colors = get_scenario_colors()  # Use centralized colors from parameters.py
        
        # Auto-detect available scenarios from data
        self.available_scenarios = self._detect_scenarios()
    
    def _detect_scenarios(self):
        """Auto-detect available scenarios from column names"""
        scenarios = set()
        
        # Look for scenario-specific adoption rate columns
        for col in self.metrics.model_data.columns:
            if col.endswith('_AdoptionRate'):
                scenario = col.replace('_AdoptionRate', '')
                scenarios.add(scenario)
        
        # Look for scenario-specific adopted columns in agent data  
        for col in self.metrics.agent_data.columns:
            if col.endswith('_Adopted'):
                scenario = col.replace('_Adopted', '')
                scenarios.add(scenario)
        
        available = sorted(list(scenarios))
        print(f"Auto-detected scenarios: {available}")
        return available
    
    def create_all_system_visuals(self, output_dir="results/system_level"):
        """Generate all system-level visualizations"""
        os.makedirs(output_dir, exist_ok=True)
        
        print("Creating System-Level Visualization Suite...")
        
        # 1. Energy Cost Evolution
        self.plot_energy_cost_evolution(output_dir)
        
        # 2. Energy Mix Evolution  
        self.plot_energy_mix_evolution(output_dir)
        
        # 3. Income Class System Analysis
        self.plot_income_class_system_analysis(output_dir)
        
        # 4. Adoption Velocity Analysis
        self.plot_adoption_velocity_analysis(output_dir)
        
        # 5. Enhanced Grid Stress Index (if data available)
        self.plot_enhanced_grid_stress_index(output_dir)
        
        print(f"All system-level visualizations saved to {output_dir}")
    
    def plot_energy_cost_evolution(self, output_dir):
        """1. Energy Cost Evolution (2×1)"""
        cost_data = self.metrics.get_energy_cost_evolution()
        if cost_data.empty:
            print("Warning: No cost evolution data available")
            return
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Top: Price Evolution with Crossover
        ax = axes[0]
        price_cols = [col for col in cost_data.columns if 'Price' in col]
        
        for col in price_cols:
            if 'Fossil' in col or 'NonRenewable' in col:
                ax.plot(cost_data['Year'], cost_data[col], 
                       label='Non-renewable Price', linewidth=3, color='#d62728', marker='o', markersize=4)
            elif 'Renewable' in col:
                ax.plot(cost_data['Year'], cost_data[col], 
                       label='Renewable Price', linewidth=3, color='#2ca02c', marker='s', markersize=4)
        
        ax.set_title('Energy Cost Evolution & Price Parity', fontweight='bold')
        ax.set_ylabel('Price ($/kWh)')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Bottom: NPV Positive Households
        ax = axes[1]
        if 'NPVPositivePercent' in cost_data.columns:
            ax.fill_between(cost_data['Year'], 0, cost_data['NPVPositivePercent'], 
                           alpha=0.6, color='green', label='Households with Positive NPV')
            ax.plot(cost_data['Year'], cost_data['NPVPositivePercent'], 
                   linewidth=2, color='darkgreen', marker='^', markersize=4)
        
        ax.set_title('Economic Adoption Potential', fontweight='bold')
        ax.set_xlabel('Year')
        ax.set_ylabel('Households with Positive NPV (%)')
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/energy_cost_evolution.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("  Energy Cost Evolution completed")
    
    def plot_energy_mix_evolution(self, output_dir):
        """2. Energy Mix Evolution - Stacked Area Chart with REAL yearly aggregated data"""
        # Find main scenarios to compare
        main_scenarios = ['rational']
        combined_scenarios = [s for s in self.available_scenarios if 'all' in s.lower() or 'combined' in s.lower()]
        if combined_scenarios:
            main_scenarios.append(combined_scenarios[0])
        
        fig, axes = plt.subplots(1, len(main_scenarios), figsize=(8*len(main_scenarios), 8))
        if len(main_scenarios) == 1:
            axes = [axes]
        
        for idx, scenario in enumerate(main_scenarios):
            ax = axes[idx]
            
            print(f"\nProcessing energy mix for {scenario}...")
            mix_data = self.metrics.get_energy_mix_evolution(scenario)
            
            if mix_data.empty:
                ax.text(0.5, 0.5, f'No data\nfor {scenario}', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=14)
                ax.set_title(f'{scenario.replace("_", " ").title()}', fontweight='bold')
                continue
            
            # Verify we have the calculated layers
            required_layers = ['Layer1_DirectSolar', 'Layer2_DirectPlusCredit', 'Layer3_Total']
            if not all(layer in mix_data.columns for layer in required_layers):
                ax.text(0.5, 0.5, f'Missing energy\ndata layers\nfor {scenario}', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=12)
                ax.set_title(f'{scenario.replace("_", " ").title()}', fontweight='bold')
                print(f"  Missing required layers: {[l for l in required_layers if l not in mix_data.columns]}")
                continue
            
            years = mix_data['Year']
            
            # Verify data integrity before plotting
            print(f"  Data verification for {scenario}:")
            print(f"    Years: {len(years)} ({years.min()}-{years.max()})")
            print(f"    Total consumption range: {mix_data['TotalConsumption'].min():.0f} - {mix_data['TotalConsumption'].max():.0f}")
            
            # Check if components sum to total (data integrity)
            sample_year = mix_data.iloc[0]
            direct = sample_year['DirectSolarConsumption']
            credit = sample_year['CreditConsumption'] 
            grid = sample_year['GridConsumption']
            total = sample_year['TotalConsumption']
            calculated_total = direct + credit + grid
            difference = abs(total - calculated_total)
            
            print(f"    Year {sample_year['Year']} verification: Total={total:.0f}, Components={calculated_total:.0f}, Diff={difference:.1f}")
            
            if difference > total * 0.05:  # More than 5% difference
                print(f"    WARNING: Large discrepancy in energy balance for {scenario}")
            
            # Create stacked areas (from bottom to top as requested)
            # Layer 1: Direct Solar Consumption (green, bottom)
            ax.fill_between(years, 0, mix_data['Layer1_DirectSolar'], 
                          alpha=0.8, color='#2ca02c', label='Direct Solar Consumption')
            
            # Layer 2: Direct Solar + Credit Consumption (yellow, middle)  
            ax.fill_between(years, mix_data['Layer1_DirectSolar'], mix_data['Layer2_DirectPlusCredit'], 
                          alpha=0.8, color='#ffcc00', label='Credit Consumption')
            
            # Layer 3: Total Consumption with Grid (red, top)
            ax.fill_between(years, mix_data['Layer2_DirectPlusCredit'], mix_data['Layer3_Total'], 
                          alpha=0.8, color='#d62728', label='Grid Consumption')
            
            # Add boundary line for total consumption
            ax.plot(years, mix_data['Layer3_Total'], color='black', linewidth=2, alpha=0.7)
            
            ax.set_title(f'Energy Mix Evolution\n{scenario.replace("_", " ").title()}', fontweight='bold')
            ax.set_xlabel('Year')
            ax.set_ylabel('Annual Energy (kWh/year)')
            ax.legend(loc='upper left', framealpha=0.9)
            ax.grid(True, alpha=0.3)
            
            # Add summary statistics for final year
            if len(mix_data) > 0:
                final_year_data = mix_data.iloc[-1]
                final_total = final_year_data['TotalConsumption']
                final_direct = final_year_data['DirectSolarConsumption']
                final_credit = final_year_data['CreditConsumption']
                final_grid = final_year_data['GridConsumption']
                
                # Calculate percentages
                direct_pct = (final_direct / final_total * 100) if final_total > 0 else 0
                credit_pct = (final_credit / final_total * 100) if final_total > 0 else 0
                grid_pct = (final_grid / final_total * 100) if final_total > 0 else 0
                
                # Add text box with final year breakdown
                stats_text = f'Year {final_year_data["Year"]:.0f} Mix:\n'
                stats_text += f'Direct Solar: {direct_pct:.1f}%\n'
                stats_text += f'Credit Use: {credit_pct:.1f}%\n'  
                stats_text += f'Grid Power: {grid_pct:.1f}%'
                
                ax.text(0.98, 0.02, stats_text, transform=ax.transAxes, 
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9),
                       ha='right', va='bottom', fontsize=10, family='monospace')
                
                print(f"  Final energy mix: Direct={direct_pct:.1f}%, Credit={credit_pct:.1f}%, Grid={grid_pct:.1f}%")
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/energy_mix_evolution.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("  Energy Mix Evolution completed (using REAL yearly aggregated data)")
    
    def plot_income_class_system_analysis(self, output_dir):
        """4. Income Class System Analysis (2×2 vs 2×2)"""
        # Get data for main scenarios
        main_scenarios = ['rational']
        combined_scenarios = [s for s in self.available_scenarios if 'all' in s.lower() or 'combined' in s.lower()]
        if combined_scenarios:
            main_scenarios.append(combined_scenarios[0])
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        for idx, scenario in enumerate(main_scenarios[:2]):  # Limit to 2 scenarios
            class_data = self.metrics.get_income_class_system_metrics(scenario)
            
            if class_data.empty:
                # Handle empty data
                for i in range(2):
                    ax = axes[i, idx]
                    ax.text(0.5, 0.5, f'No data\nfor {scenario}', ha='center', va='center', 
                           transform=ax.transAxes, fontsize=12)
                    ax.set_title(f'{scenario.replace("_", " ").title()}')
                continue
            
            # Top row: Adoption Rate by Income Class
            ax = axes[0, idx]
            for income_class in sorted(class_data['IncomeClass'].unique()):
                class_subset = class_data[class_data['IncomeClass'] == income_class]
                ax.plot(class_subset['Year'], class_subset['AdoptionRate'], 
                       marker='o', label=f'Class {income_class}', linewidth=2)
            
            ax.set_title(f'Adoption by Income Class\n({scenario.replace("_", " ").title()})', 
                        fontweight='bold')
            ax.set_ylabel('Adoption Rate')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            
            # Bottom row: Mean NPV by Income Class
            ax = axes[1, idx]
            for income_class in sorted(class_data['IncomeClass'].unique()):
                class_subset = class_data[class_data['IncomeClass'] == income_class]
                ax.plot(class_subset['Year'], class_subset['MeanNPV'], 
                       marker='s', label=f'Class {income_class}', linewidth=2)
            
            ax.set_title(f'NPV by Income Class\n({scenario.replace("_", " ").title()})', 
                        fontweight='bold')
            ax.set_xlabel('Year')
            ax.set_ylabel('Mean NPV ($)')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/income_class_system_analysis.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("  Income Class System Analysis completed")
    
    def plot_adoption_velocity_analysis(self, output_dir):
        """5. Adoption Velocity Analysis (2×1)"""
        velocity_data = self.metrics.get_adoption_velocity_metrics()
        if velocity_data.empty:
            print("Warning: No velocity data available")
            return
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Velocity Curves by Scenario
        ax = axes[0]
        for _, row in velocity_data.iterrows():
            scenario = row['Scenario']
            velocity_curve = row['VelocityCurve']
            
            if isinstance(velocity_curve, np.ndarray) and len(velocity_curve) > 0:
                years = np.arange(1, len(velocity_curve) + 1)
                color = self.colors.get(scenario, '#000000')
                
                ax.plot(years, velocity_curve, label=scenario.replace('_', ' ').title(), 
                       color=color, linewidth=2, marker='o', markersize=3)
        
        ax.set_title('Adoption Velocity Evolution by Scenario', fontweight='bold')
        ax.set_ylabel('Adoption Velocity (%/year)')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Plot 2: Time to 50% Threshold
        ax = axes[1]
        threshold_data = velocity_data[['Scenario', 'Threshold50Year']].dropna()
        
        if not threshold_data.empty:
            scenarios = threshold_data['Scenario']
            years_to_50 = threshold_data['Threshold50Year']
            colors = [self.colors.get(s, '#000000') for s in scenarios]
            
            bars = ax.bar(range(len(scenarios)), years_to_50, color=colors, alpha=0.7)
            
            # Add value labels
            for i, (bar, years) in enumerate(zip(bars, years_to_50)):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                       f'{years:.1f}y', ha='center', va='bottom', fontweight='bold', fontsize=10)
            
            ax.set_xticks(range(len(scenarios)))
            ax.set_xticklabels([s.replace('_', ' ').title() for s in scenarios], rotation=45)
            ax.set_title('Time to Reach 50% Adoption', fontweight='bold')
            ax.set_ylabel('Years to 50% Threshold')
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/adoption_velocity_analysis.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("  Adoption Velocity Analysis completed")
    
    def plot_enhanced_grid_stress_index(self, output_dir):
        """5. Enhanced Grid Stress Index (if data available)"""
        try:
            stress_data = self.metrics.get_seasonal_stress_patterns()
            
            if isinstance(stress_data, dict) and 'monthly_data' in stress_data:
                monthly_data = stress_data['monthly_data']
            elif not stress_data.empty:
                monthly_data = stress_data
            else:
                print("Warning: No grid stress data available")
                return
            
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            
            # Create heatmap by year and month
            if 'Year' in monthly_data.columns and 'Month' in monthly_data.columns:
                pivot_data = monthly_data.pivot_table(
                    values='GridStressIndex' if 'GridStressIndex' in monthly_data.columns else 'StressIndex', 
                    index='Year', 
                    columns='Month', 
                    aggfunc='mean'
                )
                
                sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='YlOrRd', 
                           ax=ax, cbar_kws={'label': 'Grid Stress Index'})
                ax.set_title('Grid Stress Index Evolution', fontweight='bold')
                ax.set_xlabel('Month')
                ax.set_ylabel('Year')
            else:
                ax.text(0.5, 0.5, 'Insufficient data\nfor grid stress heatmap', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=14)
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/enhanced_grid_stress_index.png", dpi=300, bbox_inches="tight")
            plt.close()
            print("  Enhanced Grid Stress Index completed")
            
        except Exception as e:
            print(f"Warning: Could not create grid stress visualization: {e}")
    
    def validate_metrics_extraction(self, export_csv=True):
        """
        Validate that all new metrics methods work correctly.
        Tests each method and exports CSV files for manual inspection.
        """
        print("Validating Enhanced Metrics Extraction...")
        
        validation_results = {
            'success': True,
            'methods_tested': 0,
            'methods_passed': 0,
            'data_summaries': {},
            'issues': []
        }
        
        # Test each metric method
        methods_to_test = [
            ('Energy Cost Evolution', lambda: self.metrics.get_energy_cost_evolution()),
            ('Energy Mix (rational)', lambda: self.metrics.get_energy_mix_evolution('rational')),
            ('Income Class (rational)', lambda: self.metrics.get_income_class_system_metrics('rational')),
            ('Adoption Velocity', lambda: self.metrics.get_adoption_velocity_metrics()),
        ]
        
        for method_name, method_func in methods_to_test:
            try:
                validation_results['methods_tested'] += 1
                data = method_func()
                
                if not data.empty:
                    print(f"  Success: {method_name} - {len(data)} records")
                    validation_results['methods_passed'] += 1
                else:
                    print(f"  Warning: {method_name} - No data")
                    validation_results['issues'].append(f"{method_name} returned empty")
                    
            except Exception as e:
                print(f"  Error: {method_name} - {e}")
                validation_results['issues'].append(f"{method_name} failed: {e}")
                validation_results['success'] = False
        
        # Export CSV files if requested
        if export_csv:
            try:
                self.metrics.export_all_system_metrics("data/metrics_validation")
                print("CSV files exported to data/metrics_validation/")
            except Exception as e:
                print(f"CSV export failed: {e}")
                validation_results['csv_exported'] = False
        
        print(f"Validation complete: {validation_results['methods_passed']}/{validation_results['methods_tested']} methods working")
        return validation_results