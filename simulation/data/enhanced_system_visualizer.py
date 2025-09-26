# data/enhanced_system_visualizer.py V1.0 - PHASE 2 IMPLEMENTATION
"""
Enhanced system-level analysis adapter for multi-scenario behavioral study.
Adapts existing system visualizations for multi-scenario comparative analysis.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ..utils.parameters import get_all_scenarios, get_scenario_colors, get_scenario_metadata

class EnhancedSystemVisualizer:
    """
    Enhanced system visualizer adapted for multi-scenario analysis.
    
    Adapts existing system-level plots (energy balance, credit utilization, etc.)
    to work with multi-scenario data from MultiExperimentCollector.
    """
    
    def __init__(self, data_collector, config):
        """
        Initialize enhanced system visualizer.
        
        Args:
            data_collector: MultiExperimentCollector instance
            config: Simulation configuration
        """
        self.data_collector = data_collector
        self.config = config
        self.scenarios = get_all_scenarios()
        self.colors = get_scenario_colors()
        self.metadata = get_scenario_metadata()
        
        # Set plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        print(f"EnhancedSystemVisualizer initialized for {len(self.scenarios)} scenarios")
    
    def plot_all_system_analyses(self, output_dir="results/phase2_system"):
        """
        Generate all Phase 2 system-level analyses.
        
        Args:
            output_dir: Directory to save visualizations
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print("Generating Phase 2 System-Level Analyses...")
        
        # System-Level Analysis (ADAPT from existing)
        self.plot_energy_balance_evolution(output_dir)
        self.plot_credit_system_utilization(output_dir)
        self.plot_energy_cost_distribution_evolution(output_dir)
        self.plot_seasonal_grid_stress(output_dir)
        
        print(f"✅ All Phase 2 system analyses completed and saved to {output_dir}")
    
    def plot_energy_balance_evolution(self, output_dir="results"):
        """
        Enhanced energy balance evolution across all scenarios (2x2 grid).
        Adapted from existing energy balance plot for multi-scenario comparison.
        """
        system_df = self.data_collector.get_system_metrics_dataframe()
        
        if system_df.empty:
            print("Warning: No system metrics available for energy balance analysis")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Total Energy Generation Comparison
        ax = axes[0, 0]
        for scenario in self.scenarios:
            if f'{scenario}_TotalGeneration' in system_df.columns:
                generation_data = system_df[f'{scenario}_TotalGeneration']
                color = self.colors.get(scenario, '#000000')
                display_name = self.metadata.get(scenario, {}).get('display_name', scenario)
                ax.plot(system_df['Year'], generation_data, 
                       color=color, label=display_name, linewidth=2, marker='o', markersize=4)
        
        ax.set_title('Total Solar Generation Evolution', fontweight='bold', fontsize=14)
        ax.set_xlabel('Year')
        ax.set_ylabel('Total Generation (kWh)')
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Solar Penetration Rate
        ax = axes[0, 1]
        for scenario in self.scenarios:
            total_gen_col = f'{scenario}_TotalGeneration'
            total_cons_col = f'{scenario}_TotalConsumption'
            
            if total_gen_col in system_df.columns and total_cons_col in system_df.columns:
                penetration = system_df[total_gen_col] / system_df[total_cons_col]
                penetration = penetration.fillna(0)
                
                color = self.colors.get(scenario, '#000000')
                display_name = self.metadata.get(scenario, {}).get('display_name', scenario)
                ax.plot(system_df['Year'], penetration, 
                       color=color, label=display_name, linewidth=2, marker='s', markersize=4)
        
        ax.set_title('Solar Penetration Rate by Scenario', fontweight='bold', fontsize=14)
        ax.set_xlabel('Year')
        ax.set_ylabel('Solar Gen / Total Consumption')
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Grid Consumption Evolution
        ax = axes[1, 0]
        for scenario in self.scenarios:
            if f'{scenario}_GridConsumption' in system_df.columns:
                grid_data = system_df[f'{scenario}_GridConsumption']
                color = self.colors.get(scenario, '#000000')
                display_name = self.metadata.get(scenario, {}).get('display_name', scenario)
                ax.plot(system_df['Year'], grid_data, 
                       color=color, label=display_name, linewidth=2, marker='^', markersize=4)
        
        ax.set_title('Grid Consumption by Scenario', fontweight='bold', fontsize=14)
        ax.set_xlabel('Year')
        ax.set_ylabel('Grid Consumption (kWh)')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Plot 4: System Self-Sufficiency
        ax = axes[1, 1]
        for scenario in self.scenarios:
            grid_cons_col = f'{scenario}_GridConsumption'
            total_cons_col = f'{scenario}_TotalConsumption'
            
            if grid_cons_col in system_df.columns and total_cons_col in system_df.columns:
                self_sufficiency = 1 - (system_df[grid_cons_col] / system_df[total_cons_col])
                self_sufficiency = self_sufficiency.fillna(0).clip(0, 1)
                
                color = self.colors.get(scenario, '#000000')
                display_name = self.metadata.get(scenario, {}).get('display_name', scenario)
                ax.plot(system_df['Year'], self_sufficiency, 
                       color=color, label=display_name, linewidth=2, marker='d', markersize=4)
        
        ax.set_title('System Self-Sufficiency Rate', fontweight='bold', fontsize=14)
        ax.set_xlabel('Year')
        ax.set_ylabel('Self-Sufficiency Rate')
        ax.set_ylim(0, 1)
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/energy_balance_evolution_multi_scenario.png", 
                   dpi=300, bbox_inches="tight")
        plt.close()
        
        print("  ✅ Energy balance evolution visualization completed")
    
    def plot_credit_system_utilization(self, output_dir="results"):
        """
        Enhanced credit system utilization across scenarios (2x2 grid).
        """
        system_df = self.data_collector.get_system_metrics_dataframe()
        
        if system_df.empty:
            print("Warning: No system metrics available for credit analysis")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Credits Earned vs Used
        ax = axes[0, 0]
        for scenario in self.scenarios:
            earned_col = f'{scenario}_CreditsEarned'
            used_col = f'{scenario}_CreditsUsed'
            
            if earned_col in system_df.columns:
                color = self.colors.get(scenario, '#000000')
                display_name = self.metadata.get(scenario, {}).get('display_name', scenario)
                
                # Plot earned credits (solid line)
                ax.plot(system_df['Year'], system_df[earned_col], 
                       color=color, label=f'{display_name} (Earned)', 
                       linewidth=2, linestyle='-')
                
                # Plot used credits (dashed line)
                if used_col in system_df.columns:
                    ax.plot(system_df['Year'], system_df[used_col], 
                           color=color, label=f'{display_name} (Used)', 
                           linewidth=2, linestyle='--', alpha=0.7)
        
        ax.set_title('Credit System Activity by Scenario', fontweight='bold', fontsize=14)
        ax.set_xlabel('Year')
        ax.set_ylabel('Energy Credits (kWh)')
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Credit Utilization Rate
        ax = axes[0, 1]
        for scenario in self.scenarios:
            earned_col = f'{scenario}_CreditsEarned'
            used_col = f'{scenario}_CreditsUsed'
            
            if earned_col in system_df.columns and used_col in system_df.columns:
                # Calculate utilization rate (avoid division by zero)
                utilization = np.where(system_df[earned_col] > 0, 
                                     system_df[used_col] / system_df[earned_col], 0)
                utilization = np.clip(utilization, 0, 1)
                
                color = self.colors.get(scenario, '#000000')
                display_name = self.metadata.get(scenario, {}).get('display_name', scenario)
                ax.plot(system_df['Year'], utilization, 
                       color=color, label=display_name, linewidth=2, marker='o', markersize=4)
        
        ax.set_title('Credit Utilization Rate', fontweight='bold', fontsize=14)
        ax.set_xlabel('Year')
        ax.set_ylabel('Utilization Rate (Used/Earned)')
        ax.set_ylim(0, 1)
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Credits Expired (Waste)
        ax = axes[1, 0]
        for scenario in self.scenarios:
            if f'{scenario}_CreditsExpired' in system_df.columns:
                expired_data = system_df[f'{scenario}_CreditsExpired']
                color = self.colors.get(scenario, '#000000')
                display_name = self.metadata.get(scenario, {}).get('display_name', scenario)
                ax.plot(system_df['Year'], expired_data, 
                       color=color, label=display_name, linewidth=2, marker='s', markersize=4)
        
        ax.set_title('Credits Expired (System Waste)', fontweight='bold', fontsize=14)
        ax.set_xlabel('Year')
        ax.set_ylabel('Expired Credits (kWh)')
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Credit Waste Rate
        ax = axes[1, 1]
        for scenario in self.scenarios:
            earned_col = f'{scenario}_CreditsEarned'
            expired_col = f'{scenario}_CreditsExpired'
            
            if earned_col in system_df.columns and expired_col in system_df.columns:
                # Calculate waste rate
                waste_rate = np.where(system_df[earned_col] > 0, 
                                    system_df[expired_col] / system_df[earned_col], 0)
                waste_rate = np.clip(waste_rate, 0, 1)
                
                color = self.colors.get(scenario, '#000000')
                display_name = self.metadata.get(scenario, {}).get('display_name', scenario)
                ax.plot(system_df['Year'], waste_rate, 
                       color=color, label=display_name, linewidth=2, marker='^', markersize=4)
        
        ax.set_title('Credit Waste Rate', fontweight='bold', fontsize=14)
        ax.set_xlabel('Year')
        ax.set_ylabel('Waste Rate (Expired/Earned)')
        ax.set_ylim(0, 1)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/credit_system_utilization_multi_scenario.png", 
                   dpi=300, bbox_inches="tight")
        plt.close()
        
        print("  ✅ Credit system utilization visualization completed")
    
    def plot_energy_cost_distribution_evolution(self, output_dir="results"):
        """
        Enhanced energy cost distribution evolution (2x2 grid).
        """
        # Get combined household data for cost analysis
        combined_df = self.data_collector.get_combined_dataframe()
        
        if combined_df.empty:
            print("Warning: No combined data available for cost distribution analysis")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Average Energy Cost by Scenario
        ax = axes[0, 0]
        for scenario in self.scenarios:
            cost_col = f'{scenario}_EnergyCost'
            if cost_col in combined_df.columns:
                # Calculate yearly averages
                yearly_costs = combined_df.groupby('Year')[cost_col].mean()
                
                color = self.colors.get(scenario, '#000000')
                display_name = self.metadata.get(scenario, {}).get('display_name', scenario)
                ax.plot(yearly_costs.index, yearly_costs.values, 
                       color=color, label=display_name, linewidth=2, marker='o', markersize=4)
        
        ax.set_title('Average Energy Cost Evolution', fontweight='bold', fontsize=14)
        ax.set_xlabel('Year')
        ax.set_ylabel('Average Energy Cost ($)')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Cost Savings (relative to rational scenario)
        ax = axes[0, 1]
        if 'rational_EnergyCost' in combined_df.columns:
            baseline_costs = combined_df.groupby('Year')['rational_EnergyCost'].mean()
            
            for scenario in self.scenarios:
                if scenario != 'rational':
                    cost_col = f'{scenario}_EnergyCost'
                    if cost_col in combined_df.columns:
                        scenario_costs = combined_df.groupby('Year')[cost_col].mean()
                        cost_savings = baseline_costs - scenario_costs
                        
                        color = self.colors.get(scenario, '#000000')
                        display_name = self.metadata.get(scenario, {}).get('display_name', scenario)
                        ax.plot(cost_savings.index, cost_savings.values, 
                               color=color, label=display_name, linewidth=2, marker='s', markersize=4)
        
        ax.set_title('Energy Cost Savings vs Rational', fontweight='bold', fontsize=14)
        ax.set_xlabel('Year')
        ax.set_ylabel('Cost Savings ($)')
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Cost Distribution by Income Class
        ax = axes[1, 0]
        final_year_data = combined_df[combined_df['Year'] == combined_df['Year'].max()]
        
        # Create income class vs scenario cost heatmap
        cost_matrix = []
        scenarios_for_heatmap = []
        income_classes = sorted(final_year_data['IncomeClass'].unique())
        
        for scenario in self.scenarios:
            cost_col = f'{scenario}_EnergyCost'
            if cost_col in final_year_data.columns:
                class_costs = []
                for income_class in income_classes:
                    class_data = final_year_data[final_year_data['IncomeClass'] == income_class]
                    avg_cost = class_data[cost_col].mean()
                    class_costs.append(avg_cost)
                cost_matrix.append(class_costs)
                scenarios_for_heatmap.append(self.metadata.get(scenario, {}).get('display_name', scenario))
        
        if cost_matrix:
            im = ax.imshow(cost_matrix, cmap='YlOrRd', aspect='auto')
            ax.set_xticks(range(len(income_classes)))
            ax.set_xticklabels([f'Class {i}' for i in income_classes])
            ax.set_yticks(range(len(scenarios_for_heatmap)))
            ax.set_yticklabels(scenarios_for_heatmap)
            ax.set_title('Final Year Cost by Income Class', fontweight='bold', fontsize=14)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Average Energy Cost ($)')
            
            # Add text annotations
            for i in range(len(scenarios_for_heatmap)):
                for j in range(len(income_classes)):
                    text = ax.text(j, i, f'${cost_matrix[i][j]:.0f}',
                                 ha="center", va="center", color="black", fontsize=8)
        
        # Plot 4: Cost Variability (Standard Deviation)
        ax = axes[1, 1]
        for scenario in self.scenarios:
            cost_col = f'{scenario}_EnergyCost'
            if cost_col in combined_df.columns:
                # Calculate yearly standard deviation
                yearly_std = combined_df.groupby('Year')[cost_col].std()
                
                color = self.colors.get(scenario, '#000000')
                display_name = self.metadata.get(scenario, {}).get('display_name', scenario)
                ax.plot(yearly_std.index, yearly_std.values, 
                       color=color, label=display_name, linewidth=2, marker='^', markersize=4)
        
        ax.set_title('Energy Cost Variability', fontweight='bold', fontsize=14)
        ax.set_xlabel('Year')
        ax.set_ylabel('Cost Standard Deviation ($)')
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/energy_cost_distribution_evolution.png", 
                   dpi=300, bbox_inches="tight")
        plt.close()
        
        print("  ✅ Energy cost distribution evolution visualization completed")
    
    def plot_seasonal_grid_stress(self, output_dir="results"):
        """
        Enhanced seasonal grid stress patterns (2x2 grid).
        """
        system_df = self.data_collector.get_system_metrics_dataframe()
        
        if system_df.empty:
            print("Warning: No system metrics available for seasonal analysis")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Peak Load Evolution
        ax = axes[0, 0]
        for scenario in self.scenarios:
            if f'{scenario}_PeakLoad' in system_df.columns:
                peak_data = system_df[f'{scenario}_PeakLoad']
                color = self.colors.get(scenario, '#000000')
                display_name = self.metadata.get(scenario, {}).get('display_name', scenario)
                ax.plot(system_df['Year'], peak_data, 
                       color=color, label=display_name, linewidth=2, marker='o', markersize=4)
        
        ax.set_title('Grid Peak Load Evolution', fontweight='bold', fontsize=14)
        ax.set_xlabel('Year')
        ax.set_ylabel('Peak Load (kW)')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Load Factor (Average/Peak)
        ax = axes[0, 1]
        for scenario in self.scenarios:
            peak_col = f'{scenario}_PeakLoad'
            cons_col = f'{scenario}_TotalConsumption'
            
            if peak_col in system_df.columns and cons_col in system_df.columns:
                # Approximate load factor (need to convert monthly consumption to hourly average)
                hours_per_month = 730  # Approximate
                avg_load = system_df[cons_col] / hours_per_month
                load_factor = avg_load / system_df[peak_col]
                load_factor = load_factor.fillna(0).clip(0, 1)
                
                color = self.colors.get(scenario, '#000000')
                display_name = self.metadata.get(scenario, {}).get('display_name', scenario)
                ax.plot(system_df['Year'], load_factor, 
                       color=color, label=display_name, linewidth=2, marker='s', markersize=4)
        
        ax.set_title('System Load Factor', fontweight='bold', fontsize=14)
        ax.set_xlabel('Year')
        ax.set_ylabel('Load Factor (Avg/Peak)')
        ax.set_ylim(0, 1)
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Monthly Peak Load Patterns (Final Year)
        ax = axes[1, 0]
        if 'Month' in system_df.columns:
            final_year = system_df['Year'].max()
            final_year_data = system_df[system_df['Year'] == final_year]
            
            for scenario in self.scenarios:
                if f'{scenario}_PeakLoad' in final_year_data.columns:
                    monthly_peaks = final_year_data.groupby('Month')[f'{scenario}_PeakLoad'].mean()
                    
                    color = self.colors.get(scenario, '#000000')
                    display_name = self.metadata.get(scenario, {}).get('display_name', scenario)
                    ax.plot(monthly_peaks.index, monthly_peaks.values, 
                           color=color, label=display_name, linewidth=2, marker='d', markersize=4)
        
        ax.set_title(f'Monthly Peak Load Patterns (Year {final_year})', fontweight='bold', fontsize=14)
        ax.set_xlabel('Month')
        ax.set_ylabel('Peak Load (kW)')
        ax.set_xticks(range(1, 13))
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Grid Stress Index (Peak Load / Generation Capacity)
        ax = axes[1, 1]
        for scenario in self.scenarios:
            peak_col = f'{scenario}_PeakLoad'
            gen_col = f'{scenario}_TotalGeneration'
            
            if peak_col in system_df.columns and gen_col in system_df.columns:
                # Calculate stress index (higher values = more stress)
                # Assuming generation capacity is proportional to total generation
                capacity_factor = 0.2  # Typical solar capacity factor
                estimated_capacity = system_df[gen_col] / (capacity_factor * 8760 / 12)  # Monthly to capacity
                stress_index = system_df[peak_col] / (estimated_capacity + 1)  # +1 to avoid division by zero
                
                color = self.colors.get(scenario, '#000000')
                display_name = self.metadata.get(scenario, {}).get('display_name', scenario)
                ax.plot(system_df['Year'], stress_index, 
                       color=color, label=display_name, linewidth=2, marker='^', markersize=4)
        
        ax.set_title('Grid Stress Index', fontweight='bold', fontsize=14)
        ax.set_xlabel('Year')
        ax.set_ylabel('Stress Index (Peak/Capacity)')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/seasonal_grid_stress_multi_scenario.png", 
                   dpi=300, bbox_inches="tight")
        plt.close()
        
        print("  ✅ Seasonal grid stress visualization completed")