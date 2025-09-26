# data/individual_bias_analyzer.py V1.0 - PHASE 2 IMPLEMENTATION
"""
Individual bias mechanisms analyzer for behavioral prosumer adoption study.
Creates new visualizations for understanding individual bias effects and decision processes.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ..utils.parameters import get_all_scenarios, get_scenario_colors, get_scenario_metadata, get_enabled_biases

class IndividualBiasAnalyzer:
    """
    Analyzer for individual bias mechanisms and decision processes.
    
    Creates new visualizations for Phase 2:
    - Individual Bias Time Series (2x2 Grid)
    - Decision Process Analysis (2x3 Grid)  
    - Adoption Velocity by Class (2x2 Grid)
    """
    
    def __init__(self, data_collector, config):
        """
        Initialize individual bias analyzer.
        
        Args:
            data_collector: MultiExperimentCollector instance
            config: Simulation configuration
        """
        self.data_collector = data_collector
        self.config = config
        self.scenarios = get_all_scenarios()
        self.enabled_biases = get_enabled_biases()
        self.colors = get_scenario_colors()
        self.metadata = get_scenario_metadata()
        
        # Set plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        print(f"IndividualBiasAnalyzer initialized for {len(self.enabled_biases)} biases")
    
    def plot_all_individual_analyses(self, output_dir="results/phase2_individual"):
        """
        Generate all Phase 2 individual bias analyses.
        
        Args:
            output_dir: Directory to save visualizations
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print("Generating Phase 2 Individual Bias Analyses...")
        
        # Individual Bias Mechanisms (NEW)
        self.plot_individual_bias_time_series(output_dir)
        self.plot_decision_process_analysis(output_dir)
        self.plot_adoption_velocity_by_class(output_dir)
        
        print(f"✅ All Phase 2 individual analyses completed and saved to {output_dir}")
    
    def plot_individual_bias_time_series(self, output_dir="results"):
        """
        Plot individual bias effects over time (2x2 grid).
        Shows how each bias affects decision-making throughout the simulation.
        """
        bias_df = self.data_collector.get_bias_effects_dataframe()
        
        if bias_df.empty:
            print("Warning: No bias effects data available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Average Bias Multipliers Over Time
        ax = axes[0, 0]
        
        for bias_name in self.enabled_biases:
            if f'{bias_name}_Multiplier' in bias_df.columns:
                # Calculate yearly averages
                yearly_multipliers = bias_df.groupby('Year')[f'{bias_name}_Multiplier'].mean()
                
                color = self.colors.get(bias_name, '#000000')
                display_name = self.metadata.get(bias_name, {}).get('display_name', bias_name)
                ax.plot(yearly_multipliers.index, yearly_multipliers.values,
                       color=color, label=display_name, linewidth=2, marker='o', markersize=4)
        
        ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='No Bias Effect')
        ax.set_title('Average Bias Effect Multipliers Over Time', fontweight='bold', fontsize=14)
        ax.set_xlabel('Year')
        ax.set_ylabel('Probability Multiplier')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Bias Effect Distribution by Income Class
        ax = axes[0, 1]
        
        # Create income class analysis for final year
        final_year = bias_df['Year'].max()
        final_data = bias_df[bias_df['Year'] == final_year]
        
        if not final_data.empty and 'IncomeClass' in final_data.columns:
            income_classes = sorted(final_data['IncomeClass'].unique())
            bias_effects_by_class = {}
            
            for bias_name in self.enabled_biases:
                if f'{bias_name}_Multiplier' in final_data.columns:
                    class_effects = []
                    for income_class in income_classes:
                        class_data = final_data[final_data['IncomeClass'] == income_class]
                        avg_effect = class_data[f'{bias_name}_Multiplier'].mean()
                        class_effects.append(avg_effect)
                    bias_effects_by_class[bias_name] = class_effects
            
            # Create grouped bar chart
            x = np.arange(len(income_classes))
            width = 0.8 / len(self.enabled_biases)
            
            for i, bias_name in enumerate(self.enabled_biases):
                if bias_name in bias_effects_by_class:
                    color = self.colors.get(bias_name, '#000000')
                    display_name = self.metadata.get(bias_name, {}).get('display_name', bias_name)
                    ax.bar(x + i * width, bias_effects_by_class[bias_name], 
                          width, label=display_name, color=color, alpha=0.8)
            
            ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
            ax.set_title(f'Bias Effects by Income Class (Year {final_year})', fontweight='bold', fontsize=14)
            ax.set_xlabel('Income Class')
            ax.set_ylabel('Probability Multiplier')
            ax.set_xticks(x + width * (len(self.enabled_biases) - 1) / 2)
            ax.set_xticklabels([f'Class {i}' for i in income_classes])
            ax.legend(loc='upper right', fontsize=10)
            ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Individual Household Bias Trajectories (Sample)
        ax = axes[1, 0]
        
        # Sample a few households for trajectory analysis
        sample_households = bias_df['HouseholdID'].unique()[:5]  # First 5 households
        
        for bias_name in self.enabled_biases:
            if f'{bias_name}_Multiplier' in bias_df.columns:
                color = self.colors.get(bias_name, '#000000')
                
                for i, household_id in enumerate(sample_households):
                    household_data = bias_df[bias_df['HouseholdID'] == household_id]
                    
                    if not household_data.empty:
                        alpha = 0.3 if i > 0 else 1.0  # Highlight first household
                        linestyle = '-' if i == 0 else ':'
                        
                        ax.plot(household_data['Year'], household_data[f'{bias_name}_Multiplier'],
                               color=color, alpha=alpha, linestyle=linestyle, linewidth=1.5)
                
                # Add legend entry for the bias (using first household's line)
                if len(sample_households) > 0:
                    display_name = self.metadata.get(bias_name, {}).get('display_name', bias_name)
                    ax.plot([], [], color=color, label=display_name, linewidth=2)
        
        ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
        ax.set_title('Individual Household Bias Trajectories (Sample)', fontweight='bold', fontsize=14)
        ax.set_xlabel('Year')
        ax.set_ylabel('Probability Multiplier')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Bias Effect Variance Over Time
        ax = axes[1, 1]
        
        for bias_name in self.enabled_biases:
            if f'{bias_name}_Multiplier' in bias_df.columns:
                # Calculate yearly variance
                yearly_variance = bias_df.groupby('Year')[f'{bias_name}_Multiplier'].var()
                
                color = self.colors.get(bias_name, '#000000')
                display_name = self.metadata.get(bias_name, {}).get('display_name', bias_name)
                ax.plot(yearly_variance.index, yearly_variance.values,
                       color=color, label=display_name, linewidth=2, marker='s', markersize=4)
        
        ax.set_title('Bias Effect Variance Over Time', fontweight='bold', fontsize=14)
        ax.set_xlabel('Year')
        ax.set_ylabel('Variance in Probability Multiplier')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/individual_bias_time_series.png", 
                   dpi=300, bbox_inches="tight")
        plt.close()
        
        print("  ✅ Individual bias time series visualization completed")
    
    def plot_decision_process_analysis(self, output_dir="results"):
        """
        Plot decision process analysis (2x3 grid).
        Shows how bias affects the decision-making process at different stages.
        """
        combined_df = self.data_collector.get_combined_dataframe()
        bias_df = self.data_collector.get_bias_effects_dataframe()
        
        if combined_df.empty:
            print("Warning: No combined data available for decision process analysis")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        # Plot 1: NPV Distribution by Scenario
        ax = axes[0, 0]
        
        npv_data = []
        scenario_labels = []
        
        for scenario in self.scenarios:
            npv_col = f'{scenario}_NPV'
            if npv_col in combined_df.columns:
                scenario_npv = combined_df[npv_col].dropna()
                if not scenario_npv.empty:
                    npv_data.append(scenario_npv)
                    display_name = self.metadata.get(scenario, {}).get('display_name', scenario)
                    scenario_labels.append(display_name)
        
        if npv_data:
            bp = ax.boxplot(npv_data, labels=scenario_labels, patch_artist=True)
            
            # Color the boxes
            for patch, scenario in zip(bp['boxes'], self.scenarios[:len(bp['boxes'])]):
                color = self.colors.get(scenario, '#000000')
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        
        ax.set_title('NPV Distribution by Scenario', fontweight='bold', fontsize=14)
        ax.set_ylabel('Net Present Value ($)')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Adoption Probability Distribution
        ax = axes[0, 1]
        
        prob_data = []
        scenario_labels = []
        
        for scenario in self.scenarios:
            prob_col = f'{scenario}_Probability'
            if prob_col in combined_df.columns:
                scenario_prob = combined_df[prob_col].dropna()
                if not scenario_prob.empty:
                    prob_data.append(scenario_prob)
                    display_name = self.metadata.get(scenario, {}).get('display_name', scenario)
                    scenario_labels.append(display_name)
        
        if prob_data:
            bp = ax.boxplot(prob_data, labels=scenario_labels, patch_artist=True)
            
            # Color the boxes
            for patch, scenario in zip(bp['boxes'], self.scenarios[:len(bp['boxes'])]):
                color = self.colors.get(scenario, '#000000')
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        
        ax.set_title('Adoption Probability Distribution', fontweight='bold', fontsize=14)
        ax.set_ylabel('Adoption Probability')
        ax.set_ylim(0, 1)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Decision Threshold Analysis
        ax = axes[0, 2]
        
        # Analyze relationship between NPV and adoption
        for scenario in self.scenarios:
            npv_col = f'{scenario}_NPV'
            adopted_col = f'{scenario}_Adopted'
            
            if npv_col in combined_df.columns and adopted_col in combined_df.columns:
                scenario_data = combined_df[[npv_col, adopted_col]].dropna()
                
                if not scenario_data.empty:
                    # Create NPV bins and calculate adoption rates
                    npv_bins = np.linspace(scenario_data[npv_col].min(), 
                                         scenario_data[npv_col].max(), 20)
                    bin_centers = (npv_bins[:-1] + npv_bins[1:]) / 2
                    
                    adoption_rates = []
                    for i in range(len(npv_bins) - 1):
                        bin_mask = ((scenario_data[npv_col] >= npv_bins[i]) & 
                                  (scenario_data[npv_col] < npv_bins[i + 1]))
                        bin_data = scenario_data[bin_mask]
                        
                        if len(bin_data) > 0:
                            adoption_rate = bin_data[adopted_col].mean()
                            adoption_rates.append(adoption_rate)
                        else:
                            adoption_rates.append(0)
                    
                    color = self.colors.get(scenario, '#000000')
                    display_name = self.metadata.get(scenario, {}).get('display_name', scenario)
                    ax.plot(bin_centers, adoption_rates, 
                           color=color, label=display_name, linewidth=2, marker='o', markersize=4)
        
        ax.set_title('NPV vs Adoption Rate (Decision Threshold)', fontweight='bold', fontsize=14)
        ax.set_xlabel('Net Present Value ($)')
        ax.set_ylabel('Adoption Rate')
        ax.set_ylim(0, 1)
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Adoption Timing Distribution
        ax = axes[1, 0]
        
        adoption_months = []
        scenario_labels = []
        
        for scenario in self.scenarios:
            adoption_col = f'{scenario}_AdoptionMonth'
            if adoption_col in combined_df.columns:
                scenario_months = combined_df[adoption_col].dropna()
                if not scenario_months.empty:
                    adoption_months.append(scenario_months)
                    display_name = self.metadata.get(scenario, {}).get('display_name', scenario)
                    scenario_labels.append(display_name)
        
        if adoption_months:
            ax.hist(adoption_months, bins=24, alpha=0.7, 
                   label=scenario_labels, color=[self.colors.get(s, '#000000') for s in self.scenarios[:len(adoption_months)]])
        
        ax.set_title('Adoption Timing Distribution', fontweight='bold', fontsize=14)
        ax.set_xlabel('Adoption Month')
        ax.set_ylabel('Number of Adoptions')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 5: Bias Impact on High-NPV Households
        ax = axes[1, 1]
        
        if not bias_df.empty and 'BaseNPV' in bias_df.columns:
            # Focus on households with high NPV (top quartile)
            high_npv_threshold = bias_df['BaseNPV'].quantile(0.75)
            high_npv_data = bias_df[bias_df['BaseNPV'] >= high_npv_threshold]
            
            if not high_npv_data.empty:
                for bias_name in self.enabled_biases:
                    if f'{bias_name}_Multiplier' in high_npv_data.columns:
                        multipliers = high_npv_data[f'{bias_name}_Multiplier']
                        
                        color = self.colors.get(bias_name, '#000000')
                        display_name = self.metadata.get(bias_name, {}).get('display_name', bias_name)
                        
                        ax.hist(multipliers, bins=20, alpha=0.7, color=color, 
                               label=display_name, density=True)
                
                ax.axvline(x=1.0, color='black', linestyle='--', alpha=0.5, label='No Effect')
        
        ax.set_title('Bias Impact on High-NPV Households', fontweight='bold', fontsize=14)
        ax.set_xlabel('Probability Multiplier')
        ax.set_ylabel('Density')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 6: Decision Confidence Analysis
        ax = axes[1, 2]
        
        # Analyze the "strength" of decisions based on probability values
        for scenario in self.scenarios:
            prob_col = f'{scenario}_Probability'
            if prob_col in combined_df.columns:
                probabilities = combined_df[prob_col].dropna()
                
                if not probabilities.empty:
                    # Calculate "confidence" as distance from 0.5 (indecision point)
                    confidence = np.abs(probabilities - 0.5)
                    
                    color = self.colors.get(scenario, '#000000')
                    display_name = self.metadata.get(scenario, {}).get('display_name', scenario)
                    
                    ax.hist(confidence, bins=20, alpha=0.7, color=color, 
                           label=display_name, density=True)
        
        ax.set_title('Decision Confidence Distribution', fontweight='bold', fontsize=14)
        ax.set_xlabel('Decision Confidence (|Prob - 0.5|)')
        ax.set_ylabel('Density')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/decision_process_analysis.png", 
                   dpi=300, bbox_inches="tight")
        plt.close()
        
        print("  ✅ Decision process analysis visualization completed")
    
    def plot_adoption_velocity_by_class(self, output_dir="results"):
        """
        Plot adoption velocity by income class (2x2 grid).
        Shows how quickly different income classes adopt under different scenarios.
        """
        combined_df = self.data_collector.get_combined_dataframe()
        
        if combined_df.empty or 'IncomeClass' not in combined_df.columns:
            print("Warning: No income class data available for velocity analysis")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        income_classes = sorted(combined_df['IncomeClass'].unique())
        
        # Plot 1: Cumulative Adoption by Class and Scenario
        ax = axes[0, 0]
        
        for scenario in self.scenarios:
            adoption_col = f'{scenario}_AdoptionMonth'
            if adoption_col in combined_df.columns:
                scenario_data = combined_df[combined_df[adoption_col].notna()]
                
                if not scenario_data.empty:
                    # Calculate cumulative adoption over time
                    max_month = scenario_data[adoption_col].max()
                    months = range(1, int(max_month) + 1)
                    
                    for income_class in income_classes:
                        class_data = scenario_data[scenario_data['IncomeClass'] == income_class]
                        
                        if not class_data.empty:
                            cumulative_adoption = []
                            for month in months:
                                adopted_by_month = (class_data[adoption_col] <= month).sum()
                                adoption_rate = adopted_by_month / len(class_data)
                                cumulative_adoption.append(adoption_rate)
                            
                            color = self.colors.get(scenario, '#000000')
                            alpha = 0.3 + 0.7 * (income_class / max(income_classes))  # Vary transparency
                            
                            if scenario == 'rational':  # Only label rational scenario to avoid clutter
                                label = f'Class {income_class}'
                            else:
                                label = None
                            
                            ax.plot(months, cumulative_adoption, 
                                   color=color, alpha=alpha, linewidth=2, label=label)
        
        ax.set_title('Cumulative Adoption Rate by Income Class', fontweight='bold', fontsize=14)
        ax.set_xlabel('Month')
        ax.set_ylabel('Cumulative Adoption Rate')
        ax.set_ylim(0, 1)
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Average Adoption Month by Class
        ax = axes[0, 1]
        
        avg_adoption_months = {}
        for scenario in self.scenarios:
            adoption_col = f'{scenario}_AdoptionMonth'
            if adoption_col in combined_df.columns:
                scenario_months = []
                for income_class in income_classes:
                    class_data = combined_df[combined_df['IncomeClass'] == income_class]
                    class_adoptions = class_data[adoption_col].dropna()
                    
                    if not class_adoptions.empty:
                        avg_month = class_adoptions.mean()
                        scenario_months.append(avg_month)
                    else:
                        scenario_months.append(np.nan)
                
                avg_adoption_months[scenario] = scenario_months
        
        # Create grouped bar chart
        x = np.arange(len(income_classes))
        width = 0.8 / len(self.scenarios)
        
        for i, scenario in enumerate(self.scenarios):
            if scenario in avg_adoption_months:
                color = self.colors.get(scenario, '#000000')
                display_name = self.metadata.get(scenario, {}).get('display_name', scenario)
                
                ax.bar(x + i * width, avg_adoption_months[scenario], 
                      width, label=display_name, color=color, alpha=0.8)
        
        ax.set_title('Average Adoption Month by Income Class', fontweight='bold', fontsize=14)
        ax.set_xlabel('Income Class')
        ax.set_ylabel('Average Adoption Month')
        ax.set_xticks(x + width * (len(self.scenarios) - 1) / 2)
        ax.set_xticklabels([f'Class {i}' for i in income_classes])
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Adoption Rate Differences (vs Rational)
        ax = axes[1, 0]
        
        if 'rational' in self.scenarios:
            rational_rates = {}
            for income_class in income_classes:
                rational_data = combined_df[combined_df['IncomeClass'] == income_class]
                rational_adoptions = rational_data['rational_Adopted'].mean()
                rational_rates[income_class] = rational_adoptions
            
            for scenario in self.scenarios:
                if scenario != 'rational':
                    adoption_col = f'{scenario}_Adopted'
                    if adoption_col in combined_df.columns:
                        rate_differences = []
                        
                        for income_class in income_classes:
                            class_data = combined_df[combined_df['IncomeClass'] == income_class]
                            scenario_rate = class_data[adoption_col].mean()
                            rational_rate = rational_rates.get(income_class, 0)
                            
                            rate_diff = scenario_rate - rational_rate
                            rate_differences.append(rate_diff)
                        
                        color = self.colors.get(scenario, '#000000')
                        display_name = self.metadata.get(scenario, {}).get('display_name', scenario)
                        
                        ax.bar(x + i * width, rate_differences, 
                              width, label=display_name, color=color, alpha=0.8)
        
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.set_title('Adoption Rate Difference vs Rational', fontweight='bold', fontsize=14)
        ax.set_xlabel('Income Class')
        ax.set_ylabel('Adoption Rate Difference')
        ax.set_xticks(range(len(income_classes)))
        ax.set_xticklabels([f'Class {i}' for i in income_classes])
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Adoption Velocity (Adoption Rate / Time)
        ax = axes[1, 1]
        
        for scenario in self.scenarios:
            adoption_col = f'{scenario}_AdoptionMonth'
            if adoption_col in combined_df.columns:
                velocities = []
                
                for income_class in income_classes:
                    class_data = combined_df[combined_df['IncomeClass'] == income_class]
                    class_adoptions = class_data[adoption_col].dropna()
                    
                    if not class_adoptions.empty:
                        # Calculate velocity as adoption rate / average adoption time
                        adoption_rate = len(class_adoptions) / len(class_data)
                        avg_adoption_month = class_adoptions.mean()
                        velocity = adoption_rate / avg_adoption_month if avg_adoption_month > 0 else 0
                        velocities.append(velocity)
                    else:
                        velocities.append(0)
                
                color = self.colors.get(scenario, '#000000')
                display_name = self.metadata.get(scenario, {}).get('display_name', scenario)
                
                ax.plot(income_classes, velocities, 
                       color=color, label=display_name, linewidth=2, marker='o', markersize=6)
        
        ax.set_title('Adoption Velocity by Income Class', fontweight='bold', fontsize=14)
        ax.set_xlabel('Income Class')
        ax.set_ylabel('Velocity (Rate/Month)')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/adoption_velocity_by_class.png", 
                   dpi=300, bbox_inches="tight")
        plt.close()
        
        print("  ✅ Adoption velocity by class visualization completed")