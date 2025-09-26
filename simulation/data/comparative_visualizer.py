# data/comparative_visualizer.py (Cleaned V3.0)
"""
Clean comparative visualization system focused purely on visualization.
Uses ComparativeMetrics for all data calculations.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from .comparative_metrics import ComparativeMetrics
from ..utils.parameters import (
    get_all_scenarios, get_scenario_colors, get_scenario_metadata
)

class ComparativeVisualizer:
    """
    Clean visualization class focused purely on creating plots.
    All metrics calculations delegated to ComparativeMetrics.
    """
    
    def __init__(self, model_data, agent_data):
        """Initialize with data and create metrics calculator."""
        self.model_data = model_data
        self.agent_data = agent_data
        self.scenarios = get_all_scenarios()
        self.colors = get_scenario_colors()
        self.metadata = get_scenario_metadata()
        
        # Create metrics calculator
        try:
            self.metrics = ComparativeMetrics(model_data, agent_data)
            self.data_available = True
        except ValueError as e:
            print(f"Warning: {e}")
            self.data_available = False
        
        # Set up plot styling
        self._setup_plot_style()
        
        print(f"ComparativeVisualizer initialized (data_available: {self.data_available})")
    
    def _setup_plot_style(self):
        """Set up consistent plot styling."""
        sns.set_style("whitegrid")
        plt.rcParams.update({
            'font.size': 12,
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 11,
            'figure.titlesize': 16,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight'
        })
    
    def create_adoption_comparison(self, output_dir="results/visualizations", include_scurve=False):
        """
        Create adoption comparison plot with optional S-curve benchmark.
        
        Args:
            output_dir: Directory to save plots
            include_scurve: Whether to include Rogers' S-curve benchmark
        """
        if not self.data_available:
            print("Cannot create adoption comparison: No data available")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Get adoption data from metrics
        adoption_data = self.metrics.adoption_data
        
        plt.figure(figsize=(14, 10) if include_scurve else (12, 8))
        
        # Plot each scenario
        scenarios_plotted = 0
        for scenario in self.scenarios:
            if scenario in adoption_data.columns:
                color = self.colors.get(scenario, '#000000')
                display_name = self.metadata.get(scenario, {}).get('display_name', scenario)
                
                adoption_rate_pct = adoption_data[scenario] * 100
                plt.plot(adoption_data.index, adoption_rate_pct, 
                        color=color, label=display_name, linewidth=3, 
                        marker='o', markersize=6)
                scenarios_plotted += 1
        
        if scenarios_plotted == 0:
            print("Cannot create adoption comparison: No valid scenario data found")
            return
        
        # Add S-curve benchmark if requested
        if include_scurve:
            self._add_scurve_benchmark(adoption_data.index)
        
        # Styling
        title = 'Adoption Rates vs Rogers\' S-Curve Benchmark' if include_scurve else 'Adoption Rates Across Behavioral Scenarios'
        plt.title(title, fontsize=18, fontweight='bold', pad=20)
        plt.xlabel('Year', fontsize=14)
        plt.ylabel('Adoption Rate (%)', fontsize=14)
        plt.legend(loc='center right' if include_scurve else 'upper left', 
                  fontsize=12, frameon=True, fancybox=True, shadow=True)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 100)
        
        # Add final rates
        self._add_final_rates_text(adoption_data)
        
        plt.tight_layout()
        
        suffix = '_with_scurve' if include_scurve else ''
        output_path = os.path.join(output_dir, f'adoption_comparison{suffix}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Adoption comparison saved to: {output_path}")
    
    def create_bias_impact_analysis(self, output_dir="results/visualizations", baseline='rational'):
        """
        Create bias impact analysis showing final adoption rates only.
        
        Args:
            output_dir: Directory to save plots
            baseline: Baseline scenario for comparison
        """
        if not self.data_available:
            print("Cannot create bias impact analysis: No data available")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        adoption_data = self.metrics.adoption_data
        
        if baseline not in adoption_data.columns:
            print(f"Cannot create bias impact analysis: Missing {baseline} baseline data")
            return
        
        # Get bias scenarios (excluding baseline)
        bias_scenarios = [s for s in ['optimism_bias', 'present_bias', 'loss_aversion',  
                                     'herding', 'status_quo', 'all_biases'] 
                         if s in adoption_data.columns]
        
        if not bias_scenarios:
            print("Cannot create bias impact analysis: No bias scenario data found")
            return
        
        # Create 2x3 subplot grid
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        for i, bias in enumerate(bias_scenarios):
            row = i // 3
            col = i % 3
            ax = axes[row, col]
            
            # Plot baseline and bias scenario
            baseline_rate = adoption_data[baseline] * 100
            bias_rate = adoption_data[bias] * 100
            
            ax.plot(adoption_data.index, baseline_rate, 
                   color='#1f77b4', label=f'{baseline.title()} Baseline', 
                   linewidth=2, linestyle='--')
            ax.plot(adoption_data.index, bias_rate, 
                   color=self.colors.get(bias, '#000000'), 
                   label=self.metadata.get(bias, {}).get('display_name', bias), 
                   linewidth=3, marker='o', markersize=4)
            
            # Calculate and display final effect
            final_effect = bias_rate.iloc[-1] - baseline_rate.iloc[-1]
            display_name = self.metadata.get(bias, {}).get('display_name', bias)
            ax.set_title(f'{display_name}\nFinal Effect: {final_effect:+.1f}pp', 
                       fontsize=12, fontweight='bold')
            
            ax.set_xlabel('Year')
            ax.set_ylabel('Adoption Rate (%)')
            ax.legend(loc='upper left', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 100)
        
        # Hide unused subplots
        for i in range(len(bias_scenarios), 6):
            row = i // 3
            col = i % 3
            axes[row, col].set_visible(False)
        
        fig.suptitle(f'Bias Effects on Adoption (vs {baseline.title()})', 
                    fontsize=18, fontweight='bold', y=0.95)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        output_path = os.path.join(output_dir, 'bias_impact_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Bias impact analysis saved to: {output_path}")
    
    def create_rational_comparison(self, output_dir="results/visualizations"):
        """
        Create rational baseline comparison using metrics layer data.
        
        Args:
            output_dir: Directory to save plots
        """
        if not self.data_available:
            print("Cannot create rational comparison: No data available")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        adoption_data = self.metrics.adoption_data
        
        # Find rational variants
        deterministic_col = None
        rational_col = None
        
        for col in adoption_data.columns:
            if 'deterministic' in col.lower() and 'rational' in col.lower():
                deterministic_col = col
            elif col.lower() == 'rational':
                rational_col = col
        
        if not deterministic_col or not rational_col:
            print("Cannot create rational comparison: Missing rational scenario variants")
            return
        
        plt.figure(figsize=(12, 8))
        
        # Plot both rational variants
        det_data = adoption_data[deterministic_col] * 100
        rational_data = adoption_data[rational_col] * 100
        
        plt.plot(adoption_data.index, det_data, 
                label='Deterministic Rational (NPV > 0)', 
                linewidth=3, marker='s', markersize=6, color='#1f77b4')
        plt.plot(adoption_data.index, rational_data, 
                label='Non-Deterministic Rational (Sigmoid)', 
                linewidth=3, marker='o', markersize=6, color='#ff7f0e')
        
        # Add NPV Evolution if available (RESTORED FUNCTIONALITY)
        npv_info = ""
        
        # Debug data availability
        if self.agent_data.empty:
            print("Warning: Agent data is empty - NPV plot will not be shown")
        elif 'NPV' not in self.agent_data.columns:
            print(f"Warning: NPV column not found in agent data. Available columns: {list(self.agent_data.columns)}")
        else:
            try:
                positive_npv_pct = self.agent_data.groupby('Year').apply(
                    lambda x: (x['NPV'] > 0).mean() * 100
                )
                
                plt.plot(positive_npv_pct.index, positive_npv_pct.values, 
                        linewidth=3, marker='^', markersize=6, color='green', 
                        label='Households with Positive NPV (%)', linestyle='--')
                
                final_positive_pct = positive_npv_pct.iloc[-1] if not positive_npv_pct.empty else 0
                npv_info = f"\n• Final Positive NPV: {final_positive_pct:.1f}%"
                print(f"Successfully added NPV evolution line (final: {final_positive_pct:.1f}%)")
                
            except Exception as e:
                print(f"Warning: Could not create NPV evolution plot: {e}")
                npv_info = ""
        
        # Calculate final difference
        final_det = det_data.iloc[-1]
        final_rational = rational_data.iloc[-1]
        difference = final_rational - final_det
        
        # Add comparison text with NPV info
        textstr = f'Final Rates:\nDeterministic: {final_det:.1f}%\nSigmoid: {final_rational:.1f}%\nDifference: {difference:+.1f}%{npv_info}'
        plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=11,
                 verticalalignment='top', 
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
        
        plt.title('Rational Decision Models Comparison', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Year', fontsize=14)
        plt.ylabel('Adoption Rate (%)', fontsize=14)
        plt.legend(loc='lower right', fontsize=12, frameon=True, fancybox=True, shadow=True)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, max(final_det, final_rational) * 1.1)
        
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, 'rational_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Rational comparison saved to: {output_path}")
    
    def create_income_class_analysis(self, output_dir="results/visualizations"):
        """Create income class analysis visualization."""
        if not self.data_available:
            print("Cannot create income class analysis: No data available")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Check if income class data is available
        income_data = self.metrics.income_data
        if income_data.empty:
            print("Cannot create income class analysis: No income class time series data")
            return
        
        # Create 2x3 subplot grid
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        scenarios_to_plot = ['optimism_bias', 'present_bias', 'loss_aversion',  
                             'herding', 'status_quo', 'all_biases']
        
        for i, scenario in enumerate(scenarios_to_plot):
            row = i // 3
            col = i % 3
            ax = axes[row, col]
            
            # Check if scenario has income class data
            scenario_class_cols = [col for col in income_data.columns 
                                 if col.startswith(f'{scenario}_Class') and col.endswith('_Rate')]
            
            if scenario_class_cols:
                # Plot adoption rate by income class
                for income_class in range(1, 6):
                    class_col = f'{scenario}_Class{income_class}_Rate'
                    if class_col in income_data.columns:
                        class_rates = income_data[class_col] * 100
                        color = plt.cm.RdYlBu_r(income_class / 5.0)
                        label = f'Class {income_class}'
                        if income_class == 1:
                            label += ' (Lowest)'
                        elif income_class == 5:
                            label += ' (Highest)'
                        
                        ax.plot(income_data.index, class_rates, 
                               color=color, label=label, linewidth=2, marker='o', markersize=4)
                
                # Customize subplot
                display_name = self.metadata.get(scenario, {}).get('display_name', scenario)
                ax.set_title(display_name, fontsize=12, fontweight='bold')
                ax.set_xlabel('Year')
                ax.set_ylabel('Adoption Rate (%)')
                ax.legend(loc='upper left', fontsize=9)
                ax.grid(True, alpha=0.3)
                ax.set_ylim(0, 100)
                
                # Add final adoption inequality metric
                final_rates = []
                for ic in range(1, 6):
                    class_col = f'{scenario}_Class{ic}_Rate'
                    if class_col in income_data.columns and len(income_data[class_col]) > 0:
                        final_rates.append(income_data[class_col].iloc[-1] * 100)
                
                if len(final_rates) >= 5:
                    inequality = final_rates[4] - final_rates[0]  # Class 5 - Class 1
                    ax.text(0.02, 0.98, f'Inequality: {inequality:.1f}pp', 
                           transform=ax.transAxes, fontsize=10, 
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                           verticalalignment='top')
            else:
                ax.text(0.5, 0.5, f'Income Class Data\nNot Available\nfor {scenario}', 
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=12, alpha=0.7, color='red')
                ax.set_xticks([])
                ax.set_yticks([])
        
        fig.suptitle('Adoption by Income Class Across Scenarios', 
                    fontsize=18, fontweight='bold', y=0.95)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        output_path = os.path.join(output_dir, 'income_class_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Income class analysis saved to: {output_path}")
    
    def create_system_impact_comparison(self, output_dir="results/visualizations"):
        """Create system impact comparison visualization."""
        if not self.data_available:
            print("Cannot create system impact comparison: No data available")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Use pre-extracted system metrics from metrics layer
        system_metrics = self.metrics.system_metrics
        
        if system_metrics.empty:
            print("Cannot create system impact comparison: No system metrics data")
            return
        
        # Create 2x2 subplot grid for system impacts
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Grid Peak Load
        ax = axes[0, 0]
        if 'MonthlyPeakLoad' in system_metrics.columns:
            ax.plot(system_metrics['Year'], system_metrics['MonthlyPeakLoad'], 
                   color='#ff7f0e', linewidth=2, marker='o')
            ax.set_title('Grid Peak Load Evolution', fontweight='bold')
            ax.set_xlabel('Year')
            ax.set_ylabel('Peak Load (kW)')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Peak Load\nData Not Available', 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=14, alpha=0.7, color='red')
        
        # Plot 2: Credit System Utilization
        ax = axes[0, 1]
        if 'CreditUtilizationRate' in system_metrics.columns:
            ax.plot(system_metrics['Year'], system_metrics['CreditUtilizationRate'] * 100, 
                   color='#2ca02c', linewidth=2, marker='s')
            ax.set_title('Credit System Utilization', fontweight='bold')
            ax.set_xlabel('Year')
            ax.set_ylabel('Utilization Rate (%)')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Credit Utilization\nData Not Available', 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=14, alpha=0.7, color='red')
        
        # Plot 3: Fossil Dependency
        ax = axes[1, 0]
        if 'FossilDependency' in system_metrics.columns:
            ax.plot(system_metrics['Year'], system_metrics['FossilDependency'] * 100, 
                   color='#d62728', linewidth=2, marker='^')
            ax.set_title('Fossil Fuel Dependency', fontweight='bold')
            ax.set_xlabel('Year')
            ax.set_ylabel('Fossil Dependency (%)')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Fossil Dependency\nData Not Available', 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=14, alpha=0.7, color='red')
        
        # Plot 4: Energy Price Evolution
        ax = axes[1, 1]
        if 'FossilPrice' in system_metrics.columns and 'RenewablePrice' in system_metrics.columns:
            ax.plot(system_metrics['Year'], system_metrics['FossilPrice'], 
                   color='#8c564b', label='Fossil Price', linewidth=2)
            ax.plot(system_metrics['Year'], system_metrics['RenewablePrice'], 
                   color='#17becf', label='Renewable Price', linewidth=2)
            ax.set_title('Energy Price Evolution', fontweight='bold')
            ax.set_xlabel('Year')
            ax.set_ylabel('Price ($/kWh)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Price Data\nNot Available', 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=14, alpha=0.7, color='red')
        
        fig.suptitle('System-Level Impacts Across Behavioral Scenarios', 
                    fontsize=18, fontweight='bold', y=0.95)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        output_path = os.path.join(output_dir, 'system_impact_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"System impact comparison saved to: {output_path}")
    
    def create_rational_baseline_income_analysis(self, output_dir="results/visualizations"):
        """Create separate rational baseline income class analysis."""
        if not self.data_available:
            print("Cannot create rational baseline income analysis: No data available")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        income_data = self.metrics.income_data
        if income_data.empty:
            print("Cannot create rational baseline income analysis: No income class time series data")
            return
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        scenario_class_cols = [col for col in income_data.columns 
                             if col.startswith('rational_Class') and col.endswith('_Rate')]
        
        if scenario_class_cols:
            for income_class in range(1, 6):
                class_col = f'rational_Class{income_class}_Rate'
                if class_col in income_data.columns:
                    ax.plot(income_data.index, income_data[class_col] * 100,
                           label=f'Class {income_class}', linewidth=2, marker='o', markersize=4)
            
            ax.set_title('Rational Baseline - Solar PV Adoption by Income Class', 
                        fontweight='bold', fontsize=14)
            ax.set_xlabel('Year')
            ax.set_ylabel('Adoption Rate (%)')
            ax.legend(title='Income Classes', loc='upper left', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 100)
        else:
            ax.text(0.5, 0.5, 'Rational Income Class\nData Not Available', 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=14, alpha=0.7, color='red')
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, 'rational_baseline_income_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Rational baseline income analysis saved to: {output_path}")
    
    def create_all_visualizations(self, output_dir="results/visualizations"):
        """
        Create ALL visualizations - preserving existing functionality.
        
        Args:
            output_dir: Directory to save plots
        """
        if not self.data_available:
            print("Cannot create visualizations: No data available")
            return
        
        print(f"Creating all comparative visualizations in {output_dir}...")
        
        # Core visualizations (preserving original method calls)
        self.create_adoption_comparison(output_dir, include_scurve=False)
        self.create_adoption_comparison(output_dir, include_scurve=True)  # S-curve variant
        self.create_bias_impact_analysis(output_dir)
        self.create_income_class_analysis(output_dir)  # RESTORED
        self.create_system_impact_comparison(output_dir)  # RESTORED
        self.create_rational_baseline_income_analysis(output_dir)  # RESTORED
        self.create_rational_comparison(output_dir)
        
        print("All comparative visualizations completed!")
    
    def create_metrics_visualizations(self, metrics_dir, output_dir="results/visualizations"):
        """
        Create visualizations for the 4 exported metrics CSV files.
        
        Args:
            metrics_dir: Directory containing the CSV files
            output_dir: Directory to save visualization plots
        """
        if not self.data_available:
            print("Cannot create metrics visualizations: No data available")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Creating metrics visualizations from {metrics_dir}...")
        
        # 1. Critical Mass Timing Table
        self.create_critical_mass_timing_table(metrics_dir, output_dir)
        
        # 2. Area Analysis Table  
        self.create_area_analysis_table(metrics_dir, output_dir)
        
        # 3. Scenario Comparison Summary Table
        self.create_scenario_comparison_summary_table(metrics_dir, output_dir)
        
        # 4. Adoption Snapshots Table
        self.create_adoption_snapshots_table_visual(metrics_dir, output_dir)
        
        print("All metrics visualizations completed!")
    
    def create_critical_mass_timing_table(self, metrics_dir, output_dir):
        """Create visual table for critical mass timing analysis."""
        csv_path = os.path.join(metrics_dir, 'critical_mass_timing.csv')
        
        if not os.path.exists(csv_path):
            print(f"Cannot create critical mass timing table: {csv_path} not found")
            return
        
        try:
            df = pd.read_csv(csv_path)
            
            # Filter for key scenarios and pivot for better display
            key_scenarios = ['rational', 'loss_aversion', 'present_bias', 'status_quo', 'optimism_bias', 'herding', 'all_biases']
            df_filtered = df[df['scenario'].isin(key_scenarios)].copy()
            
            if df_filtered.empty:
                print("No data available for critical mass timing table")
                return
            
            # Create pivot table
            pivot_df = df_filtered.pivot(index='scenario', columns='threshold_pct', values='threshold_year')
            
            # Add time differences vs rational
            rational_timing = df_filtered[df_filtered['scenario'] == 'rational'].set_index('threshold_pct')['threshold_year']
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
            
            # Table 1: Threshold Years
            ax1.axis('tight')
            ax1.axis('off')
            ax1.set_title('Critical Mass Timing (Years to Reach Threshold)', fontsize=14, fontweight='bold', pad=20)
            
            # Format data for display
            display_data = pivot_df.round(1).fillna('Never')
            table1 = ax1.table(cellText=display_data.values,
                              rowLabels=[self.metadata.get(idx, {}).get('display_name', idx) for idx in display_data.index],
                              colLabels=[f'{int(col)}%' for col in display_data.columns],
                              cellLoc='center', loc='center')
            
            table1.auto_set_font_size(False)
            table1.set_fontsize(10)
            table1.scale(1, 2)
            
            # Style the table
            for i in range(len(display_data.columns)):
                table1[(0, i)].set_facecolor('#4CAF50')
                table1[(0, i)].set_text_props(weight='bold', color='white')
            
            # Table 2: Time Differences vs Rational
            ax2.axis('tight')  
            ax2.axis('off')
            ax2.set_title('Time Difference vs Rational Baseline (Years)', fontsize=14, fontweight='bold', pad=20)
            
            # Calculate differences
            diff_data = pivot_df.copy()
            for threshold in diff_data.columns:
                if threshold in rational_timing.index:
                    diff_data[threshold] = diff_data[threshold] - rational_timing[threshold]
            
            # Remove rational row (all zeros)
            diff_data = diff_data.drop('rational', errors='ignore')
            diff_display = diff_data.round(1).fillna('N/A')
            
            table2 = ax2.table(cellText=diff_display.values,
                              rowLabels=[self.metadata.get(idx, {}).get('display_name', idx) for idx in diff_display.index],
                              colLabels=[f'{int(col)}%' for col in diff_display.columns],
                              cellLoc='center', loc='center')
            
            table2.auto_set_font_size(False)
            table2.set_fontsize(10)
            table2.scale(1, 2)
            
            # Style the table with color coding
            for i in range(len(diff_display.columns)):
                table2[(0, i)].set_facecolor('#FF7F0E')
                table2[(0, i)].set_text_props(weight='bold', color='white')
            
            # Color code delays (positive) vs acceleration (negative)
            for i, row_data in enumerate(diff_display.values, 1):
                for j, val in enumerate(row_data):
                    if val != 'N/A' and str(val).replace('-', '').replace('.', '').isdigit():
                        val_float = float(val)
                        if val_float > 0:
                            table2[(i, j)].set_facecolor('#FFEBEE')  # Light red for delays
                        elif val_float < 0:
                            table2[(i, j)].set_facecolor('#E8F5E8')  # Light green for acceleration
            
            plt.tight_layout()
            
            output_path = os.path.join(output_dir, 'critical_mass_timing_table.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print(f"Critical mass timing table saved to: {output_path}")
            
        except Exception as e:
            print(f"Error creating critical mass timing table: {e}")
    
    def create_area_analysis_table(self, metrics_dir, output_dir):
        """Create visual table for area analysis with multiple interpretations."""
        csv_path = os.path.join(metrics_dir, 'area_analysis.csv')
        
        if not os.path.exists(csv_path):
            print(f"Cannot create area analysis table: {csv_path} not found")
            return
        
        try:
            df = pd.read_csv(csv_path)
            
            if df.empty:
                print("No data available for area analysis table")
                return
            
            fig, ax = plt.subplots(figsize=(16, 10))
            ax.axis('tight')
            ax.axis('off')
            
            # Prepare data for display
            display_data = []
            headers = ['Scenario', 'Cumulative Gap\n(adoption-years)', 'Average Annual\nDifference (%)', 
                      'Total Additional\nAdoptions', 'Final Adoption\nDifference (pp)']
            
            for _, row in df.iterrows():
                scenario_name = self.metadata.get(row['scenario'], {}).get('display_name', row['scenario'])
                display_data.append([
                    scenario_name,
                    f"{row['cumulative_adoption_gap']:.2f}",
                    f"{row['average_annual_difference']:.2f}",
                    f"{row['total_additional_adoptions']:.0f}",
                    f"{row['final_adoption_difference_pct']:+.1f}"
                ])
            
            table = ax.table(cellText=display_data,
                           colLabels=headers,
                           cellLoc='center',
                           loc='center')
            
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 2.5)
            
            # Style the header
            for i in range(len(headers)):
                table[(0, i)].set_facecolor('#2196F3')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            # Color code positive/negative values in the last column
            for i in range(1, len(display_data) + 1):
                final_diff = display_data[i-1][-1]  # Last column value
                if '+' in final_diff:
                    table[(i, len(headers)-1)].set_facecolor('#E8F5E8')  # Light green
                elif '-' in final_diff:
                    table[(i, len(headers)-1)].set_facecolor('#FFEBEE')  # Light red
            
            ax.set_title('Area Analysis: Multiple Interpretations of Adoption Gaps', 
                        fontsize=16, fontweight='bold', pad=30)
            
            # Add explanation text
            explanation = ("Cumulative Gap: Area between scenario and rational curves (adoption-years)\n"
                          "Average Annual Difference: Gap divided by simulation years\n" 
                          "Total Additional Adoptions: Gap multiplied by population size\n"
                          "Final Adoption Difference: End-state difference in percentage points")
            
            plt.figtext(0.5, 0.02, explanation, ha='center', fontsize=9, style='italic')
            
            plt.tight_layout()
            
            output_path = os.path.join(output_dir, 'area_analysis_table.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print(f"Area analysis table saved to: {output_path}")
            
        except Exception as e:
            print(f"Error creating area analysis table: {e}")
    
    def create_scenario_comparison_summary_table(self, metrics_dir, output_dir):
        """Create visual table for scenario comparison summary statistics."""
        csv_path = os.path.join(metrics_dir, 'scenario_comparison_summary.csv')
        
        if not os.path.exists(csv_path):
            print(f"Cannot create scenario comparison summary table: {csv_path} not found")
            return
        
        try:
            df = pd.read_csv(csv_path)
            
            if df.empty:
                print("No data available for scenario comparison summary table")
                return
            
            fig, ax = plt.subplots(figsize=(18, 10))
            ax.axis('tight')
            ax.axis('off')
            
            # Prepare data for display
            display_data = []
            headers = ['Scenario', 'Final Adoption\nRate (%)', 'Avg Difference\nvs Rational (pp)', 
                      'Min Difference\n(pp)', 'Max Difference\n(pp)', 'Std Deviation\n(pp)']
            
            for _, row in df.iterrows():
                scenario_name = self.metadata.get(row['scenario'], {}).get('display_name', row['scenario'])
                display_data.append([
                    scenario_name,
                    f"{row['final_adoption_rate_pct']:.1f}",
                    f"{row['avg_difference_vs_rational']:+.1f}",
                    f"{row['min_difference_vs_rational']:+.1f}",
                    f"{row['max_difference_vs_rational']:+.1f}",
                    f"{row['std_difference_vs_rational']:.1f}"
                ])
            
            table = ax.table(cellText=display_data,
                           colLabels=headers,
                           cellLoc='center',
                           loc='center')
            
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 2.5)
            
            # Style the header
            for i in range(len(headers)):
                table[(0, i)].set_facecolor('#9C27B0')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            # Highlight rational baseline row
            for i, row_data in enumerate(display_data, 1):
                if 'rational' in row_data[0].lower() and 'deterministic' not in row_data[0].lower():
                    for j in range(len(headers)):
                        table[(i, j)].set_facecolor('#F3E5F5')
                        table[(i, j)].set_text_props(weight='bold')
            
            ax.set_title('Scenario Comparison Summary Statistics', 
                        fontsize=16, fontweight='bold', pad=30)
            
            # Add explanation
            explanation = ("Statistics comparing each scenario to rational baseline over full simulation period\n"
                          "pp = percentage points difference")
            
            plt.figtext(0.5, 0.02, explanation, ha='center', fontsize=9, style='italic')
            
            plt.tight_layout()
            
            output_path = os.path.join(output_dir, 'scenario_comparison_summary_table.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print(f"Scenario comparison summary table saved to: {output_path}")
            
        except Exception as e:
            print(f"Error creating scenario comparison summary table: {e}")
    
    def create_adoption_snapshots_table_visual(self, metrics_dir, output_dir):
        """Create visual table for adoption snapshots at specific time points."""
        csv_path = os.path.join(metrics_dir, 'adoption_snapshots_table.csv')
        
        if not os.path.exists(csv_path):
            print(f"Cannot create adoption snapshots table: {csv_path} not found")
            return
        
        try:
            df = pd.read_csv(csv_path)
            
            if df.empty:
                print("No data available for adoption snapshots table")
                return
            
            fig, ax = plt.subplots(figsize=(14, 10))
            ax.axis('tight')
            ax.axis('off')
            
            # Prepare data for display (exclude scenario column, use display_name)
            display_data = []
            time_columns = [col for col in df.columns if 'pct_time' in col]
            headers = ['Scenario'] + [col.replace('pct_time', '% Time') for col in time_columns]
            
            for _, row in df.iterrows():
                scenario_name = row.get('display_name', row['scenario'])
                row_data = [scenario_name]
                for time_col in time_columns:
                    row_data.append(f"{row[time_col]:.1f}%")
                display_data.append(row_data)
            
            table = ax.table(cellText=display_data,
                           colLabels=headers,
                           cellLoc='center',
                           loc='center')
            
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 2.2)
            
            # Style the header
            for i in range(len(headers)):
                table[(0, i)].set_facecolor('#FF9800')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            # Alternate row colors for readability
            for i in range(1, len(display_data) + 1):
                if i % 2 == 0:
                    for j in range(len(headers)):
                        table[(i, j)].set_facecolor('#F5F5F5')
            
            # Highlight rational baseline
            for i, row_data in enumerate(display_data, 1):
                if 'rational' in row_data[0].lower() and 'deterministic' not in row_data[0].lower():
                    for j in range(len(headers)):
                        table[(i, j)].set_facecolor('#FFF3E0')
                        table[(i, j)].set_text_props(weight='bold')
            
            ax.set_title('Adoption Rate Snapshots at Key Time Points', 
                        fontsize=16, fontweight='bold', pad=30)
            
            # Add explanation
            explanation = ("Adoption rates at 20%, 30%, 50%, and 90% of total simulation time\n"
                          "Shows adoption progression across different behavioral scenarios")
            
            plt.figtext(0.5, 0.02, explanation, ha='center', fontsize=9, style='italic')
            
            plt.tight_layout()
            
            output_path = os.path.join(output_dir, 'adoption_snapshots_table.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print(f"Adoption snapshots table saved to: {output_path}")
            
        except Exception as e:
            print(f"Error creating adoption snapshots table: {e}")
    
    def _extract_system_metrics(self):
        """Extract system-level metrics (needed for system impact comparison)."""
        if not hasattr(self, 'model_data') or self.model_data.empty:
            return pd.DataFrame()
        
        # Look for system metric columns
        system_cols = ['Year', 'FossilPrice', 'RenewablePrice', 'rational_TotalCreditsEarned', 
                    'rational_TotalCreditsUsed', 'rational_CreditUtilizationRate', 'rational_GridStressIndex',
                    'MonthlyPeakLoad', 'rational_FossilDependency']
        
        available_cols = [col for col in system_cols if col in self.model_data.columns]
        
        if len(available_cols) <= 1:  # Only Year column
            return pd.DataFrame()
        
        system_data = self.model_data[available_cols].copy()
        
        if 'Year' in system_data.columns:
            system_data = system_data.groupby('Year').last().reset_index()
        
        return system_data
    
    def _add_scurve_benchmark(self, years):
        """Add Rogers' S-curve benchmark to current plot."""
        s_curve = self._generate_rogers_scurve(years)
        plt.plot(years, s_curve * 100, 'k--', linewidth=2, 
                label="Rogers' S-curve", alpha=0.8)
        
        # Add adopter category markers
        thresholds = {'Innovators': 0.025, 'Early Adopters': 0.16, 
                     'Early Majority': 0.50, 'Late Majority': 0.84}
        
        for category, threshold in thresholds.items():
            crossing_indices = np.where(s_curve >= threshold)[0]
            if len(crossing_indices) > 0:
                crossing_year = years[crossing_indices[0]]
                plt.axvline(crossing_year, color='gray', linestyle=':', alpha=0.6)
                plt.axhline(threshold * 100, color='gray', linestyle=':', alpha=0.4)
    
    def _generate_rogers_scurve(self, years):
        """Generate Rogers' diffusion S-curve."""
        # Standard S-curve parameters
        growth_rate = 6
        midpoint = 0.5
        
        # Normalize years to 0-1 range
        t_norm = (years - years[0]) / (years[-1] - years[0]) if len(years) > 1 else np.array([0.5])
        
        # Logistic function
        s_curve = 1 / (1 + np.exp(-growth_rate * (t_norm - midpoint)))
        
        return s_curve
    
    def _add_final_rates_text(self, adoption_data):
        """Add final adoption rates as text annotations."""
        final_year = adoption_data.index[-1]
        y_pos = 95
        
        for scenario in self.scenarios:
            if scenario in adoption_data.columns:
                final_rate = adoption_data[scenario].iloc[-1] * 100
                color = self.colors.get(scenario, '#000000')
                plt.text(final_year + 0.5, y_pos, f'{final_rate:.1f}%', 
                        color=color, fontweight='bold', fontsize=10)
                y_pos -= 5


# Simple test function
def test_clean_visualizer():
    """Test the cleaned visualizer with minimal data."""
    import pandas as pd
    
    # Create minimal test data
    test_model_data = pd.DataFrame({
        'Year': [1, 2, 3, 4, 5],
        'rational_AdoptionRate': [0.1, 0.2, 0.4, 0.6, 0.8],
        'loss_aversion_AdoptionRate': [0.05, 0.1, 0.2, 0.3, 0.4],
        'deterministic_rational_AdoptionRate': [0.15, 0.3, 0.5, 0.7, 0.9]
    })
    
    test_agent_data = pd.DataFrame({
        'Year': [1, 2, 3, 4, 5],
        'AgentType': ['Household'] * 5
    })
    
    try:
        visualizer = ComparativeVisualizer(test_model_data, test_agent_data)
        print("Clean visualizer test passed!")
        return True
    except Exception as e:
        print(f"Clean visualizer test failed: {e}")
        return False

if __name__ == "__main__":
    test_clean_visualizer()