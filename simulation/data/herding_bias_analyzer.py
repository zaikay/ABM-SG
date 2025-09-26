"""
Herding Bias Analysis Module

This module provides specialized analysis tools for understanding herding bias effects
in prosumer adoption decisions, focusing on the gap between rational and herding-influenced
adoption patterns using step-level data for precise analysis.

Author: Based on ABM simulation framework
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class HerdingBiasAnalyzer:
    """
    Specialized analyzer for herding bias effects in prosumer adoption.
    
    Provides tools to:
    1. Quantify step-by-step herding gaps vs rational baseline
    2. Analyze cumulative herding amplification effects
    3. Generate comprehensive herding impact metrics
    
    Uses the same initialization pattern as ComparativeVisualizer for consistency.
    """
    
    def __init__(self, model_data, agent_data):
        """
        Initialize herding bias analyzer following ComparativeVisualizer pattern.
        
        Args:
            model_data: System metrics DataFrame from data_collector.get_system_metrics_dataframe()
            agent_data: Agent DataFrame from data_collector.get_combined_dataframe()
        """
        self.model_data = model_data
        self.agent_data = agent_data
        self.output_dir = "results/herding_analysis"
        
        # Extract step-level adoption data 
        self.adoption_data = self._extract_adoption_time_series()
        
        # Validate required scenarios exist
        self._validate_scenarios()
        
        # Set up styling consistent with existing visualizations
        self.colors = {
            'rational': '#1f77b4',
            'herding': '#ff7f0e', 
            'gap_positive': '#2ca02c',
            'gap_negative': '#d62728',
            'cumulative': '#9467bd'
        }
        
        # Configure matplotlib for publication-quality plots
        plt.style.use('default')
        self.setup_plot_style()
    
    def setup_plot_style(self):
        """Configure consistent plot styling."""
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 11,
            'figure.titlesize': 16
        })
    
    def _extract_adoption_time_series(self):
        """
        Extract step-level adoption time series for precise gap analysis.
        
        Returns:
            DataFrame: Step-level adoption rates indexed by Step with scenario columns
        """
        if self.model_data.empty:
            return pd.DataFrame()
        
        # Check available columns
        print(f"🔍 Available columns in model_data: {list(self.model_data.columns)}")
        
        adoption_data = pd.DataFrame()
        required_scenarios = ['rational', 'herding']
        
        # Extract step-level adoption rates (no year aggregation!)
        for scenario in required_scenarios:
            rate_col = f'{scenario}_AdoptionRate'
            if rate_col in self.model_data.columns:
                adoption_data[scenario] = self.model_data[rate_col]
            else:
                print(f"⚠️  Column {rate_col} not found")
        
        # Use Step as index for step-by-step analysis
        if not adoption_data.empty and 'Step' in self.model_data.columns:
            adoption_data.index = self.model_data['Step']
            # Add Year and Month for reference
            if 'Year' in self.model_data.columns:
                adoption_data['Year'] = self.model_data['Year']
            if 'Month' in self.model_data.columns:
                adoption_data['Month'] = self.model_data['Month']
        
        print(f"📊 Extracted adoption data shape: {adoption_data.shape}")
        print(f"📈 Step range: {adoption_data.index.min()} to {adoption_data.index.max()}")
        
        return adoption_data
    
    def _validate_scenarios(self):
        """Validate that required scenarios exist in the extracted adoption data."""
        required_scenarios = ['rational', 'herding']
        available_scenarios = list(self.adoption_data.columns) if not self.adoption_data.empty else []
        
        # Remove Year and Month from scenario list if they exist
        available_scenarios = [s for s in available_scenarios if s not in ['Year', 'Month']]
        
        missing_scenarios = [s for s in required_scenarios if s not in available_scenarios]
        if missing_scenarios:
            raise ValueError(f"Missing required scenarios: {missing_scenarios}. "
                           f"Available scenarios: {available_scenarios}")
        
        if self.adoption_data.empty:
            raise ValueError("No adoption data extracted from model_data")
        
        print(f"✅ Herding bias analysis initialized with scenarios: {available_scenarios}")
        print(f"✅ Data shape: {self.adoption_data.shape[0]} time points, {len(available_scenarios)} scenarios")
    
    def _get_total_households(self):
        """
        Calculate total households from actual data instead of assuming.
        
        Returns:
            int: Total number of households in the simulation
        """
        # Method 1: From parameters.py (if we want to import it)
        try:
            from ..utils.parameters import NUM_HOUSEHOLDS
            total_from_params = NUM_HOUSEHOLDS
            print(f"📊 From parameters.py: {total_from_params} households")
        except ImportError:
            total_from_params = None
            print("⚠️  Could not import NUM_HOUSEHOLDS from parameters.py")
        
        # Method 2: Calculate from agent_data (most robust)
        total_from_data = None
        if not self.agent_data.empty and 'AgentID' in self.agent_data.columns:
            # Get unique household agents (exclude CentralProvider if present)
            unique_agents = self.agent_data['AgentID'].nunique()
            # Filter out non-household agents if AgentType column exists
            if 'AgentType' in self.agent_data.columns:
                household_agents = self.agent_data[self.agent_data['AgentType'] == 'Household']['AgentID'].nunique()
                total_from_data = household_agents
                print(f"📊 From agent_data: {total_from_data} households (filtered by AgentType)")
            else:
                total_from_data = unique_agents
                print(f"📊 From agent_data: {total_from_data} total agents (assuming all households)")
        
        # Method 3: Calculate from final adoption rate in model_data
        total_from_model = None
        if not self.model_data.empty and 'rational_AdoptionRate' in self.model_data.columns:
            # Get adoption rate from a recent step with some adoption
            recent_data = self.model_data.tail(60)  # Last 5 years
            adoption_data = recent_data[recent_data['rational_AdoptionRate'] > 0]
            if not adoption_data.empty:
                # This would require knowing absolute number of adopters, which we don't have directly
                # Skip this method for now
                pass
        
        # Decide which value to use (priority: data > params > fallback)
        if total_from_data is not None:
            total_households = total_from_data
            print(f"✅ Using total households from agent_data: {total_households}")
        elif total_from_params is not None:
            total_households = total_from_params
            print(f"✅ Using total households from parameters.py: {total_households}")
        else:
            # Fallback - make an educated guess from typical simulation size
            total_households = 1000  # Most common value based on the code I've seen
            print(f"⚠️  Fallback: Using estimated {total_households} households")
            print("    To fix this, ensure AgentID column exists in agent_data")
        
        return total_households
    
    def create_herding_gap_analysis(self, output_dir=None):
        """
        Tool 1: Step-by-step herding vs rational comparison.
        Shows 2 graphics: step-by-step and cumulative herding amplification over time.
        
        Creates:
        1. Step-by-step adoption rates comparison with gap visualization
        2. Cumulative herding effect analysis over time
        
        Args:
            output_dir: Directory to save plots (optional)
        """
        if output_dir is None:
            output_dir = self.output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract scenario data (step-level)
        rational_data = self.adoption_data['rational'] * 100  # Convert to percentage
        herding_data = self.adoption_data['herding'] * 100
        
        # Create step-to-year mapping for x-axis display
        if 'Year' in self.adoption_data.columns:
            years = self.adoption_data['Year']
        else:
            # Fallback: convert steps to years (12 steps = 1 year)
            years = (self.adoption_data.index // 12) + 1
            
        # Calculate step-by-step gap
        step_gap = herding_data - rational_data
        
        # Calculate cumulative gap (cumulative sum of differences)
        cumulative_gap = step_gap.cumsum()
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
        
        # === PLOT 1: Step-by-Step Comparison ===
        ax1_twin = ax1.twinx()  # Secondary y-axis for gap
        
        # Plot adoption rates using years for x-axis
        line_rational = ax1.plot(years, rational_data, 
                                color=self.colors['rational'], linewidth=3, 
                                label='Rational Baseline', marker='o', markersize=3)
        line_herding = ax1.plot(years, herding_data, 
                               color=self.colors['herding'], linewidth=3, 
                               label='Herding Influenced', marker='s', markersize=3)
        
        # Plot step-by-step gap on secondary axis
        gap_colors = [self.colors['gap_positive'] if x >= 0 else self.colors['gap_negative'] 
                     for x in step_gap]
        bars = ax1_twin.bar(years, step_gap, alpha=0.6, 
                           color=gap_colors, width=0.5, label='Herding Gap (pp)')
        
        # Styling for plot 1
        ax1.set_xlabel('Year', fontsize=12)
        ax1.set_ylabel('Adoption Rate (%)', fontsize=12, color='black')
        ax1_twin.set_ylabel('Herding Gap (percentage points)', fontsize=12, color='gray')
        ax1.set_title('Step-by-Step Herding vs Rational Adoption Comparison', 
                     fontsize=14, fontweight='bold', pad=20)
        
        # Add horizontal line at y=0 for gap reference
        ax1_twin.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
        
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_twin.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', 
                  frameon=True, fancybox=True, shadow=True)
        
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, max(max(rational_data), max(herding_data)) * 1.1)
        
        # === PLOT 2: Cumulative Herding Effect ===
        ax2.plot(years, cumulative_gap, 
                color=self.colors['cumulative'], linewidth=4, 
                label='Cumulative Herding Effect', marker='D', markersize=4)
        
        # Fill area under curve to show cumulative effect magnitude
        ax2.fill_between(years, 0, cumulative_gap, 
                        color=self.colors['cumulative'], alpha=0.3)
        
        # Add zero reference line
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
        
        # Annotations for key points
        max_effect_idx = cumulative_gap.idxmax()
        max_effect_value = cumulative_gap.max()
        final_effect_value = cumulative_gap.iloc[-1]
        
        # Convert step index to year for annotation
        max_effect_year = years.loc[max_effect_idx] if max_effect_idx in years.index else years.iloc[-1]
        final_year = years.iloc[-1]
        
        ax2.annotate(f'Peak Effect: {max_effect_value:.1f}pp\n(Year {max_effect_year})',
                    xy=(max_effect_year, max_effect_value), 
                    xytext=(max_effect_year + 2, max_effect_value + 5),
                    arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                    fontsize=10, ha='center')
        
        ax2.annotate(f'Final Effect: {final_effect_value:.1f}pp',
                    xy=(final_year, final_effect_value),
                    xytext=(final_year - 3, final_effect_value + 3),
                    arrowprops=dict(arrowstyle='->', color='blue', lw=1.5),
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
                    fontsize=10, ha='center')
        
        # Styling for plot 2
        ax2.set_xlabel('Year', fontsize=12)
        ax2.set_ylabel('Cumulative Herding Effect (percentage points)', fontsize=12)
        ax2.set_title('Cumulative Herding Effect Over Time', 
                     fontsize=14, fontweight='bold', pad=20)
        ax2.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
        ax2.grid(True, alpha=0.3)
        
        # Overall figure title
        fig.suptitle('Herding Bias Gap Analysis: Step-by-Step and Cumulative Effects', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9, hspace=0.3)
        
        # Save the plot
        output_path = os.path.join(output_dir, 'herding_gap_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"✅ Herding gap analysis saved to: {output_path}")
        
        return {
            'max_effect_year': max_effect_year,
            'max_effect_value': max_effect_value,
            'final_effect_value': final_effect_value,
            'step_gap_data': step_gap,
            'cumulative_gap_data': cumulative_gap
        }
    
    def create_herding_impact_metrics(self, output_dir=None):
        """
        Summary metrics: absolute numbers, percentages, multipliers.
        
        Generates comprehensive metrics table and summary statistics
        for herding bias impact assessment.
        
        Args:
            output_dir: Directory to save metrics (optional)
            
        Returns:
            dict: Comprehensive metrics dictionary
        """
        if output_dir is None:
            output_dir = self.output_dir
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract scenario data (step-level)
        rational_data = self.adoption_data['rational']
        herding_data = self.adoption_data['herding']
        
        # Calculate total households from actual data (don't assume!)
        total_households = self._get_total_households()
        
        # Calculate absolute numbers
        rational_adopters = rational_data * total_households
        herding_adopters = herding_data * total_households
        absolute_gap = herding_adopters - rational_adopters
        
        # Calculate percentage gaps
        percentage_gap = (herding_data - rational_data) * 100  # percentage points
        
        # Calculate multipliers (avoid division by zero)
        multipliers = herding_data / rational_data.replace(0, np.nan)
        
        # Create step-to-year mapping for temporal analysis
        if 'Year' in self.adoption_data.columns:
            years = self.adoption_data['Year']
        else:
            years = (self.adoption_data.index // 12) + 1
        
        # Calculate key summary metrics
        metrics = {
            # === Simulation Info ===
            'total_households': total_households,
            
            # === Absolute Numbers ===
            'final_rational_adopters': int(rational_adopters.iloc[-1]),
            'final_herding_adopters': int(herding_adopters.iloc[-1]),
            'final_absolute_gap': int(absolute_gap.iloc[-1]),
            'max_absolute_gap': int(absolute_gap.max()),
            'min_absolute_gap': int(absolute_gap.min()),
            'mean_absolute_gap': float(absolute_gap.mean()),
            
            # === Percentage Effects ===
            'final_percentage_gap': float(percentage_gap.iloc[-1]),
            'max_percentage_gap': float(percentage_gap.max()),
            'min_percentage_gap': float(percentage_gap.min()),
            'mean_percentage_gap': float(percentage_gap.mean()),
            'std_percentage_gap': float(percentage_gap.std()),
            
            # === Multiplier Effects ===
            'final_herding_multiplier': float(multipliers.iloc[-1]) if not pd.isna(multipliers.iloc[-1]) else 1.0,
            'max_herding_multiplier': float(multipliers.max()) if not multipliers.empty else 1.0,
            'mean_herding_multiplier': float(multipliers.mean()) if not multipliers.empty else 1.0,
            
            # === Temporal Analysis (using years) ===
            'peak_effect_year': int(years.loc[absolute_gap.idxmax()]) if absolute_gap.idxmax() in years.index else int(years.iloc[-1]),
            'peak_effect_magnitude': float(absolute_gap.max()),
            'time_to_peak_effect': int(years.loc[absolute_gap.idxmax()]) if absolute_gap.idxmax() in years.index else int(years.iloc[-1]),
            
            # === Relative Impact Analysis ===
            'relative_final_impact': float((herding_adopters.iloc[-1] - rational_adopters.iloc[-1]) / rational_adopters.iloc[-1] * 100) if rational_adopters.iloc[-1] > 0 else 0.0,
            'total_cumulative_effect': float(absolute_gap.sum()),
            'average_monthly_effect': float(absolute_gap.mean()),
            
            # === Early vs Late Analysis (by steps) ===
            'early_phase_effect': float(absolute_gap.iloc[:72].mean()),  # First 6 years (72 steps)
            'middle_phase_effect': float(absolute_gap.iloc[72:216].mean()),  # Years 7-18 (144 steps)
            'late_phase_effect': float(absolute_gap.iloc[216:].mean()),  # Years 19-30 (144 steps)
        }
        
        # Create metrics summary table
        self._create_metrics_table(metrics, output_dir)
        
        # Create metrics visualization
        self._create_metrics_visualization(metrics, absolute_gap, percentage_gap, multipliers, output_dir)
        
        print(f"✅ Herding impact metrics calculated and saved to: {output_dir}")
        
        return metrics
    
    def _create_metrics_table(self, metrics, output_dir):
        """Create and save a formatted metrics summary table."""
        
        # Create formatted table data
        table_data = [
            ["SIMULATION CONFIGURATION", "", ""],
            ["Total Households", f"{metrics['total_households']:,}", "households"],
            ["", "", ""],
            ["ABSOLUTE IMPACT METRICS", "", ""],
            ["Final Rational Adopters", f"{metrics['final_rational_adopters']:,}", "households"],
            ["Final Herding Adopters", f"{metrics['final_herding_adopters']:,}", "households"],
            ["Final Absolute Gap", f"{metrics['final_absolute_gap']:+,}", "households"],
            ["Maximum Absolute Gap", f"{metrics['max_absolute_gap']:,}", "households"],
            ["Average Monthly Gap", f"{metrics['mean_absolute_gap']:+.1f}", "households"],
            ["", "", ""],
            ["PERCENTAGE IMPACT METRICS", "", ""],
            ["Final Percentage Gap", f"{metrics['final_percentage_gap']:+.2f}", "percentage points"],
            ["Maximum Percentage Gap", f"{metrics['max_percentage_gap']:+.2f}", "percentage points"],
            ["Average Percentage Gap", f"{metrics['mean_percentage_gap']:+.2f}", "percentage points"],
            ["Gap Volatility (Std Dev)", f"{metrics['std_percentage_gap']:.2f}", "percentage points"],
            ["", "", ""],
            ["MULTIPLIER EFFECTS", "", ""],
            ["Final Herding Multiplier", f"{metrics['final_herding_multiplier']:.3f}", "ratio"],
            ["Maximum Herding Multiplier", f"{metrics['max_herding_multiplier']:.3f}", "ratio"],
            ["Average Herding Multiplier", f"{metrics['mean_herding_multiplier']:.3f}", "ratio"],
            ["", "", ""],
            ["TEMPORAL ANALYSIS", "", ""],
            ["Peak Effect Year", f"{metrics['peak_effect_year']}", "year"],
            ["Peak Effect Magnitude", f"{metrics['peak_effect_magnitude']:+.1f}", "households"],
            ["Relative Final Impact", f"{metrics['relative_final_impact']:+.1f}", "% vs rational"],
            ["", "", ""],
            ["PHASE ANALYSIS", "", ""],
            ["Early Phase Effect (Y1-6)", f"{metrics['early_phase_effect']:+.1f}", "households/month"],
            ["Middle Phase Effect (Y7-18)", f"{metrics['middle_phase_effect']:+.1f}", "households/month"],
            ["Late Phase Effect (Y19-30)", f"{metrics['late_phase_effect']:+.1f}", "households/month"],
        ]
        
        # Save as CSV
        df = pd.DataFrame(table_data, columns=['Metric', 'Value', 'Unit'])
        csv_path = os.path.join(output_dir, 'herding_impact_metrics.csv')
        df.to_csv(csv_path, index=False)
        
        # Save formatted text summary
        txt_path = os.path.join(output_dir, 'herding_impact_summary.txt')
        with open(txt_path, 'w') as f:
            f.write("HERDING BIAS IMPACT ANALYSIS - SUMMARY METRICS\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("SIMULATION CONFIGURATION:\n")
            f.write(f"• Total households: {metrics['total_households']:,}\n")
            f.write(f"• Data points: {self.adoption_data.shape[0]} monthly steps\n\n")
            
            f.write("KEY FINDINGS:\n")
            f.write(f"• Final herding effect: {metrics['final_absolute_gap']:+,} households ({metrics['final_percentage_gap']:+.1f}pp)\n")
            f.write(f"• Peak effect at year {metrics['peak_effect_year']}: {metrics['peak_effect_magnitude']:+.1f} households\n")
            f.write(f"• Average herding multiplier: {metrics['mean_herding_multiplier']:.3f}x\n")
            f.write(f"• Relative impact: {metrics['relative_final_impact']:+.1f}% vs rational baseline\n\n")
            
            f.write("PHASE COMPARISON:\n")
            if metrics['early_phase_effect'] > metrics['late_phase_effect']:
                f.write("• Herding effects strongest in EARLY phase\n")
                f.write("• Policy implication: Target early adopter programs\n")
            else:
                f.write("• Herding effects persist or strengthen in LATE phase\n") 
                f.write("• Policy implication: Sustained intervention needed\n")
            
            f.write(f"\nDETAILED METRICS:\n")
            for row in table_data:
                if row[0] and not any(row[0].startswith(prefix) for prefix in 
                                    ["PERCENTAGE", "ABSOLUTE", "MULTIPLIER", "TEMPORAL", "PHASE", "SIMULATION"]):
                    f.write(f"{row[0]:<30} {row[1]:>15} {row[2]}\n")
        
        print(f"✅ Metrics table saved: {csv_path}")
        print(f"✅ Summary report saved: {txt_path}")
    
    def _create_metrics_visualization(self, metrics, absolute_gap, percentage_gap, multipliers, output_dir):
        """Create visualization dashboard for herding impact metrics."""
        
        # Create step-to-year mapping for x-axis
        if 'Year' in self.adoption_data.columns:
            years = self.adoption_data['Year']
        else:
            years = (self.adoption_data.index // 12) + 1
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # === Plot 1: Absolute Gap Distribution ===
        ax1.hist(absolute_gap, bins=30, alpha=0.7, color=self.colors['cumulative'], 
                edgecolor='black', density=True)
        ax1.axvline(metrics['mean_absolute_gap'], color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {metrics["mean_absolute_gap"]:+.1f}')
        ax1.axvline(metrics['final_absolute_gap'], color='orange', linestyle='-', linewidth=2,
                   label=f'Final: {metrics["final_absolute_gap"]:+}')
        ax1.set_xlabel('Absolute Gap (households)')
        ax1.set_ylabel('Density')
        ax1.set_title('Distribution of Herding Gap\n(Absolute Numbers)', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # === Plot 2: Percentage Gap Over Time ===
        ax2.plot(years, percentage_gap, color=self.colors['herding'], 
                linewidth=3, marker='o', markersize=3)
        ax2.axhline(metrics['mean_percentage_gap'], color='red', linestyle='--', 
                   label=f'Mean: {metrics["mean_percentage_gap"]:+.2f}pp')
        ax2.axhline(0, color='black', linestyle='-', alpha=0.5)
        ax2.set_xlabel('Year')
        ax2.set_ylabel('Herding Gap (percentage points)')
        ax2.set_title('Herding Gap Evolution\n(Percentage Points)', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # === Plot 3: Multiplier Effects ===
        valid_multipliers = multipliers.dropna()
        if not valid_multipliers.empty:
            # Map multiplier index to years
            multiplier_years = years.loc[valid_multipliers.index] if not years.empty else valid_multipliers.index
            ax3.plot(multiplier_years, valid_multipliers, color=self.colors['gap_positive'], 
                    linewidth=3, marker='s', markersize=3)
            ax3.axhline(1.0, color='black', linestyle='-', alpha=0.5, label='No Effect (1.0x)')
            ax3.axhline(metrics['mean_herding_multiplier'], color='red', linestyle='--',
                       label=f'Mean: {metrics["mean_herding_multiplier"]:.3f}x')
        ax3.set_xlabel('Year')
        ax3.set_ylabel('Herding Multiplier (ratio)')
        ax3.set_title('Herding Multiplier Over Time\n(Herding/Rational)', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # === Plot 4: Phase Comparison Bar Chart ===
        phases = ['Early\n(Y1-6)', 'Middle\n(Y7-18)', 'Late\n(Y19-30)']
        phase_values = [metrics['early_phase_effect'], metrics['middle_phase_effect'], 
                       metrics['late_phase_effect']]
        
        colors = [self.colors['gap_positive'] if x >= 0 else self.colors['gap_negative'] 
                 for x in phase_values]
        bars = ax4.bar(phases, phase_values, color=colors, alpha=0.7, edgecolor='black')
        
        # Add value labels on bars
        for bar, value in zip(bars, phase_values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1 if height >= 0 else height - 0.5,
                    f'{value:+.1f}', ha='center', va='bottom' if height >= 0 else 'top', 
                    fontweight='bold')
        
        ax4.axhline(0, color='black', linestyle='-', alpha=0.5)
        ax4.set_ylabel('Average Effect (households/month)')
        ax4.set_title('Herding Effects by Phase\n(Average Monthly Impact)', fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Overall figure styling
        fig.suptitle(f'Herding Bias Impact Metrics Dashboard\n(Total: {metrics["total_households"]} households)', 
                    fontsize=16, fontweight='bold', y=0.95)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9, hspace=0.3, wspace=0.3)
        
        # Save visualization
        viz_path = os.path.join(output_dir, 'herding_impact_metrics_dashboard.png')
        plt.savefig(viz_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"✅ Metrics visualization saved: {viz_path}")