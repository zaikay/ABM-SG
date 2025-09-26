# data/spatial_visualizer.py V2.0 - PHASE 1 ENHANCED
"""
Enhanced spatial network visualization for behavioral prosumer adoption.
PHASE 1 ENHANCEMENTS:
- Simplified blue/green binary color scheme (instead of 5-color income overlay)
- 6x2 and 6x4 grid layouts (scenarios × timepoints) 
- Income homophily matrices for herding validation
- Herding component validation plots
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from collections import defaultdict

from ..utils.parameters import (
    SPATIAL_VISUALIZATION_CONFIG, get_all_scenarios, get_scenario_colors,
    BEHAVIORAL_BIASES
)
from ..utils.sampling_utils import VisualizationSampler

class SpatialVisualizer:
    """
    Enhanced spatial visualizations with simplified color scheme and improved layouts.
    
    Key changes from original:
    1. Binary blue/green color scheme (prosumer/non-prosumer)
    2. 6x2 and 6x4 grid layouts (scenarios as rows, timepoints as columns)
    3. Income homophily matrices for bias validation
    4. Herding component analysis plots
    """
    
    def __init__(self, model=None, config=None, data_path=None):
        """
        Initialize the enhanced spatial visualizer.
        
        Args:
            model: Mesa model instance (for live visualization)
            config: Simulation configuration
            data_path: Path to saved data (for post-hoc visualization)
        """
        self.model = model
        self.config = config or {}
        self.data_path = data_path
        
        # Visualization configuration
        self.viz_config = SPATIAL_VISUALIZATION_CONFIG
        all_scenarios = get_all_scenarios()
        self.scenarios = [s for s in all_scenarios if s != 'deterministic_rational']
        self.scenario_colors = get_scenario_colors()
        
        # Initialize sampler
        self.sampler = VisualizationSampler(
            sample_size=self.viz_config['sample_size']
        )
        
        # Set up matplotlib styling
        self._setup_plot_style()
        
        # Enhanced color scheme (simplified binary)
        self.colors = {
            'prosumer': '#00C851',      # Green for prosumers
            'nonprosumer': '#FF4444',   # Blue for non-prosumers
            'background': '#f8f9fa',    # Light background
            'grid': '#dee2e6'           # Light grid lines
        }
        
        print(f"EnhancedSpatialVisualizer initialized for {len(self.scenarios)} scenarios")
        print(f"Using simplified binary color scheme and enhanced layouts")
    
    def _setup_plot_style(self):
        """Set up consistent plot styling for publication quality."""
        plt.rcParams.update({
            'font.size': 10,
            'font.family': 'sans-serif',
            'axes.labelsize': 10,
            'axes.titlesize': 12,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'legend.fontsize': 9,
            'figure.titlesize': 14,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'figure.facecolor': 'white',
            'axes.facecolor': 'white'
        })
        plt.rcParams['font.family'] = 'DejaVu Sans'  # Or 'Times New Roman'
    
    def create_enhanced_spatial_grid_6x2(self, output_dir="results/spatial_analysis"):
        """
        Create 6x2 spatial grid: 6 scenarios × 2 timepoints.
        ENHANCED LAYOUT: Scenarios as rows, timepoints as columns.
        
        Layout:
        - Rows: rational, loss_aversion, present_bias, status_quo, herding, all_biases
        - Columns: Year 5, Year 20
        
        Args:
            output_dir: Directory to save visualizations
        """
        os.makedirs(output_dir, exist_ok=True)
        
        timepoints = [5, 20]  # Years to visualize
        
        # Create 6x2 grid
        fig, axes = plt.subplots(7, 2, figsize=(12, 18))
        
        for row, scenario in enumerate(self.scenarios):
            for col, year in enumerate(timepoints):
                ax = axes[row, col]
                step = year * 12  # Convert years to simulation steps
                
                try:
                    # Get household data for this scenario and timepoint
                    households = self._get_households_for_scenario_timepoint(scenario, step)
                    
                    if households:
                        # Sample households for visualization
                        sampled_households = self.sampler.stratified_spatial_sample(households)
                        
                        # Create the spatial plot with binary colors
                        self._create_binary_spatial_plot(ax, sampled_households, scenario, year)
                    else:
                        # No data available
                        ax.text(0.5, 0.5, f'No data\navailable', 
                               ha='center', va='center', transform=ax.transAxes,
                               fontsize=10, alpha=0.7)
                        ax.set_xlim(0, 1)
                        ax.set_ylim(0, 1)
                
                except Exception as e:
                    print(f"Error creating plot for {scenario} Year {year}: {e}")
                    ax.text(0.5, 0.5, f'Error:\n{str(e)[:30]}...', 
                           ha='center', va='center', transform=ax.transAxes,
                           fontsize=8, alpha=0.7)
                
                # Set column titles (top row only)
                if row == 0:
                    ax.set_title(f'Year {year}', fontsize=14, fontweight='bold', pad=15)
                
                # Set row labels (first column only)
                if col == 0:
                    scenario_display = scenario.replace('_', ' ').title()
                    if scenario == 'all_biases':
                        scenario_display = 'All Biases\nCombined'
                    ax.set_ylabel(scenario_display, fontsize=12, fontweight='bold', 
                                 rotation=90, labelpad=15)
                
                # Remove x/y tick labels for cleaner look
                ax.set_xticks([])
                ax.set_yticks([])
        
        # Add overall title
        fig.suptitle('Spatial Adoption Patterns: Enhanced Binary View\n(Scenarios × Timepoints)', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        # Create custom legend for binary scheme
        self._add_binary_legend(fig)
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.9, bottom=0.12, left=0.12, right=0.95)
        
        # Save the figure
        output_path = os.path.join(output_dir, 'enhanced_spatial_grid_6x2.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"Enhanced 6x2 spatial grid saved to: {output_path}")
    
    def create_enhanced_spatial_grid_6x4(self, output_dir="results/spatial_analysis"):
        """
        Create 6x4 spatial grid: 6 scenarios × 4 timepoints.
        ENHANCED LAYOUT: Shows temporal evolution more clearly.
        
        Layout:
        - Rows: rational, loss_aversion, present_bias, status_quo, herding, all_biases
        - Columns: Year 1, Year 5, Year 10, Year 20
        
        Args:
            output_dir: Directory to save visualizations
        """
        os.makedirs(output_dir, exist_ok=True)
        
        timepoints = [1, 5, 10, 20]  # Years to visualize
        
        # Create 6x4 grid
        fig, axes = plt.subplots(7, 4, figsize=(16, 18))
        
        for row, scenario in enumerate(self.scenarios):
            for col, year in enumerate(timepoints):
                ax = axes[row, col]
                step = year * 12  # Convert years to simulation steps
                
                try:
                    # Get household data for this scenario and timepoint
                    households = self._get_households_for_scenario_timepoint(scenario, step)
                    
                    if households:
                        # Sample households for visualization
                        sampled_households = self.sampler.stratified_spatial_sample(households)
                        
                        # Create the spatial plot with binary colors
                        self._create_binary_spatial_plot(ax, sampled_households, scenario, year)
                    else:
                        # No data available
                        ax.text(0.5, 0.5, f'No data\navailable', 
                               ha='center', va='center', transform=ax.transAxes,
                               fontsize=10, alpha=0.7)
                        ax.set_xlim(0, 1)
                        ax.set_ylim(0, 1)
                
                except Exception as e:
                    print(f"Error creating plot for {scenario} Year {year}: {e}")
                    ax.text(0.5, 0.5, f'Error:\n{str(e)[:30]}...', 
                           ha='center', va='center', transform=ax.transAxes,
                           fontsize=8, alpha=0.7)
                
                # Set column titles (top row only)
                if row == 0:
                    ax.set_title(f'Year {year}', fontsize=12, fontweight='bold', pad=10)
                
                # Set row labels (first column only)
                if col == 0:
                    scenario_display = scenario.replace('_', ' ').title()
                    if scenario == 'all_biases':
                        scenario_display = 'All Biases'
                    ax.set_ylabel(scenario_display, fontsize=11, fontweight='bold', 
                                 rotation=90, labelpad=10)
                
                # Remove x/y tick labels for cleaner look
                ax.set_xticks([])
                ax.set_yticks([])
        
        # Add overall title
        fig.suptitle('Temporal Evolution of Spatial Adoption Patterns\n(6 Scenarios × 4 Timepoints)', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        # Create custom legend for binary scheme
        self._add_binary_legend(fig)
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.95)
        
        # Save the figure
        output_path = os.path.join(output_dir, 'enhanced_spatial_grid_6x4.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"Enhanced 6x4 spatial grid saved to: {output_path}")
    
    def create_income_homophily_matrices(self, output_dir="results/spatial_analysis"):
        """
        Create 5x5 income homophily matrices for each scenario.
        CRITICAL for herding bias validation - shows class-based clustering effects.
        
        Creates matrices showing adoption rates between income classes,
        essential for validating the income homophily component of herding bias.
        
        Args:
            output_dir: Directory to save visualizations
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Create subplot grid for scenarios
        n_scenarios = len(self.scenarios)
        cols = 3
        rows = (n_scenarios + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        # Final timepoint for analysis
        final_year = 20
        final_step = final_year * 12
        
        for i, scenario in enumerate(self.scenarios):
            row = i // cols
            col = i % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            
            try:
                # Calculate homophily matrix for this scenario
                homophily_matrix = self._calculate_income_homophily_matrix(scenario, final_step)
                
                # Create heatmap
                im = ax.imshow(homophily_matrix, cmap='RdYlBu_r', vmin=0, vmax=1, 
                              interpolation='nearest')
                
                # Add text annotations
                for i_class in range(5):
                    for j_class in range(5):
                        value = homophily_matrix[i_class, j_class]
                        text_color = 'white' if value > 0.5 else 'black'
                        ax.text(j_class, i_class, f'{value:.2f}',
                               ha='center', va='center', color=text_color,
                               fontweight='bold', fontsize=10)
                
                # Formatting
                ax.set_xticks(range(5))
                ax.set_yticks(range(5))
                ax.set_xticklabels([f'Class {i+1}' for i in range(5)], rotation=45)
                ax.set_yticklabels([f'Class {i+1}' for i in range(5)])
                ax.set_xlabel('Target Income Class')
                ax.set_ylabel('Source Income Class')
                
                scenario_display = scenario.replace('_', ' ').title()
                if scenario == 'all_biases':
                    scenario_display = 'All Biases'
                ax.set_title(f'{scenario_display}\n(Year {final_year})', fontweight='bold')
                
                # Add colorbar to first plot
                if i == 0:
                    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    cbar.set_label('Adoption Rate', rotation=270, labelpad=15)
            
            except Exception as e:
                print(f"Error creating homophily matrix for {scenario}: {e}")
                ax.text(0.5, 0.5, f'Error:\n{str(e)[:50]}...', 
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=10, alpha=0.7)
                ax.set_xticks([])
                ax.set_yticks([])
        
        # Handle empty subplots
        for i in range(len(self.scenarios), rows * cols):
            row = i // cols
            col = i % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            ax.axis('off')
        
        # Overall title
        fig.suptitle('Income Class Homophily Matrices\n(Herding Bias Validation)', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        # Save the figure
        output_path = os.path.join(output_dir, 'income_homophily_matrices.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"Income homophily matrices saved to: {output_path}")
    
    def create_herding_component_validation(self, output_dir="results/spatial_analysis"):
        """
        Create dual-panel herding component validation plots.
        CRITICAL for herding bias validation - shows both spatial and class components.
        
        Args:
            output_dir: Directory to save visualizations
        """
        os.makedirs(output_dir, exist_ok=True)
        
        if 'herding' not in BEHAVIORAL_BIASES:
            print("Warning: Herding bias not enabled, skipping validation")
            return
        
        # Create 2x2 subplot grid
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Spatial influence vs distance
        ax = axes[0, 0]
        self._plot_spatial_influence_validation(ax)
        
        # Plot 2: Class influence over time
        ax = axes[0, 1]
        self._plot_class_influence_validation(ax)
        
        # Plot 3: Combined influence example
        ax = axes[1, 0]
        self._plot_combined_influence_example(ax)
        
        # Plot 4: Herding effect distribution
        ax = axes[1, 1]
        self._plot_herding_effect_distribution(ax)
        
        # Overall title
        fig.suptitle('Herding Bias Component Validation\n(Spatial + Income Class Effects)', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        # Save the figure
        output_path = os.path.join(output_dir, 'herding_component_validation.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"Herding component validation saved to: {output_path}")
    
    def _create_binary_spatial_plot(self, ax, households, scenario, year):
        """
        Create spatial plot with simplified binary color scheme.
        ENHANCED: Uses only blue/green colors for cleaner visualization.
        
        Args:
            ax: Matplotlib axes object
            households: List of household agents (sampled)
            scenario: Scenario name
            year: Year number
        """
        if not households:
            return
        
        # Extract positions and adoption status
        prosumer_positions = []
        nonprosumer_positions = []
        
        for h in households:
            if hasattr(h, 'pos') and h.pos is not None:
                # Check scenario-specific adoption status
                if hasattr(h, 'scenario_adoption'):
                    is_prosumer = h.scenario_adoption.get(scenario, h.is_prosumer)
                else:
                    is_prosumer = getattr(h, 'is_prosumer', False)
                
                if is_prosumer:
                    prosumer_positions.append(h.pos)
                else:
                    nonprosumer_positions.append(h.pos)
        
        # Convert to arrays
        if nonprosumer_positions:
            nonprosumer_positions = np.array(nonprosumer_positions)
            ax.scatter(nonprosumer_positions[:, 0], nonprosumer_positions[:, 1],
                      c=self.colors['nonprosumer'], s=5, alpha=0.3, 
                      marker='o', edgecolors='white', linewidths=0.5,
                      label='Non-Prosumer')
        
        if prosumer_positions:
            prosumer_positions = np.array(prosumer_positions)
            ax.scatter(prosumer_positions[:, 0], prosumer_positions[:, 1],
                      c=self.colors['prosumer'], s=10, alpha=0.8, 
                      marker='o', edgecolors='black', linewidths=0.8,
                      label='Prosumer', zorder=5)
        
        # Set equal aspect ratio and clean axes
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3, color=self.colors['grid'])
        ax.set_facecolor(self.colors['background'])
        
        # Add adoption rate annotation
        total_households = len(prosumer_positions) + len(nonprosumer_positions)
        adoption_rate = len(prosumer_positions) / total_households * 100 if total_households > 0 else 0
        
        ax.text(0.02, 0.98, f'{adoption_rate:.1f}%', 
               transform=ax.transAxes, fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
               verticalalignment='top')
    
    def _add_binary_legend(self, fig):
        """
        Add legend for binary color scheme.
        
        Args:
            fig: Matplotlib figure object
        """
        # Create legend elements
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor=self.colors['nonprosumer'],
                      markersize=8, label='Non-Prosumer', linestyle='None',
                      markeredgewidth=0.5, markeredgecolor='white'),
            plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor=self.colors['prosumer'],
                      markersize=10, label='Prosumer', linestyle='None',
                      markeredgewidth=0.8, markeredgecolor='black')
        ]
        
        # Add legend to figure
        fig.legend(handles=legend_elements, 
                  loc='lower center', 
                  ncol=2,
                  bbox_to_anchor=(0.5, 0.02),
                  fontsize=12,
                  frameon=True,
                  fancybox=True,
                  shadow=True)
    
    def _calculate_income_homophily_matrix(self, scenario, step):
        """
        Calculate 5x5 income homophily matrix for a scenario.
        
        Args:
            scenario: Scenario name
            step: Simulation step
            
        Returns:
            np.array: 5x5 matrix of adoption rates
        """
        # Get households for this scenario and step
        households = self._get_households_for_scenario_timepoint(scenario, step)
        
        if not households:
            return np.zeros((5, 5))
        
        # Initialize matrix
        homophily_matrix = np.zeros((5, 5))
        
        # Calculate adoption rates for each income class
        for source_class in range(1, 6):
            for target_class in range(1, 6):
                # Find households in source class
                source_households = [h for h in households 
                                   if getattr(h, 'income_class', 1) == source_class]
                
                if not source_households:
                    continue
                
                # Count adoptions in target class among source class neighbors
                # For simplicity, we'll use the adoption rate of target class
                target_households = [h for h in households 
                                   if getattr(h, 'income_class', 1) == target_class]
                
                if target_households:
                    # Check scenario-specific adoption
                    target_adopters = 0
                    for h in target_households:
                        if hasattr(h, 'scenario_adoption'):
                            is_adopter = h.scenario_adoption.get(scenario, h.is_prosumer)
                        else:
                            is_adopter = getattr(h, 'is_prosumer', False)
                        
                        if is_adopter:
                            target_adopters += 1
                    
                    adoption_rate = target_adopters / len(target_households)
                    homophily_matrix[source_class-1, target_class-1] = adoption_rate
        
        return homophily_matrix
    
    def _plot_spatial_influence_validation(self, ax):
        """Plot spatial influence vs distance validation."""
        if 'herding' not in BEHAVIORAL_BIASES:
            return
        
        params = BEHAVIORAL_BIASES['herding']['parameters']
        d_0 = params['distance_normalization']
        
        # Distance decay function
        distances = np.linspace(0.1, 10, 1000)
        weights = 1 / (1 + (distances / d_0) ** 2)
        
        ax.plot(distances, weights, linewidth=3, color='#9467bd', label='Weight Function')
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='50% Weight')
        ax.axvline(x=d_0, color='green', linestyle='--', alpha=0.7, label=f'd₀ = {d_0}')
        
        ax.set_xlabel('Distance (grid units)')
        ax.set_ylabel('Influence Weight')
        ax.set_title('Spatial Influence vs Distance\nw = 1/(1 + (d/d₀)²)', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_class_influence_validation(self, ax):
        """Plot class influence validation over time."""
        # Example time series data
        years = np.arange(1, 21)
        
        # Simulate class adoption rates over time
        class_rates = {}
        for income_class in range(1, 6):
            # Higher income classes adopt earlier and more
            base_rate = 0.05 + (income_class - 1) * 0.02
            growth_rate = 0.8 + (income_class - 1) * 0.1
            rates = base_rate * (1 - np.exp(-growth_rate * years / 10))
            class_rates[income_class] = rates
        
        # Plot adoption curves
        colors = plt.cm.viridis(np.linspace(0, 1, 5))
        for i, (income_class, rates) in enumerate(class_rates.items()):
            ax.plot(years, rates * 100, linewidth=2, color=colors[i], 
                   marker='o', markersize=4, label=f'Class {income_class}')
        
        ax.set_xlabel('Year')
        ax.set_ylabel('Class Adoption Rate (%)')
        ax.set_title('Income Class Influence Over Time', fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
    
    def _plot_combined_influence_example(self, ax):
        """Plot combined spatial + class influence example."""
        if 'herding' not in BEHAVIORAL_BIASES:
            return
        
        params = BEHAVIORAL_BIASES['herding']['parameters']
        
        # Example influence values
        spatial_adoption_rates = np.linspace(0, 1, 11)
        class_adoption_rate = 0.3  # Fixed example
        
        # Beta distribution means
        spatial_beta = params['spatial_beta_shape_a'] / (
            params['spatial_beta_shape_a'] + params['spatial_beta_shape_b'])
        class_beta = params['class_beta_shape_a'] / (
            params['class_beta_shape_a'] + params['class_beta_shape_b'])
        
        # Calculate multipliers
        multipliers = 1 + spatial_beta * spatial_adoption_rates + class_beta * class_adoption_rate
        
        ax.plot(spatial_adoption_rates * 100, multipliers, linewidth=3, 
               color='#9467bd', marker='o', markersize=6)
        ax.axhline(y=1, color='black', linestyle='--', alpha=0.7, label='No Effect')
        
        ax.set_xlabel('Spatial Neighbor Adoption Rate (%)')
        ax.set_ylabel('Probability Multiplier')
        ax.set_title(f'Combined Influence Effect\n(Class Rate: {class_adoption_rate:.1%})', 
                    fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_herding_effect_distribution(self, ax):
        """Plot distribution of herding effects."""
        if 'herding' not in BEHAVIORAL_BIASES:
            return
        
        params = BEHAVIORAL_BIASES['herding']['parameters']
        
        # Generate random herding effects
        np.random.seed(42)
        n_samples = 10000
        
        # Sample beta values
        spatial_betas = np.random.beta(params['spatial_beta_shape_a'], 
                                     params['spatial_beta_shape_b'], n_samples)
        class_betas = np.random.beta(params['class_beta_shape_a'], 
                                   params['class_beta_shape_b'], n_samples)
        
        # Sample influence rates
        spatial_rates = np.random.uniform(0, 1, n_samples)
        class_rates = np.random.uniform(0, 1, n_samples)
        
        # Calculate multipliers
        multipliers = 1 + spatial_betas * spatial_rates + class_betas * class_rates
        
        ax.hist(multipliers, bins=50, density=True, alpha=0.7, color='#9467bd', 
               edgecolor='black', linewidth=0.2)
        ax.axvline(x=1, color='black', linestyle='--', alpha=0.7, label='No Effect')
        ax.axvline(x=np.mean(multipliers), color='red', linestyle='--', alpha=0.7, 
                  label=f'Mean: {np.mean(multipliers):.2f}')
        
        ax.set_xlabel('Probability Multiplier')
        ax.set_ylabel('Density')
        ax.set_title('Distribution of Herding Effects\n(Population Simulation)', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _get_households_for_scenario_timepoint(self, scenario, step):
        """
        Get household data for specific scenario and timepoint.
        Enhanced version with better multi-scenario support.
        """
        if self.model is not None:
            # Live model access
            households = [agent for agent in self.model.schedule.agents 
                        if hasattr(agent, 'daily_consumption')]
            return households
            
        elif self.data_path is not None:
            # Load from saved data with enhanced error handling
            try:
                base_dir = os.path.dirname(self.data_path)
                scenario_file = os.path.join(base_dir, f"{scenario}_data.csv")
                
                if os.path.exists(scenario_file):
                    scenario_data = pd.read_csv(scenario_file)
                    step_data = scenario_data[scenario_data['Step'] == step]
                else:
                    # Fallback to combined data
                    if os.path.exists(self.data_path):
                        data = pd.read_csv(self.data_path)
                        step_data = data[(data['Scenario'] == scenario) & (data['Step'] == step)]
                    else:
                        print(f"No data found for {scenario} at step {step}")
                        return []
                
                if step_data.empty:
                    print(f"No data found for {scenario} at step {step}")
                    return []
                
                # Convert to mock household objects
                households = []
                for _, row in step_data.iterrows():
                    household = MockHouseholdForVisualization(row)
                    household.scenario_adoption = {scenario: row.get('IsProsumer', False)}
                    households.append(household)
                
                return households
                
            except Exception as e:
                print(f"Error loading data for {scenario} step {step}: {e}")
                return []
        
        else:
            print("No data source available")
            return []

    def create_spatial_network_grid_10cm(self, output_dir="results/spatial_analysis", scenario="rational", step=120):
        """
        Create a 10cm x 10cm spatial network visualization showing household positions 
        and neighbor connections.
        
        Features:
        - Green dots: Prosumer households 
        - Blue dots: Non-prosumer households
        - Black lines: Edges between closest 10 neighbors
        - Grid scaled to approximately 10cm x 10cm
        
        Args:
            output_dir: Directory to save visualizations
            scenario: Scenario to visualize (default: "rational")
            step: Simulation step (default: 120 = Year 10)
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Get household data for the specified scenario and timepoint
        households = self._get_households_for_scenario_timepoint(scenario, step)
        
        if not households:
            print(f"No household data available for scenario '{scenario}' at step {step}")
            return
        
        # Create the figure
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        
        # Extract household positions
        positions = []
        household_objects = []
        
        for household in households:
            if hasattr(household, 'pos') and household.pos is not None:
                positions.append(household.pos)
                household_objects.append(household)
        
        if not positions:
            print("No household positions found. Network may not be built yet.")
            return
        
        positions = np.array(positions)
        
        # Determine grid scaling
        x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
        y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
        
        grid_width = x_max - x_min
        grid_height = y_max - y_min
        
        # Check if current grid is approximately square and around 10 units
        # If so, we can interpret each unit as 1cm for a 10cm x 10cm grid
        target_size_cm = 10
        current_size = max(grid_width, grid_height)
        
        if abs(current_size - target_size_cm) < 2:  # Within 2 units of target
            print(f"Current grid size ({current_size:.1f} units) approximates {target_size_cm}cm. Using original coordinates.")
            scale_factor = 1.0
            grid_label = f"{target_size_cm}cm × {target_size_cm}cm"
        else:
            # Scale to 10cm x 10cm
            scale_factor = target_size_cm / current_size
            grid_label = f"{target_size_cm}cm × {target_size_cm}cm (scaled)"
            print(f"Scaling grid from {current_size:.1f} units to {target_size_cm}cm")
        
        # Apply scaling
        scaled_positions = positions * scale_factor
        scaled_x_min, scaled_x_max = scaled_positions[:, 0].min(), scaled_positions[:, 0].max()
        scaled_y_min, scaled_y_max = scaled_positions[:, 1].min(), scaled_positions[:, 1].max()
        
        # Draw neighbor connections first (so they appear behind points)
        connection_count = 0
        for i, household in enumerate(household_objects):
            if hasattr(household, 'spatial_neighbors'):
                household_pos = scaled_positions[i]
                
                for neighbor_household, distance in household.spatial_neighbors:
                    # Find the neighbor's position
                    neighbor_idx = None
                    for j, h in enumerate(household_objects):
                        if h.unique_id == neighbor_household.unique_id:
                            neighbor_idx = j
                            break
                    
                    if neighbor_idx is not None:
                        neighbor_pos = scaled_positions[neighbor_idx]
                        
                        # Draw line between household and neighbor
                        ax.plot([household_pos[0], neighbor_pos[0]], 
                            [household_pos[1], neighbor_pos[1]], 
                            'k-', linewidth=0.2, alpha=0.3, zorder=1)
                        connection_count += 1
        
        # Draw household points with color coding (green=prosumer, blue=non-prosumer)
        colors = []
        prosumer_count = 0
        
        for household in household_objects:
            # Check if household is prosumer for this scenario at this step
            is_prosumer = False
            
            # Method 1: Check adoption month (most reliable for timeline)
            if hasattr(household, 'get_adoption_month'):
                adoption_month = household.get_adoption_month(scenario)
                is_prosumer = adoption_month is not None and adoption_month <= step
            # Method 2: Check scenario_adoption dictionary
            elif hasattr(household, 'scenario_adoption') and scenario in household.scenario_adoption:
                is_prosumer = household.scenario_adoption[scenario]
            # Method 3: Fallback to general is_prosumer attribute
            else:
                is_prosumer = getattr(household, 'is_prosumer', False)
            
            if is_prosumer:
                prosumer_count += 1
                
            colors.append('#00C851' if is_prosumer else '#FF4444')  # Green for prosumer, blue for non-prosumer
        
        # Debug print
        print(f"DEBUG: Step {step}, Scenario {scenario}: {prosumer_count}/{len(household_objects)} prosumers")
        
        ax.scatter(scaled_positions[:, 0], scaled_positions[:, 1], 
                c=colors, s=7, alpha=1.0, zorder=2, edgecolors='black', linewidth=0.2)
        
        # Set up the plot
        ax.set_xlim(scaled_x_min - 0.5, scaled_x_max + 0.5)
        ax.set_ylim(scaled_y_min - 0.5, scaled_y_max + 0.5)
        ax.set_aspect('equal')
        
        # Add grid
        ax.grid(True, alpha=0.3, linewidth=0.2)
        
        # Labels and title
        ax.set_xlabel('X Position (cm)', fontsize=12)
        ax.set_ylabel('Y Position (cm)', fontsize=12)
        ax.set_title(f'Spatial Network & Adoption Patterns - {grid_label}\n'
                    f'Scenario: {scenario.replace("_", " ").title()}, Step: {step}\n'
                    f'{len(household_objects)} Households, {connection_count} Connections', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#00C851', 
                markersize=8, label='Prosumer Households', markeredgecolor='black'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF4444', 
                markersize=8, label='Non-Prosumer Households', markeredgecolor='black'),
            Line2D([0], [0], color='black', linewidth=1, alpha=0.3, 
                label='Neighbor Connections')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        # Add network statistics as text
        avg_connections = connection_count / len(household_objects) if household_objects else 0
        
        stats_text = f'''Network Statistics:
        Households: {len(household_objects)}
        Prosumers: {prosumer_count} ({prosumer_count/len(household_objects)*100:.1f}%)
        Avg connections/household: {avg_connections:.1f}
        Total connections: {connection_count}'''
        stats_text=''
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Save the figure
        filename = f'spatial_network_10cm_{scenario}_step_{step}.png'
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"✅ Spatial network visualization saved to: {output_path}")
        print(f"   Grid: {grid_label}")
        print(f"   Households: {len(household_objects)}")
        print(f"   Prosumers: {prosumer_count}")
        print(f"   Connections: {connection_count}")
        print(f"   Avg connections per household: {avg_connections:.1f}")


    def create_multi_scenario_network_grid(self, output_dir="results/spatial_analysis", step=120):
        """
        Create a 2x3 grid showing spatial network visualizations for all scenarios.
        
        Args:
            output_dir: Directory to save visualizations
            step: Simulation step (default: 120 = Year 10)
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Define scenarios to visualize (excluding 'all_biases' for clarity)
        scenarios_to_plot = ['rational', 'herding','all_biases']
        
        # Create 2x3 subplot grid
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, scenario in enumerate(scenarios_to_plot):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            
            # Get household data for this scenario
            households = self._get_households_for_scenario_timepoint(scenario, step)
            
            if not households:
                ax.text(0.5, 0.5, f'No data\navailable\nfor {scenario}', 
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=12, alpha=0.7)
                ax.set_title(scenario.replace('_', ' ').title(), fontsize=12, fontweight='bold')
                continue
            
            # Extract positions
            positions = []
            household_objects = []
            
            for household in households:
                if hasattr(household, 'pos') and household.pos is not None:
                    positions.append(household.pos)
                    household_objects.append(household)
            
            if not positions:
                ax.text(0.5, 0.5, 'No position data', ha='center', va='center', 
                    transform=ax.transAxes, fontsize=10, alpha=0.7)
                ax.set_title(scenario.replace('_', ' ').title(), fontsize=12, fontweight='bold')
                continue
            
            positions = np.array(positions)
            
            # Scale to 10cm grid
            x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
            y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
            current_size = max(x_max - x_min, y_max - y_min)
            scale_factor = 10.0 / current_size if current_size > 0 else 1.0
            scaled_positions = positions * scale_factor
            
            # Draw connections
            connection_count = 0
            for i, household in enumerate(household_objects):
                if hasattr(household, 'spatial_neighbors'):
                    household_pos = scaled_positions[i]
                    
                    for neighbor_household, distance in household.spatial_neighbors:
                        # Find neighbor position
                        neighbor_idx = None
                        for j, h in enumerate(household_objects):
                            if h.unique_id == neighbor_household.unique_id:
                                neighbor_idx = j
                                break
                        
                        if neighbor_idx is not None:
                            neighbor_pos = scaled_positions[neighbor_idx]
                            ax.plot([household_pos[0], neighbor_pos[0]], 
                                [household_pos[1], neighbor_pos[1]], 
                                'k-', linewidth=0.15, alpha=0.25, zorder=1)
                            connection_count += 1
            
            # Draw household points with color coding
            colors = []
            prosumer_count = 0
            
            for household in household_objects:
                # Check if household is prosumer for this scenario at this step
                is_prosumer = False
                
                # Method 1: Check adoption month (most reliable for timeline)
                if hasattr(household, 'get_adoption_month'):
                    adoption_month = household.get_adoption_month(scenario)
                    is_prosumer = adoption_month is not None and adoption_month <= step
                # Method 2: Check scenario_adoption dictionary
                elif hasattr(household, 'scenario_adoption') and scenario in household.scenario_adoption:
                    is_prosumer = household.scenario_adoption[scenario]
                # Method 3: Fallback to general is_prosumer attribute
                else:
                    is_prosumer = getattr(household, 'is_prosumer', False)
                
                if is_prosumer:
                    prosumer_count += 1
                    
                colors.append('#00C851' if is_prosumer else '#FF4444')  # Green for prosumer, blue for non-prosumer
            
            # Debug print
            print(f"DEBUG: Step {step}, Scenario {scenario}: {prosumer_count}/{len(household_objects)} prosumers")
            
            ax.scatter(scaled_positions[:, 0], scaled_positions[:, 1], 
                    c=colors, s=5, alpha=0.8, zorder=2, edgecolors='black', linewidth=0.15)
            
            # Set up subplot
            scaled_x_min, scaled_x_max = scaled_positions[:, 0].min(), scaled_positions[:, 0].max()
            scaled_y_min, scaled_y_max = scaled_positions[:, 1].min(), scaled_positions[:, 1].max()
            
            ax.set_xlim(scaled_x_min - 0.3, scaled_x_max + 0.3)
            ax.set_ylim(scaled_y_min - 0.3, scaled_y_max + 0.3)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.2, linewidth=0.15)
            
            # Title with statistics
            prosumer_pct = prosumer_count / len(household_objects) * 100 if household_objects else 0
            title = f'{scenario.replace("_", " ").title()}\n{prosumer_count}/{len(household_objects)} prosumers ({prosumer_pct:.1f}%)'
            ax.set_title(title, fontsize=10, fontweight='bold')
            
            # Labels for edge subplots
            if idx >= 3:  # Bottom row
                ax.set_xlabel('X Position (cm)', fontsize=9)
            if idx % 3 == 0:  # Left column
                ax.set_ylabel('Y Position (cm)', fontsize=9)
        
        # Hide unused subplot
        if len(scenarios_to_plot) < len(axes):
            for idx in range(len(scenarios_to_plot), len(axes)):
                axes[idx].set_visible(False)
        
        # Add overall title and legend
        fig.suptitle(f'Spatial Network & Adoption Patterns (10cm × 10cm) - Step {step}\nGreen: Prosumers, Blue: Non-Prosumers, Black Lines: Neighbor Connections', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        # Add a single legend for all subplots
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#00C851', 
                markersize=6, label='Prosumer Households', markeredgecolor='black'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF4444', 
                markersize=6, label='Non-Prosumer Households', markeredgecolor='black'),
            Line2D([0], [0], color='black', linewidth=1, alpha=0.25, 
                label='Neighbor Connections')
        ]
        fig.legend(handles=legend_elements, loc='lower right', fontsize=11, 
                bbox_to_anchor=(0.98, 0.02))
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.88, bottom=0.08, right=0.94)
        
        # Save the figure
        filename = f'multi_scenario_network_grids_step_{step}.png'
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"✅ Multi-scenario network visualization saved to: {output_path}")


    def create_interactive_network_analysis(self, output_dir="results/spatial_analysis", scenario="rational", step=120):
        """
        Create an enhanced network analysis with detailed statistics and network metrics.
        
        Args:
            output_dir: Directory to save visualizations  
            scenario: Scenario to analyze
            step: Simulation step
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import os
        from collections import defaultdict
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Get household data
        households = self._get_households_for_scenario_timepoint(scenario, step)
        
        if not households:
            print(f"No household data available for scenario '{scenario}' at step {step}")
            return
        
        # Extract network data
        positions = []
        household_objects = []
        distances = []
        degree_distribution = defaultdict(int)
        
        for household in households:
            if hasattr(household, 'pos') and household.pos is not None:
                positions.append(household.pos)
                household_objects.append(household)
                
                # Collect distance and degree data
                if hasattr(household, 'spatial_neighbors'):
                    degree = len(household.spatial_neighbors)
                    degree_distribution[degree] += 1
                    
                    for _, distance in household.spatial_neighbors:
                        distances.append(distance)
        
        if not positions:
            print("No household positions found.")
            return
        
        positions = np.array(positions)
        
        # Create a 2x2 analysis figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Main spatial network plot (top-left)
        # Scale to 10cm
        current_size = max(positions[:, 0].max() - positions[:, 0].min(),
                        positions[:, 1].max() - positions[:, 1].min())
        scale_factor = 10.0 / current_size if current_size > 0 else 1.0
        scaled_positions = positions * scale_factor
        
        # Draw connections
        connection_count = 0
        prosumer_count = 0
        colors = []
        
        for i, household in enumerate(household_objects):
            # Check if household is prosumer for this scenario at this step
            is_prosumer = False
            
            # Method 1: Check adoption month (most reliable for timeline)
            if hasattr(household, 'get_adoption_month'):
                adoption_month = household.get_adoption_month(scenario)
                is_prosumer = adoption_month is not None and adoption_month <= step
            # Method 2: Check scenario_adoption dictionary
            elif hasattr(household, 'scenario_adoption') and scenario in household.scenario_adoption:
                is_prosumer = household.scenario_adoption[scenario]
            # Method 3: Fallback to general is_prosumer attribute
            else:
                is_prosumer = getattr(household, 'is_prosumer', False)
            
            if is_prosumer:
                prosumer_count += 1
            
            colors.append('#00C851' if is_prosumer else '#FF4444')  # Green for prosumer, blue for non-prosumer


            # Draw connections
            if hasattr(household, 'spatial_neighbors'):
                household_pos = scaled_positions[i]
                for neighbor_household, distance in household.spatial_neighbors:
                    neighbor_idx = None
                    for j, h in enumerate(household_objects):
                        if h.unique_id == neighbor_household.unique_id:
                            neighbor_idx = j
                            break
                    
                    if neighbor_idx is not None:
                        neighbor_pos = scaled_positions[neighbor_idx]
                        ax1.plot([household_pos[0], neighbor_pos[0]], 
                            [household_pos[1], neighbor_pos[1]], 
                            'k-', linewidth=0.2, alpha=0.3, zorder=1)
                        connection_count += 1
        
        # Debug print
        print(f"DEBUG: Step {step}, Scenario {scenario}: {prosumer_count}/{len(household_objects)} prosumers")
        
        ax1.scatter(scaled_positions[:, 0], scaled_positions[:, 1], 
                c=colors, s=40, alpha=0.8, zorder=2, edgecolors='black', linewidth=0.2)
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)
        ax1.set_title('Spatial Network (10cm × 10cm)', fontweight='bold')
        ax1.set_xlabel('X Position (cm)')
        ax1.set_ylabel('Y Position (cm)')
        
        # 2. Distance distribution (top-right)
        if distances:
            ax2.hist(distances, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax2.set_title('Neighbor Distance Distribution', fontweight='bold')
            ax2.set_xlabel('Distance')
            ax2.set_ylabel('Frequency')
            ax2.axvline(np.mean(distances), color='red', linestyle='--', 
                    label=f'Mean: {np.mean(distances):.2f}')
            ax2.legend()
        
        # 3. Degree distribution (bottom-left)
        if degree_distribution:
            degrees = list(degree_distribution.keys())
            counts = list(degree_distribution.values())
            ax3.bar(degrees, counts, alpha=0.7, color='lightgreen', edgecolor='black')
            ax3.set_title('Degree Distribution', fontweight='bold')
            ax3.set_xlabel('Number of Neighbors (Degree)')
            ax3.set_ylabel('Number of Households')
        
        # 4. Network statistics (bottom-right)
        ax4.axis('off')
        
        # Calculate network statistics
        total_households = len(household_objects)
        total_connections = connection_count
        avg_degree = np.mean(list(degree_distribution.keys())) if degree_distribution else 0
        avg_distance = np.mean(distances) if distances else 0
        min_distance = np.min(distances) if distances else 0
        max_distance = np.max(distances) if distances else 0
        
        stats_text = f"""
        Network Statistics

        Scenario: {scenario.replace('_', ' ').title()}
        Simulation Step: {step}

        Households: {total_households:,}
        Prosumers: {prosumer_count:,} ({prosumer_count/total_households*100:.1f}%)
        Non-Prosumers: {total_households-prosumer_count:,} ({(total_households-prosumer_count)/total_households*100:.1f}%)

        Total Connections: {total_connections:,}
        Average Degree: {avg_degree:.2f}

        Distance Metrics:
        Average: {avg_distance:.3f} units
        Minimum: {min_distance:.3f} units  
        Maximum: {max_distance:.3f} units

        Grid Properties:
        Original Size: {current_size:.2f} units
        Scaled Size: 10.0 cm × 10.0 cm
        Scale Factor: {scale_factor:.3f}
        """
        stats_text=''
        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        # Save the figure
        filename = f'network_analysis_{scenario}_step_{step}.png'
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"✅ Network analysis saved to: {output_path}")
        print(f"   Network density: {total_connections / (total_households * (total_households - 1) / 2) * 100:.2f}%")
        print(f"   Average clustering: {avg_degree / (total_households - 1) * 100:.2f}%")

    def create_spatial_network_grid_10cm_income_class(self, output_dir="results/spatial_analysis", 
                                                    scenario="rational", step=120, 
                                                    highlight_prosumers_only=False):
        """
        Create a 10cm x 10cm spatial network visualization showing household positions 
        colored by income class.
        
        Features:
        - Households colored by income class (Class 1=Red to Class 5=Green)
        - Boolean option to highlight only prosumers with income colors
        - Black lines: Edges between closest 10 neighbors
        - Grid scaled to approximately 10cm x 10cm
        
        Args:
            output_dir: Directory to save visualizations
            scenario: Scenario to visualize (default: "rational")
            step: Simulation step (default: 120 = Year 10)
            highlight_prosumers_only: If True, show only prosumers with income colors, 
                                    non-prosumers in light gray
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Get household data for the specified scenario and timepoint
        households = self._get_households_for_scenario_timepoint(scenario, step)
        
        if not households:
            print(f"No household data available for scenario '{scenario}' at step {step}")
            return
        
        # Create the figure
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        
        # Extract household positions
        positions = []
        household_objects = []
        
        for household in households:
            if hasattr(household, 'pos') and household.pos is not None:
                positions.append(household.pos)
                household_objects.append(household)
        
        if not positions:
            print("No household positions found. Network may not be built yet.")
            return
        
        positions = np.array(positions)
        
        # Determine grid scaling
        x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
        y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
        
        grid_width = x_max - x_min
        grid_height = y_max - y_min
        
        # Check if current grid is approximately square and around 10 units
        target_size_cm = 10
        current_size = max(grid_width, grid_height)
        
        if abs(current_size - target_size_cm) < 2:  # Within 2 units of target
            print(f"Current grid size ({current_size:.1f} units) approximates {target_size_cm}cm. Using original coordinates.")
            scale_factor = 1.0
            grid_label = f"{target_size_cm}cm × {target_size_cm}cm"
        else:
            # Scale to 10cm x 10cm
            scale_factor = target_size_cm / current_size
            grid_label = f"{target_size_cm}cm × {target_size_cm}cm (scaled)"
            print(f"Scaling grid from {current_size:.1f} units to {target_size_cm}cm")
        
        # Apply scaling
        scaled_positions = positions * scale_factor
        scaled_x_min, scaled_x_max = scaled_positions[:, 0].min(), scaled_positions[:, 0].max()
        scaled_y_min, scaled_y_max = scaled_positions[:, 1].min(), scaled_positions[:, 1].max()
        
        # Draw neighbor connections first (so they appear behind points)
        connection_count = 0
        for i, household in enumerate(household_objects):
            if hasattr(household, 'spatial_neighbors'):
                household_pos = scaled_positions[i]
                
                for neighbor_household, distance in household.spatial_neighbors:
                    # Find the neighbor's position
                    neighbor_idx = None
                    for j, h in enumerate(household_objects):
                        if h.unique_id == neighbor_household.unique_id:
                            neighbor_idx = j
                            break
                    
                    if neighbor_idx is not None:
                        neighbor_pos = scaled_positions[neighbor_idx]
                        
                        # Draw line between household and neighbor
                        ax.plot([household_pos[0], neighbor_pos[0]], 
                            [household_pos[1], neighbor_pos[1]], 
                            'k-', linewidth=0.2, alpha=0.3, zorder=1)
                        connection_count += 1
        
        # Income class color palette (Class 1=Red to Class 5=Green)
        income_class_colors = {
            1: '#8B0000',  # Dark red (lowest income)
            2: '#FF8C00',  # Orange
            3: '#FFD700',  # Gold/Yellow  
            4: '#32CD32',  # Lime green
            5: '#006400'   # Dark green (highest income)
        }
        light_gray = '#D3D3D3'  # For non-prosumers when highlighting
        
        # Draw household points with income class coloring
        colors = []
        prosumer_count = 0
        income_class_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        prosumer_by_class = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        
        for household in household_objects:
            # Get income class
            income_class = getattr(household, 'income_class', 1)  # Default to class 1 if missing
            income_class_counts[income_class] += 1
            
            # Check if household is prosumer for this scenario at this step
            is_prosumer = False
            
            # Method 1: Check adoption month (most reliable for timeline)
            if hasattr(household, 'get_adoption_month'):
                adoption_month = household.get_adoption_month(scenario)
                is_prosumer = adoption_month is not None and adoption_month <= step
            # Method 2: Check scenario_adoption dictionary
            elif hasattr(household, 'scenario_adoption') and scenario in household.scenario_adoption:
                is_prosumer = household.scenario_adoption[scenario]
            # Method 3: Fallback to general is_prosumer attribute
            else:
                is_prosumer = getattr(household, 'is_prosumer', False)
            
            if is_prosumer:
                prosumer_count += 1
                prosumer_by_class[income_class] += 1
            
            # Determine color based on highlight option
            if highlight_prosumers_only:
                if is_prosumer:
                    colors.append(income_class_colors[income_class])
                else:
                    colors.append(light_gray)
            else:
                colors.append(income_class_colors[income_class])
        
        ax.scatter(scaled_positions[:, 0], scaled_positions[:, 1], 
                c=colors, s=7, alpha=1.0, zorder=2, edgecolors='black', linewidth=0.15)
        
        # Set up the plot
        ax.set_xlim(scaled_x_min - 0.5, scaled_x_max + 0.5)
        ax.set_ylim(scaled_y_min - 0.5, scaled_y_max + 0.5)
        ax.set_aspect('equal')
        
        # Add grid
        ax.grid(True, alpha=0.3, linewidth=0.2)
        
        # Labels and title
        ax.set_xlabel('X Position (cm)', fontsize=12)
        ax.set_ylabel('Y Position (cm)', fontsize=12)
        
        highlight_text = " (Prosumers Highlighted)" if highlight_prosumers_only else ""
        ax.set_title(f'Spatial Network by Income Class - {grid_label}{highlight_text}\n'
                    f'Scenario: {scenario.replace("_", " ").title()}, Step: {step}\n'
                    f'{len(household_objects)} Households, {connection_count} Connections', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = []
        
        # Income class legend elements
        for income_class in range(1, 6):
            if income_class_counts[income_class] > 0:  # Only show classes that exist
                label = f'Class {income_class}'
                if income_class == 1:
                    label += ' (Lowest Income)'
                elif income_class == 5:
                    label += ' (Highest Income)'
                
                legend_elements.append(
                    Line2D([0], [0], marker='o', color='w', 
                        markerfacecolor=income_class_colors[income_class], 
                        markersize=8, label=label, markeredgecolor='black')
                )
        
        # Add non-prosumer legend if highlighting
        if highlight_prosumers_only:
            legend_elements.append(
                Line2D([0], [0], marker='o', color='w', markerfacecolor=light_gray, 
                    markersize=8, label='Non-Prosumers', markeredgecolor='black')
            )
        
        # Add connection legend
        legend_elements.append(
            Line2D([0], [0], color='black', linewidth=1, alpha=0.3, 
                label='Neighbor Connections')
        )
        
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        # Add statistics text box
        stats_lines = [
            f'Households by Income Class:',
            f'Class 1: {income_class_counts[1]} ({prosumer_by_class[1]} prosumers)',
            f'Class 2: {income_class_counts[2]} ({prosumer_by_class[2]} prosumers)',
            f'Class 3: {income_class_counts[3]} ({prosumer_by_class[3]} prosumers)',
            f'Class 4: {income_class_counts[4]} ({prosumer_by_class[4]} prosumers)',
            f'Class 5: {income_class_counts[5]} ({prosumer_by_class[5]} prosumers)',
            f'',
            f'Total prosumers: {prosumer_count}/{len(household_objects)} ({prosumer_count/len(household_objects)*100:.1f}%)',
            f'Connections: {connection_count}'
        ]
        
        stats_text = ''
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=8, 
                verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        # Save the figure
        highlight_suffix = "_prosumers_highlighted" if highlight_prosumers_only else ""
        filename = f'spatial_network_10cm_income_class_{scenario}_step_{step}{highlight_suffix}.png'
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"✅ Income class spatial network visualization saved to: {output_path}")
        print(f"   Grid: {grid_label}")
        print(f"   Households: {len(household_objects)}")
        print(f"   Prosumers: {prosumer_count}")
        print(f"   Connections: {connection_count}")
        print(f"   Highlight prosumers only: {highlight_prosumers_only}")


    def create_multi_scenario_network_grid_income_class(self, output_dir="results/spatial_analysis", 
                                                    step=120, highlight_prosumers_only=False):
        """
        Create a 2x3 grid showing spatial network visualizations for all scenarios,
        colored by income class.
        
        Args:
            output_dir: Directory to save visualizations
            step: Simulation step (default: 120 = Year 10)
            highlight_prosumers_only: If True, show only prosumers with income colors,
                                    non-prosumers in light gray
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Define scenarios to visualize (excluding 'all_biases' for clarity)
        scenarios_to_plot = ['rational', 'herding','all_biases']
        
        # Create 2x3 subplot grid
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # Income class color palette (Class 1=Red to Class 5=Green)
        income_class_colors = {
            1: '#8B0000',  # Dark red (lowest income)
            2: '#FF8C00',  # Orange
            3: '#FFD700',  # Gold/Yellow  
            4: '#32CD32',  # Lime green
            5: '#006400'   # Dark green (highest income)
        }
        light_gray = "#D3D3D33C"  # For non-prosumers when highlighting
        
        for idx, scenario in enumerate(scenarios_to_plot):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            
            # Get household data for this scenario
            households = self._get_households_for_scenario_timepoint(scenario, step)
            
            if not households:
                ax.text(0.5, 0.5, f'No data\navailable\nfor {scenario}', 
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=12, alpha=0.7)
                ax.set_title(scenario.replace('_', ' ').title(), fontsize=12, fontweight='bold')
                continue
            
            # Extract positions
            positions = []
            household_objects = []
            
            for household in households:
                if hasattr(household, 'pos') and household.pos is not None:
                    positions.append(household.pos)
                    household_objects.append(household)
            
            if not positions:
                ax.text(0.5, 0.5, 'No position data', ha='center', va='center', 
                    transform=ax.transAxes, fontsize=10, alpha=0.7)
                ax.set_title(scenario.replace('_', ' ').title(), fontsize=12, fontweight='bold')
                continue
            
            positions = np.array(positions)
            
            # Scale to 10cm grid
            x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
            y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
            current_size = max(x_max - x_min, y_max - y_min)
            scale_factor = 10.0 / current_size if current_size > 0 else 1.0
            scaled_positions = positions * scale_factor
            
            # Draw connections
            connection_count = 0
            for i, household in enumerate(household_objects):
                if hasattr(household, 'spatial_neighbors'):
                    household_pos = scaled_positions[i]
                    
                    for neighbor_household, distance in household.spatial_neighbors:
                        # Find neighbor position
                        neighbor_idx = None
                        for j, h in enumerate(household_objects):
                            if h.unique_id == neighbor_household.unique_id:
                                neighbor_idx = j
                                break
                        
                        if neighbor_idx is not None:
                            neighbor_pos = scaled_positions[neighbor_idx]
                            ax.plot([household_pos[0], neighbor_pos[0]], 
                                [household_pos[1], neighbor_pos[1]], 
                                'k-', linewidth=0.15, alpha=0.25, zorder=1)
                            connection_count += 1
            
            # Draw household points with income class coloring
            colors = []
            prosumer_count = 0
            income_class_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
            prosumer_by_class = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
            
            for household in household_objects:
                # Get income class
                income_class = getattr(household, 'income_class', 1)  # Default to class 1 if missing
                income_class_counts[income_class] += 1
                
                # Check if household is prosumer for this scenario at this step
                is_prosumer = False
                
                # Method 1: Check adoption month (most reliable for timeline)
                if hasattr(household, 'get_adoption_month'):
                    adoption_month = household.get_adoption_month(scenario)
                    is_prosumer = adoption_month is not None and adoption_month <= step
                # Method 2: Check scenario_adoption dictionary
                elif hasattr(household, 'scenario_adoption') and scenario in household.scenario_adoption:
                    is_prosumer = household.scenario_adoption[scenario]
                # Method 3: Fallback to general is_prosumer attribute
                else:
                    is_prosumer = getattr(household, 'is_prosumer', False)
                
                if is_prosumer:
                    prosumer_count += 1
                    prosumer_by_class[income_class] += 1
                
                # Determine color based on highlight option
                if highlight_prosumers_only:
                    if is_prosumer:
                        colors.append(income_class_colors[income_class])
                    else:
                        colors.append(light_gray)
                else:
                    colors.append(income_class_colors[income_class])
            
            ax.scatter(scaled_positions[:, 0], scaled_positions[:, 1], 
                    c=colors, s=5, alpha=0.8, zorder=2, edgecolors='black', linewidth=0.15)
            
            # Set up subplot
            scaled_x_min, scaled_x_max = scaled_positions[:, 0].min(), scaled_positions[:, 0].max()
            scaled_y_min, scaled_y_max = scaled_positions[:, 1].min(), scaled_positions[:, 1].max()
            
            ax.set_xlim(scaled_x_min - 0.3, scaled_x_max + 0.3)
            ax.set_ylim(scaled_y_min - 0.3, scaled_y_max + 0.3)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.2, linewidth=0.15)
            
            # Title with statistics
            prosumer_pct = prosumer_count / len(household_objects) * 100 if household_objects else 0
            
            # Calculate class distribution for title
            most_prosumers_class = max(prosumer_by_class, key=prosumer_by_class.get)
            most_prosumers_count = prosumer_by_class[most_prosumers_class]
            
            title = f'{scenario.replace("_", " ").title()}\n{prosumer_count}/{len(household_objects)} prosumers ({prosumer_pct:.1f}%)\nMost adopters: Class {most_prosumers_class} ({most_prosumers_count})'
            ax.set_title(title, fontsize=10, fontweight='bold')
            
            # Labels for edge subplots
            if idx >= 3:  # Bottom row
                ax.set_xlabel('X Position (cm)', fontsize=9)
            if idx % 3 == 0:  # Left column
                ax.set_ylabel('Y Position (cm)', fontsize=9)
        
        # Hide unused subplot
        if len(scenarios_to_plot) < len(axes):
            for idx in range(len(scenarios_to_plot), len(axes)):
                axes[idx].set_visible(False)
        
        # Add overall title and legend
        highlight_text = " (Prosumers Highlighted)" if highlight_prosumers_only else ""
        fig.suptitle(f'Income Class Spatial Networks (10cm × 10cm) - Step {step}{highlight_text}\nRed→Green: Income Class 1→5, Black Lines: Neighbor Connections', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        # Add a comprehensive legend for all subplots
        from matplotlib.lines import Line2D
        legend_elements = []
        
        # Income class legend elements
        for income_class in range(1, 6):
            label = f'Class {income_class}'
            if income_class == 1:
                label += ' (Lowest)'
            elif income_class == 5:
                label += ' (Highest)'
            
            legend_elements.append(
                Line2D([0], [0], marker='o', color='w', 
                    markerfacecolor=income_class_colors[income_class], 
                    markersize=6, label=label, markeredgecolor='black')
            )
        
        # Add non-prosumer legend if highlighting
        if highlight_prosumers_only:
            legend_elements.append(
                Line2D([0], [0], marker='o', color='w', markerfacecolor=light_gray, 
                    markersize=6, label='Non-Prosumers', markeredgecolor='black')
            )
        
        # Add connection legend
        legend_elements.append(
            Line2D([0], [0], color='black', linewidth=1, alpha=0.25, 
                label='Neighbor Connections')
        )
        
        fig.legend(handles=legend_elements, loc='lower right', fontsize=11, 
                bbox_to_anchor=(0.98, 0.02))
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.88, bottom=0.1, right=0.94)
        
        # Save the figure
        highlight_suffix = "_prosumers_highlighted" if highlight_prosumers_only else ""
        filename = f'multi_scenario_income_class_grids_step_{step}{highlight_suffix}.png'
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"✅ Multi-scenario income class network visualization saved to: {output_path}")
        print(f"   Highlight prosumers only: {highlight_prosumers_only}")

    def create_adoption_timeline_grid(self, output_dir="results/spatial_analysis", scenario="rational", 
                                    step=120, show_only_new_prosumer_networks=False):
        """
        Create a spatial grid showing prosumer adoption timeline with color-coded timing.
        
        Color Coding:
        - Lime Green: NEW prosumers (adopted in last 12 months: step-11 to step)
        - Gold/Yellow: RECENT prosumers (adopted in year before: step-23 to step-12) 
        - Blue: OLDER prosumers (adopted at step-24 or earlier)
        - Light Gray: NON-prosumers (haven't adopted yet)
        
        Args:
            output_dir: Directory to save visualizations
            scenario: Scenario to analyze (e.g., "rational", "herding", etc.)
            step: Current simulation step to analyze
            show_only_new_prosumer_networks: If True, only show new prosumers + their neighbors
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Get household data for the specified scenario and timepoint
        households = self._get_households_for_scenario_timepoint(scenario, step)
        
        if not households:
            print(f"No household data available for scenario '{scenario}' at step {step}")
            return
        
        # Create the figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        # Extract household positions and classify by adoption timing
        positions = []
        household_objects = []
        colors = []
        
        new_prosumers = []  # Track new prosumers for filtering option
        
        # Color definitions
        lime_green = '#32CD32'  # New prosumers (most highlighted)
        gold_yellow = '#FFD700'  # Recent prosumers  
        blue = "#41C6E1"        # Older prosumers
        light_gray = '#D3D3D3'  # Non-prosumers (barely visible)
        
        for household in households:
            if hasattr(household, 'pos') and household.pos is not None:
                positions.append(household.pos)
                household_objects.append(household)
                
                # Get adoption month for this scenario
                adoption_month = household.get_adoption_month(scenario)
                
                # Classify by adoption timing
                if adoption_month is None:
                    # Non-prosumer
                    colors.append(light_gray)
                elif step - 11 <= adoption_month <= step:
                    # NEW prosumer (adopted in last 12 months)
                    colors.append(lime_green)
                    new_prosumers.append(household)
                elif step - 23 <= adoption_month <= step - 12:
                    # RECENT prosumer (adopted in year before: step-23 to step-12)
                    colors.append(gold_yellow)
                elif adoption_month <= step - 24:
                    # OLDER prosumer (adopted at step-24 or earlier)
                    colors.append(blue)
                else:
                    # Future adopter (shouldn't happen, but safety check)
                    colors.append(light_gray)
        
        if not positions:
            print("No household positions found. Network may not be built yet.")
            return
        
        positions = np.array(positions)
        
        # Determine grid scaling to 10cm
        x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
        y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
        current_size = max(x_max - x_min, y_max - y_min)
        scale_factor = 10.0 / current_size if current_size > 0 else 1.0
        scaled_positions = positions * scale_factor
        
        # Apply filtering if requested
        if show_only_new_prosumer_networks:
            # Find all neighbors of new prosumers
            visible_household_ids = set()
            
            # Add all new prosumers
            for household in new_prosumers:
                visible_household_ids.add(household.unique_id)
                
                # Add their 10 closest neighbors
                if hasattr(household, 'spatial_neighbors'):
                    for neighbor_household, distance in household.spatial_neighbors:
                        visible_household_ids.add(neighbor_household.unique_id)
            
            # Filter households and positions to only visible ones
            filtered_positions = []
            filtered_household_objects = []
            filtered_colors = []
            
            for i, household in enumerate(household_objects):
                if household.unique_id in visible_household_ids:
                    filtered_positions.append(scaled_positions[i])
                    filtered_household_objects.append(household)
                    filtered_colors.append(colors[i])
            
            if filtered_positions:
                filtered_positions = np.array(filtered_positions)
                household_objects = filtered_household_objects
                colors = filtered_colors
                scaled_positions = filtered_positions
            else:
                print(f"No visible households found with filtering option.")
                return
        
        # Draw neighbor connections
        connection_count = 0
        for i, household in enumerate(household_objects):
            if hasattr(household, 'spatial_neighbors'):
                household_pos = scaled_positions[i]
                
                for neighbor_household, distance in household.spatial_neighbors:
                    # Find the neighbor's position in our visible households
                    neighbor_idx = None
                    for j, h in enumerate(household_objects):
                        if h.unique_id == neighbor_household.unique_id:
                            neighbor_idx = j
                            break
                    
                    if neighbor_idx is not None:
                        neighbor_pos = scaled_positions[neighbor_idx]
                        
                        # Draw line between household and neighbor
                        ax.plot([household_pos[0], neighbor_pos[0]], 
                            [household_pos[1], neighbor_pos[1]], 
                            'k-', linewidth=0.2, alpha=0.3, zorder=1)
                        connection_count += 1
        
        # Draw household points with adoption timeline colors
        ax.scatter(scaled_positions[:, 0], scaled_positions[:, 1], 
                c=colors, s=20, alpha=1.0, zorder=3, edgecolors='black', linewidth=0.2)
        
        # Set up the plot
        scaled_x_min, scaled_x_max = scaled_positions[:, 0].min(), scaled_positions[:, 0].max()
        scaled_y_min, scaled_y_max = scaled_positions[:, 1].min(), scaled_positions[:, 1].max()
        
        ax.set_xlim(scaled_x_min - 0.5, scaled_x_max + 0.5)
        ax.set_ylim(scaled_y_min - 0.5, scaled_y_max + 0.5)
        ax.set_aspect('equal')
        
        # Add grid
        ax.grid(True, alpha=0.3, linewidth=0.2)
        
        # Labels and title
        ax.set_xlabel('X Position (cm)', fontsize=12)
        ax.set_ylabel('Y Position (cm)', fontsize=12)
        
        # Count households by type
        new_count = sum(1 for c in colors if c == lime_green)
        recent_count = sum(1 for c in colors if c == gold_yellow)
        older_count = sum(1 for c in colors if c == blue)
        non_prosumer_count = sum(1 for c in colors if c == light_gray)
        
        filter_text = " (New Prosumer Networks Only)" if show_only_new_prosumer_networks else ""
        
        ax.set_title(f'Prosumer Adoption Timeline - 10cm × 10cm{filter_text}\n'
                    f'Scenario: {scenario.replace("_", " ").title()}, Step: {step}\n'
                    f'New: {new_count}, Recent: {recent_count}, Older: {older_count}, Non-prosumers: {non_prosumer_count}', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=lime_green, 
               markersize=10, label=f'NEW Prosumers (Steps {step-11}-{step})', markeredgecolor='black'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=gold_yellow, 
               markersize=10, label=f'RECENT Prosumers (Steps {step-23}-{step-12})', markeredgecolor='black'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=blue, 
               markersize=10, label=f'OLDER Prosumers (≤Step {step-24})', markeredgecolor='black'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=light_gray, 
                markersize=10, label='Non-Prosumers', markeredgecolor='black'),
            Line2D([0], [0], color='black', linewidth=1, alpha=0.3, 
                label='Neighbor Connections')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10, 
                bbox_to_anchor=(1.0, 1.0))
        
        # Add statistics text box
        stats_text = f'''Timeline Statistics:
        NEW prosumers (steps {step-11}-{step}): {new_count}
        RECENT prosumers (steps {step-23}-{step-12}): {recent_count}
        OLDER prosumers (≤step {step-24}): {older_count}
        Non-prosumers: {non_prosumer_count}
        Total households: {len(household_objects)}
        Connections shown: {connection_count}'''
        
        if show_only_new_prosumer_networks:
            stats_text += f'\n\nFiltered to show only {len(new_prosumers)} new prosumers + neighbors'
        stats_text=''
        ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, fontsize=9, 
                verticalalignment='bottom', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        # Save the figure
        filter_suffix = "_filtered" if show_only_new_prosumer_networks else ""
        filename = f'adoption_timeline_{scenario}_step_{step}{filter_suffix}.png'
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"✅ Adoption timeline visualization saved to: {output_path}")
        print(f"   Scenario: {scenario}")
        print(f"   Step: {step} (Year {(step//12)+1}, Month {(step%12)+1})")
        print(f"   NEW prosumers: {new_count}")
        print(f"   RECENT prosumers: {recent_count}")
        print(f"   OLDER prosumers: {older_count}")
        print(f"   Non-prosumers: {non_prosumer_count}")
        print(f"   Filter applied: {show_only_new_prosumer_networks}")
        print(f"   Total visible households: {len(household_objects)}")

    def create_adoption_timeline_grid_income_similarity(self, output_dir="results/spatial_analysis", 
                                                    scenario="rational", step=120, 
                                                    show_only_new_prosumer_networks=False):
        """
        Create a spatial grid showing prosumer adoption timeline with income similarity analysis.
        
        Color Coding:
        - Lime Green: NEW prosumers (adopted in last 12 months: step-11 to step)
        - Yellow: RECENT prosumers (step-23 to step-12) who are BOTH neighbors AND have same income class as NEW prosumers
        - Orange: RECENT prosumers (step-23 to step-12) who DON'T meet both conditions  
        - Grey: OLDER prosumers (adopted at step-24 or earlier)
        - Light Grey: NON-prosumers (haven't adopted yet)
        
        Args:
            output_dir: Directory to save visualizations
            scenario: Scenario to analyze (e.g., "rational", "herding", etc.)
            step: Current simulation step to analyze
            show_only_new_prosumer_networks: If True, only show new prosumers + their neighbors
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Get household data for the specified scenario and timepoint
        households = self._get_households_for_scenario_timepoint(scenario, step)
        
        if not households:
            print(f"No household data available for scenario '{scenario}' at step {step}")
            return
        
        # Create the figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        # Extract household positions and classify by adoption timing
        positions = []
        household_objects = []
        colors = []
        
        new_prosumers = []  # Track new prosumers for filtering and income similarity
        recent_prosumers = []  # Track recent prosumers for income similarity analysis
        
        # Color definitions
        lime_green = '#32CD32'  # New prosumers (most highlighted)
        gold_yellow = '#FFD700'  # Recent prosumers meeting both conditions
        orange = '#FF8C00'      # Recent prosumers NOT meeting both conditions
        grey = '#808080'        # Older prosumers (not light grey)
        light_gray = '#D3D3D3'  # Non-prosumers (barely visible)
        
        # First pass: classify all households by adoption timing
        for household in households:
            if hasattr(household, 'pos') and household.pos is not None:
                positions.append(household.pos)
                household_objects.append(household)
                
                # Get adoption month for this scenario
                adoption_month = household.get_adoption_month(scenario)
                
                # Classify by adoption timing
                if adoption_month is None:
                    # Non-prosumer
                    colors.append(light_gray)
                elif step - 11 <= adoption_month <= step:
                    # NEW prosumer (adopted in last 12 months)
                    colors.append(lime_green)
                    new_prosumers.append(household)
                elif step - 23 <= adoption_month <= step - 12:
                    # RECENT prosumer (adopted in year before) - will be re-classified below
                    colors.append(gold_yellow)  # Temporary, will be updated
                    recent_prosumers.append(household)
                elif adoption_month <= step - 24:
                    # OLDER prosumer (adopted at step-24 or earlier)
                    colors.append(grey)
                else:
                    # Future adopter (shouldn't happen, but safety check)
                    colors.append(light_gray)
        
        if not positions:
            print("No household positions found. Network may not be built yet.")
            return
        
        positions = np.array(positions)
        
        # Get income classes of new prosumers for similarity check
        new_prosumer_income_classes = set()
        for household in new_prosumers:
            income_class = getattr(household, 'income_class', 1)
            new_prosumer_income_classes.add(income_class)
        
        print(f"DEBUG: New prosumer income classes: {new_prosumer_income_classes}")
        
        # Second pass: Re-classify recent prosumers based on income similarity and neighbor conditions
        yellow_count = 0  # Recent prosumers meeting both conditions
        orange_count = 0  # Recent prosumers not meeting both conditions
        
        for i, household in enumerate(household_objects):
            if household in recent_prosumers:
                # Check both conditions for recent prosumers
                household_income_class = getattr(household, 'income_class', 1)
                
                # Condition 1: Same income class as at least one new prosumer
                has_same_income_class = household_income_class in new_prosumer_income_classes
                
                # Condition 2: Is neighbor of at least one new prosumer
                is_neighbor_of_new_prosumer = False
                if hasattr(household, 'spatial_neighbors'):
                    for neighbor_household, distance in household.spatial_neighbors:
                        if neighbor_household in new_prosumers:
                            is_neighbor_of_new_prosumer = True
                            break
                
                # Update color based on both conditions
                if has_same_income_class and is_neighbor_of_new_prosumer:
                    colors[i] = gold_yellow  # Keep yellow
                    yellow_count += 1
                else:
                    colors[i] = orange  # Change to orange
                    orange_count += 1
        
        print(f"DEBUG: Recent prosumers - Yellow (both conditions): {yellow_count}, Orange (failed conditions): {orange_count}")
        
        # Determine grid scaling to 10cm
        x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
        y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
        current_size = max(x_max - x_min, y_max - y_min)
        scale_factor = 10.0 / current_size if current_size > 0 else 1.0
        scaled_positions = positions * scale_factor
        
        # Apply filtering if requested
        if show_only_new_prosumer_networks:
            # Find all neighbors of new prosumers
            visible_household_ids = set()
            
            # Add all new prosumers
            for household in new_prosumers:
                visible_household_ids.add(household.unique_id)
                
                # Add their 10 closest neighbors
                if hasattr(household, 'spatial_neighbors'):
                    for neighbor_household, distance in household.spatial_neighbors:
                        visible_household_ids.add(neighbor_household.unique_id)
            
            # Filter households and positions to only visible ones
            filtered_positions = []
            filtered_household_objects = []
            filtered_colors = []
            
            for i, household in enumerate(household_objects):
                if household.unique_id in visible_household_ids:
                    filtered_positions.append(scaled_positions[i])
                    filtered_household_objects.append(household)
                    filtered_colors.append(colors[i])
            
            if filtered_positions:
                filtered_positions = np.array(filtered_positions)
                household_objects = filtered_household_objects
                colors = filtered_colors
                scaled_positions = filtered_positions
            else:
                print(f"No visible households found with filtering option.")
                return
        
        # Draw neighbor connections
        connection_count = 0
        for i, household in enumerate(household_objects):
            if hasattr(household, 'spatial_neighbors'):
                household_pos = scaled_positions[i]
                
                for neighbor_household, distance in household.spatial_neighbors:
                    # Find the neighbor's position in our visible households
                    neighbor_idx = None
                    for j, h in enumerate(household_objects):
                        if h.unique_id == neighbor_household.unique_id:
                            neighbor_idx = j
                            break
                    
                    if neighbor_idx is not None:
                        neighbor_pos = scaled_positions[neighbor_idx]
                        
                        # Draw line between household and neighbor
                        ax.plot([household_pos[0], neighbor_pos[0]], 
                            [household_pos[1], neighbor_pos[1]], 
                            'k-', linewidth=0.2, alpha=0.3, zorder=1)
                        connection_count += 1
        
        # Draw household points with adoption timeline colors
        ax.scatter(scaled_positions[:, 0], scaled_positions[:, 1], 
                c=colors, s=10, alpha=0.9, zorder=2, edgecolors='black', linewidth=0.2)
        
        # Set up the plot
        scaled_x_min, scaled_x_max = scaled_positions[:, 0].min(), scaled_positions[:, 0].max()
        scaled_y_min, scaled_y_max = scaled_positions[:, 1].min(), scaled_positions[:, 1].max()
        
        ax.set_xlim(scaled_x_min - 0.5, scaled_x_max + 0.5)
        ax.set_ylim(scaled_y_min - 0.5, scaled_y_max + 0.5)
        ax.set_aspect('equal')
        
        # Add grid
        ax.grid(True, alpha=0.3, linewidth=0.2)
        
        # Labels and title
        ax.set_xlabel('X Position (cm)', fontsize=12)
        ax.set_ylabel('Y Position (cm)', fontsize=12)
        
        # Count households by type
        new_count = sum(1 for c in colors if c == lime_green)
        yellow_count_final = sum(1 for c in colors if c == gold_yellow)
        orange_count_final = sum(1 for c in colors if c == orange)
        older_count = sum(1 for c in colors if c == grey)
        non_prosumer_count = sum(1 for c in colors if c == light_gray)
        
        filter_text = " (New Prosumer Networks Only)" if show_only_new_prosumer_networks else ""
        
        ax.set_title(f'Adoption Timeline with Income Similarity - 10cm × 10cm{filter_text}\n'
                    f'Scenario: {scenario.replace("_", " ").title()}, Step: {step}\n'
                    f'New: {new_count}, Recent+Similar: {yellow_count_final}, Recent+Different: {orange_count_final}, Older: {older_count}', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=lime_green, 
                markersize=10, label=f'NEW Prosumers (Steps {step-11}-{step})', markeredgecolor='black'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=gold_yellow, 
                markersize=10, label='RECENT: Neighbors + Same Income Class', markeredgecolor='black'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=orange, 
                markersize=10, label='RECENT: Missing Neighbor/Income Condition', markeredgecolor='black'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=grey, 
                markersize=10, label=f'OLDER Prosumers (≤Step {step-24})', markeredgecolor='black'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=light_gray, 
                markersize=10, label='Non-Prosumers', markeredgecolor='black'),
            Line2D([0], [0], color='black', linewidth=1, alpha=0.3, 
                label='Neighbor Connections')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9, 
                bbox_to_anchor=(1.0, 1.0))
        
        # Add detailed statistics text box
        stats_lines = [
            f'Income Similarity Analysis:',
            f'NEW prosumers (steps {step-11}-{step}): {new_count}',
            f'Income classes of NEW: {sorted(new_prosumer_income_classes)}',
            f'',
            f'RECENT prosumers (steps {step-23}-{step-12}):',
            f'  Meeting both conditions: {yellow_count_final}',
            f'  Missing condition(s): {orange_count_final}',
            f'',
            f'OLDER prosumers (≤step {step-24}): {older_count}',
            f'Non-prosumers: {non_prosumer_count}',
            f'',
            f'Total households: {len(household_objects)}',
            f'Connections shown: {connection_count}'
        ]
        
        if show_only_new_prosumer_networks:
            stats_lines.append(f'')
            stats_lines.append(f'Filtered to show only {len(new_prosumers)} new prosumers + neighbors')
        
        stats_text = ''
        ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, fontsize=8, 
                verticalalignment='bottom', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.95))
        
        # Save the figure
        filter_suffix = "_filtered" if show_only_new_prosumer_networks else ""
        filename = f'adoption_timeline_income_similarity_{scenario}_step_{step}{filter_suffix}.png'
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"✅ Adoption timeline with income similarity saved to: {output_path}")
        print(f"   Scenario: {scenario}")
        print(f"   Step: {step} (Year {(step//12)+1}, Month {(step%12)+1})")
        print(f"   NEW prosumers: {new_count}")
        print(f"   RECENT (both conditions): {yellow_count_final}")
        print(f"   RECENT (missing conditions): {orange_count_final}")
        print(f"   OLDER prosumers: {older_count}")
        print(f"   Non-prosumers: {non_prosumer_count}")
        print(f"   Income classes of NEW prosumers: {sorted(new_prosumer_income_classes)}")
        print(f"   Filter applied: {show_only_new_prosumer_networks}")
        print(f"   Total visible households: {len(household_objects)}")

    def create_information_cascade_analysis(self, output_dir="results/spatial_analysis", 
                                        behavioral_scenario="herding", counterfactual_scenario="rational"):
        """
        Analyze information cascade effects in prosumer adoption.
        
        Rolling cascade analysis where each period's adopters become next period's influencers.
        Exports data to CSV and creates stacked area visualization.
        
        Args:
            output_dir: Directory to save analysis and plots
            behavioral_scenario: Scenario with social influence (default: "herding")
            counterfactual_scenario: Baseline scenario (default: "rational")
        """
        import matplotlib.pyplot as plt
        import pandas as pd
        import numpy as np
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Get household data
        households = self._get_households_for_scenario_timepoint(behavioral_scenario, 240)  # End of simulation
        
        if not households:
            print(f"No household data available for scenario '{behavioral_scenario}'")
            return
        
        # Validate required data
        missing_data = []
        for h in households[:5]:  # Check first 5 households
            if not hasattr(h, 'spatial_neighbors') or not h.spatial_neighbors:
                missing_data.append(f"spatial_neighbors missing")
            if not hasattr(h, 'income_class'):
                missing_data.append(f"income_class missing")
            if not hasattr(h, 'get_adoption_month'):
                missing_data.append(f"get_adoption_month method missing")
        
        if missing_data:
            print(f"Missing required data: {set(missing_data)}")
            return
        
        # Initialize data collection
        cascade_data = {
            'year_range': [],
            'new_adopters_count': [],
            'next_adopters_total': [],
            'behavioral_only_adopters': [],
            'other_adopters': [],
            'spatial_influenced': [],
            'income_influenced': [],
            'both_influenced': [],
            'neither_influenced': []
        }
        
        # Rolling cascade analysis
        current_new_adopters = []
        
        # Get initial adopters (baseline influencers)
        for household in households:
            behavioral_month = household.get_adoption_month(behavioral_scenario)
            if behavioral_month is not None and behavioral_month < 12:
                current_new_adopters.append(household)
        
        print(f"Initial adopters (Year 0): {len(current_new_adopters)}")
        
        # Analyze cascades for each year
        for year in range(1, 21):  # Years 1-20
            step_start = (year - 1) * 12
            step_end = year * 12 - 1
            
            print(f"Analyzing Year {year} cascade (steps {step_start}-{step_end})...")
            
            # Get next adopters in behavioral scenario
            next_adopters = []
            for household in households:
                behavioral_month = household.get_adoption_month(behavioral_scenario)
                if behavioral_month is not None and step_start <= behavioral_month <= step_end:
                    next_adopters.append(household)
            
            # Split next adopters: behavioral_only vs other
            behavioral_only = []
            other_adopters = []
            
            for household in next_adopters:
                counterfactual_month = household.get_adoption_month(counterfactual_scenario)
                # Would have adopted in counterfactual anyway
                if counterfactual_month is not None and step_start <= counterfactual_month <= step_end:
                    other_adopters.append(household)
                else:
                    behavioral_only.append(household)
            
            # Classify behavioral_only adopters by influence mechanism
            spatial_influenced = []
            income_influenced = []
            both_influenced = []
            neither_influenced = []
            
            if current_new_adopters:  # Only classify if there are potential influencers
                # Get income classes of new adopters
                new_adopter_income_classes = {getattr(h, 'income_class', 1) for h in current_new_adopters}
                
                for household in behavioral_only:
                    has_spatial_connection = False
                    has_income_connection = False
                    
                    # Check spatial connection
                    if hasattr(household, 'spatial_neighbors'):
                        for neighbor_household, distance in household.spatial_neighbors:
                            if neighbor_household in current_new_adopters:
                                has_spatial_connection = True
                                break
                    
                    # Check income class connection
                    household_income = getattr(household, 'income_class', 1)
                    if household_income in new_adopter_income_classes:
                        has_income_connection = True
                    
                    # Classify
                    if has_spatial_connection and has_income_connection:
                        both_influenced.append(household)
                    elif has_spatial_connection:
                        spatial_influenced.append(household)
                    elif has_income_connection:
                        income_influenced.append(household)
                    else:
                        neither_influenced.append(household)
            else:
                # No potential influencers, all go to neither
                neither_influenced = behavioral_only.copy()
            
            # Store data
            cascade_data['year_range'].append(f'Year {year}')
            cascade_data['new_adopters_count'].append(len(current_new_adopters))
            cascade_data['next_adopters_total'].append(len(next_adopters))
            cascade_data['behavioral_only_adopters'].append(len(behavioral_only))
            cascade_data['other_adopters'].append(len(other_adopters))
            cascade_data['spatial_influenced'].append(len(spatial_influenced))
            cascade_data['income_influenced'].append(len(income_influenced))
            cascade_data['both_influenced'].append(len(both_influenced))
            cascade_data['neither_influenced'].append(len(neither_influenced))
            
            print(f"  New adopters (influencers): {len(current_new_adopters)}")
            print(f"  Next adopters total: {len(next_adopters)}")
            print(f"  Behavioral only: {len(behavioral_only)} (spatial:{len(spatial_influenced)}, income:{len(income_influenced)}, both:{len(both_influenced)}, neither:{len(neither_influenced)})")
            
            # Update for next iteration: all next adopters become new adopters
            current_new_adopters = next_adopters.copy()
        
        # Convert to DataFrame and calculate cumulative values
        df = pd.DataFrame(cascade_data)
        
        # Calculate cumulative values for stacking
        df['cum_other'] = df['other_adopters'].cumsum()
        df['cum_neither'] = df['cum_other'] + df['neither_influenced'].cumsum()
        df['cum_spatial'] = df['cum_neither'] + df['spatial_influenced'].cumsum()
        df['cum_income'] = df['cum_spatial'] + df['income_influenced'].cumsum()
        df['cum_both'] = df['cum_income'] + df['both_influenced'].cumsum()
        
        # Export to CSV
        csv_filename = f'information_cascade_analysis_{behavioral_scenario}_vs_{counterfactual_scenario}.csv'
        csv_path = os.path.join(output_dir, csv_filename)
        df.to_csv(csv_path, index=False)
        print(f"Data exported to: {csv_path}")
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Panel 1: Cascade Volume Over Time (Line Plot)
        years = range(1, len(df) + 1)
        # ax1.plot(years, df['new_adopters_count'], 'o-', linewidth=2, label='Influencers (New Adopters)', color='red')
        ax1.plot(years, df['behavioral_only_adopters'], 's-', linewidth=2, label='Influence cascade (BInfluenced Adopters)', color='blue')
        ax1.plot(years, df['next_adopters_total'], '^-', linewidth=2, label='Total New Adopters', color='gray', alpha=0.7)
        
        ax1.set_title(f'Information Cascade Volume: {behavioral_scenario.replace("_", " ").title()} vs {counterfactual_scenario.replace("_", " ").title()}', 
                    fontsize=14, fontweight='bold')
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Number of Households')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Panel 2: Influence Composition (Stacked Area)
        ax2.fill_between(years, 0, df['cum_other'], alpha=0.8, color='lightgray', label='Economic Fundamentals')
        ax2.fill_between(years, df['cum_other'], df['cum_neither'], alpha=0.8, color='orange', label='Unmeasured Behavioral')
        ax2.fill_between(years, df['cum_neither'], df['cum_spatial'], alpha=0.8, color='lightblue', label='Spatial Influence')
        ax2.fill_between(years, df['cum_spatial'], df['cum_income'], alpha=0.8, color='lightgreen', label='Income Class Influence')
        ax2.fill_between(years, df['cum_income'], df['cum_both'], alpha=0.8, color='darkblue', label='Both Channels')
        
        ax2.set_title('Cumulative Adoption by Influence Mechanism', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Year')
        ax2.set_ylabel('Cumulative Adopters')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = f'information_cascade_analysis_{behavioral_scenario}_vs_{counterfactual_scenario}.png'
        plot_path = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"Analysis completed and saved to: {plot_path}")
        
        # Summary statistics
        total_behavioral = df['behavioral_only_adopters'].sum()
        total_spatial = df['spatial_influenced'].sum()
        total_income = df['income_influenced'].sum()
        total_both = df['both_influenced'].sum()
        total_neither = df['neither_influenced'].sum()
        
        print(f"\nCascade Summary:")
        print(f"  Total behavioral adopters: {total_behavioral}")
        print(f"  Spatial influence: {total_spatial} ({total_spatial/total_behavioral*100:.1f}%)")
        print(f"  Income influence: {total_income} ({total_income/total_behavioral*100:.1f}%)")
        print(f"  Both channels: {total_both} ({total_both/total_behavioral*100:.1f}%)")
        print(f"  Neither/unmeasured: {total_neither} ({total_neither/total_behavioral*100:.1f}%)")

    def create_all_visualizations(self, output_dir="results/spatial_analysis"):
        """
        Create all enhanced spatial visualizations for Phase 1.
        
        Args:
            output_dir: Directory to save all visualizations
        """
        print(f"Creating all enhanced spatial visualizations in {output_dir}...")
        
        # Core enhanced spatial grids
        self.create_enhanced_spatial_grid_6x2(output_dir)
        self.create_enhanced_spatial_grid_6x4(output_dir)
        
        # Herding bias validation plots
        self.create_income_homophily_matrices(output_dir)
        self.create_herding_component_validation(output_dir)

        # Single scenario 10cm grid
        self.create_spatial_network_grid_10cm(output_dir,
            scenario="rational",  # or any other scenario
            step=1  # Year 10
        )
        self.create_spatial_network_grid_10cm(output_dir,
            scenario="rational",
            step=48  # Year 10
        )

        # Multi-scenario comparison
        self.create_multi_scenario_network_grid(output_dir, 
            step=120
        )

        # Detailed network analysis
        self.create_interactive_network_analysis(output_dir,
            scenario="herding",
            step=120  # Year 20
        )

        # Single scenario - all households by income class
        self.create_spatial_network_grid_10cm_income_class(output_dir,
            scenario="herding", 
            step=60,
            highlight_prosumers_only=False
        )

        # Single scenario - highlight prosumers only
        self.create_spatial_network_grid_10cm_income_class(output_dir,
            scenario="herding",
            step=60, 
            highlight_prosumers_only=True
        )

        # Multi-scenario comparison all households by income class
        self.create_multi_scenario_network_grid_income_class(output_dir,
            step=60,
            highlight_prosumers_only=True
        )
        # Multi-scenario comparison - highlight prosumers only
        self.create_multi_scenario_network_grid_income_class(output_dir,
            step=60,
            highlight_prosumers_only=True
        )
        # ---
        self.create_adoption_timeline_grid(output_dir,
            scenario="rational",  # or any scenario
            step=24,  # Year 10
            show_only_new_prosumer_networks=False
        )

        # Filtered view showing only new prosumer networks
        self.create_adoption_timeline_grid(output_dir,
            scenario="rational",
            step=60,
            show_only_new_prosumer_networks=True
        )
                # Filtered view showing only new prosumer networks
        self.create_adoption_timeline_grid(output_dir,
            scenario="rational",
            step=120,
            show_only_new_prosumer_networks=True
        )
        self.create_adoption_timeline_grid(output_dir,
            scenario="rational",
            step=180,
            show_only_new_prosumer_networks=True
        )
        self.create_adoption_timeline_grid(output_dir,
            scenario="rational",
            step=240,
            show_only_new_prosumer_networks=True
        )

        self.create_adoption_timeline_grid(output_dir,
            scenario="herding",  # or any scenario
            step=24,  # Year 10
            show_only_new_prosumer_networks=False
        )

        # Filtered view showing only new prosumer networks
        self.create_adoption_timeline_grid(output_dir,
            scenario="herding",
            step=60,
            show_only_new_prosumer_networks=True
        )
                # Filtered view showing only new prosumer networks
        self.create_adoption_timeline_grid(output_dir,
            scenario="herding",
            step=120,
            show_only_new_prosumer_networks=True
        )
        self.create_adoption_timeline_grid(output_dir,
            scenario="herding",
            step=180,
            show_only_new_prosumer_networks=True
        )
        self.create_adoption_timeline_grid(output_dir,
            scenario="herding",
            step=240,
            show_only_new_prosumer_networks=True
        )

        self.create_adoption_timeline_grid(output_dir,
            scenario="all_biases",  # or any scenario
            step=24,  # Year 10
            show_only_new_prosumer_networks=False
        )
        self.create_adoption_timeline_grid(output_dir,
            scenario="all_biases",
            step=60,
            show_only_new_prosumer_networks=True
        )
        self.create_adoption_timeline_grid(output_dir,
            scenario="all_biases",
            step=120,
            show_only_new_prosumer_networks=True
        )
        self.create_adoption_timeline_grid(output_dir,
            scenario="all_biases",
            step=180,
            show_only_new_prosumer_networks=True
        )
        self.create_adoption_timeline_grid(output_dir,
            scenario="all_biases",
            step=240,
            show_only_new_prosumer_networks=True
        )

        # Full view with income similarity analysis
        self.create_adoption_timeline_grid_income_similarity(output_dir,
            scenario="herding",  # Perfect for testing social effects
            step=60,  # Year 10
            show_only_new_prosumer_networks=False
        )

        # Filtered view focusing on new prosumer networks
        self.create_adoption_timeline_grid_income_similarity(output_dir,
            scenario="all_biases",
            step=60,
            show_only_new_prosumer_networks=True
        )
        
        # Add to spatial_visualizer.py and use:
        self.create_information_cascade_analysis(output_dir,
            behavioral_scenario="herding",      # default
            counterfactual_scenario="rational"  # default
        )
        self.create_information_cascade_analysis(output_dir,
            behavioral_scenario="herding",      # default
            counterfactual_scenario="all_biases"  # default
        )
        self.create_information_cascade_analysis(output_dir,
            behavioral_scenario="all_biases",      # default
            counterfactual_scenario="rational"  # default
        )
        print(f"✅ All enhanced spatial visualizations completed!")


class MockHouseholdForVisualization:
    """Enhanced mock household class with scenario support."""
    
    def __init__(self, row_data):
        """Initialize from pandas row data."""
        self.unique_id = row_data.get('HouseholdID', row_data.get('unique_id', 0))
        self.income_class = row_data.get('IncomeClass', 1)
        self.is_prosumer = row_data.get('IsProsumer', False)
        self.income = row_data.get('Income', 50000)
        
        # Position handling
        if 'PosX' in row_data and 'PosY' in row_data:
            self.pos = (row_data['PosX'], row_data['PosY'])
        else:
            # Fallback: reconstruct from ID
            grid_size = int(np.sqrt(1000))
            row = self.unique_id // grid_size
            col = self.unique_id % grid_size
            self.pos = (col, row)
        
        # Scenario-specific attributes
        self.scenario_adoption = {}


# =============================================================================
# TESTING FUNCTIONS
# =============================================================================

def test_enhanced_spatial_visualizer():
    """Test the enhanced spatial visualizer."""
    print("Testing EnhancedSpatialVisualizer...")
    
    try:
        # Create mock households
        households = []
        for i in range(50):
            household = MockHouseholdForVisualization({
                'HouseholdID': i,
                'IncomeClass': (i % 5) + 1,
                'IsProsumer': i % 3 == 0,
                'Income': 30000 + (i % 5) * 15000,
                'PosX': np.random.uniform(-5, 5),
                'PosY': np.random.uniform(-5, 5)
            })
            household.daily_consumption = 20.0
            households.append(household)
        
        # Test visualizer
        visualizer = SpatialVisualizer()
        
        # Test binary spatial plot
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        visualizer._create_binary_spatial_plot(ax, households, 'rational', 10)
        
        # Save test plot
        os.makedirs('test_results', exist_ok=True)
        plt.savefig('test_results/test_enhanced_spatial_plot.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("✅ EnhancedSpatialVisualizer test passed!")
        print("   - Created test enhanced spatial plot in test_results/")
        
        return True
        
    except Exception as e:
        print(f"❌ EnhancedSpatialVisualizer test failed: {e}")
        return False

if __name__ == "__main__":
    test_enhanced_spatial_visualizer()