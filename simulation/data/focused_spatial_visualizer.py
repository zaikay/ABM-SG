# data/focused_visualizer.py V2.0 - CORRECTED FOCUSED BEHAVIORAL VISUALIZATIONS
"""
CORRECTED: Focused visualizer with proper time series plots and temporal network classification.
Fixes: line plots for income evolution, temporal node classification, proper data loading.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import networkx as nx
from ..data.focused_spatial_metrics_calculator import FocusedSpatialMetricsCalculator
from ..utils.parameters import get_all_scenarios, get_scenario_colors, get_scenario_metadata

class FocusedSpatialVisualizer:
    """
    CORRECTED: Focused visualizer for behavioral prosumer analysis.
    """
    
    def __init__(self, data_path=None, model=None):
        """Initialize the focused visualizer."""
        self.data_path = data_path
        self.model = model
        self.scenarios = get_all_scenarios()
        self.colors = get_scenario_colors()
        self.metadata = get_scenario_metadata()
        
        # Load data from CSV files or model
        self.data_loaded = False
        if data_path:
            self._load_data_from_csv()
        elif model and hasattr(model, 'focused_spatial_metrics'):
            self._load_data_from_model()
        
        # Set up plot styling
        self._setup_plot_style()
        
        print(f"FocusedVisualizer initialized (data_loaded: {self.data_loaded})")
    
    def _setup_plot_style(self):
        """Set up consistent plot styling following project pattern."""
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
    
    def _load_data_from_csv(self):
        """CORRECTED: Load focused metrics data from CSV files."""
        try:
            # Load income adoption time series (CORRECTED filename)
            income_path = os.path.join(self.data_path, "focused_spatial_income_adoption_timeseries.csv")
            if os.path.exists(income_path):
                self.income_data = pd.read_csv(income_path)
            else:
                self.income_data = pd.DataFrame()
            
            # Load neighbor evolution data
            neighbor_path = os.path.join(self.data_path, "focused_spatial_neighbor_evolution.csv")
            if os.path.exists(neighbor_path):
                self.neighbor_data = pd.read_csv(neighbor_path)
            else:
                self.neighbor_data = pd.DataFrame()
            
            # Load adoption context data
            context_path = os.path.join(self.data_path, "focused_spatial_adoption_context.csv")
            if os.path.exists(context_path):
                self.context_data = pd.read_csv(context_path)
            else:
                self.context_data = pd.DataFrame()
            
            # Load network snapshots data
            snapshot_path = os.path.join(self.data_path, "focused_spatial_network_snapshots.csv")
            if os.path.exists(snapshot_path):
                self.snapshot_data = pd.read_csv(snapshot_path)
            else:
                self.snapshot_data = pd.DataFrame()
            
            self.data_loaded = True
            print(f"Loaded corrected focused metrics data from {self.data_path}")
            
        except Exception as e:
            print(f"Error loading data from CSV: {e}")
            self.data_loaded = False
    
    def _load_data_from_model(self):
        """CORRECTED: Load focused metrics data directly from model."""
        try:
            focused_spatial_metrics = self.model.focused_spatial_metrics
            
            # Convert income adoption time series to DataFrame
            income_records = []
            for scenario, scenario_data in focused_spatial_metrics.income_adoption_timeseries.items():
                for income_class, timeseries in scenario_data.items():
                    for datapoint in timeseries:
                        income_records.append({
                            'Year': datapoint['year'],
                            'Step': datapoint['step'],
                            'Scenario': scenario,
                            'IncomeClass': income_class,
                            'AdoptionRate': datapoint['adoption_rate']
                        })
            self.income_data = pd.DataFrame(income_records)
            
            # Convert neighbor evolution data
            self.neighbor_data = pd.DataFrame(focused_spatial_metrics.neighbor_evolution_data)
            
            # Convert adoption context data
            self.context_data = pd.DataFrame(focused_spatial_metrics.adoption_context_data)
            
            # Convert network snapshots data (CORRECTED temporal classification)
            snapshot_records = []
            for year, year_data in focused_spatial_metrics.network_snapshots.items():
                for scenario, node_types in year_data.items():
                    for node_type, household_ids in node_types.items():
                        for household_id in household_ids:
                            snapshot_records.append({
                                'Year': year,
                                'Scenario': scenario,
                                'NodeType': node_type,
                                'HouseholdId': household_id
                            })
            self.snapshot_data = pd.DataFrame(snapshot_records)
            
            self.data_loaded = True
            print("Loaded corrected focused metrics data from model")
            
        except Exception as e:
            print(f"Error loading data from model: {e}")
            self.data_loaded = False
    
    def create_all_focused_plots(self, output_dir="results/focused_spatial_visualizations"):
        """Create all focused visualization plots."""
        if not self.data_loaded:
            print("Cannot create visualizations: No data loaded")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        print(f"Creating corrected focused visualizations in {output_dir}...")
        
        # Create all focused plots
        self.plot_income_adoption_evolution(output_dir)  # CORRECTED: time series
        self.plot_neighbor_evolution(output_dir)
        self.plot_network_snapshots(output_dir)         # CORRECTED: temporal classification
        self.plot_adoption_context(output_dir)
        
        print("All corrected focused visualizations completed!")
    
    def plot_income_adoption_evolution(self, output_dir):
        """CORRECTED: Create time series evolution plots for income adoption."""
        if self.income_data.empty:
            print("Cannot create income adoption evolution: No income data")
            return
        
        # Create 2x1 subplot for rational vs herding, rational vs all_biases
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot 1: Rational vs Herding
        ax1 = axes[0]
        scenarios_to_compare = ['rational', 'herding']
        
        if all(scenario in self.income_data['Scenario'].values for scenario in scenarios_to_compare):
            income_classes = sorted(self.income_data['IncomeClass'].unique())
            
            for income_class in income_classes:
                for scenario in scenarios_to_compare:
                    # Get time series data for this scenario and income class
                    mask = (self.income_data['Scenario'] == scenario) & (self.income_data['IncomeClass'] == income_class)
                    scenario_data = self.income_data[mask].sort_values('Year')
                    
                    if not scenario_data.empty:
                        # Create line style: solid for rational, dashed for herding
                        linestyle = '-' if scenario == 'rational' else '--'
                        alpha = 0.8 if scenario == 'rational' else 0.9
                        
                        ax1.plot(scenario_data['Year'], scenario_data['AdoptionRate'] * 100, 
                               label=f'{self.metadata[scenario]["display_name"]} - Q{income_class}',
                               color=self.colors[scenario], linestyle=linestyle, 
                               linewidth=2, alpha=alpha, marker='o', markersize=4)
            
            ax1.set_xlabel('Year')
            ax1.set_ylabel('Adoption Rate (%)')
            ax1.set_title('Income-Based Adoption Evolution: Rational vs Social Influence', fontweight='bold')
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax1.grid(True, alpha=0.3)
        else:
            ax1.text(0.5, 0.5, 'Data not available\nfor comparison', 
                    ha='center', va='center', transform=ax1.transAxes,
                    fontsize=14, color='red')
        
        # Plot 2: Rational vs All Biases
        ax2 = axes[1]
        scenarios_to_compare = ['rational', 'all_biases']
        
        if all(scenario in self.income_data['Scenario'].values for scenario in scenarios_to_compare):
            for income_class in income_classes:
                for scenario in scenarios_to_compare:
                    # Get time series data for this scenario and income class
                    mask = (self.income_data['Scenario'] == scenario) & (self.income_data['IncomeClass'] == income_class)
                    scenario_data = self.income_data[mask].sort_values('Year')
                    
                    if not scenario_data.empty:
                        # Create line style: solid for rational, dashed for all_biases
                        linestyle = '-' if scenario == 'rational' else '--'
                        alpha = 0.8 if scenario == 'rational' else 0.9
                        
                        ax2.plot(scenario_data['Year'], scenario_data['AdoptionRate'] * 100, 
                               label=f'{self.metadata[scenario]["display_name"]} - Q{income_class}',
                               color=self.colors[scenario], linestyle=linestyle, 
                               linewidth=2, alpha=alpha, marker='o', markersize=4)
            
            ax2.set_xlabel('Year')
            ax2.set_ylabel('Adoption Rate (%)')
            ax2.set_title('Income-Based Adoption Evolution: Rational vs Combined Biases', fontweight='bold')
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'Data not available\nfor comparison', 
                    ha='center', va='center', transform=ax2.transAxes,
                    fontsize=14, color='red')
        
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, 'focused_spatial_income_adoption_evolution.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Income adoption evolution (time series) saved to: {output_path}")
    
    def plot_neighbor_evolution(self, output_dir):
        """Create neighbor evolution time series plot."""
        if self.neighbor_data.empty:
            print("Cannot create neighbor evolution plot: No neighbor data")
            return
        
        # Create figure for 6 lines (3 scenarios × 2 neighbor types)
        fig, ax = plt.subplots(figsize=(14, 8))
        
        scenarios_to_plot = ['rational', 'herding', 'all_biases']
        
        for scenario in scenarios_to_plot:
            if scenario in self.scenarios:
                # Plot average prosumer neighbors (all households)
                prosumer_col = f'{scenario}_avg_prosumer_neighbors'
                nonprosumer_col = f'{scenario}_avg_nonprosumer_prosumer_neighbors'
                
                if prosumer_col in self.neighbor_data.columns:
                    ax.plot(self.neighbor_data['year'], self.neighbor_data[prosumer_col],
                           label=f'{self.metadata[scenario]["display_name"]} - All Households',
                           color=self.colors[scenario], linewidth=2, linestyle='-', 
                           marker='o', markersize=4)
                
                if nonprosumer_col in self.neighbor_data.columns:
                    ax.plot(self.neighbor_data['year'], self.neighbor_data[nonprosumer_col],
                           label=f'{self.metadata[scenario]["display_name"]} - Non-Prosumers Only',
                           color=self.colors[scenario], linewidth=2, linestyle='--',
                           marker='s', markersize=4)
        
        ax.set_xlabel('Year')
        ax.set_ylabel('Average Prosumer Neighbors')
        ax.set_title('Evolution of Prosumer Neighbors Over Time', fontsize=16, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, 'focused_spatial_neighbor_evolution.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Neighbor evolution plot saved to: {output_path}")
    
    def plot_adoption_context(self, output_dir):
        """Create dual adoption context plots."""
        if self.context_data.empty:
            print("Cannot create adoption context plots: No context data")
            return
        
        # Create 2x1 subplot for spatial and class adoption context
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        scenarios_to_plot = ['rational', 'herding', 'all_biases']
        
        # Plot 1: Spatial Adoption Context
        ax1 = axes[0]
        for scenario in scenarios_to_plot:
            if scenario in self.context_data['scenario'].values:
                scenario_data = self.context_data[self.context_data['scenario'] == scenario]
                if not scenario_data.empty:
                    ax1.scatter(scenario_data['spatial_adoption_rate'], 
                               scenario_data['adoption_year'],
                               label=self.metadata[scenario]['display_name'],
                               color=self.colors[scenario], alpha=0.6, s=50)
        
        ax1.set_xlabel('Spatial Adoption Rate (Neighbors)')
        ax1.set_ylabel('Adoption Year')
        ax1.set_title('Spatial Context at Adoption Time', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Class Adoption Context
        ax2 = axes[1]
        for scenario in scenarios_to_plot:
            if scenario in self.context_data['scenario'].values:
                scenario_data = self.context_data[self.context_data['scenario'] == scenario]
                if not scenario_data.empty:
                    ax2.scatter(scenario_data['class_adoption_rate'], 
                               scenario_data['adoption_year'],
                               label=self.metadata[scenario]['display_name'],
                               color=self.colors[scenario], alpha=0.6, s=50)
        
        ax2.set_xlabel('Income Class Adoption Rate')
        ax2.set_ylabel('Adoption Year')
        ax2.set_title('Income Class Context at Adoption Time', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, 'focused_spatial_adoption_context.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Adoption context plots saved to: {output_path}")
    
    def plot_network_snapshots(self, output_dir):
        """CORRECTED: Create 4×3 grid network snapshots with temporal classification."""
        if self.snapshot_data.empty:
            print("Cannot create network snapshots: No snapshot data")
            return
        
        years_available = sorted(self.snapshot_data['Year'].unique())
        scenarios_to_plot = ['rational', 'herding', 'all_biases']
        
        # Create 4×3 grid
        fig, axes = plt.subplots(4, 3, figsize=(18, 20))
        
        # CORRECTED: Node colors for temporal classification
        node_colors = {
            'non_prosumers': '#1f77b4',      # Blue - Non-adopters
            'new_prosumers': '#ff7f0e',      # Orange - Adopted current month
            'old_prosumers': '#2ca02c'       # Green - Adopted previous month
        }
        
        for year_idx, year in enumerate(years_available[:4]):  # Max 4 years
            year_data = self.snapshot_data[self.snapshot_data['Year'] == year]
            
            for scenario_idx, scenario in enumerate(scenarios_to_plot):
                if scenario_idx < 3:  # Max 3 scenarios
                    ax = axes[year_idx, scenario_idx]
                    
                    scenario_data = year_data[year_data['Scenario'] == scenario]
                    
                    if not scenario_data.empty:
                        # Create a simple network layout
                        G = nx.Graph()
                        
                        # Add nodes by temporal category
                        node_positions = {}
                        color_list = []
                        
                        for node_type in ['non_prosumers', 'new_prosumers', 'old_prosumers']:
                            type_data = scenario_data[scenario_data['NodeType'] == node_type]
                            household_ids = type_data['HouseholdId'].tolist()
                            
                            for i, hh_id in enumerate(household_ids[:15]):  # Limit to 15 nodes per type
                                G.add_node(hh_id)
                                # Simple circular layout by type
                                angle = 2 * np.pi * i / max(len(household_ids[:15]), 1)
                                
                                if node_type == 'non_prosumers':
                                    # Outer ring for non-prosumers
                                    node_positions[hh_id] = (np.cos(angle), np.sin(angle))
                                elif node_type == 'new_prosumers':
                                    # Middle ring for new prosumers
                                    node_positions[hh_id] = (0.6 * np.cos(angle), 0.6 * np.sin(angle))
                                else:  # old_prosumers
                                    # Inner ring for old prosumers
                                    node_positions[hh_id] = (0.3 * np.cos(angle), 0.3 * np.sin(angle))
                                
                                color_list.append(node_colors[node_type])
                        
                        # Draw the network
                        if G.nodes():
                            nx.draw(G, pos=node_positions, ax=ax, 
                                   node_color=color_list, node_size=40, 
                                   with_labels=False, edge_color='gray', alpha=0.7)
                    
                    ax.set_title(f'Year {year} - {self.metadata[scenario]["display_name"]}', 
                               fontsize=10, fontweight='bold')
                    ax.set_aspect('equal')
        
        # CORRECTED: Add legend with temporal classification
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor=color, markersize=10, label=label)
                          for label, color in [
                              ('Non-Prosumers', node_colors['non_prosumers']),
                              ('New Prosumers (Current Month)', node_colors['new_prosumers']),
                              ('Old Prosumers (Previous Month)', node_colors['old_prosumers'])
                          ]]
        
        fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.95), 
                  ncol=3, fontsize=12)
        
        plt.suptitle('Network Snapshots: Temporal Adoption Patterns by Scenario and Year', 
                    fontsize=18, fontweight='bold', y=0.97)
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        
        output_path = os.path.join(output_dir, 'focused_spatial_network_snapshots_temporal.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Network snapshots (temporal classification) saved to: {output_path}")


# =============================================================================
# TESTING FUNCTIONS
# =============================================================================

def test_corrected_focused_spatial_visualizer():
    """Test the corrected FocusedVisualizer class."""
    print("Testing corrected FocusedVisualizer...")
    
    try:
        # Test with dummy corrected data
        import tempfile
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create dummy time series CSV files
            income_data = pd.DataFrame({
                'Year': [1, 1, 1, 2, 2, 2, 3, 3, 3],
                'Step': [12, 12, 12, 24, 24, 24, 36, 36, 36],
                'Scenario': ['rational', 'herding', 'all_biases'] * 3,
                'IncomeClass': [1, 1, 1, 2, 2, 2, 3, 3, 3],
                'AdoptionRate': [0.1, 0.15, 0.2, 0.3, 0.35, 0.4, 0.5, 0.55, 0.6]
            })
            income_data.to_csv(os.path.join(temp_dir, "focused_income_adoption_timeseries.csv"), index=False)
            
            neighbor_data = pd.DataFrame({
                'step': [1, 2, 3],
                'year': [1, 1, 1],
                'rational_avg_prosumer_neighbors': [0.5, 1.0, 1.5],
                'herding_avg_prosumer_neighbors': [0.6, 1.2, 1.8],
                'rational_avg_nonprosumer_prosumer_neighbors': [0.3, 0.8, 1.3],
                'herding_avg_nonprosumer_prosumer_neighbors': [0.4, 1.0, 1.6]
            })
            neighbor_data.to_csv(os.path.join(temp_dir, "focused_neighbor_evolution.csv"), index=False)
            
            # Create adoption context with proper values
            context_data = pd.DataFrame({
                'household_id': [1, 2, 3],
                'scenario': ['rational', 'herding', 'all_biases'],
                'spatial_adoption_rate': [0.3, 0.5, 0.7],
                'total_neighbors': [10, 10, 10],
                'adoption_year': [2, 3, 4]
            })
            context_data.to_csv(os.path.join(temp_dir, "focused_adoption_context.csv"), index=False)
            
            # Create temporal network snapshots
            snapshot_data = pd.DataFrame({
                'Year': [2, 2, 2, 5, 5, 5],
                'Scenario': ['rational', 'rational', 'herding', 'herding', 'all_biases', 'all_biases'],
                'NodeType': ['non_prosumers', 'new_prosumers', 'non_prosumers', 'old_prosumers', 'new_prosumers', 'old_prosumers'],
                'HouseholdId': [1, 2, 3, 4, 5, 6]
            })
            snapshot_data.to_csv(os.path.join(temp_dir, "focused_spatial_network_snapshots.csv"), index=False)
            
            # Test visualizer
            visualizer = FocusedSpatialVisualizer(data_path=temp_dir)
            
            if not visualizer.data_loaded:
                print("❌ Data not loaded from CSV")
                return False
            
            # Test creating plots
            output_dir = os.path.join(temp_dir, "visualizations")
            visualizer.create_all_focused_plots(output_dir)
            
            # Check if plot files were created
            expected_plots = [
                "focused_spatial_income_adoption_evolution.png",   # CORRECTED name
                "focused_spatial_neighbor_evolution.png",
                "focused_spatial_adoption_context.png",
                "focused_spatial_network_snapshots_temporal.png"  # CORRECTED name
            ]
            
            for plot_name in expected_plots:
                plot_path = os.path.join(output_dir, plot_name)
                if not os.path.exists(plot_path):
                    print(f"❌ Missing plot: {plot_name}")
                    return False
        
        print("✅ Corrected FocusedVisualizer tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Corrected FocusedVisualizer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_corrected_focused_spatial_visualizer()