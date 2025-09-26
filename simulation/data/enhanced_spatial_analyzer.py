# data/enhanced_spatial_analyzer.py - FIXED VERSION
"""
Enhanced Spatial-Social Analysis Framework - CORRECTED VERSION
==============================================================

Follows project visualizer patterns:
- Uses output_dir parameter (not save_path)
- Automatic filename generation  
- Method signature compatibility
- File saving like other visualizers
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import networkx as nx
from collections import defaultdict, Counter
import warnings
from ..utils.parameters import get_spatial_analysis_params, get_spatial_distance_threshold, get_dbscan_params, get_statistical_thresholds
warnings.filterwarnings('ignore')

class IntegratedSpatialAnalyzer:
    """
    Enhanced Spatial-Social Analysis integrated directly with MultiExperimentModel.
    
    CORRECTED: Follows project visualizer patterns with output_dir and automatic filenames.
    """
    
    def __init__(self, model):
        """Initialize analyzer with running model instance."""
        self.model = model
        self.households = model.get_households()
        self.network_graph = model.grid.G
        self.data_collector = model.data_collector
        
        # Load spatial analysis parameters from parameters.py
        self.spatial_params = get_spatial_analysis_params()
        self.distance_thresholds = self.spatial_params['distance_thresholds']
        self.statistical_params = get_statistical_thresholds()
        self.clustering_params = get_dbscan_params()
        
        # Analysis results storage
        self.results = {
            'spatial_autocorrelation': {},
            'homophily_metrics': {},
            'velocity_analysis': {},
            'clustering_results': {},
            'influence_decomposition': {}
        }
        
        print(f"IntegratedSpatialAnalyzer initialized")
        print(f"  Households: {len(self.households)}")
        print(f"  Network nodes: {len(self.network_graph.nodes()) if self.network_graph else 0}")
        print(f"  Available scenarios: {self._detect_scenarios()}")
    
    def _detect_scenarios(self):
        """Detect scenarios from MultiScenarioHousehold agents."""
        if not self.households:
            return []
        return list(self.households[0].scenario_adoption.keys())
    
    def _get_household_dataframe(self, target_step=None):
        """
        FIXED: Create household DataFrame with optional target_step parameter.
        
        Args:
            target_step: Step to evaluate adoption status at (default: current model step)
        """
        if target_step is None:
            target_step = getattr(self.model, 'current_step', self.model.schedule.steps)
        
        records = []
        
        for household in self.households:
            # Base record
            record = {
                'HouseholdID': household.unique_id,
                'PosX': household.pos[0] if household.pos else 0,
                'PosY': household.pos[1] if household.pos else 0,
                'IncomeClass': household.income_class,
                'Income': household.income,
                'Step': target_step,
                'AnalysisStep': target_step
            }
            
            # FIXED: Calculate adoption status dynamically from adoption_months
            for scenario in household.adoption_months:
                adoption_month = household.adoption_months.get(scenario, None)
                is_adopted = (adoption_month is not None and adoption_month <= target_step)
                
                record[f'IsProsumer_{scenario}'] = is_adopted
                record[f'AdoptionMonth_{scenario}'] = adoption_month
            
            # Legacy compatibility
            rational_adoption_month = household.adoption_months.get('rational', None)
            record['IsProsumer'] = (rational_adoption_month is not None and 
                                   rational_adoption_month <= target_step)
            
            records.append(record)
        
        return pd.DataFrame(records)
    
    def create_all_spatial_analyses(self, output_dir="results/spatial_analysis"):
        """
        FIXED: Create all spatial analyses with proper function calls.
        
        Args:
            output_dir: Directory to save all spatial analysis files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Creating comprehensive spatial analysis in {output_dir}...")
        
        # 1. Network Structure Analysis (RESTORED)
        network_results = self.create_social_network_analysis()
        
        # 2. Spatial Autocorrelation Analysis
        autocorr_results = self.calculate_spatial_autocorrelation()
        
        # 3. Income Homophily Analysis
        homophily_results = self.analyze_income_homophily_strength()
        
        # 4. Create visualizations following project pattern
        self.plot_integrated_social_grid(output_dir)
        self.plot_spatial_autocorrelation_analysis(output_dir)
        self.plot_income_homophily_analysis(output_dir)
        self.plot_temporal_spatial_evolution(output_dir)
        
        print(f"All spatial analyses completed and saved to {output_dir}")
        
        # Print diagnostic summary
        self._print_diagnostic_summary(network_results, autocorr_results, homophily_results)
    
    def create_social_network_analysis(self, scenarios=None, k_neighbors=None):
        """
        RESTORED: Analyze social network structure using existing NetworkBuilder graph.
        
        Args:
            scenarios: List of scenarios to analyze (default: all)
            k_neighbors: Number of neighbors for analysis (default: from parameters)
            
        Returns:
            Dictionary with network analysis results
        """
        if self.network_graph is None:
            print("No network graph found - creating basic spatial network")
            self._create_basic_spatial_network()
        
        if scenarios is None:
            scenarios = self._detect_scenarios()
        
        if k_neighbors is None:
            k_neighbors = self.spatial_params['network_analysis']['k_neighbors']
        
        print(f"Analyzing social network for scenarios: {scenarios}")
        print(f"Using k_neighbors: {k_neighbors} (from parameters)")
        
        # FIXED: Debug network structure first
        self._debug_network_structure()
        
        # Get household data
        household_df = self._get_household_dataframe()
        
        # Network structure analysis
        network_stats = {
            'nodes': self.network_graph.number_of_nodes(),
            'edges': self.network_graph.number_of_edges(),
            'density': nx.density(self.network_graph),
            'avg_clustering': nx.average_clustering(self.network_graph) if self.network_graph.number_of_edges() > 0 else 0,
            'avg_degree': sum(dict(self.network_graph.degree()).values()) / self.network_graph.number_of_nodes() if self.network_graph.number_of_nodes() > 0 else 0
        }
        
        # Identify edge types (spatial vs homophily)
        #edge_analysis = self._analyze_edge_types()
        
        results = {
            'network_statistics': network_stats,
            #'edge_analysis': edge_analysis,
            'household_data': household_df,
            'scenarios_analyzed': scenarios,
            'analysis_parameters': {
                'k_neighbors': k_neighbors,
                'distance_thresholds': self.distance_thresholds,
                'statistical_significance': self.statistical_params['significance_level']
            }
        }
        
        print(f"Network Analysis Complete:")
        print(f"  Nodes: {network_stats['nodes']:,}")
        print(f"  Edges: {network_stats['edges']:,}")
        print(f"  Density: {network_stats['density']:.4f}")
        
        return results
    
    def _debug_network_structure(self):
        """Enhanced debug to identify exact mismatch between network and households."""
        print(f"\n--- ENHANCED NETWORK STRUCTURE DEBUG ---")
        
        # Network information
        print(f"NetworkGraph type: {type(self.network_graph)}")
        print(f"Network nodes: {self.network_graph.number_of_nodes()}")
        print(f"Network edges: {self.network_graph.number_of_edges()}")
        
        # Household information
        print(f"Households count: {len(self.households)}")
        print(f"Household IDs: {[h.unique_id for h in self.households[:10]]}...")
        
        # Check network node IDs vs household IDs
        network_node_ids = set(self.network_graph.nodes())
        household_ids = set(h.unique_id for h in self.households)
        
        print(f"Network node IDs (first 10): {list(network_node_ids)[:10]}")
        print(f"Household IDs (first 10): {list(household_ids)[:10]}")
        
        # Find mismatches
        nodes_not_in_households = network_node_ids - household_ids
        households_not_in_network = household_ids - network_node_ids
        
        if nodes_not_in_households:
            print(f"❌ Network nodes WITHOUT corresponding households: {list(nodes_not_in_households)[:10]}")
        
        if households_not_in_network:
            print(f"❌ Households NOT in network: {list(households_not_in_network)[:10]}")
        
        if not nodes_not_in_households and not households_not_in_network:
            print(f"✅ Network nodes and household IDs match perfectly")
        
        # Check position attributes on households
        households_with_pos = sum(1 for h in self.households if hasattr(h, 'pos') and h.pos is not None)
        print(f"Households with pos attribute: {households_with_pos}/{len(self.households)}")
        
        # Sample position values
        print(f"Sample household positions:")
        for household in self.households[:5]:
            pos_value = getattr(household, 'pos', 'NO_POS_ATTR')
            pos_type = type(pos_value) if hasattr(household, 'pos') else 'N/A'
            print(f"  Household {household.unique_id}: pos={pos_value}, type={pos_type}")
        
        # Check network node data structure
        print(f"Sample network node data:")
        sample_nodes = list(self.network_graph.nodes(data=True))[:3]
        for node_id, node_data in sample_nodes:
            print(f"  Node {node_id}: keys={list(node_data.keys())}")
            if 'agent' in node_data:
                agent = node_data['agent']
                agent_pos = getattr(agent, 'pos', 'NO_POS') if agent else 'NO_AGENT'
                print(f"    Agent pos: {agent_pos}")
        
        print(f"--- END ENHANCED DEBUG ---\n")
    
    def plot_integrated_social_grid(self, output_dir, scenario='herding', target_step=None):
        """
        FIXED: Create spatial network visualization with corrected position handling.
        
        Args:
            output_dir: Directory to save visualization
            scenario: Scenario to visualize
            target_step: Time step to analyze (default: current step)
        """
        if target_step is None:
            target_step = getattr(self.model, 'current_step', self.model.schedule.steps)
        
        household_df = self._get_household_dataframe(target_step=target_step)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # FIXED: Get positions correctly from network structure
        pos = self._get_network_positions()
        
        if not pos:
            print("Warning: No network positions found. Cannot create spatial visualization.")
            plt.close(fig)
            return None
        
        print(f"Found {len(pos)} node positions for visualization")
        
        # Plot 1: Network Structure
        if self.network_graph and pos:
            ax1.set_title('Social Network Structure', fontsize=14, fontweight='bold')
            
            # Draw edges
            try:
                nx.draw_networkx_edges(self.network_graph, pos, edge_color='gray', 
                                      alpha=0.4, width=0.5, ax=ax1)
                
                # Color nodes by income class
                income_colors = plt.cm.viridis(np.linspace(0, 1, 5))
                node_colors = []
                
                for node in self.network_graph.nodes():
                    household = self._get_household_by_id(node)
                    if household:
                        color_idx = min(max(household.income_class - 1, 0), 4)  # Clamp to 0-4
                        node_colors.append(income_colors[color_idx])
                    else:
                        node_colors.append('gray')
                
                nx.draw_networkx_nodes(self.network_graph, pos, node_color=node_colors, 
                                      node_size=40, alpha=0.8, ax=ax1)
                ax1.set_aspect('equal')
                ax1.grid(True, alpha=0.3)
            
            except Exception as e:
                print(f"Error drawing network: {e}")
                ax1.text(0.5, 0.5, f'Network visualization error:\n{str(e)}', 
                        transform=ax1.transAxes, ha='center', va='center')
        
        # Plot 2: Adoption Pattern
        adoption_col = f'IsProsumer_{scenario}'
        
        if adoption_col in household_df.columns and self.network_graph and pos:
            ax2.set_title(f'Adoption Pattern ({scenario.title()}) - Step {target_step}', 
                         fontsize=14, fontweight='bold')
            
            try:
                adoption_dict = dict(zip(household_df['HouseholdID'], household_df[adoption_col]))
                node_colors = ['#2ca02c' if adoption_dict.get(node, 0) else '#d62728' 
                              for node in self.network_graph.nodes()]
                
                nx.draw_networkx_edges(self.network_graph, pos, edge_color='gray', 
                                      alpha=0.3, width=0.3, ax=ax2)
                nx.draw_networkx_nodes(self.network_graph, pos, node_color=node_colors, 
                                      node_size=50, alpha=0.8, ax=ax2)
                
                adoption_rate = household_df[adoption_col].mean()
                ax2.text(0.05, 0.95, f'Adoption Rate: {adoption_rate:.1%}\n(at step {target_step})', 
                        transform=ax2.transAxes, fontsize=12, fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                ax2.set_aspect('equal')
                ax2.grid(True, alpha=0.3)
            
            except Exception as e:
                print(f"Error drawing adoption pattern: {e}")
                ax2.text(0.5, 0.5, f'Adoption pattern error:\n{str(e)}', 
                        transform=ax2.transAxes, ha='center', va='center')
        
        # Plot 3: Income Distribution
        ax3.set_title('Income Class Distribution', fontsize=14, fontweight='bold')
        
        class_counts = household_df['IncomeClass'].value_counts().sort_index()
        income_colors = plt.cm.viridis(np.linspace(0, 1, 5))
        bars = ax3.bar(class_counts.index, class_counts.values, 
                      color=[income_colors[i-1] for i in class_counts.index], alpha=0.7)
        
        ax3.set_xlabel('Income Class')
        ax3.set_ylabel('Number of Households')
        ax3.grid(True, alpha=0.3)
        
        # Add percentages
        total = len(household_df)
        for bar, count in zip(bars, class_counts.values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + total*0.01,
                    f'{count/total:.1%}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 4: Analysis Summary with diagnostics
        ax4.set_title('Analysis Summary & Diagnostics', fontsize=14, fontweight='bold')
        
        summary_stats = [
            f'Analysis Step: {target_step}',
            f'Year: {(target_step // 12) + 1}',
            f'Total Households: {len(household_df):,}',
            f'Network Nodes: {self.network_graph.number_of_nodes() if self.network_graph else 0}',
            f'Network Edges: {self.network_graph.number_of_edges() if self.network_graph else 0}',
            f'Positions Found: {len(pos)}',
            f'Scenarios Available: {len(self._detect_scenarios())}'
        ]
        
        for i, stat in enumerate(summary_stats):
            ax4.text(0.1, 0.9 - i*0.12, stat, transform=ax4.transAxes,
                    fontsize=10, fontweight='bold')
        ax4.axis('off')
        
        plt.tight_layout()
        fig.suptitle(f'Spatial Social Grid Analysis - {scenario.title()} Scenario\nStep {target_step} ({len(household_df):,} Households)', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        # Save with automatic filename
        output_path = os.path.join(output_dir, f'spatial_social_grid_{scenario}_step_{target_step}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Spatial social grid saved to: {output_path}")
        return output_path
    
    def _get_network_positions(self):
        """
        CORRECTED: Get node positions using Method 2 (direct household lookup) only.
        NetworkBuilder sets household.pos directly, so we should retrieve from there.
        
        Returns:
            dict: {node_id: (x, y)} position mapping
        """
        pos = {}
        
        if self.network_graph is None:
            return pos
        
        print(f"\n--- POSITION RETRIEVAL DEBUG (Method 2 Only) ---")
        print(f"Network nodes: {self.network_graph.number_of_nodes()}")
        print(f"Available households: {len(self.households)}")
        
        # Debug: Check a few household positions first
        print(f"Sample household positions:")
        for i, household in enumerate(self.households[:3]):
            print(f"  Household {household.unique_id}: pos = {getattr(household, 'pos', 'MISSING')}")
        
        # Method 2: Direct household lookup (preferred method)
        successful_positions = 0
        missing_positions = []
        
        for node_id in self.network_graph.nodes():
            # Find household by ID
            household = None
            for h in self.households:
                if h.unique_id == node_id:
                    household = h
                    break
            
            if household is None:
                print(f"  ERROR: No household found for node {node_id}")
                missing_positions.append(node_id)
                continue
                
            # Check if household has position
            if hasattr(household, 'pos') and household.pos is not None:
                pos[node_id] = household.pos
                successful_positions += 1
            else:
                print(f"  ERROR: Household {node_id} missing pos attribute: {getattr(household, 'pos', 'NO_ATTR')}")
                missing_positions.append(node_id)
        
        print(f"Position retrieval results:")
        print(f"  Successful: {successful_positions}/{self.network_graph.number_of_nodes()}")
        print(f"  Missing: {len(missing_positions)} nodes")
        if missing_positions:
            print(f"  Missing node IDs: {missing_positions[:10]}..." if len(missing_positions) > 10 else f"  Missing node IDs: {missing_positions}")
        
        # Debug: Verify position format
        if pos:
            sample_positions = list(pos.items())[:3]
            print(f"Sample retrieved positions:")
            for node_id, position in sample_positions:
                print(f"  Node {node_id}: {position} (type: {type(position)})")
        
        print(f"--- END POSITION DEBUG ---\n")
        
        return pos
    
    def _get_household_by_id(self, household_id):
        """
        Get household agent by ID with enhanced debugging.
        
        Args:
            household_id: Household unique ID
            
        Returns:
            MultiScenarioHousehold or None
        """
        for household in self.households:
            if household.unique_id == household_id:
                return household
        
        # Debug: If not found, print available IDs for debugging
        available_ids = [h.unique_id for h in self.households[:10]]
        print(f"WARNING: Household {household_id} not found. Available IDs: {available_ids}...")
        return None
    
    def _print_diagnostic_summary(self, network_results, autocorr_results, homophily_results):
        """Print diagnostic summary of analysis results."""
        print(f"\n--- SPATIAL ANALYSIS DIAGNOSTIC SUMMARY ---")
        
        # Network diagnostics
        if network_results:
            print(f"Network Structure:")
            print(f"  Density: {network_results['network_statistics']['density']:.4f}")
            print(f"  Avg Clustering: {network_results['network_statistics']['avg_clustering']:.4f}")
        
        # Autocorrelation diagnostics
        if autocorr_results:
            print(f"Spatial Autocorrelation:")
            non_zero_morans = {s: r['I'] for s, r in autocorr_results.items() if abs(r['I']) > 0.001}
            if non_zero_morans:
                print(f"  Scenarios with spatial clustering: {list(non_zero_morans.keys())}")
                for scenario, morans_i in non_zero_morans.items():
                    print(f"    {scenario}: {morans_i:.4f}")
            else:
                print(f"  WARNING: No spatial clustering detected (all Moran's I ≈ 0)")
                print(f"  This could indicate:")
                print(f"    - Insufficient adoption variation between scenarios")
                print(f"    - Network structure issues")
                print(f"    - All households adopting in similar patterns")
        
        # Homophily diagnostics
        if homophily_results:
            print(f"Income Homophily:")
            valid_cramers = {s: r['cramers_v'] for s, r in homophily_results.items() 
                           if not pd.isna(r['cramers_v'])}
            if valid_cramers:
                print(f"  Scenarios with valid Cramer's V: {list(valid_cramers.keys())}")
            else:
                print(f"  WARNING: Most Cramer's V values are NaN")
                print(f"  This indicates insufficient adoption variation within income classes")
        
        print(f"--- END DIAGNOSTIC SUMMARY ---\n")
    
    
    def plot_spatial_autocorrelation_analysis(self, output_dir):
        """
        Create spatial autocorrelation analysis visualization.
        
        Args:
            output_dir: Directory to save visualization
        """
        if 'spatial_autocorrelation' not in self.results or not self.results['spatial_autocorrelation']:
            print("No spatial autocorrelation results found. Running analysis...")
            self.calculate_spatial_autocorrelation()
        
        results = self.results['spatial_autocorrelation']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Moran's I values
        scenarios = list(results.keys())
        morans_i_values = [results[s]['I'] for s in scenarios]
        p_values = [results[s]['p_value'] for s in scenarios]
        
        bars = ax1.bar(scenarios, morans_i_values, 
                      color=['blue' if p < 0.05 else 'gray' for p in p_values], alpha=0.7)
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax1.set_ylabel("Moran's I")
        ax1.set_title('Spatial Autocorrelation (Moran\'s I)', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add significance indicators
        for i, (bar, p_val) in enumerate(zip(bars, p_values)):
            significance = "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{morans_i_values[i]:.3f}{significance}', 
                   ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: Statistical significance
        ax2.bar(scenarios, [-np.log10(p) for p in p_values], 
               color=['red' if p < 0.05 else 'orange' for p in p_values], alpha=0.7)
        ax2.axhline(y=-np.log10(0.05), color='black', linestyle='--', alpha=0.7, 
                   label='p=0.05 threshold')
        ax2.set_ylabel('-log₁₀(p-value)')
        ax2.set_title('Statistical Significance', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save with automatic filename
        output_path = os.path.join(output_dir, 'spatial_autocorrelation_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Spatial autocorrelation analysis saved to: {output_path}")
        return output_path
    
    def plot_income_homophily_analysis(self, output_dir):
        """
        Create income homophily analysis visualization.
        
        Args:
            output_dir: Directory to save visualization
        """
        if 'homophily_metrics' not in self.results or not self.results['homophily_metrics']:
            print("No homophily results found. Running analysis...")
            self.analyze_income_homophily_strength()
        
        results = self.results['homophily_metrics']
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        scenarios = list(results.keys())
        
        # Plot 1: EI Index
        ei_values = [results[s]['ei_index'] for s in scenarios]
        bars1 = ax1.bar(scenarios, ei_values, alpha=0.7, color='lightcoral')
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax1.set_ylabel('EI Index')
        ax1.set_title('External-Internal Index\n(Closer to -1 = More Homophily)', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Cramer's V
        cramers_v_values = [results[s]['cramers_v'] for s in scenarios]
        bars2 = ax2.bar(scenarios, cramers_v_values, alpha=0.7, color='lightblue')
        ax2.set_ylabel('Cramer\'s V')
        ax2.set_title('Effect Size (Cramer\'s V)', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Adoption by Income Class (first scenario)
        if scenarios:
            first_scenario = scenarios[0]
            adoption_by_class = results[first_scenario]['adoption_by_class']
            
            ax3.bar(adoption_by_class['IncomeClass'], adoption_by_class['adoption_rate'], 
                   alpha=0.7, color=plt.cm.viridis(np.linspace(0, 1, len(adoption_by_class))))
            ax3.set_xlabel('Income Class')
            ax3.set_ylabel('Adoption Rate')
            ax3.set_title(f'Adoption by Income Class\n({first_scenario.title()})', fontweight='bold')
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Significance Summary
        significant_scenarios = [s for s in scenarios if results[s]['significant_homophily']]
        significance_counts = [len(significant_scenarios), len(scenarios) - len(significant_scenarios)]
        labels = ['Significant', 'Not Significant']
        
        ax4.pie(significance_counts, labels=labels, autopct='%1.0f%%', startangle=90)
        ax4.set_title('Homophily Significance\n(Chi-square test)', fontweight='bold')
        
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, 'income_homophily_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Income homophily analysis saved to: {output_path}")
        return output_path
    
    def plot_temporal_spatial_evolution(self, output_dir, scenario='herding'):
        """
        FIXED: Create temporal evolution using the working _get_network_positions method.
        
        Args:
            output_dir: Directory to save visualization
            scenario: Scenario to analyze
        """
        analysis_points = self.spatial_params['temporal_analysis']['default_analysis_points']
        max_step = getattr(self.model, 'current_step', self.model.schedule.steps)
        
        valid_points = [step for step in analysis_points if step <= max_step]
        
        if len(valid_points) < 2:
            print(f"Insufficient temporal data points for evolution analysis")
            return None
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # FIXED: Use the working _get_network_positions method instead of broken inline logic
        pos = self._get_network_positions()
        
        print(f"Temporal evolution using {len(pos)} positions for {self.network_graph.number_of_nodes()} nodes")
        
        # Verify we have all positions before proceeding
        if len(pos) != self.network_graph.number_of_nodes():
            print(f"ERROR: Position count mismatch in temporal evolution!")
            print(f"  Network nodes: {self.network_graph.number_of_nodes()}")
            print(f"  Positions: {len(pos)}")
            return None
        
        for i, step in enumerate(valid_points[:6]):  # Max 6 time points
            if i >= len(axes):
                break
                
            ax = axes[i]
            household_df = self._get_household_dataframe(target_step=step)
            
            if self.network_graph and f'IsProsumer_{scenario}' in household_df.columns:
                try:
                    adoption_dict = dict(zip(household_df['HouseholdID'], household_df[f'IsProsumer_{scenario}']))
                    node_colors = ['#2ca02c' if adoption_dict.get(node, 0) else '#d62728' 
                                for node in self.network_graph.nodes()]
                    
                    # This should work now since pos has all nodes
                    nx.draw_networkx_edges(self.network_graph, pos, edge_color='gray', 
                                        alpha=0.2, width=0.3, ax=ax)
                    nx.draw_networkx_nodes(self.network_graph, pos, node_color=node_colors, 
                                        node_size=30, alpha=0.8, ax=ax)
                    
                    adoption_rate = household_df[f'IsProsumer_{scenario}'].mean()
                    year = (step // 12) + 1
                    ax.set_title(f'Year {year} (Step {step})\nAdoption: {adoption_rate:.1%}', 
                            fontsize=12, fontweight='bold')
                    ax.set_aspect('equal')
                    ax.axis('off')
                    
                    print(f"  ✅ Successfully plotted step {step}")
                    
                except Exception as e:
                    print(f"  ❌ Error plotting step {step}: {e}")
                    ax.text(0.5, 0.5, f'Error Step {step}\n{str(e)}', 
                        transform=ax.transAxes, ha='center', va='center')
                    ax.axis('off')
        
        # Hide unused subplots
        for i in range(len(valid_points), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        fig.suptitle(f'Temporal Spatial Evolution - {scenario.title()} Scenario', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        output_path = os.path.join(output_dir, f'temporal_spatial_evolution_{scenario}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Temporal spatial evolution saved to: {output_path}")
        return output_path
    
    def calculate_spatial_autocorrelation(self, scenarios=None, target_step=None):
        """Calculate Moran's I spatial autocorrelation for adoption patterns."""
        if scenarios is None:
            scenarios = self._detect_scenarios()
        
        if target_step is None:
            target_step = getattr(self.model, 'current_step', self.model.schedule.steps)
        
        print(f"Calculating spatial autocorrelation at step {target_step} for {len(scenarios)} scenarios...")
        
        results = {}
        household_df = self._get_household_dataframe(target_step=target_step)
        
        for scenario in scenarios:
            adoption_col = f'IsProsumer_{scenario}'
            
            if adoption_col not in household_df.columns:
                print(f"Warning: {adoption_col} not found")
                continue
            
            morans_i = self._calculate_morans_i_from_network(household_df, adoption_col)
            morans_i['analysis_step'] = target_step
            
            results[scenario] = morans_i
            
            print(f"  {scenario}: Moran's I = {morans_i['I']:.4f} (p={morans_i['p_value']:.4f})")
        
        self.results['spatial_autocorrelation'] = results
        return results
    
    def analyze_income_homophily_strength(self, scenarios=None, target_step=None):
        """Quantify income homophily effects."""
        if scenarios is None:
            scenarios = self._detect_scenarios()
        
        if target_step is None:
            target_step = getattr(self.model, 'current_step', self.model.schedule.steps)
        
        print(f"Analyzing income homophily at step {target_step} for {len(scenarios)} scenarios...")
        
        results = {}
        household_df = self._get_household_dataframe(target_step=target_step)
        
        for scenario in scenarios:
            adoption_col = f'IsProsumer_{scenario}'
            
            if adoption_col not in household_df.columns:
                continue
            
            # Calculate EI index
            ei_index = self._calculate_ei_index(household_df, adoption_col)
            
            # Adoption rates by income class
            adoption_by_class = household_df.groupby('IncomeClass')[adoption_col].agg(['mean', 'count']).reset_index()
            adoption_by_class.columns = ['IncomeClass', 'adoption_rate', 'count']
            
            # Chi-square test
            contingency_table = pd.crosstab(household_df['IncomeClass'], household_df[adoption_col])
            chi2, chi2_p, dof, expected = stats.chi2_contingency(contingency_table)
            
            # Effect size (Cramer's V)
            n = contingency_table.sum().sum()
            cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))
            
            results[scenario] = {
                'ei_index': ei_index,
                'homophily_strength': -ei_index,
                'adoption_by_class': adoption_by_class,
                'chi2_statistic': chi2,
                'chi2_p_value': chi2_p,
                'cramers_v': cramers_v,
                'significant_homophily': chi2_p < 0.05,
                'analysis_step': target_step
            }
            
            print(f"  {scenario}: EI = {ei_index:.3f}, Cramer's V = {cramers_v:.3f}")
        
        self.results['homophily_metrics'] = results
        return results
    
    def _calculate_morans_i_from_network(self, household_df, adoption_col):
        """Calculate Moran's I using NetworkX graph structure."""
        if self.network_graph is None:
            return {'I': 0, 'expected_I': 0, 'z_score': 0, 'p_value': 1.0, 'significant': False}
        
        nodes = list(self.network_graph.nodes())
        n = len(nodes)
        
        # Get adoption values in node order
        adoption_dict = dict(zip(household_df['HouseholdID'], household_df[adoption_col]))
        y = np.array([adoption_dict.get(node, 0) for node in nodes], dtype=float)
        
        # Create weighted adjacency matrix
        W = nx.adjacency_matrix(self.network_graph, nodelist=nodes, weight='weight').astype(float)
        W = W.toarray()
        
        # Row-normalize weights
        row_sums = W.sum(axis=1)
        W = np.divide(W, row_sums[:, np.newaxis], out=np.zeros_like(W), where=row_sums[:, np.newaxis]!=0)
        
        # Calculate Moran's I
        y_mean = np.mean(y)
        y_centered = y - y_mean
        
        numerator = np.sum(W * np.outer(y_centered, y_centered))
        denominator = np.sum(y_centered**2)
        
        S0 = np.sum(W)
        I = (n / S0) * (numerator / denominator) if S0 > 0 and denominator > 0 else 0
        
        # Expected value and significance testing
        expected_I = -1 / (n - 1)
        
        # Simplified variance calculation
        S1 = 0.5 * np.sum((W + W.T)**2)
        S2 = np.sum((W.sum(axis=1) + W.sum(axis=0))**2)
        
        var_I = ((n * ((n**2 - 3*n + 3) * S1 - n * S2 + 3 * S0**2)) / 
                ((n - 1) * (n - 2) * (n - 3) * S0**2) - expected_I**2)
        
        if var_I > 0:
            z_score = (I - expected_I) / np.sqrt(var_I)
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        else:
            z_score = 0
            p_value = 1.0
        
        return {
            'I': I,
            'expected_I': expected_I,
            'variance': var_I,
            'z_score': z_score,
            'p_value': p_value,
            'significant': p_value < self.statistical_params['significance_level']
        }
    
    def _calculate_ei_index(self, household_df, adoption_col):
        """Calculate EI index for income homophily."""
        total_internal = 0
        total_external = 0
        
        adopters = household_df[household_df[adoption_col] == True]
        
        for _, household in adopters.iterrows():
            household_class = household['IncomeClass']
            
            same_class_adopters = len(household_df[
                (household_df['IncomeClass'] == household_class) & 
                (household_df[adoption_col] == True) &
                (household_df['HouseholdID'] != household['HouseholdID'])
            ])
            
            diff_class_adopters = len(household_df[
                (household_df['IncomeClass'] != household_class) & 
                (household_df[adoption_col] == True)
            ])
            
            total_internal += same_class_adopters
            total_external += diff_class_adopters
        
        if (total_internal + total_external) > 0:
            ei_index = (total_external - total_internal) / (total_external + total_internal)
        else:
            ei_index = 0
        
        return ei_index


