# data/immediate_spatial_analyzer.py - FIXED VERSION
"""
Immediate Spatial Analysis Tools - CORRECTED VERSION
====================================================

Quick-start analysis tools following project patterns:
- Uses model integration (like enhanced_spatial_analyzer.py)
- Uses output_dir parameter (not save_path)
- Automatic filename generation  
- Method signature compatibility
- Focused on immediate insights with minimal setup

Scientific Foundation:
- Aral et al. (2009): "homophily explains >50% of perceived behavioral contagion"
- Bollinger & Gillingham (2012): 0.78 percentage point increase per neighbor adoption
- Graziano & Gillingham (2015): Spatial patterns in solar adoption
- McPherson et al. (2001): Birds of a feather homophily analysis
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from scipy.stats import chi2_contingency, pearsonr, spearmanr
import networkx as nx
from collections import defaultdict, Counter
import warnings
from ..utils.parameters import get_spatial_analysis_params, get_spatial_distance_threshold, get_dbscan_params, get_statistical_thresholds
warnings.filterwarnings('ignore')

class ImmediateSpatialAnalyzer:
    """
    Ready-to-use spatial analysis tools integrated with MultiExperimentModel.
    
    CORRECTED: Follows project visualizer patterns with output_dir and model integration.
    Designed for immediate insights with minimal setup required.
    """
    
    def __init__(self, model):
        """Initialize with running model instance."""
        self.model = model
        self.households = model.get_households()
        self.network_graph = model.grid.G
        
        # Load spatial analysis parameters from parameters.py
        self.spatial_params = get_spatial_analysis_params()
        self.distance_thresholds = self.spatial_params['distance_thresholds']
        self.statistical_params = get_statistical_thresholds()
        self.clustering_params = get_dbscan_params()
        
        # Analysis results storage
        self.results = {
            'network_analysis': {},
            'spatial_effects': {},
            'homophily_effects': {},
            'clustering_results': {},
            'velocity_metrics': {}
        }
        
        print(f"ImmediateSpatialAnalyzer initialized")
        print(f"  Households: {len(self.households)}")
        print(f"  Network nodes: {len(self.network_graph.nodes()) if self.network_graph else 0}")
        print(f"  Available scenarios: {self._detect_scenarios()}")
        
        # Validate network-household alignment
        self._validate_network_integration()
    
    def _detect_scenarios(self):
        """Detect scenarios from MultiScenarioHousehold agents."""
        if not self.households:
            return []
        return list(self.households[0].scenario_adoption.keys())
    
    def _validate_network_integration(self):
        """Validate that network and households are properly aligned."""
        if self.network_graph is None:
            print("WARNING: No network graph found")
            return False
        
        network_nodes = set(self.network_graph.nodes())
        household_ids = set(h.unique_id for h in self.households)
        
        missing_in_network = household_ids - network_nodes
        missing_in_households = network_nodes - household_ids
        
        if missing_in_network:
            print(f"WARNING: {len(missing_in_network)} households not in network")
        
        if missing_in_households:
            print(f"WARNING: {len(missing_in_households)} network nodes without households")
        
        return len(missing_in_network) == 0 and len(missing_in_households) == 0
    
    def _get_household_dataframe(self, target_step=None):
        """
        Create household DataFrame with optional target_step parameter.
        
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
            
            # Calculate adoption status dynamically from adoption_months
            for scenario in household.adoption_months:
                adoption_month = household.adoption_months.get(scenario, None)
                is_adopted = (adoption_month is not None and adoption_month <= target_step)
                
                record[f'IsProsumer_{scenario}'] = is_adopted
                record[f'AdoptionMonth_{scenario}'] = adoption_month
            
            records.append(record)
        
        return pd.DataFrame(records)
    
    def create_all_immediate_analyses(self, output_dir="results/immediate_spatial_analysis", max_households=500):
        """
        Create all immediate spatial analyses with automatic output handling.
        
        Args:
            output_dir: Directory to save all analysis files
            max_households: Maximum households for visualization (performance)
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Creating immediate spatial analysis in {output_dir}...")
        print(f"Performance limit: {max_households} households for visualization")
        
        # 1. Quick network overview
        self.create_network_overview(output_dir, max_households)
        
        # 2. Spatial vs homophily quantification
        self.quantify_spatial_vs_homophily_effects()
        
        # 3. Policy-relevant clustering
        self.identify_policy_clusters()
        
        # 4. Adoption velocity comparison
        self.compare_adoption_velocities()
        
        # 5. Create comprehensive summary
        self.create_immediate_summary_report(output_dir)
        
        print(f"All immediate analyses completed and saved to {output_dir}")
        
        # Print diagnostic summary
        self._print_analysis_summary()
    
    def create_network_overview(self, output_dir, max_households=500, scenario='herding'):
        """
        Create quick network structure overview following project patterns.
        
        Args:
            output_dir: Directory to save visualization
            max_households: Maximum households to visualize for performance
            scenario: Scenario to show adoption for
        """
        household_df = self._get_household_dataframe()
        
        # Sample data if too large for performance
        if len(household_df) > max_households:
            sample_df = household_df.sample(n=max_households, random_state=42)
            sample_household_ids = set(sample_df['HouseholdID'])
            print(f"Sampled {max_households} households from {len(household_df)} total")
        else:
            sample_df = household_df.copy()
            sample_household_ids = set(sample_df['HouseholdID'])
        
        # Create subgraph for sampled households
        if self.network_graph:
            subgraph_nodes = [node for node in self.network_graph.nodes() if node in sample_household_ids]
            network_subgraph = self.network_graph.subgraph(subgraph_nodes)
        else:
            print("No network graph available")
            return None
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Get positions from households
        pos = {}
        for _, row in sample_df.iterrows():
            pos[row['HouseholdID']] = (row['PosX'], row['PosY'])
        
        # Plot 1: Network Structure
        ax1.set_title('Social Network Structure', fontsize=14, fontweight='bold')
        
        if network_subgraph and pos:
            try:
                # Draw edges
                nx.draw_networkx_edges(network_subgraph, pos, edge_color='gray', 
                                      alpha=0.4, width=0.5, ax=ax1)
                
                # Color nodes by income class
                income_colors = plt.cm.viridis(np.linspace(0, 1, 5))
                node_colors = []
                
                for node in network_subgraph.nodes():
                    household_data = sample_df[sample_df['HouseholdID'] == node]
                    if not household_data.empty:
                        income_class = household_data['IncomeClass'].iloc[0]
                        color_idx = min(max(income_class - 1, 0), 4)  # Clamp to 0-4
                        node_colors.append(income_colors[color_idx])
                    else:
                        node_colors.append('gray')
                
                nx.draw_networkx_nodes(network_subgraph, pos, node_color=node_colors, 
                                      node_size=40, alpha=0.8, ax=ax1)
                
                ax1.set_aspect('equal')
                ax1.grid(True, alpha=0.3)
                
                # Add network stats
                n_nodes = network_subgraph.number_of_nodes()
                n_edges = network_subgraph.number_of_edges()
                avg_degree = 2 * n_edges / n_nodes if n_nodes > 0 else 0
                
                ax1.text(0.05, 0.95, f'Nodes: {n_nodes:,}\nEdges: {n_edges:,}\nAvg Degree: {avg_degree:.1f}', 
                        transform=ax1.transAxes, fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            except Exception as e:
                ax1.text(0.5, 0.5, f'Network visualization error:\n{str(e)}', 
                        transform=ax1.transAxes, ha='center', va='center')
        
        # Plot 2: Adoption Pattern
        adoption_col = f'IsProsumer_{scenario}'
        
        if adoption_col in sample_df.columns and network_subgraph and pos:
            ax2.set_title(f'Adoption Pattern ({scenario.title()})', fontsize=14, fontweight='bold')
            
            try:
                adoption_dict = dict(zip(sample_df['HouseholdID'], sample_df[adoption_col]))
                node_colors = ['#2ca02c' if adoption_dict.get(node, 0) else '#d62728' 
                              for node in network_subgraph.nodes()]
                
                nx.draw_networkx_edges(network_subgraph, pos, edge_color='gray', 
                                      alpha=0.3, width=0.3, ax=ax2)
                nx.draw_networkx_nodes(network_subgraph, pos, node_color=node_colors, 
                                      node_size=50, alpha=0.8, ax=ax2)
                
                adoption_rate = sample_df[adoption_col].mean()
                ax2.text(0.05, 0.95, f'Adoption Rate: {adoption_rate:.1%}', 
                        transform=ax2.transAxes, fontsize=12, fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                ax2.set_aspect('equal')
                ax2.grid(True, alpha=0.3)
            
            except Exception as e:
                ax2.text(0.5, 0.5, f'Adoption pattern error:\n{str(e)}', 
                        transform=ax2.transAxes, ha='center', va='center')
        
        # Plot 3: Income Distribution
        ax3.set_title('Income Class Distribution', fontsize=14, fontweight='bold')
        
        class_counts = sample_df['IncomeClass'].value_counts().sort_index()
        income_colors = plt.cm.viridis(np.linspace(0, 1, 5))
        bars = ax3.bar(class_counts.index, class_counts.values, 
                      color=[income_colors[i-1] for i in class_counts.index], alpha=0.7)
        
        ax3.set_xlabel('Income Class')
        ax3.set_ylabel('Number of Households')
        ax3.grid(True, alpha=0.3)
        
        # Add percentages
        total = len(sample_df)
        for bar, count in zip(bars, class_counts.values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + total*0.01,
                    f'{count/total:.1%}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 4: Quick Analysis Summary
        ax4.set_title('Quick Analysis Summary', fontsize=14, fontweight='bold')
        
        # Basic statistics
        current_step = getattr(self.model, 'current_step', self.model.schedule.steps)
        scenarios = self._detect_scenarios()
        
        summary_stats = [
            f'Analysis Step: {current_step}',
            f'Total Households: {len(self.households):,}',
            f'Visualized: {len(sample_df):,}',
            f'Network Nodes: {len(self.network_graph.nodes()) if self.network_graph else 0:,}',
            f'Network Edges: {len(self.network_graph.edges()) if self.network_graph else 0:,}',
            f'Available Scenarios: {len(scenarios)}',
            f'Analysis Ready: {"✅" if self._validate_network_integration() else "⚠️"}'
        ]
        
        for i, stat in enumerate(summary_stats):
            ax4.text(0.1, 0.9 - i*0.12, stat, transform=ax4.transAxes,
                    fontsize=11, fontweight='bold')
        ax4.axis('off')
        
        plt.tight_layout()
        fig.suptitle(f'Immediate Network Overview - {len(sample_df):,} Households', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        # Save with automatic filename
        output_path = os.path.join(output_dir, f'network_overview_{scenario}_step_{current_step}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Network overview saved to: {output_path}")
        
        # Store network statistics
        if network_subgraph:
            self.results['network_analysis'] = {
                'sampled_nodes': network_subgraph.number_of_nodes(),
                'sampled_edges': network_subgraph.number_of_edges(),
                'total_households': len(self.households),
                'network_density': nx.density(network_subgraph),
                'avg_clustering': nx.average_clustering(network_subgraph) if network_subgraph.number_of_edges() > 0 else 0
            }
        
        return output_path
    
    def quantify_spatial_vs_homophily_effects(self, scenarios=None):
        """
        Quantify spatial proximity vs. income homophily effects following Aral et al. (2009).
        
        Args:
            scenarios: List of scenarios to analyze (default: all detected)
            
        Returns:
            Dictionary with effect sizes and statistical significance
        """
        if scenarios is None:
            scenarios = self._detect_scenarios()
        
        print(f"Quantifying spatial vs homophily effects for {len(scenarios)} scenarios...")
        
        results = {}
        household_df = self._get_household_dataframe()
        
        for scenario in scenarios:
            adoption_col = f'IsProsumer_{scenario}'
            
            if adoption_col not in household_df.columns:
                print(f"Warning: {adoption_col} not found")
                continue
            
            print(f"  Analyzing {scenario}...")
            
            # 1. Spatial Proximity Effect
            spatial_effect = self._calculate_spatial_proximity_effect(household_df, adoption_col)
            
            # 2. Income Homophily Effect  
            homophily_effect = self._calculate_income_homophily_effect(household_df, adoption_col)
            
            # 3. Combined Effect Assessment (following Aral et al.)
            combined_effect = self._calculate_combined_effects(spatial_effect, homophily_effect)
            
            results[scenario] = {
                'spatial_proximity': spatial_effect,
                'income_homophily': homophily_effect,
                'combined_assessment': combined_effect
            }
            
            # Print results
            print(f"    Spatial effect: {spatial_effect['effect_size']:.3f} (p={spatial_effect['p_value']:.3f})")
            print(f"    Homophily effect: {homophily_effect['effect_size']:.3f} (p={homophily_effect['p_value']:.3f})")
            print(f"    Homophily explains {combined_effect['homophily_percentage']:.1f}% of correlation")
        
        self.results['spatial_effects'] = results
        return results
    
    def _calculate_spatial_proximity_effect(self, household_df, adoption_col):
        """Calculate spatial proximity effect on adoption using network structure."""
        if self.network_graph is None:
            return {'correlation': 0, 'p_value': 1, 'effect_size': 0, 'n_observations': 0}
        
        spatial_effects = []
        
        for _, household_row in household_df.iterrows():
            household_id = household_row['HouseholdID']
            
            if household_id not in self.network_graph:
                continue
            
            # Get spatial neighbors from network
            neighbors = list(self.network_graph.neighbors(household_id))
            
            if neighbors:
                # Calculate neighbor adoption rate
                neighbor_adoptions = []
                for neighbor_id in neighbors:
                    neighbor_data = household_df[household_df['HouseholdID'] == neighbor_id]
                    if not neighbor_data.empty:
                        neighbor_adoptions.append(neighbor_data[adoption_col].iloc[0])
                
                if neighbor_adoptions:
                    neighbor_rate = np.mean(neighbor_adoptions)
                    household_adoption = household_row[adoption_col]
                    spatial_effects.append((household_adoption, neighbor_rate))
        
        # Calculate correlation
        if len(spatial_effects) > 10:  # Need sufficient observations
            household_adoptions, neighbor_rates = zip(*spatial_effects)
            correlation, p_value = pearsonr(household_adoptions, neighbor_rates)
            
            # Effect size calculation (difference in neighbor rates)
            adopter_neighbor_rate = np.mean([nr for ha, nr in spatial_effects if ha == 1])
            non_adopter_neighbor_rate = np.mean([nr for ha, nr in spatial_effects if ha == 0])
            
            if adopter_neighbor_rate is not None and non_adopter_neighbor_rate is not None:
                effect_size = adopter_neighbor_rate - non_adopter_neighbor_rate
            else:
                effect_size = 0
        else:
            correlation, p_value, effect_size = 0, 1, 0
        
        return {
            'correlation': correlation,
            'p_value': p_value,
            'effect_size': effect_size,
            'n_observations': len(spatial_effects)
        }
    
    def _calculate_income_homophily_effect(self, household_df, adoption_col):
        """Calculate income homophily effect using chi-square analysis."""
        # Chi-square test for independence
        try:
            contingency_table = pd.crosstab(household_df['IncomeClass'], household_df[adoption_col])
            chi2, p_value, dof, expected = chi2_contingency(contingency_table)
            
            # Effect size (Cramer's V)
            n = contingency_table.sum().sum()
            cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))
            
            # Variance in adoption rates across income classes
            class_adoption_rates = household_df.groupby('IncomeClass')[adoption_col].mean()
            variance_between_classes = np.var(class_adoption_rates.values)
            
        except Exception as e:
            print(f"    Warning: Income homophily calculation failed: {e}")
            chi2, p_value, cramers_v, variance_between_classes = 0, 1, 0, 0
        
        return {
            'chi2_statistic': chi2,
            'p_value': p_value,
            'cramers_v': cramers_v,
            'effect_size': variance_between_classes
        }
    
    def _calculate_combined_effects(self, spatial_effect, homophily_effect):
        """
        Calculate combined effect assessment following Aral et al. methodology.
        
        Estimates what proportion of spatial correlation is due to homophily
        versus true spatial influence.
        """
        spatial_correlation = abs(spatial_effect['correlation'])
        homophily_strength = homophily_effect['cramers_v']
        
        # Estimate homophily contribution to spatial correlation
        if spatial_correlation > 0.001:  # Avoid division by very small numbers
            homophily_percentage = min(100, (homophily_strength / spatial_correlation) * 100)
        else:
            homophily_percentage = 0
        
        return {
            'spatial_correlation': spatial_correlation,
            'homophily_strength': homophily_strength,
            'homophily_percentage': homophily_percentage,
            'estimated_true_influence': max(0, spatial_correlation - homophily_strength)
        }
    
    def identify_policy_clusters(self, adoption_threshold=0.3, min_cluster_size=10):
        """
        Identify spatially-clustered groups with policy implications using DBSCAN.
        
        Args:
            adoption_threshold: Threshold for categorizing clusters as hot/cold spots
            min_cluster_size: Minimum households per cluster
            
        Returns:
            Dictionary with cluster analysis and policy recommendations
        """
        print(f"Identifying policy-relevant clusters...")
        
        household_df = self._get_household_dataframe()
        
        # Use spatial positions for clustering
        positions = household_df[['PosX', 'PosY']].values
        
        # Normalize positions for clustering
        scaler = StandardScaler()
        positions_scaled = scaler.fit_transform(positions)
        
        # Try different eps values to find optimal clustering
        best_score = -1
        best_eps = self.clustering_params['eps']
        
        eps_range = np.arange(0.3, 2.0, 0.1)
        for eps in eps_range:
            clustering = DBSCAN(eps=eps, min_samples=min_cluster_size)
            cluster_labels = clustering.fit_predict(positions_scaled)
            
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            if n_clusters >= 2:  # Need at least 2 clusters
                try:
                    score = silhouette_score(positions_scaled, cluster_labels)
                    if score > best_score:
                        best_score = score
                        best_eps = eps
                except:
                    continue
        
        # Final clustering with best parameters
        clustering = DBSCAN(eps=best_eps, min_samples=min_cluster_size)
        cluster_labels = clustering.fit_predict(positions_scaled)
        
        # Analyze clusters by adoption patterns
        cluster_analysis = {}
        unique_clusters = set(cluster_labels)
        
        if -1 in unique_clusters:
            unique_clusters.remove(-1)  # Remove noise points
        
        for cluster_id in unique_clusters:
            cluster_mask = cluster_labels == cluster_id
            cluster_data = household_df[cluster_mask]
            
            if len(cluster_data) < min_cluster_size:
                continue
            
            # Calculate cluster metrics for all scenarios
            cluster_metrics = {
                'household_count': len(cluster_data),
                'avg_income': cluster_data['Income'].mean(),
                'dominant_income_class': cluster_data['IncomeClass'].mode().iloc[0],
                'income_classes': cluster_data['IncomeClass'].value_counts().to_dict(),
                'center_position': (cluster_data['PosX'].mean(), cluster_data['PosY'].mean()),
                'adoption_rates': {}
            }
            
            # Calculate adoption rates for different scenarios
            for col in cluster_data.columns:
                if col.startswith('IsProsumer_'):
                    scenario = col.replace('IsProsumer_', '')
                    adoption_rate = cluster_data[col].mean()
                    cluster_metrics['adoption_rates'][scenario] = adoption_rate
            
            # Categorize cluster based on average adoption rate
            if cluster_metrics['adoption_rates']:
                avg_adoption = np.mean(list(cluster_metrics['adoption_rates'].values()))
                
                if avg_adoption >= adoption_threshold * 1.5:
                    category = 'hotspot'
                elif avg_adoption <= adoption_threshold * 0.5:
                    category = 'coldspot' 
                else:
                    category = 'mixed'
            else:
                category = 'unknown'
            
            cluster_metrics['category'] = category
            cluster_analysis[cluster_id] = cluster_metrics
        
        # Generate policy recommendations
        policy_recommendations = self._generate_cluster_policy_recommendations(cluster_analysis)
        
        results = {
            'cluster_labels': cluster_labels.tolist(),
            'cluster_analysis': cluster_analysis,
            'clustering_parameters': {
                'eps': best_eps, 
                'min_samples': min_cluster_size,
                'n_clusters': len(cluster_analysis)
            },
            'silhouette_score': best_score,
            'policy_recommendations': policy_recommendations
        }
        
        self.results['clustering_results'] = results
        
        # Print summary
        n_clusters = len(cluster_analysis)
        categories = Counter([c['category'] for c in cluster_analysis.values()])
        
        print(f"  Identified {n_clusters} clusters (silhouette score: {best_score:.3f})")
        print(f"  Policy categories: {dict(categories)}")
        
        return results
    
    def _generate_cluster_policy_recommendations(self, cluster_analysis):
        """Generate policy recommendations based on cluster analysis."""
        recommendations = {
            'hotspot_strategies': [],
            'coldspot_interventions': [], 
            'mixed_peer_programs': [],
            'summary': {}
        }
        
        for cluster_id, metrics in cluster_analysis.items():
            category = metrics['category']
            
            base_recommendation = {
                'cluster_id': cluster_id,
                'households': metrics['household_count'],
                'avg_income': metrics['avg_income'],
                'dominant_class': metrics['dominant_income_class'],
                'center_position': metrics['center_position']
            }
            
            if category == 'hotspot':
                rec = base_recommendation.copy()
                rec['recommendation'] = 'Showcase as demonstration site for peer learning and diffusion'
                rec['strategy'] = 'Leverage high adoption for education and awareness campaigns'
                recommendations['hotspot_strategies'].append(rec)
            
            elif category == 'coldspot':
                rec = base_recommendation.copy()
                rec['recommendation'] = 'Target with intensive outreach and financial incentives'
                rec['strategy'] = 'Address barriers through subsidies and personalized engagement'
                recommendations['coldspot_interventions'].append(rec)
            
            elif category == 'mixed':
                rec = base_recommendation.copy()
                rec['recommendation'] = 'Implement peer influence programs leveraging early adopters'
                rec['strategy'] = 'Use social networks to accelerate adoption through existing prosumers'
                recommendations['mixed_peer_programs'].append(rec)
        
        # Create summary statistics
        recommendations['summary'] = {
            'total_clusters': len(cluster_analysis),
            'hotspots': len(recommendations['hotspot_strategies']),
            'coldspots': len(recommendations['coldspot_interventions']),
            'mixed': len(recommendations['mixed_peer_programs']),
            'total_households_clustered': sum(m['household_count'] for m in cluster_analysis.values())
        }
        
        return recommendations
    
    def compare_adoption_velocities(self, scenarios=None):
        """
        Compare adoption penetration velocities between income classes and scenarios.
        
        Args:
            scenarios: List of scenarios to compare (default: all detected)
            
        Returns:
            Dictionary with velocity analysis results
        """
        if scenarios is None:
            scenarios = self._detect_scenarios()
        
        print(f"Comparing adoption velocities for {len(scenarios)} scenarios...")
        
        results = {}
        household_df = self._get_household_dataframe()
        
        for scenario in scenarios:
            adoption_col = f'IsProsumer_{scenario}'
            
            if adoption_col not in household_df.columns:
                continue
            
            print(f"  Analyzing velocity for {scenario}...")
            
            # Velocity by income class
            income_velocities = {}
            for income_class in sorted(household_df['IncomeClass'].unique()):
                class_data = household_df[household_df['IncomeClass'] == income_class]
                adoption_rate = class_data[adoption_col].mean()
                
                income_velocities[income_class] = {
                    'adoption_rate': adoption_rate,
                    'household_count': len(class_data),
                    'adopter_count': int(class_data[adoption_col].sum())
                }
            
            # Calculate velocity metrics
            velocity_metrics = self._calculate_velocity_metrics(income_velocities)
            
            results[scenario] = {
                'income_velocities': income_velocities,
                'velocity_metrics': velocity_metrics
            }
        
        # Cross-scenario comparison
        if len(results) >= 2:
            comparison = self._compare_scenario_velocities(results)
            results['cross_scenario_comparison'] = comparison
        
        self.results['velocity_metrics'] = results
        
        # Print summary
        for scenario, data in results.items():
            if scenario != 'cross_scenario_comparison':
                metrics = data['velocity_metrics']
                fastest_class, fastest_rate = metrics['fastest_class']
                slowest_class, slowest_rate = metrics['slowest_class']
                
                print(f"  {scenario}: Fastest=Class {fastest_class} ({fastest_rate:.1%}), "
                      f"Slowest=Class {slowest_class} ({slowest_rate:.1%})")
        
        return results
    
    def _calculate_velocity_metrics(self, income_velocities):
        """Calculate velocity-related metrics."""
        rates = [v['adoption_rate'] for v in income_velocities.values()]
        
        if not rates:
            return {'velocity_range': 0, 'velocity_coefficient_variation': 0,
                   'fastest_class': (0, 0), 'slowest_class': (0, 0)}
        
        fastest_item = max(income_velocities.items(), key=lambda x: x[1]['adoption_rate'])
        slowest_item = min(income_velocities.items(), key=lambda x: x[1]['adoption_rate'])
        
        return {
            'velocity_range': max(rates) - min(rates),
            'velocity_coefficient_variation': np.std(rates) / np.mean(rates) if np.mean(rates) > 0 else 0,
            'fastest_class': (fastest_item[0], fastest_item[1]['adoption_rate']),
            'slowest_class': (slowest_item[0], slowest_item[1]['adoption_rate'])
        }
    
    def _compare_scenario_velocities(self, results):
        """Compare velocities across scenarios."""
        scenario_keys = [k for k in results.keys() if k != 'cross_scenario_comparison']
        
        if len(scenario_keys) < 2:
            return None
        
        comparisons = {}
        for i, scenario_1 in enumerate(scenario_keys):
            for scenario_2 in scenario_keys[i+1:]:
                rates_1 = [v['adoption_rate'] for v in results[scenario_1]['income_velocities'].values()]
                rates_2 = [v['adoption_rate'] for v in results[scenario_2]['income_velocities'].values()]
                
                comparison_key = f"{scenario_1}_vs_{scenario_2}"
                comparisons[comparison_key] = {
                    'scenario_1': scenario_1,
                    'scenario_2': scenario_2,
                    'avg_rate_1': np.mean(rates_1),
                    'avg_rate_2': np.mean(rates_2),
                    'rate_difference': np.mean(rates_2) - np.mean(rates_1),
                    'variance_difference': np.var(rates_2) - np.var(rates_1)
                }
        
        return comparisons
    
    def create_immediate_summary_report(self, output_dir):
        """
        Create comprehensive summary report with all key findings.
        
        Args:
            output_dir: Directory to save the summary report
            
        Returns:
            Path to saved report file
        """
        print("Creating immediate summary report...")
        
        fig = plt.figure(figsize=(16, 12))
        
        # Create grid layout for subplots
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Plot 1: Network Overview (top-left)
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_network_summary(ax1)
        
        # Plot 2: Effect Sizes Comparison (top-middle-left) 
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_effect_sizes_summary(ax2)
        
        # Plot 3: Homophily Percentage (top-middle-right)
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_homophily_percentage_summary(ax3)
        
        # Plot 4: Adoption Rates by Scenario (top-right)
        ax4 = fig.add_subplot(gs[0, 3])
        self._plot_scenario_adoption_rates(ax4)
        
        # Plot 5-6: Policy Clusters (middle row, left half)
        ax5 = fig.add_subplot(gs[1, :2])
        self._plot_policy_clusters_summary(ax5)
        
        # Plot 7-8: Velocity Comparison (middle row, right half)
        ax7 = fig.add_subplot(gs[1, 2:])
        self._plot_velocity_summary(ax7)
        
        # Plot 9: Key Findings (bottom-left)
        ax9 = fig.add_subplot(gs[2, 0])
        self._plot_key_findings_summary(ax9)
        
        # Plot 10: Methodology (bottom-middle-left)
        ax10 = fig.add_subplot(gs[2, 1])
        self._plot_methodology_summary(ax10)
        
        # Plot 11: Policy Recommendations (bottom-middle-right)
        ax11 = fig.add_subplot(gs[2, 2])
        self._plot_policy_recommendations_summary(ax11)
        
        # Plot 12: Next Steps (bottom-right)
        ax12 = fig.add_subplot(gs[2, 3])
        self._plot_next_steps_summary(ax12)
        
        # Overall title
        current_step = getattr(self.model, 'current_step', self.model.schedule.steps)
        fig.suptitle(f'Immediate Spatial Analysis Report\nStep {current_step} - {len(self.households):,} Households', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        # Save report
        output_path = os.path.join(output_dir, f'immediate_spatial_analysis_report_step_{current_step}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Immediate summary report saved to: {output_path}")
        return output_path
    
    # Helper plotting methods for the summary report
    def _plot_network_summary(self, ax):
        """Plot network structure summary."""
        if 'network_analysis' in self.results:
            stats = self.results['network_analysis']
            labels = ['Network\nStructure']
            values = [stats.get('sampled_nodes', 0)]
            
            ax.bar(labels, values, alpha=0.7, color='lightblue')
            ax.set_ylabel('Nodes')
            ax.set_title('Network Overview', fontweight='bold', fontsize=10)
            
            # Add text annotations
            density = stats.get('network_density', 0)
            clustering = stats.get('avg_clustering', 0)
            ax.text(0.5, 0.8, f'Density: {density:.3f}', transform=ax.transAxes, 
                   ha='center', fontsize=9, fontweight='bold')
            ax.text(0.5, 0.7, f'Clustering: {clustering:.3f}', transform=ax.transAxes, 
                   ha='center', fontsize=9, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'Network\nAnalysis\nPending', transform=ax.transAxes, 
                   ha='center', va='center', fontweight='bold')
            ax.set_title('Network Overview', fontweight='bold', fontsize=10)
    
    def _plot_effect_sizes_summary(self, ax):
        """Plot spatial vs homophily effect sizes."""
        if 'spatial_effects' in self.results:
            scenarios = list(self.results['spatial_effects'].keys())[:3]  # Limit to first 3 for space
            
            spatial_effects = [self.results['spatial_effects'][s]['spatial_proximity']['effect_size'] 
                             for s in scenarios]
            homophily_effects = [self.results['spatial_effects'][s]['income_homophily']['effect_size'] 
                               for s in scenarios]
            
            x = np.arange(len(scenarios))
            width = 0.35
            
            ax.bar(x - width/2, spatial_effects, width, label='Spatial', alpha=0.7, color='orange')
            ax.bar(x + width/2, homophily_effects, width, label='Homophily', alpha=0.7, color='green')
            
            ax.set_ylabel('Effect Size')
            ax.set_title('Effect Comparison', fontweight='bold', fontsize=10)
            ax.set_xticks(x)
            ax.set_xticklabels([s[:8] for s in scenarios], rotation=45, fontsize=8)  # Truncate long names
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, 'Effect\nAnalysis\nPending', transform=ax.transAxes, 
                   ha='center', va='center', fontweight='bold')
            ax.set_title('Effect Comparison', fontweight='bold', fontsize=10)
    
    def _plot_homophily_percentage_summary(self, ax):
        """Plot homophily percentage following Aral et al."""
        if 'spatial_effects' in self.results:
            scenarios = list(self.results['spatial_effects'].keys())[:3]  # Limit for space
            homophily_percentages = [
                self.results['spatial_effects'][s]['combined_assessment']['homophily_percentage'] 
                for s in scenarios
            ]
            
            bars = ax.bar(scenarios, homophily_percentages, color='purple', alpha=0.7)
            ax.axhline(y=50, color='red', linestyle='--', alpha=0.7, linewidth=2)
            
            ax.set_ylabel('Percentage (%)')
            ax.set_title('Homophily\nExplanation', fontweight='bold', fontsize=10)
            ax.set_xticklabels([s[:8] for s in scenarios], rotation=45, fontsize=8)
            
            # Add Aral reference line label
            ax.text(0.02, 0.95, 'Aral et al.\n>50% line', transform=ax.transAxes, 
                   fontsize=8, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax.text(0.5, 0.5, 'Homophily\nAnalysis\nPending', transform=ax.transAxes, 
                   ha='center', va='center', fontweight='bold')
            ax.set_title('Homophily\nExplanation', fontweight='bold', fontsize=10)
    
    def _plot_scenario_adoption_rates(self, ax):
        """Plot adoption rates by scenario."""
        household_df = self._get_household_dataframe()
        scenarios = []
        rates = []
        
        for col in household_df.columns:
            if col.startswith('IsProsumer_'):
                scenario = col.replace('IsProsumer_', '')
                rate = household_df[col].mean()
                scenarios.append(scenario[:8])  # Truncate for display
                rates.append(rate)
        
        if scenarios:
            bars = ax.bar(scenarios, rates, alpha=0.7, color='steelblue')
            ax.set_ylabel('Adoption Rate')
            ax.set_title('Adoption Rates', fontweight='bold', fontsize=10)
            ax.set_xticklabels(scenarios, rotation=45, fontsize=8)
            
            # Add percentage labels on bars
            for bar, rate in zip(bars, rates):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{rate:.1%}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No Adoption\nData Found', transform=ax.transAxes, 
                   ha='center', va='center', fontweight='bold')
            ax.set_title('Adoption Rates', fontweight='bold', fontsize=10)
    
    def _plot_policy_clusters_summary(self, ax):
        """Plot policy clusters overview."""
        if 'clustering_results' in self.results:
            cluster_analysis = self.results['clustering_results']['cluster_analysis']
            
            if cluster_analysis:
                # Count categories
                categories = [cluster['category'] for cluster in cluster_analysis.values()]
                category_counts = Counter(categories)
                
                labels = list(category_counts.keys())
                sizes = list(category_counts.values())
                colors = {'hotspot': 'green', 'coldspot': 'red', 'mixed': 'orange', 'unknown': 'gray'}
                pie_colors = [colors.get(label, 'gray') for label in labels]
                
                wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=pie_colors, 
                                                 autopct='%1.0f%%', startangle=90)
                ax.set_title('Policy Clusters', fontweight='bold', fontsize=12)
                
                # Add cluster summary text
                total_households = sum(cluster['household_count'] for cluster in cluster_analysis.values())
                ax.text(1.3, 0.5, f'Total Clusters: {len(cluster_analysis)}\n'
                                  f'Households: {total_households:,}\n'
                                  f'Silhouette: {self.results["clustering_results"]["silhouette_score"]:.2f}',
                       transform=ax.transAxes, fontsize=10, va='center',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            else:
                ax.text(0.5, 0.5, 'No Clusters\nIdentified', transform=ax.transAxes, 
                       ha='center', va='center', fontweight='bold', fontsize=12)
                ax.set_title('Policy Clusters', fontweight='bold', fontsize=12)
        else:
            ax.text(0.5, 0.5, 'Clustering\nAnalysis\nPending', transform=ax.transAxes, 
                   ha='center', va='center', fontweight='bold', fontsize=12)
            ax.set_title('Policy Clusters', fontweight='bold', fontsize=12)
    
    def _plot_velocity_summary(self, ax):
        """Plot adoption velocity summary."""
        if 'velocity_metrics' in self.results:
            # Get first scenario for demonstration
            scenarios = [k for k in self.results['velocity_metrics'].keys() if k != 'cross_scenario_comparison']
            
            if scenarios:
                scenario = scenarios[0]
                income_velocities = self.results['velocity_metrics'][scenario]['income_velocities']
                
                classes = list(income_velocities.keys())
                rates = [income_velocities[c]['adoption_rate'] for c in classes]
                
                ax.plot(classes, rates, marker='o', linewidth=3, markersize=8, color='blue')
                ax.set_xlabel('Income Class', fontsize=10)
                ax.set_ylabel('Adoption Rate', fontsize=10)
                ax.set_title(f'Velocity by Income\n({scenario})', fontweight='bold', fontsize=10)
                ax.grid(True, alpha=0.3)
                
                # Add trend line
                if len(classes) > 1:
                    z = np.polyfit(classes, rates, 1)
                    p = np.poly1d(z)
                    ax.plot(classes, p(classes), "r--", alpha=0.8, linewidth=2,
                           label=f'Trend (slope={z[0]:.3f})')
                    ax.legend(fontsize=8)
            else:
                ax.text(0.5, 0.5, 'No Velocity\nData', transform=ax.transAxes, 
                       ha='center', va='center', fontweight='bold')
                ax.set_title('Velocity by Income', fontweight='bold', fontsize=10)
        else:
            ax.text(0.5, 0.5, 'Velocity\nAnalysis\nPending', transform=ax.transAxes, 
                   ha='center', va='center', fontweight='bold')
            ax.set_title('Velocity by Income', fontweight='bold', fontsize=10)
    
    def _plot_key_findings_summary(self, ax):
        """Plot key findings summary."""
        findings = [
            "Key Findings:",
            "• Network structure analyzed",
            "• Spatial vs homophily quantified", 
            "• Policy clusters identified",
            "• Adoption velocities compared",
            "• Literature-grounded insights"
        ]
        
        for i, finding in enumerate(findings):
            weight = 'bold' if i == 0 else 'normal'
            size = 11 if i == 0 else 9
            ax.text(0.05, 0.9 - i*0.15, finding, transform=ax.transAxes, 
                   fontweight=weight, fontsize=size)
        
        ax.set_title('Key Findings', fontweight='bold', fontsize=10)
        ax.axis('off')
    
    def _plot_methodology_summary(self, ax):
        """Plot methodology summary."""
        methods = [
            "Methodology:",
            "• Aral et al. (2009) framework",
            "• Spatial autocorrelation", 
            "• Chi-square homophily tests",
            "• DBSCAN clustering",
            "• Network-based analysis"
        ]
        
        for i, method in enumerate(methods):
            weight = 'bold' if i == 0 else 'normal'
            size = 11 if i == 0 else 9
            ax.text(0.05, 0.9 - i*0.15, method, transform=ax.transAxes, 
                   fontweight=weight, fontsize=size)
        
        ax.set_title('Methodology', fontweight='bold', fontsize=10)
        ax.axis('off')
    
    def _plot_policy_recommendations_summary(self, ax):
        """Plot policy recommendations summary."""
        recommendations = [
            "Policy Insights:",
            "• Target hotspots for demos",
            "• Intensive outreach for coldspots", 
            "• Peer programs for mixed areas",
            "• Income-based strategies",
            "• Spatial-aware interventions"
        ]
        
        for i, rec in enumerate(recommendations):
            weight = 'bold' if i == 0 else 'normal'
            size = 11 if i == 0 else 9
            ax.text(0.05, 0.9 - i*0.15, rec, transform=ax.transAxes, 
                   fontweight=weight, fontsize=size)
        
        ax.set_title('Policy Insights', fontweight='bold', fontsize=10)
        ax.axis('off')
    
    def _plot_next_steps_summary(self, ax):
        """Plot next steps summary."""
        next_steps = [
            "Next Steps:",
            "• Run enhanced analyzer",
            "• Temporal evolution analysis", 
            "• Detailed cascade tracking",
            "• Publication preparation",
            "• Policy implementation"
        ]
        
        for i, step in enumerate(next_steps):
            weight = 'bold' if i == 0 else 'normal'
            size = 11 if i == 0 else 9
            ax.text(0.05, 0.9 - i*0.15, step, transform=ax.transAxes, 
                   fontweight=weight, fontsize=size)
        
        ax.set_title('Next Steps', fontweight='bold', fontsize=10)
        ax.axis('off')
    
    def _print_analysis_summary(self):
        """Print comprehensive analysis summary."""
        print(f"\n--- IMMEDIATE SPATIAL ANALYSIS SUMMARY ---")
        
        current_step = getattr(self.model, 'current_step', self.model.schedule.steps)
        print(f"Analysis completed at step {current_step}")
        print(f"Total households analyzed: {len(self.households):,}")
        
        # Network analysis summary
        if 'network_analysis' in self.results:
            net_stats = self.results['network_analysis']
            print(f"\nNetwork Analysis:")
            print(f"  Density: {net_stats.get('network_density', 0):.4f}")
            print(f"  Avg Clustering: {net_stats.get('avg_clustering', 0):.4f}")
        
        # Spatial effects summary
        if 'spatial_effects' in self.results:
            print(f"\nSpatial vs Homophily Effects:")
            for scenario, effects in self.results['spatial_effects'].items():
                spatial_effect = effects['spatial_proximity']['effect_size']
                homophily_pct = effects['combined_assessment']['homophily_percentage']
                print(f"  {scenario}: Spatial={spatial_effect:.3f}, Homophily explains {homophily_pct:.1f}%")
        
        # Clustering summary
        if 'clustering_results' in self.results:
            cluster_results = self.results['clustering_results']
            recommendations = cluster_results['policy_recommendations']['summary']
            print(f"\nPolicy Clustering:")
            print(f"  Total clusters: {recommendations.get('total_clusters', 0)}")
            print(f"  Hotspots: {recommendations.get('hotspots', 0)}")
            print(f"  Coldspots: {recommendations.get('coldspots', 0)}")
            print(f"  Mixed areas: {recommendations.get('mixed', 0)}")
        
        # Velocity summary
        if 'velocity_metrics' in self.results:
            print(f"\nAdoption Velocities:")
            for scenario, velocity_data in self.results['velocity_metrics'].items():
                if scenario != 'cross_scenario_comparison':
                    metrics = velocity_data['velocity_metrics']
                    velocity_range = metrics['velocity_range']
                    print(f"  {scenario}: Range={velocity_range:.3f}")
        
        print(f"--- END ANALYSIS SUMMARY ---\n")


# =============================================================================
# QUICK USAGE FUNCTIONS
# =============================================================================

def run_immediate_analysis_on_model(model, output_dir="results/immediate_spatial_analysis", max_households=500):
    """
    Run immediate analysis on an existing model instance.
    
    Args:
        model: Running MultiExperimentModel instance
        output_dir: Directory to save results
        max_households: Maximum households for visualization performance
        
    Returns:
        ImmediateSpatialAnalyzer instance with completed analysis
    """
    print(f"Running immediate spatial analysis on model...")
    
    try:
        # Create analyzer
        analyzer = ImmediateSpatialAnalyzer(model)
        
        # Run complete analysis
        analyzer.create_all_immediate_analyses(output_dir, max_households)
        
        print(f"✅ Immediate analysis complete! Results saved to {output_dir}/")
        return analyzer
        
    except Exception as e:
        print(f"❌ Immediate analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_quick_spatial_overview(model, output_dir="results/quick_spatial", scenario='herding'):
    """
    Create just the network overview visualization for quick insights.
    
    Args:
        model: Running MultiExperimentModel instance
        output_dir: Directory to save visualization
        scenario: Scenario to visualize
        
    Returns:
        Path to saved visualization or None if failed
    """
    try:
        analyzer = ImmediateSpatialAnalyzer(model)
        return analyzer.create_network_overview(output_dir, scenario=scenario)
    except Exception as e:
        print(f"❌ Quick overview failed: {e}")
        return None


if __name__ == "__main__":
    print("ImmediateSpatialAnalyzer - Fixed Version")
    print("This module provides quick spatial analysis integrated with MultiExperimentModel")
    print("Usage:")
    print("  from data.immediate_spatial_analyzer import run_immediate_analysis_on_model")
    print("  analyzer = run_immediate_analysis_on_model(model)")