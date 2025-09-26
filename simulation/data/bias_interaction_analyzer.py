# data/bias_interaction_analyzer.py V1.0 - PHASE 3 IMPLEMENTATION
"""
Advanced bias interaction analyzer for behavioral prosumer adoption study.
Analyzes how cognitive biases interact, combine, and create emergent effects.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from itertools import combinations
from ..utils.parameters import get_all_scenarios, get_scenario_colors, get_scenario_metadata, get_enabled_biases

class BiasInteractionAnalyzer:
    """
    Advanced analyzer for bias interactions and emergent behavioral effects.
    
    Phase 3 Analysis:
    - Bias Synergy Analysis: How biases amplify or dampen each other
    - Multiplicative vs Additive Effects: Testing interaction models
    - Bias Dominance Patterns: Which biases override others
    - Emergent Behavioral Phenotypes: Clustering households by bias response
    """
    
    def __init__(self, data_collector, config):
        """
        Initialize advanced bias interaction analyzer.
        
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
        
        # Set advanced plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        print(f"BiasInteractionAnalyzer initialized for {len(self.enabled_biases)} bias interactions")
    
    def plot_all_interaction_analyses(self, output_dir="results/phase3_interactions"):
        """
        Generate all Phase 3 bias interaction analyses.
        
        Args:
            output_dir: Directory to save visualizations
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print("Generating Phase 3 Bias Interaction Analyses...")
        
        # Advanced Bias Interaction Analysis
        self.plot_bias_synergy_matrix(output_dir)
        #self.plot_interaction_effect_decomposition(output_dir)
        self.plot_bias_dominance_analysis(output_dir)
        self.plot_emergent_behavioral_phenotypes(output_dir)
        
        print(f"✅ All Phase 3 interaction analyses completed and saved to {output_dir}")
    
    def plot_bias_synergy_matrix(self, output_dir="results"):
        """
        Analyze synergistic vs antagonistic bias interactions (2x2 grid).
        Shows how pairs of biases interact beyond simple addition.
        """
        bias_df = self.data_collector.get_bias_effects_dataframe()
        
        if bias_df.empty:
            print("Warning: No bias effects data available for synergy analysis")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Calculate pairwise bias interactions
        bias_pairs = list(combinations(self.enabled_biases, 2))
        
        # Plot 1: Bias Interaction Heatmap
        ax = axes[0, 0]
        
        if len(self.enabled_biases) >= 2:
            # Create interaction matrix
            n_biases = len(self.enabled_biases)
            interaction_matrix = np.zeros((n_biases, n_biases))
            
            for i, bias1 in enumerate(self.enabled_biases):
                for j, bias2 in enumerate(self.enabled_biases):
                    if i != j and f'{bias1}_Multiplier' in bias_df.columns and f'{bias2}_Multiplier' in bias_df.columns:
                        # Calculate correlation between bias effects
                        bias1_effects = bias_df[f'{bias1}_Multiplier']
                        bias2_effects = bias_df[f'{bias2}_Multiplier']
                        
                        # Remove NaN values
                        valid_mask = ~(np.isnan(bias1_effects) | np.isnan(bias2_effects))
                        if valid_mask.sum() > 10:  # Ensure sufficient data
                            correlation = np.corrcoef(bias1_effects[valid_mask], bias2_effects[valid_mask])[0, 1]
                            interaction_matrix[i, j] = correlation
            
            # Create heatmap
            im = ax.imshow(interaction_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
            
            # Add bias labels
            bias_labels = [self.metadata.get(bias, {}).get('display_name', bias) for bias in self.enabled_biases]
            ax.set_xticks(range(n_biases))
            ax.set_yticks(range(n_biases))
            ax.set_xticklabels(bias_labels, rotation=45, ha='right')
            ax.set_yticklabels(bias_labels)
            
            # Add correlation values
            for i in range(n_biases):
                for j in range(n_biases):
                    if i != j:
                        text = ax.text(j, i, f'{interaction_matrix[i, j]:.2f}',
                                     ha="center", va="center", color="black", fontsize=10)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Correlation Coefficient')
        
        ax.set_title('Bias Interaction Correlation Matrix', fontweight='bold', fontsize=14)
        
        # Plot 2: Synergy vs Antagonism Classification
        ax = axes[0, 1]
        
        # Calculate expected vs observed combined effects
        if 'all_biases' in self.scenarios and len(self.enabled_biases) >= 2:
            combined_data = self.data_collector.get_combined_dataframe()
            
            if not combined_data.empty and 'all_biases_Probability' in combined_data.columns:
                synergy_scores = []
                household_ids = []
                
                for _, row in combined_data.iterrows():
                    # Get individual bias probabilities
                    individual_probs = []
                    base_prob = row.get('rational_Probability', 0.1)
                    
                    for bias in self.enabled_biases:
                        bias_prob = row.get(f'{bias}_Probability', base_prob)
                        individual_probs.append(bias_prob)
                    
                    # Calculate expected combined effect (multiplicative model)
                    expected_combined = base_prob
                    for prob in individual_probs:
                        multiplier = prob / base_prob if base_prob > 0 else 1.0
                        expected_combined *= multiplier
                    
                    # Get actual combined effect
                    actual_combined = row.get('all_biases_Probability', base_prob)
                    
                    # Calculate synergy score (actual vs expected)
                    if expected_combined > 0:
                        synergy_score = (actual_combined - expected_combined) / expected_combined
                        synergy_scores.append(synergy_score)
                        household_ids.append(row.get('HouseholdID', len(household_ids)))
                
                if synergy_scores:
                    # Create histogram of synergy scores
                    synergy_scores = np.array(synergy_scores)
                    
                    ax.hist(synergy_scores, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No Interaction')
                    ax.axvline(x=np.mean(synergy_scores), color='green', linestyle='-', linewidth=2, 
                              label=f'Mean: {np.mean(synergy_scores):.3f}')
                    
                    # Add annotations
                    synergistic = (synergy_scores > 0.05).sum()
                    antagonistic = (synergy_scores < -0.05).sum()
                    neutral = len(synergy_scores) - synergistic - antagonistic
                    
                    ax.text(0.7, 0.9, f'Synergistic: {synergistic}\nAntagonistic: {antagonistic}\nNeutral: {neutral}',
                           transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat"))
        
        ax.set_title('Synergy vs Antagonism Distribution', fontweight='bold', fontsize=14)
        ax.set_xlabel('Synergy Score (Actual - Expected) / Expected')
        ax.set_ylabel('Number of Households')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Bias Effect Magnitude Comparison
        ax = axes[1, 0]
        
        if not bias_df.empty:
            bias_magnitudes = []
            bias_names = []
            
            for bias in self.enabled_biases:
                if f'{bias}_Multiplier' in bias_df.columns:
                    multipliers = bias_df[f'{bias}_Multiplier'].dropna()
                    if not multipliers.empty:
                        # Calculate magnitude as deviation from 1.0 (no effect)
                        magnitude = np.abs(multipliers - 1.0).mean()
                        bias_magnitudes.append(magnitude)
                        bias_names.append(self.metadata.get(bias, {}).get('display_name', bias))
            
            if bias_magnitudes:
                # Create bar plot
                bars = ax.bar(range(len(bias_names)), bias_magnitudes, 
                             color=[self.colors.get(bias, '#000000') for bias in self.enabled_biases[:len(bias_names)]],
                             alpha=0.8)
                
                # Add value labels on bars
                for bar, magnitude in zip(bars, bias_magnitudes):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                           f'{magnitude:.3f}', ha='center', va='bottom', fontsize=10)
        
        ax.set_title('Average Bias Effect Magnitude', fontweight='bold', fontsize=14)
        ax.set_xlabel('Bias Type')
        ax.set_ylabel('Average Effect Magnitude |Multiplier - 1|')
        ax.set_xticks(range(len(bias_names)))
        ax.set_xticklabels(bias_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Interaction Strength vs Household Characteristics
        ax = axes[1, 1]
        
        if not bias_df.empty and 'IncomeClass' in bias_df.columns:
            # Analyze interaction strength by income class
            income_classes = sorted(bias_df['IncomeClass'].unique())
            
            # Calculate interaction strength for each household
            interaction_strengths = []
            household_incomes = []
            
            for _, row in bias_df.iterrows():
                # Calculate variance in bias effects as proxy for interaction strength
                bias_effects = []
                for bias in self.enabled_biases:
                    if f'{bias}_Multiplier' in row:
                        effect = row[f'{bias}_Multiplier']
                        if not np.isnan(effect):
                            bias_effects.append(effect)
                
                if len(bias_effects) >= 2:
                    interaction_strength = np.var(bias_effects)
                    interaction_strengths.append(interaction_strength)
                    household_incomes.append(row['IncomeClass'])
            
            if interaction_strengths:
                # Create scatter plot
                ax.scatter(household_incomes, interaction_strengths, alpha=0.6, color='purple')
                
                # Add trend line
                if len(set(household_incomes)) > 1:
                    z = np.polyfit(household_incomes, interaction_strengths, 1)
                    p = np.poly1d(z)
                    ax.plot(sorted(set(household_incomes)), [p(x) for x in sorted(set(household_incomes))], 
                           "r--", alpha=0.8, linewidth=2)
        
        ax.set_title('Bias Interaction Strength by Income Class', fontweight='bold', fontsize=14)
        ax.set_xlabel('Income Class')
        ax.set_ylabel('Interaction Strength (Variance in Effects)')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/bias_synergy_matrix.png", dpi=300, bbox_inches="tight")
        plt.close()
        
        print("  ✅ Bias synergy matrix analysis completed")
    
    def plot_interaction_effect_decomposition(self, output_dir="results"):# skiped for now
        """
        Decompose bias interactions into additive vs multiplicative components (2x3 grid).
        """
            # ADD THESE LINES
        interaction_by_npv = []
        additive_predictions = []
        multiplicative_predictions = []
        
        combined_df = self.data_collector.get_combined_dataframe()
        bias_df = self.data_collector.get_bias_effects_dataframe()

                # ADD THESE SAFETY CHECKS
        if combined_df.empty or len(combined_df) < 10:
            print("Warning: Insufficient data for interaction decomposition analysis")
            # Create placeholder figure
            fig, axes = plt.subplots(2, 3, figsize=(16, 10))
            for ax in axes.flat:
                ax.text(0.5, 0.5, 'Insufficient Data\nfor Analysis', 
                    ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Interaction Effect Decomposition')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/interaction_effect_decomposition.png", dpi=150, bbox_inches="tight")
            plt.close()
            return
        
        # ADD FIGURE SIZE CONSTRAINTS
        plt.rcParams['figure.max_open_warning'] = 0
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))  # Fixed reasonable size
        
        if combined_df.empty:
            print("Warning: No combined data available for interaction decomposition")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        # Plot 1: Additive vs Multiplicative Model Comparison
        ax = axes[0, 0]
        
        if 'rational_Probability' in combined_df.columns and 'all_biases_Probability' in combined_df.columns:
            # Calculate predicted probabilities under different models
            rational_probs = combined_df['rational_Probability']
            actual_combined = combined_df['all_biases_Probability']
            
            # Additive model: P_combined = P_base + sum(P_bias - P_base)
            additive_predictions = []
            multiplicative_predictions = []
            
            for _, row in combined_df.iterrows():
                base_prob = row['rational_Probability']
                
                # Additive model
                additive_sum = 0
                multiplicative_product = 1
                
                for bias in self.enabled_biases:
                    bias_col = f'{bias}_Probability'
                    if bias_col in row:
                        bias_prob = row[bias_col]
                        additive_sum += (bias_prob - base_prob)
                        multiplier = bias_prob / base_prob if base_prob > 0 else 1.0
                        multiplicative_product *= multiplier
                
                additive_pred = base_prob + additive_sum
                multiplicative_pred = base_prob * multiplicative_product
                
                additive_predictions.append(max(0, min(1, additive_pred)))
                multiplicative_predictions.append(max(0, min(1, multiplicative_pred)))
            
            # Scatter plot comparison
            ax.scatter(actual_combined, additive_predictions, alpha=0.6, color='blue', 
                      label='Additive Model', s=20)
            ax.scatter(actual_combined, multiplicative_predictions, alpha=0.6, color='red', 
                      label='Multiplicative Model', s=20)
            
            # Perfect prediction line
            max_prob = max(actual_combined.max(), max(additive_predictions + multiplicative_predictions))
            ax.plot([0, max_prob], [0, max_prob], 'k--', alpha=0.8, label='Perfect Prediction')
            
            # Calculate R² for both models
            try:
                if len(actual_combined) > 5 and len(additive_predictions) > 5:
                    # Check for constant arrays
                    if np.std(actual_combined) > 1e-10 and np.std(additive_predictions) > 1e-10:
                        additive_r2 = stats.pearsonr(actual_combined, additive_predictions)[0]**2
                    else:
                        additive_r2 = 0.0
                    
                    if np.std(multiplicative_predictions) > 1e-10:
                        multiplicative_r2 = stats.pearsonr(actual_combined, multiplicative_predictions)[0]**2
                    else:
                        multiplicative_r2 = 0.0
                else:
                    additive_r2 = 0.0
                    multiplicative_r2 = 0.0
            except (ValueError, np.linalg.LinAlgError):
                additive_r2 = 0.0
                multiplicative_r2 = 0.0
            
            ax.text(0.05, 0.95, f'Additive R²: {additive_r2:.3f}\nMultiplicative R²: {multiplicative_r2:.3f}',
                   transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat"),
                   verticalalignment='top')
        
        ax.set_title('Interaction Model Comparison', fontweight='bold', fontsize=14)
        ax.set_xlabel('Actual Combined Probability')
        ax.set_ylabel('Predicted Probability')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Residual Analysis
        ax = axes[0, 1]
        
        if len(additive_predictions) > 0:
            additive_residuals = np.array(actual_combined) - np.array(additive_predictions)
            multiplicative_residuals = np.array(actual_combined) - np.array(multiplicative_predictions)
            
            ax.hist(additive_residuals, bins=30, alpha=0.7, color='blue', label='Additive Residuals', density=True)
            ax.hist(multiplicative_residuals, bins=30, alpha=0.7, color='red', label='Multiplicative Residuals', density=True)
            
            ax.axvline(x=0, color='black', linestyle='--', alpha=0.8)
            ax.axvline(x=np.mean(additive_residuals), color='blue', linestyle='-', alpha=0.8)
            ax.axvline(x=np.mean(multiplicative_residuals), color='red', linestyle='-', alpha=0.8)
        
        ax.set_title('Model Residual Distributions', fontweight='bold', fontsize=14)
        ax.set_xlabel('Residual (Actual - Predicted)')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Bias Contribution Analysis
        ax = axes[0, 2]
        
        # Calculate each bias's contribution to final probability
        if not combined_df.empty:
            bias_contributions = {}
            
            for bias in self.enabled_biases:
                bias_col = f'{bias}_Probability'
                if bias_col in combined_df.columns:
                    # Calculate contribution as difference from rational
                    contributions = combined_df[bias_col] - combined_df['rational_Probability']
                    bias_contributions[bias] = contributions.mean()
            
            if bias_contributions:
                biases = list(bias_contributions.keys())
                contributions = list(bias_contributions.values())
                colors = [self.colors.get(bias, '#000000') for bias in biases]
                
                bars = ax.bar(range(len(biases)), contributions, color=colors, alpha=0.8)
                
                # Add value labels
                for bar, contrib in zip(bars, contributions):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.001 if height >= 0 else height - 0.001,
                           f'{contrib:.3f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=10)
                
                ax.axhline(y=0, color='black', linestyle='-', alpha=0.8)
        
        ax.set_title('Average Bias Contributions', fontweight='bold', fontsize=14)
        ax.set_xlabel('Bias Type')
        ax.set_ylabel('Average Probability Contribution')
        if bias_contributions:
            bias_labels = [self.metadata.get(bias, {}).get('display_name', bias) for bias in biases]
            ax.set_xticks(range(len(biases)))
            ax.set_xticklabels(bias_labels, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Interaction Order Effects
        ax = axes[1, 0]
        
        # Test if order of bias application matters (using synthetic examples)
        if len(self.enabled_biases) >= 2:
            # Create sample data to show order effects
            sample_base_prob = np.linspace(0.1, 0.9, 50)
            
            # Example: Apply biases in different orders
            bias1, bias2 = self.enabled_biases[0], self.enabled_biases[1]
            
            # Order 1: bias1 then bias2
            order1_results = []
            # Order 2: bias2 then bias1  
            order2_results = []
            
            for base_prob in sample_base_prob:
                # Simulate typical bias effects (placeholder values)
                bias1_mult = 0.8  # Example: reduces probability
                bias2_mult = 1.2  # Example: increases probability
                
                # Order 1
                intermediate = base_prob * bias1_mult
                final1 = intermediate * bias2_mult
                order1_results.append(final1)
                
                # Order 2
                intermediate = base_prob * bias2_mult
                final2 = intermediate * bias1_mult
                order2_results.append(final2)
            
            ax.plot(sample_base_prob, order1_results, label=f'{bias1} → {bias2}', 
                   color=self.colors.get(bias1, 'blue'), linewidth=2)
            ax.plot(sample_base_prob, order2_results, label=f'{bias2} → {bias1}', 
                   color=self.colors.get(bias2, 'red'), linewidth=2, linestyle='--')
            
            # Difference plot
            ax_twin = ax.twinx()
            differences = np.array(order1_results) - np.array(order2_results)
            ax_twin.plot(sample_base_prob, differences, color='gray', alpha=0.7, 
                        label='Order Difference')
            ax_twin.set_ylabel('Order Effect Magnitude', color='gray')
        
        ax.set_title('Bias Application Order Effects', fontweight='bold', fontsize=14)
        ax.set_xlabel('Base Probability')
        ax.set_ylabel('Final Probability')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Plot 5: Non-linear Interaction Detection
        ax = axes[1, 1]
        
        # Use PCA to detect non-linear bias patterns
        if not bias_df.empty and len(self.enabled_biases) >= 2:
            # Prepare bias effect matrix
            bias_matrix = []
            for bias in self.enabled_biases:
                if f'{bias}_Multiplier' in bias_df.columns:
                    bias_effects = bias_df[f'{bias}_Multiplier'].fillna(1.0)
                    bias_matrix.append(bias_effects)
            
            if len(bias_matrix) >= 2:
                bias_matrix = np.array(bias_matrix).T
                
                # Standardize data
                scaler = StandardScaler()
                bias_matrix_scaled = scaler.fit_transform(bias_matrix)
                
                # Apply PCA
                pca = PCA(n_components=min(len(self.enabled_biases), 3))
                pca_result = pca.fit_transform(bias_matrix_scaled)
                
                # Plot first two principal components
                scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], 
                                   c=bias_df['IncomeClass'] if 'IncomeClass' in bias_df.columns else 'blue',
                                   cmap='viridis', alpha=0.6)
                
                # Add explained variance
                ax.text(0.05, 0.95, f'PC1: {pca.explained_variance_ratio_[0]:.2%}\nPC2: {pca.explained_variance_ratio_[1]:.2%}',
                       transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat"),
                       verticalalignment='top')
                
                if 'IncomeClass' in bias_df.columns:
                    plt.colorbar(scatter, ax=ax, label='Income Class')
        
        ax.set_title('Principal Component Analysis of Bias Effects', fontweight='bold', fontsize=14)
        ax.set_xlabel('First Principal Component')
        ax.set_ylabel('Second Principal Component')
        ax.grid(True, alpha=0.3)
        
        # Plot 6: Interaction Strength by NPV Range
        ax = axes[1, 2]
        
        if not bias_df.empty and 'BaseNPV' in bias_df.columns:
            # Bin households by NPV
            npv_bins = pd.qcut(bias_df['BaseNPV'], q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
            

            npv_labels = []
            
            for npv_category in npv_bins.cat.categories:
                npv_mask = npv_bins == npv_category
                npv_data = bias_df[npv_mask]
                
                if len(npv_data) > 5:  # Ensure sufficient data
                    # Calculate interaction strength as variance in bias effects
                    bias_effects = []
                    for bias in self.enabled_biases:
                        if f'{bias}_Multiplier' in npv_data.columns:
                            effects = npv_data[f'{bias}_Multiplier'].dropna()
                            if not effects.empty:
                                bias_effects.extend(effects)
                    
                    if len(bias_effects) > 0:
                        interaction_strength = np.var(bias_effects)
                        interaction_by_npv.append(interaction_strength)
                        npv_labels.append(npv_category)
            
            if interaction_by_npv:
                bars = ax.bar(range(len(npv_labels)), interaction_by_npv, 
                             color='orange', alpha=0.8)
                
                # Add value labels
                for bar, strength in zip(bars, interaction_by_npv):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                           f'{strength:.3f}', ha='center', va='bottom', fontsize=10)
        
        ax.set_title('Bias Interaction Strength by NPV Range', fontweight='bold', fontsize=14)
        ax.set_xlabel('NPV Category')
        ax.set_ylabel('Interaction Strength')
        if interaction_by_npv:
            ax.set_xticks(range(len(npv_labels)))
            ax.set_xticklabels(npv_labels, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/interaction_effect_decomposition.png", dpi=300, bbox_inches="tight")
        plt.close()
        
        print("  ✅ Interaction effect decomposition completed")
    
    def plot_bias_dominance_analysis(self, output_dir="results"):
        """
        Analyze which biases dominate in different contexts (2x2 grid).
        """
        bias_df = self.data_collector.get_bias_effects_dataframe()
        combined_df = self.data_collector.get_combined_dataframe()
        
        if bias_df.empty:
            print("Warning: No bias effects data available for dominance analysis")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Bias Dominance Hierarchy
        ax = axes[0, 0]
        
        # Calculate average effect magnitude for each bias
        bias_dominance = {}
        for bias in self.enabled_biases:
            if f'{bias}_Multiplier' in bias_df.columns:
                multipliers = bias_df[f'{bias}_Multiplier'].dropna()
                if not multipliers.empty:
                    # Dominance as average deviation from neutral (1.0)
                    dominance = np.abs(multipliers - 1.0).mean()
                    bias_dominance[bias] = dominance
        
        if bias_dominance:
            # Sort by dominance
            sorted_biases = sorted(bias_dominance.items(), key=lambda x: x[1], reverse=True)
            biases, dominance_values = zip(*sorted_biases)
            
            # Create horizontal bar chart
            y_pos = np.arange(len(biases))
            colors = [self.colors.get(bias, '#000000') for bias in biases]
            
            bars = ax.barh(y_pos, dominance_values, color=colors, alpha=0.8)
            
            # Add value labels
            for bar, dominance in zip(bars, dominance_values):
                width = bar.get_width()
                ax.text(width + 0.001, bar.get_y() + bar.get_height()/2,
                       f'{dominance:.3f}', ha='left', va='center', fontsize=10)
            
            # Customize axis
            bias_labels = [self.metadata.get(bias, {}).get('display_name', bias) for bias in biases]
            ax.set_yticks(y_pos)
            ax.set_yticklabels(bias_labels)
            ax.invert_yaxis()  # Highest dominance at top
        
        ax.set_title('Bias Dominance Hierarchy', fontweight='bold', fontsize=14)
        ax.set_xlabel('Average Effect Magnitude')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Plot 2: Context-Dependent Dominance
        ax = axes[0, 1]
        
        if 'IncomeClass' in bias_df.columns:
            income_classes = sorted(bias_df['IncomeClass'].unique())
            
            # Calculate dominance by income class
            dominance_by_class = {}
            for income_class in income_classes:
                class_data = bias_df[bias_df['IncomeClass'] == income_class]
                class_dominance = {}
                
                for bias in self.enabled_biases:
                    if f'{bias}_Multiplier' in class_data.columns:
                        multipliers = class_data[f'{bias}_Multiplier'].dropna()
                        if not multipliers.empty:
                            dominance = np.abs(multipliers - 1.0).mean()
                            class_dominance[bias] = dominance
                
                dominance_by_class[income_class] = class_dominance
            
            # Create stacked bar chart
            if dominance_by_class:
                bias_totals = {bias: [] for bias in self.enabled_biases}
                
                for income_class in income_classes:
                    for bias in self.enabled_biases:
                        dominance = dominance_by_class[income_class].get(bias, 0)
                        bias_totals[bias].append(dominance)
                
                # Plot stacked bars
                bottom = np.zeros(len(income_classes))
                for bias in self.enabled_biases:
                    if bias in bias_totals:
                        color = self.colors.get(bias, '#000000')
                        display_name = self.metadata.get(bias, {}).get('display_name', bias)
                        ax.bar(range(len(income_classes)), bias_totals[bias], 
                              bottom=bottom, label=display_name, color=color, alpha=0.8)
                        bottom += np.array(bias_totals[bias])
                
                ax.set_xticks(range(len(income_classes)))
                ax.set_xticklabels([f'Class {ic}' for ic in income_classes])
        
        ax.set_title('Bias Dominance by Income Class', fontweight='bold', fontsize=14)
        ax.set_xlabel('Income Class')
        ax.set_ylabel('Cumulative Dominance')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Temporal Dominance Evolution
        ax = axes[1, 0]
        
        if 'Year' in bias_df.columns:
            years = sorted(bias_df['Year'].unique())
            
            # Track dominance over time
            for bias in self.enabled_biases:
                if f'{bias}_Multiplier' in bias_df.columns:
                    yearly_dominance = []
                    
                    for year in years:
                        year_data = bias_df[bias_df['Year'] == year]
                        multipliers = year_data[f'{bias}_Multiplier'].dropna()
                        
                        if not multipliers.empty:
                            dominance = np.abs(multipliers - 1.0).mean()
                            yearly_dominance.append(dominance)
                        else:
                            yearly_dominance.append(0)
                    
                    if yearly_dominance:
                        color = self.colors.get(bias, '#000000')
                        display_name = self.metadata.get(bias, {}).get('display_name', bias)
                        ax.plot(years, yearly_dominance, color=color, label=display_name, 
                               linewidth=2, marker='o', markersize=4)
        
        ax.set_title('Temporal Evolution of Bias Dominance', fontweight='bold', fontsize=14)
        ax.set_xlabel('Year')
        ax.set_ylabel('Average Dominance')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Dominance Switch Points
        ax = axes[1, 1]
        
        # Analyze where dominance switches between biases
        if not combined_df.empty and len(self.enabled_biases) >= 2:
            switch_points = []
            npv_values = []
            
            # Sample analysis based on NPV ranges
            if 'rational_NPV' in combined_df.columns:
                npv_range = np.linspace(combined_df['rational_NPV'].min(), 
                                      combined_df['rational_NPV'].max(), 50)
                
                for npv in npv_range:
                    # Find households near this NPV
                    npv_mask = np.abs(combined_df['rational_NPV'] - npv) < (npv_range[1] - npv_range[0])
                    npv_data = combined_df[npv_mask]
                    
                    if len(npv_data) > 5:
                        # Determine dominant bias for this NPV range
                        bias_effects = {}
                        for bias in self.enabled_biases:
                            prob_col = f'{bias}_Probability'
                            if prob_col in npv_data.columns:
                                avg_prob = npv_data[prob_col].mean()
                                baseline_prob = npv_data['rational_Probability'].mean()
                                effect = abs(avg_prob - baseline_prob)
                                bias_effects[bias] = effect
                        
                        if bias_effects:
                            dominant_bias = max(bias_effects, key=bias_effects.get)
                            switch_points.append(dominant_bias)
                            npv_values.append(npv)
                
                # Plot dominance regions
                if switch_points and npv_values:
                    unique_biases = list(set(switch_points))
                    bias_indices = {bias: i for i, bias in enumerate(unique_biases)}
                    
                    y_values = [bias_indices[bias] for bias in switch_points]
                    
                    # Create step plot
                    ax.step(npv_values, y_values, where='post', linewidth=3, alpha=0.8)
                    
                    # Color regions
                    for i, bias in enumerate(unique_biases):
                        color = self.colors.get(bias, '#000000')
                        ax.axhspan(i-0.4, i+0.4, alpha=0.3, color=color)
                    
                    # Customize y-axis
                    ax.set_yticks(range(len(unique_biases)))
                    bias_labels = [self.metadata.get(bias, {}).get('display_name', bias) 
                                 for bias in unique_biases]
                    ax.set_yticklabels(bias_labels)
        
        ax.set_title('Bias Dominance Regions by NPV', fontweight='bold', fontsize=14)
        ax.set_xlabel('Net Present Value ($)')
        ax.set_ylabel('Dominant Bias')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/bias_dominance_analysis.png", dpi=300, bbox_inches="tight")
        plt.close()
        
        print("  ✅ Bias dominance analysis completed")
    
    def plot_emergent_behavioral_phenotypes(self, output_dir="results"):
        """
        Identify and analyze emergent behavioral phenotypes (2x2 grid).
        Groups households by their bias response patterns.
        """
        bias_df = self.data_collector.get_bias_effects_dataframe()
        combined_df = self.data_collector.get_combined_dataframe()
        
        if bias_df.empty:
            print("Warning: No bias effects data available for phenotype analysis")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Prepare bias effect matrix for clustering
        bias_matrix = []
        valid_indices = []
        
        for i, (_, row) in enumerate(bias_df.iterrows()):
            bias_effects = []
            for bias in self.enabled_biases:
                if f'{bias}_Multiplier' in row:
                    effect = row[f'{bias}_Multiplier']
                    if not np.isnan(effect):
                        bias_effects.append(effect)
                    else:
                        bias_effects.append(1.0)  # No effect
                else:
                    bias_effects.append(1.0)
            
            if len(bias_effects) == len(self.enabled_biases):
                bias_matrix.append(bias_effects)
                valid_indices.append(i)
        
        if len(bias_matrix) < 10:  # Need sufficient data for clustering
            print("Warning: Insufficient data for phenotype clustering")
            return
        
        bias_matrix = np.array(bias_matrix)
        
        # Plot 1: Phenotype Clustering (PCA visualization)
        ax = axes[0, 0]
        
        try:
            from sklearn.cluster import KMeans
            
            # Standardize data
            scaler = StandardScaler()
            bias_matrix_scaled = scaler.fit_transform(bias_matrix)
            
            # Apply K-means clustering
            n_clusters = min(5, len(bias_matrix) // 10)  # Reasonable cluster count
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(bias_matrix_scaled)
            
            # Apply PCA for visualization
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(bias_matrix_scaled)
            
            # Create scatter plot with clusters
            scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], 
                               c=cluster_labels, cmap='tab10', alpha=0.7, s=50)
            
            # Add cluster centers
            centers_pca = pca.transform(kmeans.cluster_centers_)
            ax.scatter(centers_pca[:, 0], centers_pca[:, 1], 
                      c='red', marker='X', s=200, linewidths=2, 
                      label='Cluster Centers')
            
            # Add explained variance
            ax.text(0.05, 0.95, f'PC1: {pca.explained_variance_ratio_[0]:.2%}\nPC2: {pca.explained_variance_ratio_[1]:.2%}',
                   transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat"),
                   verticalalignment='top')
            
            plt.colorbar(scatter, ax=ax, label='Behavioral Phenotype')
            
        except ImportError:
            ax.text(0.5, 0.5, 'sklearn required for clustering analysis', 
                   transform=ax.transAxes, ha='center', va='center')
        
        ax.set_title('Behavioral Phenotype Clustering', fontweight='bold', fontsize=14)
        ax.set_xlabel('First Principal Component')
        ax.set_ylabel('Second Principal Component')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Phenotype Characteristics
        ax = axes[0, 1]
        
        if 'cluster_labels' in locals():
            # Analyze characteristics of each cluster
            cluster_characteristics = {}
            
            for cluster_id in range(n_clusters):
                cluster_mask = cluster_labels == cluster_id
                cluster_data = bias_matrix[cluster_mask]
                
                if len(cluster_data) > 0:
                    # Calculate average bias effects for this cluster
                    avg_effects = np.mean(cluster_data, axis=0)
                    cluster_characteristics[cluster_id] = avg_effects
            
            if cluster_characteristics:
                # Create heatmap of cluster characteristics
                cluster_matrix = np.array([cluster_characteristics[i] for i in range(n_clusters)])
                
                im = ax.imshow(cluster_matrix, cmap='RdBu_r', aspect='auto')
                
                # Add labels
                ax.set_xticks(range(len(self.enabled_biases)))
                ax.set_yticks(range(n_clusters))
                
                bias_labels = [self.metadata.get(bias, {}).get('display_name', bias) 
                             for bias in self.enabled_biases]
                ax.set_xticklabels(bias_labels, rotation=45, ha='right')
                ax.set_yticklabels([f'Phenotype {i+1}' for i in range(n_clusters)])
                
                # Add value annotations
                for i in range(n_clusters):
                    for j in range(len(self.enabled_biases)):
                        text = ax.text(j, i, f'{cluster_matrix[i, j]:.2f}',
                                     ha="center", va="center", color="black", fontsize=10)
                
                # Add colorbar
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label('Average Bias Multiplier')
        
        ax.set_title('Phenotype Bias Characteristics', fontweight='bold', fontsize=14)
        
        # Plot 3: Phenotype Distribution by Income
        ax = axes[1, 0]
        
        if 'cluster_labels' in locals() and 'IncomeClass' in bias_df.columns:
            valid_bias_df = bias_df.iloc[valid_indices]
            income_classes = sorted(valid_bias_df['IncomeClass'].unique())
            
            # Create distribution matrix
            distribution_matrix = np.zeros((n_clusters, len(income_classes)))
            
            for i, income_class in enumerate(income_classes):
                income_mask = valid_bias_df['IncomeClass'] == income_class
                income_clusters = cluster_labels[income_mask]
                
                if len(income_clusters) > 0:
                    for cluster_id in range(n_clusters):
                        count = np.sum(income_clusters == cluster_id)
                        total = len(income_clusters)
                        distribution_matrix[cluster_id, i] = count / total if total > 0 else 0
            
            # Create stacked bar chart
            bottom = np.zeros(len(income_classes))
            for cluster_id in range(n_clusters):
                ax.bar(range(len(income_classes)), distribution_matrix[cluster_id], 
                      bottom=bottom, label=f'Phenotype {cluster_id+1}', alpha=0.8)
                bottom += distribution_matrix[cluster_id]
            
            ax.set_xticks(range(len(income_classes)))
            ax.set_xticklabels([f'Class {ic}' for ic in income_classes])
        
        ax.set_title('Phenotype Distribution by Income Class', fontweight='bold', fontsize=14)
        ax.set_xlabel('Income Class')
        ax.set_ylabel('Proportion')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Phenotype Adoption Outcomes
        ax = axes[1, 1]
        
        if 'cluster_labels' in locals() and not combined_df.empty:
            # Match cluster assignments to combined data
            phenotype_adoption_rates = {}
            
            for cluster_id in range(n_clusters):
                cluster_mask = cluster_labels == cluster_id
                cluster_household_ids = valid_bias_df.iloc[cluster_mask]['HouseholdID']
                
                # Get adoption rates for each scenario
                scenario_rates = {}
                for scenario in self.scenarios:
                    adopted_col = f'{scenario}_Adopted'
                    if adopted_col in combined_df.columns:
                        cluster_data = combined_df[combined_df['HouseholdID'].isin(cluster_household_ids)]
                        if not cluster_data.empty:
                            adoption_rate = cluster_data[adopted_col].mean()
                            scenario_rates[scenario] = adoption_rate
                
                phenotype_adoption_rates[cluster_id] = scenario_rates
            
            # Create grouped bar chart
            if phenotype_adoption_rates:
                scenario_names = list(phenotype_adoption_rates[0].keys())
                x = np.arange(len(scenario_names))
                width = 0.8 / n_clusters
                
                for cluster_id in range(n_clusters):
                    rates = [phenotype_adoption_rates[cluster_id].get(scenario, 0) 
                           for scenario in scenario_names]
                    ax.bar(x + cluster_id * width, rates, width, 
                          label=f'Phenotype {cluster_id+1}', alpha=0.8)
                
                ax.set_xticks(x + width * (n_clusters - 1) / 2)
                scenario_labels = [self.metadata.get(scenario, {}).get('display_name', scenario) 
                                 for scenario in scenario_names]
                ax.set_xticklabels(scenario_labels, rotation=45, ha='right')
        
        ax.set_title('Adoption Rates by Phenotype and Scenario', fontweight='bold', fontsize=14)
        ax.set_xlabel('Scenario')
        ax.set_ylabel('Adoption Rate')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/emergent_behavioral_phenotypes.png", dpi=300, bbox_inches="tight")
        plt.close()
        
        print("  ✅ Emergent behavioral phenotypes analysis completed")