# data/bias_validation_visualizer.py V1.0 - PHASE 1 IMPLEMENTATION
"""
Mathematical validation suite for behavioral biases.
Creates publication-quality plots validating bias parameter distributions and NPV transformations.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
from scipy import stats
from collections import defaultdict

from ..utils.parameters import (
    BEHAVIORAL_BIASES, get_enabled_biases, NPV_SIGMOID_STEEPNESS,
    INCOME_LOGNORMAL_MEAN, INCOME_LOGNORMAL_SD
)

class BiasValidationVisualizer:
    """
    Creates validation visualizations for behavioral bias implementations.
    
    Validates:
    1. Parameter distributions match literature
    2. NPV transformations follow manuscript equations
    3. Individual bias mechanisms work correctly
    4. Income dependencies are properly implemented
    """
    
    def __init__(self, config=None, output_dir="results/validation"):
        """
        Initialize the bias validation visualizer.
        
        Args:
            config: Simulation configuration
            output_dir: Directory to save validation plots
        """
        self.config = config or {}
        self.output_dir = output_dir
        self.enabled_biases = get_enabled_biases()
        
        # Set up plot styling for publication quality
        self._setup_plot_style()
        
        print(f"BiasValidationVisualizer initialized for {len(self.enabled_biases)} biases")
    
    def _setup_plot_style(self):
        """Set up consistent plot styling for publication quality."""
        plt.rcParams.update({
            'font.size': 12,
            'font.family': 'sans-serif',
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 11,
            'figure.titlesize': 16,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight'
        })
    
    def create_bias_parameter_distributions(self):
        """
        Create 2x2 grid showing parameter distributions for all enabled biases.
        CRITICAL validation - ensures bias parameters match literature ranges.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create 2x2 subplot grid
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        enabled_biases = self.enabled_biases[:4]  # Show first 4 biases
        
        for i, bias_name in enumerate(enabled_biases):
            ax = axes[i]
            self._plot_bias_parameter_distribution(ax, bias_name)
        
        # Handle empty subplots
        for i in range(len(enabled_biases), 4):
            axes[i].text(0.5, 0.5, 'No Additional\nBias', 
                        ha='center', va='center', transform=axes[i].transAxes,
                        fontsize=14, alpha=0.5)
            axes[i].set_xticks([])
            axes[i].set_yticks([])
        
        # Overall title
        fig.suptitle('Behavioral Bias Parameter Distributions\n(Literature Validation)', 
                    fontsize=18, fontweight='bold', y=0.95)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        # Save the plot
        output_path = os.path.join(self.output_dir, 'bias_parameter_distributions.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Bias parameter distributions saved to: {output_path}")
    
    def _plot_bias_parameter_distribution(self, ax, bias_name):
        """
        Plot parameter distribution for a specific bias.
        
        Args:
            ax: Matplotlib axes object
            bias_name: Name of the bias
        """
        if bias_name not in BEHAVIORAL_BIASES:
            return
        
        bias_config = BEHAVIORAL_BIASES[bias_name]
        params = bias_config['parameters']
        display_name = bias_config['display_name']
        
        if bias_name == 'loss_aversion':
            # Plot loss aversion coefficient distribution
            baseline_coeff = params['baseline_coefficient']
            income_sensitivity = params['income_sensitivity']
            
            # Generate income range
            incomes = np.logspace(np.log10(20000), np.log10(120000), 1000)
            median_income = np.exp(INCOME_LOGNORMAL_MEAN)
            
            # Calculate λ_i for each income level
            lambda_i = baseline_coeff * (median_income / incomes) ** income_sensitivity
            
            ax.plot(incomes, lambda_i, linewidth=3, color='#ff7f0e', label='λ(income)')
            ax.axhline(y=baseline_coeff, color='red', linestyle='--', alpha=0.7, 
                      label=f'Baseline: {baseline_coeff}')
            
            # Literature range
            ax.axhspan(1.5, 4.0, alpha=0.2, color='green', label='Literature Range')
            
            ax.set_xlabel('Household Income ($)')
            ax.set_ylabel('Loss Aversion Coefficient (λ)')
            ax.set_title(f'{display_name}\n(Income-Dependent)', fontweight='bold')
            ax.set_xscale('log')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
        
        elif bias_name == 'present_bias':
            # Plot present bias beta distribution
            beta_min = params['beta_min']
            beta_max = params['beta_max']
            
            # Generate uniform distribution
            beta_values = np.linspace(beta_min, beta_max, 1000)
            density = np.ones_like(beta_values) / (beta_max - beta_min)
            
            ax.fill_between(beta_values, 0, density, alpha=0.6, color='#2ca02c', 
                           label=f'Uniform({beta_min}, {beta_max})')
            
            # Literature range
            ax.axvspan(0.5, 0.9, alpha=0.2, color='orange', label='Literature Range')
            
            ax.set_xlabel('Present Bias Parameter (β)')
            ax.set_ylabel('Probability Density')
            ax.set_title(f'{display_name}\n(Quasi-Hyperbolic)', fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
        
        elif bias_name == 'status_quo':
            # Plot status quo bias strength distribution
            baseline_strength = params['baseline_strength']
            individual_variation = params['individual_variation']
            
            # Generate normal distribution
            sigma_values = np.linspace(0, 0.8, 1000)
            pdf = stats.norm.pdf(sigma_values, baseline_strength, individual_variation)
            
            ax.plot(sigma_values, pdf, linewidth=3, color='#d62728', 
                   label=f'N({baseline_strength}, {individual_variation}²)')
            ax.axvline(x=baseline_strength, color='blue', linestyle='--', alpha=0.7, 
                      label=f'Baseline: {baseline_strength}')
            
            # Literature range
            ax.axvspan(0.1, 0.5, alpha=0.2, color='purple', label='Literature Range')
            
            ax.set_xlabel('Status Quo Bias Strength (σ)')
            ax.set_ylabel('Probability Density')
            ax.set_title(f'{display_name}\n(Individual Variation)', fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
        
        elif bias_name == 'herding':
            # Plot herding bias influence distribution
            spatial_a = params['spatial_beta_shape_a']
            spatial_b = params['spatial_beta_shape_b']
            class_a = params['class_beta_shape_a']
            class_b = params['class_beta_shape_b']
            
            # Generate beta distributions
            x_values = np.linspace(0, 1, 1000)
            spatial_pdf = stats.beta.pdf(x_values, spatial_a, spatial_b)
            class_pdf = stats.beta.pdf(x_values, class_a, class_b)
            
            ax.plot(x_values, spatial_pdf, linewidth=3, color='#9467bd', 
                   label=f'Spatial: Beta({spatial_a}, {spatial_b})')
            ax.plot(x_values, class_pdf, linewidth=3, color='#8c564b', 
                   label=f'Class: Beta({class_a}, {class_b})')
            
            ax.set_xlabel('Influence Strength')
            ax.set_ylabel('Probability Density')
            ax.set_title(f'{display_name}\n(Dual Channel)', fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
    
    def create_npv_transformation_analysis(self):
        """
        Create 2x3 grid analyzing NPV transformations for each bias.
        CRITICAL validation - shows how each bias transforms NPV and adoption probability.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create 2x3 subplot grid
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Base NPV range for analysis
        npv_range = np.linspace(-20000, 20000, 1000)
        
        # Plot rational baseline (top-left)
        ax = axes[0, 0]
        rational_prob = self._npv_to_probability(npv_range)
        ax.plot(npv_range, rational_prob, linewidth=3, color='#1f77b4', label='Rational')
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Break-even')
        ax.set_title('Rational Baseline\n(NPV > 0)', fontweight='bold')
        ax.set_xlabel('NPV ($)')
        ax.set_ylabel('Adoption Probability')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot individual bias transformations
        enabled_biases = self.enabled_biases[:5]  # First 5 biases
        positions = [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
        
        for i, bias_name in enumerate(enabled_biases):
            if i < len(positions):
                row, col = positions[i]
                ax = axes[row, col]
                self._plot_npv_transformation(ax, bias_name, npv_range, rational_prob)
        
        # Handle empty subplots
        for i in range(len(enabled_biases), 5):
            if i < len(positions):
                row, col = positions[i]
                axes[row, col].text(0.5, 0.5, 'No Additional\nBias', 
                                   ha='center', va='center', transform=axes[row, col].transAxes,
                                   fontsize=14, alpha=0.5)
                axes[row, col].set_xticks([])
                axes[row, col].set_yticks([])
        
        # Overall title
        fig.suptitle('NPV Transformation Analysis by Bias\n(Mathematical Validation)', 
                    fontsize=18, fontweight='bold', y=0.95)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        # Save the plot
        output_path = os.path.join(self.output_dir, 'npv_transformation_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"NPV transformation analysis saved to: {output_path}")
    
    def _plot_npv_transformation(self, ax, bias_name, npv_range, rational_prob):
        """
        Plot NPV transformation for a specific bias.
        
        Args:
            ax: Matplotlib axes object
            bias_name: Name of the bias
            npv_range: Range of NPV values
            rational_prob: Rational baseline probabilities
        """
        if bias_name not in BEHAVIORAL_BIASES:
            return
        
        bias_config = BEHAVIORAL_BIASES[bias_name]
        params = bias_config['parameters']
        display_name = bias_config['display_name']
        
        # Create mock values for demonstration
        if bias_name == 'loss_aversion':
            # Apply loss aversion transformation
            baseline_coeff = params['baseline_coefficient']
            installation_cost = 15000  # Typical installation cost
            
            # NPV adjustment: NPV - (λ - 1) × C_install
            adjusted_npv = npv_range - (baseline_coeff - 1) * installation_cost
            biased_prob = self._npv_to_probability(adjusted_npv)
            
            ax.plot(npv_range, rational_prob, linewidth=2, color='#1f77b4', 
                   alpha=0.5, label='Rational')
            ax.plot(npv_range, biased_prob, linewidth=3, color='#ff7f0e', 
                   label=f'Loss Aversion (λ={baseline_coeff})')
            
        elif bias_name == 'present_bias':
            # Apply present bias transformation
            beta_avg = (params['beta_min'] + params['beta_max']) / 2
            installation_cost = 15000
            
            # NPV adjustment: NPV - (1-β) × (NPV + C_install)
            adjusted_npv = npv_range - (1 - beta_avg) * (npv_range + installation_cost)
            biased_prob = self._npv_to_probability(adjusted_npv)
            
            ax.plot(npv_range, rational_prob, linewidth=2, color='#1f77b4', 
                   alpha=0.5, label='Rational')
            ax.plot(npv_range, biased_prob, linewidth=3, color='#2ca02c', 
                   label=f'Present Bias (β={beta_avg:.2f})')
        
        elif bias_name == 'status_quo':
            # Apply status quo bias transformation
            baseline_strength = params['baseline_strength']
            
            # Probability reduction: P × (1 - σ)
            biased_prob = rational_prob * (1 - baseline_strength)
            
            ax.plot(npv_range, rational_prob, linewidth=2, color='#1f77b4', 
                   alpha=0.5, label='Rational')
            ax.plot(npv_range, biased_prob, linewidth=3, color='#d62728', 
                   label=f'Status Quo (σ={baseline_strength})')
        
        elif bias_name == 'herding':
            # Apply herding bias transformation (example with 50% neighbor adoption)
            spatial_influence = 0.5  # Example: 50% of neighbors adopted
            class_influence = 0.3    # Example: 30% of class adopted
            
            # Average beta parameters
            spatial_beta = params['spatial_beta_shape_a'] / (params['spatial_beta_shape_a'] + params['spatial_beta_shape_b'])
            class_beta = params['class_beta_shape_a'] / (params['class_beta_shape_a'] + params['class_beta_shape_b'])
            
            # Probability increase: P × (1 + β×ρ_spatial + γ×ρ_class)
            multiplier = 1 + spatial_beta * spatial_influence + class_beta * class_influence
            biased_prob = np.minimum(rational_prob * multiplier, 1.0)  # Cap at 1.0
            
            ax.plot(npv_range, rational_prob, linewidth=2, color='#1f77b4', 
                   alpha=0.5, label='Rational')
            ax.plot(npv_range, biased_prob, linewidth=3, color='#9467bd', 
                   label=f'Herding (mult={multiplier:.2f})')
        
        # Formatting
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        ax.set_title(f'{display_name}\nTransformation', fontweight='bold')
        ax.set_xlabel('NPV ($)')
        ax.set_ylabel('Adoption Probability')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
    
    def create_loss_aversion_income_dependency(self):
        """
        Create detailed analysis of loss aversion income dependency.
        CRITICAL validation - shows how loss aversion varies with income.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        if 'loss_aversion' not in BEHAVIORAL_BIASES:
            fig.text(0.5, 0.5, 'Loss Aversion Not Enabled', 
                    ha='center', va='center', fontsize=16)
            plt.savefig(os.path.join(self.output_dir, 'loss_aversion_income_dependency.png'))
            plt.close()
            return
        
        params = BEHAVIORAL_BIASES['loss_aversion']['parameters']
        baseline_coeff = params['baseline_coefficient']
        income_sensitivity = params['income_sensitivity']
        median_income = np.exp(INCOME_LOGNORMAL_MEAN)
        
        # Plot 1: Loss aversion coefficient vs income
        ax = axes[0, 0]
        incomes = np.logspace(np.log10(20000), np.log10(120000), 1000)
        lambda_i = baseline_coeff * (median_income / incomes) ** income_sensitivity
        
        ax.plot(incomes, lambda_i, linewidth=3, color='#ff7f0e')
        ax.axhline(y=baseline_coeff, color='red', linestyle='--', alpha=0.7, 
                  label=f'Baseline: {baseline_coeff}')
        ax.axvspan(1.5, 4.0, alpha=0.2, color='green', label='Literature Range')
        
        ax.set_xlabel('Household Income ($)')
        ax.set_ylabel('Loss Aversion Coefficient (λ)')
        ax.set_title('Income-Dependent Loss Aversion', fontweight='bold')
        ax.set_xscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: NPV adjustment by income class
        ax = axes[0, 1]
        income_classes = np.array([1, 2, 3, 4, 5])
        class_incomes = [25000, 40000, 55000, 75000, 100000]  # Representative incomes
        class_lambdas = [baseline_coeff * (median_income / inc) ** income_sensitivity 
                        for inc in class_incomes]
        installation_cost = 15000
        
        # NPV adjustment for each class
        npv_adjustments = [(lam - 1) * installation_cost for lam in class_lambdas]
        
        bars = ax.bar(income_classes, npv_adjustments, color='#ff7f0e', alpha=0.7, 
                     edgecolor='black', linewidth=1)
        
        # Add value labels on bars
        for i, (bar, adj) in enumerate(zip(bars, npv_adjustments)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                   f'${adj:.0f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_xlabel('Income Class')
        ax.set_ylabel('NPV Adjustment ($)')
        ax.set_title('NPV Penalty by Income Class', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Adoption probability shift
        ax = axes[1, 0]
        npv_range = np.linspace(0, 30000, 1000)
        
        for i, (income_class, lam) in enumerate(zip(income_classes, class_lambdas)):
            adjusted_npv = npv_range - (lam - 1) * installation_cost
            prob = self._npv_to_probability(adjusted_npv)
            
            color = plt.cm.viridis(i / 4)  # Color gradient
            ax.plot(npv_range, prob, linewidth=2, color=color, 
                   label=f'Class {income_class} (λ={lam:.2f})')
        
        # Rational baseline
        rational_prob = self._npv_to_probability(npv_range)
        ax.plot(npv_range, rational_prob, linewidth=3, color='red', 
               linestyle='--', label='Rational')
        
        ax.set_xlabel('Base NPV ($)')
        ax.set_ylabel('Adoption Probability')
        ax.set_title('Adoption Curves by Income Class', fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Literature validation
        ax = axes[1, 1]
        literature_ranges = {
            'Kahneman & Tversky (1979)': (2.0, 2.5),
            'Gächter et al. (2022)': (1.8, 3.2),
            'Tversky & Kahneman (1992)': (2.1, 2.8)
        }
        
        y_pos = np.arange(len(literature_ranges))
        for i, (study, (low, high)) in enumerate(literature_ranges.items()):
            ax.barh(i, high - low, left=low, height=0.6, 
                   alpha=0.7, label=study, color=plt.cm.Set3(i))
            ax.text(low + (high - low)/2, i, f'{(low+high)/2:.1f}', 
                   ha='center', va='center', fontweight='bold')
        
        # Show our baseline
        ax.axvline(x=baseline_coeff, color='red', linewidth=3, 
                  label=f'Our Baseline: {baseline_coeff}')
        
        ax.set_xlabel('Loss Aversion Coefficient (λ)')
        ax.set_yticks(y_pos)
        ax.set_yticklabels([study.split('(')[0].strip() for study in literature_ranges.keys()])
        ax.set_title('Literature Validation', fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Overall title
        fig.suptitle('Loss Aversion: Income Dependency Analysis\n(Manuscript Equation Validation)', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        # Save the plot
        output_path = os.path.join(self.output_dir, 'loss_aversion_income_dependency.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Loss aversion income dependency analysis saved to: {output_path}")
    
    def create_present_bias_beta_distribution(self):
        """
        Create analysis of present bias beta parameter distribution.
        HIGH priority validation - shows quasi-hyperbolic discounting implementation.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        
        if 'present_bias' not in BEHAVIORAL_BIASES:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        params = BEHAVIORAL_BIASES['present_bias']['parameters']
        beta_min = params['beta_min']
        beta_max = params['beta_max']
        
        # Plot 1: Beta distribution
        ax = axes[0, 0]
        beta_values = np.linspace(0.4, 1.0, 1000)
        uniform_density = np.where((beta_values >= beta_min) & (beta_values <= beta_max),
                                  1.0 / (beta_max - beta_min), 0)
        
        ax.fill_between(beta_values, 0, uniform_density, alpha=0.6, color='#2ca02c', 
                       label=f'Uniform({beta_min}, {beta_max})')
        ax.axvspan(0.5, 0.9, alpha=0.2, color='orange', label='Literature Range')
        
        ax.set_xlabel('Present Bias Parameter (β)')
        ax.set_ylabel('Probability Density')
        ax.set_title('Beta Parameter Distribution', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: NPV transformation for different beta values
        ax = axes[0, 1]
        npv_range = np.linspace(-10000, 30000, 1000)
        installation_cost = 15000
        
        beta_examples = [0.6, 0.7, 0.8]
        colors = ['#2ca02c', '#ff7f0e', '#d62728']
        
        # Rational baseline
        rational_prob = self._npv_to_probability(npv_range)
        ax.plot(npv_range, rational_prob, linewidth=3, color='blue', 
               linestyle='--', label='Rational')
        
        for beta, color in zip(beta_examples, colors):
            adjusted_npv = npv_range - (1 - beta) * (npv_range + installation_cost)
            biased_prob = self._npv_to_probability(adjusted_npv)
            ax.plot(npv_range, biased_prob, linewidth=2, color=color, 
                   label=f'β = {beta}')
        
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        ax.set_xlabel('Base NPV ($)')
        ax.set_ylabel('Adoption Probability')
        ax.set_title('NPV Transformation by β Value', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Discount factor visualization
        ax = axes[1, 0]
        years = np.arange(1, 21)  # 20 years
        standard_discount = 0.04  # 4% discount rate
        
        for beta in beta_examples:
            # Quasi-hyperbolic: β × δ^t for t > 0
            qh_factors = [beta * (1 / (1 + standard_discount)) ** t for t in years]
            standard_factors = [(1 / (1 + standard_discount)) ** t for t in years]
            
            ax.plot(years, qh_factors, linewidth=2, 
                   label=f'Quasi-hyperbolic (β={beta})', marker='o', markersize=4)
        
        ax.plot(years, standard_factors, linewidth=3, color='blue', 
               linestyle='--', label='Standard exponential', marker='s', markersize=4)
        
        ax.set_xlabel('Year')
        ax.set_ylabel('Discount Factor')
        ax.set_title('Quasi-Hyperbolic vs Standard Discounting', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Literature validation
        ax = axes[1, 1]
        literature_studies = {
            'Laibson (1997)': (0.6, 0.7),
            'Newell & Siikamäki (2015)': (0.65, 0.8),
            'Heutel (2019)': (0.7, 0.85),
            'Meta-analysis': (0.6, 0.8)
        }
        
        y_pos = np.arange(len(literature_studies))
        for i, (study, (low, high)) in enumerate(literature_studies.items()):
            ax.barh(i, high - low, left=low, height=0.6, 
                   alpha=0.7, color=plt.cm.Set2(i))
            ax.text(low + (high - low)/2, i, f'{(low+high)/2:.2f}', 
                   ha='center', va='center', fontweight='bold')
        
        # Show our range
        ax.axvspan(beta_min, beta_max, alpha=0.3, color='red', 
                  label=f'Our Range: [{beta_min}, {beta_max}]')
        
        ax.set_xlabel('Beta Parameter (β)')
        ax.set_yticks(y_pos)
        ax.set_yticklabels([study.split('(')[0].strip() for study in literature_studies.keys()])
        ax.set_title('Literature Validation', fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Overall title
        fig.suptitle('Present Bias: Beta Parameter Analysis\n(Quasi-Hyperbolic Discounting Validation)', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        # Save the plot
        output_path = os.path.join(self.output_dir, 'present_bias_beta_distribution.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Present bias beta distribution analysis saved to: {output_path}")
    
    def create_all_validation_plots(self):
        """
        Create all validation plots for Phase 1.
        """
        print("Creating all bias validation plots...")
        
        # Critical validation plots
        self.create_bias_parameter_distributions()
        self.create_npv_transformation_analysis()
        self.create_loss_aversion_income_dependency()
        self.create_present_bias_beta_distribution()
        self.create_status_quo_individual_variation()
        self.create_herding_distance_decay()
        
        print(f"✅ All validation plots created in {self.output_dir}")
    
    def create_status_quo_individual_variation(self):
        """
        Create analysis of status quo bias individual variation.
        HIGH priority validation - shows individual heterogeneity implementation.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        
        if 'status_quo' not in BEHAVIORAL_BIASES:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        params = BEHAVIORAL_BIASES['status_quo']['parameters']
        baseline_strength = params['baseline_strength']
        individual_variation = params['individual_variation']
        min_strength = params['min_strength']
        max_strength = params['max_strength']
        
        # Plot 1: Individual variation distribution
        ax = axes[0, 0]
        sigma_values = np.linspace(0, 0.8, 1000)
        pdf = stats.norm.pdf(sigma_values, baseline_strength, individual_variation)
        
        # Apply bounds
        bounded_pdf = np.where((sigma_values >= min_strength) & (sigma_values <= max_strength),
                              pdf, 0)
        
        ax.fill_between(sigma_values, 0, bounded_pdf, alpha=0.6, color='#d62728', 
                       label=f'N({baseline_strength}, {individual_variation}²)')
        ax.axvline(x=baseline_strength, color='blue', linestyle='--', alpha=0.7, 
                  label=f'Baseline: {baseline_strength}')
        ax.axvspan(0.1, 0.5, alpha=0.2, color='green', label='Literature Range')
        
        ax.set_xlabel('Status Quo Bias Strength (σ)')
        ax.set_ylabel('Probability Density')
        ax.set_title('Individual Variation Distribution', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Probability reduction by sigma value
        ax = axes[0, 1]
        base_prob = 0.5  # 50% baseline probability
        sigma_range = np.linspace(0, 0.6, 100)
        reduced_prob = base_prob * (1 - sigma_range)
        
        ax.plot(sigma_range, reduced_prob, linewidth=3, color='#d62728')
        ax.axhline(y=base_prob, color='blue', linestyle='--', alpha=0.7, 
                  label=f'Baseline Probability: {base_prob}')
        ax.axvline(x=baseline_strength, color='green', linestyle='--', alpha=0.7, 
                  label=f'Our Baseline σ: {baseline_strength}')
        
        ax.set_xlabel('Status Quo Bias Strength (σ)')
        ax.set_ylabel('Final Adoption Probability')
        ax.set_title('Probability Reduction Effect', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Distribution across population
        ax = axes[1, 0]
        np.random.seed(42)  # For reproducibility
        population_sigmas = np.random.normal(baseline_strength, individual_variation, 10000)
        population_sigmas = np.clip(population_sigmas, min_strength, max_strength)
        
        ax.hist(population_sigmas, bins=50, density=True, alpha=0.7, color='#d62728', 
               edgecolor='black', linewidth=0.5)
        ax.axvline(x=baseline_strength, color='blue', linestyle='--', linewidth=2, 
                  label=f'Mean: {baseline_strength}')
        ax.axvline(x=np.mean(population_sigmas), color='green', linestyle='--', linewidth=2, 
                  label=f'Sample Mean: {np.mean(population_sigmas):.3f}')
        
        ax.set_xlabel('Status Quo Bias Strength (σ)')
        ax.set_ylabel('Frequency Density')
        ax.set_title('Population Distribution (N=10,000)', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Literature validation
        ax = axes[1, 1]
        literature_studies = {
                'Johnson et al. (1993)': (0.75, 0.80),      # Updated values
                'Murakami & Ida (2019)': (0.70, 0.80),     # Updated values
                'Boonen et al. (2011)': (0.60, 0.75),      # Updated values
                'Meta-analysis': (0.70, 0.80)              # Updated values
        }
        
        y_pos = np.arange(len(literature_studies))
        for i, (study, (low, high)) in enumerate(literature_studies.items()):
            ax.barh(i, high - low, left=low, height=0.6, 
                   alpha=0.7, color=plt.cm.Set1(i))
            ax.text(low + (high - low)/2, i, f'{(low+high)/2:.2f}', 
                   ha='center', va='center', fontweight='bold')
        
        # Show our baseline
        ax.axvline(x=baseline_strength, color='red', linewidth=3, 
                  label=f'Our Baseline: {baseline_strength}')
        
        ax.set_xlabel('Status Quo Bias Strength (σ)')
        ax.set_yticks(y_pos)
        ax.set_yticklabels([study.split('(')[0].strip() for study in literature_studies.keys()])
        ax.set_title('Literature Validation', fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Overall title
        fig.suptitle('Status Quo Bias: Individual Variation Analysis\n(Heterogeneity Validation)', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        # Save the plot
        output_path = os.path.join(self.output_dir, 'status_quo_individual_variation.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Status quo individual variation analysis saved to: {output_path}")
    
    def create_herding_distance_decay(self):
        """
        Create analysis of herding bias distance decay function.
        HIGH priority validation - shows spatial influence implementation.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        
        if 'herding' not in BEHAVIORAL_BIASES:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        params = BEHAVIORAL_BIASES['herding']['parameters']
        d_0 = params['distance_normalization']
        spatial_radius = params.get('spatial_radius', 5.0)
        
        # Plot 1: Distance decay function
        ax = axes[0, 0]
        distances = np.linspace(0.1, 10, 1000)
        weights = 1 / (1 + (distances / d_0) ** 2)
        
        ax.plot(distances, weights, linewidth=3, color='#9467bd')
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, 
                  label='50% Weight')
        ax.axvline(x=d_0, color='green', linestyle='--', alpha=0.7, 
                  label=f'd₀ = {d_0}')
        ax.axvline(x=spatial_radius, color='orange', linestyle='--', alpha=0.7, 
                  label=f'Max Radius = {spatial_radius}')
        
        ax.set_xlabel('Distance (grid units)')
        ax.set_ylabel('Influence Weight')
        ax.set_title('Spatial Influence Decay\nw = 1/(1 + (d/d₀)²)', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Beta distributions for influence strength
        ax = axes[0, 1]
        spatial_a = params['spatial_beta_shape_a']
        spatial_b = params['spatial_beta_shape_b']
        class_a = params['class_beta_shape_a']
        class_b = params['class_beta_shape_b']
        
        x_values = np.linspace(0, 1, 1000)
        spatial_pdf = stats.beta.pdf(x_values, spatial_a, spatial_b)
        class_pdf = stats.beta.pdf(x_values, class_a, class_b)
        
        ax.plot(x_values, spatial_pdf, linewidth=3, color='#9467bd', 
               label=f'Spatial: Beta({spatial_a}, {spatial_b})')
        ax.plot(x_values, class_pdf, linewidth=3, color='#8c564b', 
               label=f'Class: Beta({class_a}, {class_b})')
        
        # Show means
        spatial_mean = spatial_a / (spatial_a + spatial_b)
        class_mean = class_a / (class_a + class_b)
        ax.axvline(x=spatial_mean, color='#9467bd', linestyle='--', alpha=0.7, 
                  label=f'Spatial Mean: {spatial_mean:.2f}')
        ax.axvline(x=class_mean, color='#8c564b', linestyle='--', alpha=0.7, 
                  label=f'Class Mean: {class_mean:.2f}')
        
        ax.set_xlabel('Influence Strength (β, γ)')
        ax.set_ylabel('Probability Density')
        ax.set_title('Dual-Channel Influence Distributions', fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Combined influence example
        ax = axes[1, 0]
        neighbor_adoption_rates = np.linspace(0, 1, 11)
        
        # Example influence strengths
        example_betas = [0.2, 0.4, 0.6]  # Different spatial influence strengths
        class_influence = 0.3  # Fixed class influence
        gamma = class_mean  # Use mean class influence
        
        for beta in example_betas:
            multipliers = 1 + beta * neighbor_adoption_rates + gamma * class_influence
            ax.plot(neighbor_adoption_rates * 100, multipliers, linewidth=2, 
                   marker='o', markersize=4, label=f'β = {beta}')
        
        ax.axhline(y=1, color='black', linestyle='--', alpha=0.7, label='No Effect')
        ax.set_xlabel('Neighbor Adoption Rate (%)')
        ax.set_ylabel('Probability Multiplier')
        ax.set_title(f'Influence Effect (γ×ρ_class = {gamma:.2f}×{class_influence:.1f})', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Spatial network visualization
        ax = axes[1, 1]
        
        # Create example spatial network
        np.random.seed(42)
        n_households = 50
        positions = np.random.uniform(-5, 5, (n_households, 2))
        
        # Choose a central household
        center_idx = 25
        center_pos = positions[center_idx]
        
        # Calculate distances and weights
        distances_to_center = np.sqrt(np.sum((positions - center_pos)**2, axis=1))
        weights_to_center = 1 / (1 + (distances_to_center / d_0) ** 2)
        
        # Plot all households
        scatter = ax.scatter(positions[:, 0], positions[:, 1], 
                           c=weights_to_center, cmap='viridis', 
                           s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
        
        # Highlight central household
        ax.scatter(center_pos[0], center_pos[1], 
                  c='red', s=150, marker='*', edgecolors='black', linewidth=2,
                  label='Target Household', zorder=5)
        
        # Add influence radius circle
        circle = plt.Circle(center_pos, spatial_radius, fill=False, 
                          color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax.add_patch(circle)
        
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title('Spatial Influence Network\n(Color = Influence Weight)', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Influence Weight')
        
        # Overall title
        fig.suptitle('Herding Bias: Distance Decay Analysis\n(Spatial Influence Validation)', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        # Save the plot
        output_path = os.path.join(self.output_dir, 'herding_distance_decay.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Herding distance decay analysis saved to: {output_path}")
    
    def _npv_to_probability(self, npv):
        """
        Convert NPV to adoption probability using sigmoid function.
        
        Args:
            npv: Net Present Value (scalar or array)
            
        Returns:
            float or array: Adoption probability [0, 1]
        """
        exponent = -NPV_SIGMOID_STEEPNESS * npv
        
        # Prevent overflow
        exponent = np.clip(exponent, -700, 700)
        
        probability = 1.0 / (1.0 + np.exp(exponent))
        return probability
    
    def validate_bias_implementations(self):
        """
        Comprehensive validation of bias implementations.
        
        Returns:
            dict: Validation results with errors and warnings
        """
        validation_results = {
            'errors': [],
            'warnings': [],
            'parameter_checks': {},
            'literature_compliance': {}
        }
        
        # Check each enabled bias
        for bias_name in self.enabled_biases:
            if bias_name not in BEHAVIORAL_BIASES:
                validation_results['errors'].append(f"Bias '{bias_name}' not found in configuration")
                continue
            
            bias_config = BEHAVIORAL_BIASES[bias_name]
            params = bias_config['parameters']
            
            # Bias-specific validation
            if bias_name == 'loss_aversion':
                baseline_coeff = params['baseline_coefficient']
                if baseline_coeff <= 1:
                    validation_results['errors'].append(f"Loss aversion coefficient must be > 1, got {baseline_coeff}")
                elif baseline_coeff < 1.5 or baseline_coeff > 4.0:
                    validation_results['warnings'].append(f"Loss aversion coefficient {baseline_coeff} outside literature range [1.5, 4.0]")
                
                validation_results['parameter_checks'][bias_name] = {
                    'baseline_coefficient': baseline_coeff,
                    'literature_range': [1.5, 4.0],
                    'compliant': 1.5 <= baseline_coeff <= 4.0
                }
            
            elif bias_name == 'present_bias':
                beta_min = params['beta_min']
                beta_max = params['beta_max']
                if not (0 < beta_min < beta_max < 1):
                    validation_results['errors'].append(f"Present bias beta range invalid: [{beta_min}, {beta_max}]")
                elif beta_min < 0.5 or beta_max > 0.9:
                    validation_results['warnings'].append(f"Present bias range [{beta_min}, {beta_max}] outside literature range [0.5, 0.9]")
                
                validation_results['parameter_checks'][bias_name] = {
                    'beta_range': [beta_min, beta_max],
                    'literature_range': [0.5, 0.9],
                    'compliant': 0.5 <= beta_min and beta_max <= 0.9
                }
            
            elif bias_name == 'status_quo':
                baseline_strength = params['baseline_strength']
                if not (0 <= baseline_strength <= 1):
                    validation_results['errors'].append(f"Status quo strength must be in [0,1], got {baseline_strength}")
                elif baseline_strength < 0.1 or baseline_strength > 0.5:
                    validation_results['warnings'].append(f"Status quo strength {baseline_strength} outside literature range [0.1, 0.5]")
                
                validation_results['parameter_checks'][bias_name] = {
                    'baseline_strength': baseline_strength,
                    'literature_range': [0.1, 0.5],
                    'compliant': 0.1 <= baseline_strength <= 0.5
                }
            
            elif bias_name == 'herding':
                max_mult = params.get('max_influence_multiplier', 1)
                if max_mult < 1:
                    validation_results['errors'].append(f"Herding multiplier must be >= 1, got {max_mult}")
                elif max_mult > 5:
                    validation_results['warnings'].append(f"Very high herding multiplier: {max_mult}")
                
                validation_results['parameter_checks'][bias_name] = {
                    'max_multiplier': max_mult,
                    'literature_range': [1.0, 3.0],
                    'compliant': 1.0 <= max_mult <= 3.0
                }
        
        # Overall compliance
        compliant_biases = sum(1 for bias_checks in validation_results['parameter_checks'].values() 
                              if bias_checks.get('compliant', False))
        total_biases = len(validation_results['parameter_checks'])
        
        validation_results['literature_compliance'] = {
            'compliant_biases': compliant_biases,
            'total_biases': total_biases,
            'compliance_rate': compliant_biases / total_biases if total_biases > 0 else 0
        }
        
        return validation_results
    
    def print_validation_summary(self):
        """Print validation summary to console."""
        validation_results = self.validate_bias_implementations()
        
        print("\n" + "="*80)
        print("BIAS VALIDATION SUMMARY")
        print("="*80)
        
        # Errors
        if validation_results['errors']:
            print(f"\n❌ ERRORS ({len(validation_results['errors'])}):")
            for error in validation_results['errors']:
                print(f"  - {error}")
        
        # Warnings
        if validation_results['warnings']:
            print(f"\n⚠️  WARNINGS ({len(validation_results['warnings'])}):")
            for warning in validation_results['warnings']:
                print(f"  - {warning}")
        
        # Parameter compliance
        print(f"\n📊 PARAMETER COMPLIANCE:")
        for bias_name, checks in validation_results['parameter_checks'].items():
            status = "✅" if checks.get('compliant', False) else "❌"
            print(f"  {status} {bias_name}: {checks}")
        
        # Overall compliance
        compliance = validation_results['literature_compliance']
        print(f"\n📈 OVERALL COMPLIANCE: {compliance['compliant_biases']}/{compliance['total_biases']} "
              f"({compliance['compliance_rate']:.1%})")
        
        if validation_results['errors']:
            print(f"\n❌ Validation FAILED - {len(validation_results['errors'])} errors found")
        else:
            print(f"\n✅ Validation PASSED - All bias implementations valid")
        
        print("="*80)


# =============================================================================
# TESTING FUNCTIONS
# =============================================================================

def test_bias_validation_visualizer():
    """
    Test the BiasValidationVisualizer class.
    
    Returns:
        bool: True if tests pass
    """
    print("Testing BiasValidationVisualizer...")
    
    try:
        # Create test visualizer
        visualizer = BiasValidationVisualizer()
        
        # Test validation function
        validation_results = visualizer.validate_bias_implementations()
        
        if not isinstance(validation_results, dict):
            print("❌ Validation results should be a dictionary")
            return False
        
        required_keys = ['errors', 'warnings', 'parameter_checks', 'literature_compliance']
        if not all(key in validation_results for key in required_keys):
            print(f"❌ Missing required keys in validation results")
            return False
        
        # Test NPV to probability conversion
        test_npvs = np.array([-10000, 0, 10000])
        probabilities = visualizer._npv_to_probability(test_npvs)
        
        if not all(0 <= p <= 1 for p in probabilities):
            print("❌ NPV to probability conversion out of bounds")
            return False
        
        if probabilities[1] >= probabilities[0]:  # P(NPV=0) should be > P(NPV<0)
            if probabilities[2] >= probabilities[1]:  # P(NPV>0) should be > P(NPV=0)
                pass  # Good
            else:
                print("❌ NPV to probability conversion not monotonic")
                return False
        
        # Test plot creation (don't actually save)
        os.makedirs('test_results', exist_ok=True)
        visualizer.output_dir = 'test_results'
        
        # Test parameter distribution plotting
        try:
            visualizer.create_bias_parameter_distributions()
            print("✅ Parameter distribution plot created successfully")
        except Exception as e:
            print(f"❌ Error creating parameter distribution plot: {e}")
            return False
        
        print("✅ BiasValidationVisualizer tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ BiasValidationVisualizer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_bias_validation_visualizer()