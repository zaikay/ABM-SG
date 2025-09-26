# data/bias_effects_analyzer.py V1.0
from matplotlib import pyplot as plt
import numpy as np
from simulation.utils.parameters import get_enabled_biases, get_scenario_colors


class EnhancedBiasEffectsAnalyzer:
    """Enhanced bias effect quantification with multiple metrics"""
    
    def calculate_comprehensive_bias_effects(self, scenario_data):
        """Calculate multiple bias effect metrics"""
        
        baseline_scenario = 'rational'  # Non-deterministic rational
        effects_summary = {}
        
        for bias_scenario in get_enabled_biases():
            effects = {}
            
            # 1. TEMPORAL EFFECT - Area between curves
            baseline_curve = scenario_data[f'{baseline_scenario}_AdoptionRate'].values
            bias_curve = scenario_data[f'{bias_scenario}_AdoptionRate'].values
            
            # Area between curves (cumulative impact)
            temporal_effect = np.trapz(bias_curve - baseline_curve, 
                                     dx=1/12)  # Monthly resolution
            effects['temporal_impact'] = temporal_effect
            
            # 2. VELOCITY EFFECT - Peak adoption rate difference
            baseline_velocity = np.diff(baseline_curve).max()
            bias_velocity = np.diff(bias_curve).max()
            effects['velocity_impact'] = bias_velocity - baseline_velocity
            
            # 3. TIME-TO-THRESHOLD - When scenarios reach 50% adoption
            threshold = 0.5
            baseline_time = self._time_to_threshold(baseline_curve, threshold)
            bias_time = self._time_to_threshold(bias_curve, threshold)
            effects['time_shift'] = bias_time - baseline_time if baseline_time and bias_time else None
            
            # 4. FINAL IMPACT - Traditional end-point difference
            effects['final_impact'] = bias_curve[-1] - baseline_curve[-1]
            
            # 5. PEAK IMPACT - Maximum difference at any time point
            effects['peak_impact'] = np.max(np.abs(bias_curve - baseline_curve))
            
            # 6. RELATIVE IMPACT - Percentage change from baseline
            effects['relative_impact'] = (bias_curve[-1] / baseline_curve[-1] - 1) * 100
            
            effects_summary[bias_scenario] = effects
        
        return effects_summary
    
    def _time_to_threshold(self, adoption_curve, threshold):
        """Find first time when adoption rate crosses threshold"""
        crossing_indices = np.where(adoption_curve >= threshold)[0]
        return crossing_indices[0] / 12 if len(crossing_indices) > 0 else None  # Convert to years
    
    def plot_enhanced_bias_effects(self, effects_summary, output_dir="results"):
        """Create comprehensive bias effects visualization"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        biases = list(effects_summary.keys())
        colors = [get_scenario_colors()[bias] for bias in biases]
        
        # Plot 1: Temporal Impact (Area between curves)
        temporal_impacts = [effects_summary[bias]['temporal_impact'] for bias in biases]
        bars1 = axes[0, 0].bar(biases, temporal_impacts, color=colors, alpha=0.7)
        axes[0, 0].set_title('Cumulative Temporal Impact', fontweight='bold')
        axes[0, 0].set_ylabel('Area Difference (adoption-years)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Velocity Impact
        velocity_impacts = [effects_summary[bias]['velocity_impact'] for bias in biases]
        bars2 = axes[0, 1].bar(biases, velocity_impacts, color=colors, alpha=0.7)
        axes[0, 1].set_title('Peak Adoption Velocity Impact', fontweight='bold')
        axes[0, 1].set_ylabel('Velocity Difference (adoption rate/month)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Time Shift Effect
        time_shifts = [effects_summary[bias]['time_shift'] for bias in biases if effects_summary[bias]['time_shift'] is not None]
        valid_biases = [bias for bias in biases if effects_summary[bias]['time_shift'] is not None]
        valid_colors = [get_scenario_colors()[bias] for bias in valid_biases]
        
        if time_shifts:
            bars3 = axes[0, 2].bar(valid_biases, time_shifts, color=valid_colors, alpha=0.7)
            axes[0, 2].set_title('Time Shift to 50% Adoption', fontweight='bold')
            axes[0, 2].set_ylabel('Time Difference (years)')
            axes[0, 2].tick_params(axis='x', rotation=45)
        
        # Plot 4: Peak Impact
        peak_impacts = [effects_summary[bias]['peak_impact'] for bias in biases]
        bars4 = axes[1, 0].bar(biases, peak_impacts, color=colors, alpha=0.7)
        axes[1, 0].set_title('Maximum Instantaneous Impact', fontweight='bold')
        axes[1, 0].set_ylabel('Peak Difference (adoption rate)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Plot 5: Relative Impact (%)
        relative_impacts = [effects_summary[bias]['relative_impact'] for bias in biases]
        bars5 = axes[1, 1].bar(biases, relative_impacts, color=colors, alpha=0.7)
        axes[1, 1].set_title('Relative Final Impact', fontweight='bold')
        axes[1, 1].set_ylabel('Percentage Change (%)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Plot 6: Comprehensive Effect Score
        # Weighted combination of all effects (normalized)
        effect_scores = []
        for bias in biases:
            score = (
                abs(effects_summary[bias]['temporal_impact']) * 0.3 +
                abs(effects_summary[bias]['velocity_impact']) * 0.2 + 
                abs(effects_summary[bias]['peak_impact']) * 0.2 +
                abs(effects_summary[bias]['final_impact']) * 0.3
            )
            effect_scores.append(score)
        
        bars6 = axes[1, 2].bar(biases, effect_scores, color=colors, alpha=0.7)
        axes[1, 2].set_title('Comprehensive Effect Score', fontweight='bold')
        axes[1, 2].set_ylabel('Weighted Effect Magnitude')
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for ax, values in zip(axes.flat, [temporal_impacts, velocity_impacts, time_shifts if time_shifts else [0]*len(biases), 
                                         peak_impacts, relative_impacts, effect_scores]):
            for bar, value in zip(ax.containers[0] if ax.containers else [], values):
                if value != 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(abs(v) for v in values),
                           f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        plt.suptitle('Enhanced Bias Effects Analysis', fontsize=16, fontweight='bold', y=0.95)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.savefig(f"{output_dir}/enhanced_bias_effects.png", dpi=300, bbox_inches='tight')
        plt.close()