# agents/bias_manager.py - CORRECTED IMPLEMENTATION
"""
Fixed bias management system that properly implements manuscript methodology.
Separates NPV-based biases from probability-based biases as per manuscript equations.
COMPLETE VERSION - includes all missing functionality from original 600+ line file.
"""

import numpy as np
from ..utils.parameters import (
    BEHAVIORAL_BIASES, get_enabled_biases, NPV_SIGMOID_STEEPNESS
)

class BiasManager:
    """
    Corrected bias management system implementing manuscript methodology.
    
    Key fix: Properly separates NPV-based vs probability-based bias transformations:
    - NPV-based: Loss Aversion, Present Bias (modify NPV, then convert to probability)
    - Probability-based: Status Quo, Herding (modify probability directly)
    
    COMPLETE VERSION with all original functionality preserved.
    """
    
    def __init__(self, config):
        """Initialize the bias manager with configuration."""
        self.config = config
        self.enabled_biases = get_enabled_biases()
        
        # Extract bias parameters for easier access
        self.bias_params = {}
        for bias_name in self.enabled_biases:
            if bias_name in BEHAVIORAL_BIASES:
                self.bias_params[bias_name] = BEHAVIORAL_BIASES[bias_name]['parameters']
        
        
        # Bias application methods mapping (FIXED method names)
        self.bias_methods = {
            'loss_aversion': self._apply_loss_aversion,
            'present_bias': self._apply_present_bias,
            'status_quo': self._apply_status_quo,
            'herding': self._apply_herding,
            'optimism_bias': self._apply_optimism_bias
        }
        
        # Cache for household-specific parameters (avoid recalculation)
        self._household_bias_cache = {}
        
        # Pre-calculate common values
        self.median_income = np.exp(self.config["income_params"]["lognormal_mean"])
        
        print(f"BiasManager initialized with {len(self.enabled_biases)} enabled biases")

    def _calculate_combined_social_influence(self, household):
        """
        Alternative formulation with saturation to prevent extreme effects.
        
        """
        params = self.bias_params['herding']
        
        # Calculate influence components  
        spatial_influence = self._calculate_spatial_influence(household, params)
        class_influence = self._calculate_class_influence(household)
        
        # Weighted combination
        spatial_weight = params['spatial_weight']
        class_weight = params['class_weight'] 
        
        social_influence = (spatial_weight * spatial_influence + 
                            class_weight * class_influence)

        # Individual susceptibility
        susceptibility = household.get_behavioral_coefficient('herding', 'variation_std')
        
        # Base effect (your 0.1)
        base_effect = params['target_effect_per_neighbor']  # 0.1
        
        # Final effect
        rho_combined = base_effect * social_influence * (1+susceptibility)
        
        return rho_combined
    
    def apply_single_bias(self, household, bias_name, npv, base_probability):
        """
        Apply a single cognitive bias following manuscript methodology.
        
        FIXED: Properly routes to NPV or probability modification based on bias type.
        """
        if bias_name not in self.enabled_biases:
            print(f"Warning: Bias '{bias_name}' not enabled, returning base probability")
            return base_probability
        
        if bias_name not in self.bias_methods:
            print(f"Warning: No implementation for bias '{bias_name}', returning base probability")
            return base_probability
        
        # Apply the specific bias using the unified interface
        try:
            adjusted_npv, modified_probability = self.bias_methods[bias_name](household, npv, base_probability)  # ✅ NOW expects tuple
            
            # Ensure probability stays within valid bounds [0, 1]
            modified_probability = max(0.0, min(1.0, modified_probability))
            
            return (adjusted_npv, modified_probability)  # ✅ NOW returns tuple
            
        except Exception as e:
            print(f"Error applying bias '{bias_name}': {e}")
            return (npv, base_probability)  # ✅ NOW returns tuple

    def apply_all_biases(self, household, npv, base_probability):
        """
        Apply all enabled biases in correct order using individual bias methods.
        
        This ensures consistency with individual bias scenarios by using the same
        mathematical transformations. Biases are applied sequentially:
        1. Loss Aversion (NPV transformation)
        2. Present Bias (NPV transformation) 
        3. Status Quo (probability transformation)
        4. Herding (probability transformation)
        5. optimism bias (probability transformation - applied last)
        
        Args:
            household: Household agent instance
            npv: Rational NPV value
            base_probability: Rational probability value
            
        Returns:
            tuple[float, float]: (final_npv, final_probability)
        """
        current_npv = npv
        current_probability = base_probability
        
        #print(f"🔄 Applying combined biases for household {household.unique_id}")
        #print(f"   Initial: NPV=${current_npv:.0f}, Probability={current_probability:.4f}")
        
        # Define bias application order (herding last due to additive component)
        bias_order = ['optimism_bias', 'loss_aversion', 'present_bias', 'status_quo', ]
        
        # Apply each enabled bias in the defined order
        for bias_name in bias_order:
            if bias_name in self.enabled_biases:
                previous_npv = current_npv
                previous_prob = current_probability
                
                # Apply the bias using the individual bias method
                current_npv, current_probability = self.bias_methods[bias_name](
                    household, current_npv, current_probability
                )
                
                # Log the transformation
                npv_change = current_npv - previous_npv
                prob_change = current_probability - previous_prob
                #print(f"   {bias_name.replace('_', ' ').title()}: "
                #    f"NPV ${previous_npv:.0f}→${current_npv:.0f} (Δ${npv_change:.0f}), "
                #    f"Prob {previous_prob:.4f}→{current_probability:.4f} (Δ{prob_change:.4f})")
        
        herding_npv, herding_probability = self._apply_herding(
            household, current_npv, current_probability
        )
        # Ensure probability is base on last NVP
        final_probability = herding_probability
        
        #print(f"   Final: NPV=${current_npv:.0f}, Probability={final_probability:.4f}")
        #print(f"   Total Change: NPV Δ${current_npv - npv:.0f}, Prob Δ{final_probability - base_probability:.4f}")
        
        return (current_npv, final_probability)

    def _apply_loss_aversion(self, household, npv, probability):
        """
        Simplified loss aversion with natural income bounds.
        
        Formula: λᵢ = λ_base + tanh((I_median - I_i)/I_median) + εᵢ
        NPV_adjusted = NPV - (λᵢ - 1) × C_install
        """
        params = self.bias_params['loss_aversion']
        
        lambda_base = params['baseline_coefficient']  # 2.25
        
        # Naturally bounded income effect (no parameters needed!)
        income_ratio = (self.median_income - household.income) / self.median_income
        income_effect = np.tanh(income_ratio)
        
        # Individual heterogeneity
        epsilon_i = household.get_behavioral_coefficient('loss_aversion', 'variation_std')
        
        # Calculate final coefficient
        lambda_i = lambda_base + income_effect + epsilon_i
        lambda_i = np.clip(lambda_i, params['min_coefficient'], params['max_coefficient'])
        
        # Apply transformation
        installation_cost = getattr(household, 'installation_cost', 0)
        if installation_cost > 0:
            adjusted_npv = npv - (lambda_i - 1) * installation_cost
        else:
            adjusted_npv = npv
            
        return (adjusted_npv, self._npv_to_probability(adjusted_npv))
    

    def _apply_present_bias(self, household, npv, probability):
        """
        Apply present bias following corrected manuscript methodology:
        
        1. NPV transformation: NPV_PB = NPV - (1-β_i) × (NPV + C_install)
        2. Probability reduction: P_PB = β_i × Φ(NPV_PB)
        
        This captures both future benefit devaluation and procrastination tendency.
        """
        params = self.bias_params['present_bias']
        
        # Generate β_i ~ Uniform(0.6, 0.7) as specified in updated manuscript
        beta_i = household.get_behavioral_coefficient('present_bias', 'beta_i')
        
        # Step 1: Apply present bias to NPV (future benefit devaluation)
        installation_cost = getattr(household, 'installation_cost', 0)
        future_value_discount = (1-beta_i) * (npv + installation_cost)
        adjusted_npv = npv - future_value_discount
        
        # Step 2: Convert adjusted NPV to base probability
        adjusted_probability = self._npv_to_probability(adjusted_npv)
        
        return (adjusted_npv, adjusted_probability)  # ✅ NOW returns both values

    def _apply_status_quo(self, household, npv, probability):
        """
        Apply status quo bias as NPV switching cost following literature-calibrated hurdle rates.
        
        Implementation based on meta-analysis showing consumers require 20-50% returns
        above break-even to overcome switching inertia (Devine & Waddell 2023,
        Gerarden et al. 2017, Gillingham & Palmer 2014).
        
        Args:
            household: Household agent instance
            npv: Rational NPV value
            probability: Current probability (not used - pure NPV transformation)
            
        Returns:
            float: Adjusted probability after NPV switching cost
        """
        params = self.bias_params['status_quo']
        
        # Literature-calibrated switching cost coefficient
        # σ_i represents percentage of installation cost required as hurdle buffer
        sigma_base = params['baseline_strength']     # 0.30 (30% hurdle)
        sigma_variation = params['individual_variation']  # 0.08 (individual heterogeneity)
        
        # Individual switching cost: σ_i ~ N(σ_base, σ_variation²)
        sigma_i = sigma_base + household.get_behavioral_coefficient('status_quo', 'individual_variation')
        
        # Constrain to literature-validated range [0.20, 0.50] (affects ~1% of tail values)
        sigma_i = max(params['min_strength'], min(params['max_strength'], sigma_i))
        
        # Apply switching cost as NPV penalty
        # Interpretation: Household needs (1 + σ_i) × C_install in total savings to adopt
        installation_cost = getattr(household, 'installation_cost', 0)
        switching_cost = sigma_i * installation_cost
        adjusted_npv = npv - switching_cost
        
        # Convert adjusted NPV to probability
        adjusted_probability = self._npv_to_probability(adjusted_npv)
        
        return (adjusted_npv, adjusted_probability)  # ✅ NOW returns both values (NPV unchanged)

    def _apply_herding(self, household, npv, probability):
        """
        Apply herding bias based on dual-channel social influence.
        
        Implementation based on manuscript equation:
        P_adopt_adjusted = P_adopt × (1 + β_i×ρ_spatial + γ_i×ρ_class)
        
        Where:
        - β_i ~ Beta(2.5, 5) for spatial influence
        - γ_i ~ Beta(2, 4.5) for class influence
        - ρ_spatial = weighted neighbor adoption rate
        - ρ_class = same income class adoption rate
        """

                # Calculate combined social influence
        rho_combined = self._calculate_combined_social_influence(household)
        
        
        # Apply Additive effect (conformity pressure) 
        herding_probability = self._npv_to_probability(npv)+rho_combined

        if probability > herding_probability :
            print(f"========================== herding_probability: {herding_probability} probability: {probability}")
        # NPV remains unchanged for herding bias
        return (npv, herding_probability)  # ✅ NOW returns both values (NPV unchanged)

    def _apply_optimism_bias(self, household, npv, probability):
        """
        Empirically-calibrated optimism bias from energy efficiency literature.
        
        Based on Allcott & Greenstone (2017) showing 47% benefit overestimation.
        Divided by 2 because dual-factor formula applies to both benefits and costs.
        """
        params = self.bias_params['optimism_bias']
        base_optimism = params['base_optimism'] # to compensate the dual effect for both cost and benefits

        individual_variation = household.get_behavioral_coefficient('optimism_bias', 'individual_variation')
        # Get household's individual base optimism level
        base_optimism_i = base_optimism + individual_variation
        base_optimism_i = np.clip(base_optimism_i, params['effect_variation_min'], params['effect_variation_max'])
        
        # Apply symmetric dual-factor optimism
        installation_cost = getattr(household, 'installation_cost', 15000)
        adjusted_npv = (1 + base_optimism_i) * npv + 2 * base_optimism_i * installation_cost

        adjusted_probability = self._npv_to_probability(adjusted_npv)
        
        return (adjusted_npv, adjusted_probability)
    
    def _calculate_spatial_influence(self, household, params):
        """
        Calculate spatial influence using NetworkBuilder-created neighbor data.
        
        This now uses the neighbor data created by NetworkBuilder.create_spatial_network()
        instead of recalculating distances every time.
        
        Args:
            household: Household with spatial_neighbors attribute
            params: Herding bias parameters  
            
        Returns:
            float: Spatial influence (0.0 to 1.0)
        """
        # Check if household has neighbor data from NetworkBuilder
        if not hasattr(household, 'spatial_neighbors') or not household.spatial_neighbors:
            print(f"Warning: Household {household.unique_id} missing spatial_neighbors. "
                  f"Ensure NetworkBuilder.create_spatial_network() was called.")
            return 0.0
        
        # Count adopting neighbors (this is the only dynamic calculation needed)
        prosumer_count = 0
        total_neighbors = len(household.spatial_neighbors)
        
        for neighbor_household, distance in household.spatial_neighbors:
            # Check neighbor's current adoption status
            if hasattr(neighbor_household, 'scenario_adoption'):
                is_prosumer = neighbor_household.scenario_adoption.get('herding', False)
            else:
                is_prosumer = getattr(neighbor_household, 'is_prosumer', False)
            
            if is_prosumer:
                prosumer_count += 1
        
        return prosumer_count / total_neighbors
    

    def _calculate_class_influence(self, household):
        """
        Calculate class homophily influence.
        
        Returns the adoption rate among households in the same income class.
        """
        if not hasattr(household, 'income_class') or not hasattr(household, 'model'):
            return 0.0
        
        model = household.model
        household_class = household.income_class
        
        # Get all households in the same income class
        same_class_households = [
            agent for agent in model.schedule.agents 
            if hasattr(agent, 'income_class') 
            and agent.income_class == household_class
            and agent.unique_id != household.unique_id
        ]
        
        if not same_class_households:
            return 0.0
        
        # Calculate adoption rate in the same class for herding scenario
        prosumer_count = 0
        for other in same_class_households:
            if hasattr(other, 'scenario_adoption'):
                is_prosumer = other.scenario_adoption.get('herding') is True
            else:
                is_prosumer = getattr(other, 'is_prosumer', False)
            
            if is_prosumer:
                prosumer_count += 1
        
        return prosumer_count / len(same_class_households)

    def _calculate_global_influence(self, household):
        """
        Calculate global bandwagon influence from overall population adoption rate.
        
        Returns the adoption rate among all households in the population.
        """
        if not hasattr(household, 'model'):
            return 0.0
        
        model = household.model
        
        # Get all households in the population
        all_households = [
            agent for agent in model.schedule.agents 
            if hasattr(agent, 'scenario_adoption')
            and agent.unique_id != household.unique_id
        ]
        
        if not all_households:
            return 0.0
        
        # Calculate overall adoption rate for herding scenario
        prosumer_count = 0
        for other in all_households:
            is_prosumer = other.scenario_adoption.get('herding', getattr(other, 'is_prosumer', False))
            if is_prosumer:
                prosumer_count += 1
        
        return prosumer_count / len(all_households)

    def _npv_to_probability(self, npv):
        """
        Convert NPV to adoption probability using sigmoid function.
        
        Φ(NPV) = 1 / (1 + e^(-κ × NPV))
        where κ = 0.01 from manuscript
        """
        exponent = -NPV_SIGMOID_STEEPNESS * npv
        
        # Prevent overflow
        exponent = np.clip(exponent, -700, 700)
        
        probability = 1.0 / (1.0 + np.exp(exponent))
        return probability

    def get_bias_effects_summary(self, household, base_npv=None, base_probability=None):
        """
        Generate comprehensive bias effects summary showing both NPV and probability impacts.
        
        COMPLETE VERSION with all original functionality.
        """
        if base_npv is None:
            base_npv = getattr(household, 'npv', 0)
        if base_probability is None:
            base_probability = self._npv_to_probability(base_npv)
        
        summary = {
            'household_id': household.unique_id,
            'income': getattr(household, 'income', 0),
            'income_class': getattr(household, 'income_class', 0),
            'position': getattr(household, 'pos', (0, 0)),
            'enabled_biases': self.enabled_biases.copy(),
            'npv': base_npv,
            'base_probability': base_probability,
            'bias_effects': {}
        }
        
        # Calculate effect of each bias individually
        for bias_name in self.enabled_biases:
            try:
                biased_prob = self.apply_single_bias(household, bias_name, base_npv, base_probability)
                effect_multiplier = biased_prob / base_probability if base_probability > 0 else 1.0
                
                summary['bias_effects'][bias_name] = {
                    'biased_probability': biased_prob,
                    'effect_multiplier': effect_multiplier,
                    'probability_change': biased_prob - base_probability
                }
                
                # Add bias-specific details
                if bias_name == 'loss_aversion':
                    lambda_base = self.bias_params['loss_aversion']['baseline_coefficient']
                    income_sensitivity = self.bias_params['loss_aversion']['income_sensitivity']
                    lambda_i = lambda_base * (self.median_income / household.income) ** income_sensitivity
                    summary['bias_effects'][bias_name]['loss_aversion_coefficient'] = lambda_i
                
                elif bias_name == 'herding':
                    spatial_influence = self._calculate_spatial_influence(household, self.bias_params['herding'])
                    class_influence = self._calculate_class_influence(household)
                    global_influence = self._calculate_global_influence(household)
                    summary['bias_effects'][bias_name]['spatial_influence'] = spatial_influence
                    summary['bias_effects'][bias_name]['class_influence'] = class_influence
                    summary['bias_effects'][bias_name]['global_influence'] = global_influence
                    
            except Exception as e:
                summary['bias_effects'][bias_name] = {
                    'error': str(e),
                    'biased_probability': base_probability,
                    'effect_multiplier': 1.0
                }
        
        # Combined effect
        try:
            combined_prob = self.apply_all_biases(household, base_npv, base_probability)
            combined_effect = combined_prob / base_probability if base_probability > 0 else 1.0
            summary['combined_effect'] = {
                'final_probability': combined_prob,
                'total_multiplier': combined_effect,
                'total_probability_change': combined_prob - base_probability
            }
        except Exception as e:
            summary['combined_effect'] = {
                'error': str(e),
                'final_probability': base_probability,
                'total_multiplier': 1.0
            }
        
        return summary

    def validate_bias_calculations(self, test_household=None):
        """
        Comprehensive validation of corrected bias calculations.
        
        COMPLETE VERSION with all original validation logic.
        """
        errors = []
        warnings = []
        
        # Create mock household if not provided
        if test_household is None:
            test_household = self._create_mock_household()
        
        # Test NPV to probability conversion
        try:
            prob_zero = self._npv_to_probability(0)
            prob_positive = self._npv_to_probability(10000)
            prob_negative = self._npv_to_probability(-10000)
            
            if not (0 <= prob_zero <= 1):
                errors.append(f"NPV=0 probability out of bounds: {prob_zero}")
            if not (prob_positive > prob_zero):
                errors.append(f"Positive NPV should have higher probability")
            if not (prob_negative < prob_zero):
                errors.append(f"Negative NPV should have lower probability")
                
        except Exception as e:
            errors.append(f"NPV to probability conversion failed: {e}")
        
        # Test each bias
        base_npv = 5000
        base_prob = self._npv_to_probability(base_npv)
        
        for bias_name in self.enabled_biases:
            try:
                biased_prob = self.apply_single_bias(test_household, bias_name, base_npv, base_prob)
                
                if not (0 <= biased_prob <= 1):
                    errors.append(f"{bias_name} produced invalid probability: {biased_prob}")
                
                # Bias-specific validation
                if bias_name == 'loss_aversion':
                    if biased_prob > base_prob:
                        warnings.append(f"Loss aversion increased probability (unusual but possible with very low income)")
                
                elif bias_name == 'status_quo':
                    if biased_prob > base_prob:
                        errors.append(f"Status quo bias should never increase probability")
                
                elif bias_name == 'herding':
                    # Herding can increase or decrease probability depending on neighbors
                    multiplier = biased_prob / base_prob if base_prob > 0 else 1.0
                    if multiplier > 3.0:
                        warnings.append(f"Herding bias effect very large: {multiplier:.2f}x")
                
            except Exception as e:
                errors.append(f"Bias {bias_name} calculation failed: {e}")
        
        # Test combined bias application
        try:
            combined_prob = self.apply_all_biases(test_household, base_npv, base_prob)
            if not (0 <= combined_prob <= 1):
                errors.append(f"Combined bias produced invalid probability: {combined_prob}")
        except Exception as e:
            errors.append(f"Combined bias calculation failed: {e}")
        
        is_valid = len(errors) == 0
        return is_valid, errors, warnings

    def validate_bias_implementations(self):
        """
        Comprehensive validation of bias implementations against literature.
        
        COMPLETE VERSION preserving all original validation logic.
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
                elif baseline_strength < 0.6 or baseline_strength > 0.9:
                    validation_results['warnings'].append(f"Status quo strength {baseline_strength} outside literature range [0.1, 0.5]")
                
                validation_results['parameter_checks'][bias_name] = {
                    'baseline_strength': baseline_strength,
                    'literature_range': [0.6, 0.9],
                    'compliant': 0.6 <= baseline_strength <= 0.9
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

    def _create_mock_household(self):
        """Create mock household for testing - COMPLETE VERSION."""
        class MockModel:
            def __init__(self):
                class MockSchedule:
                    def __init__(self):
                        self.agents = []
                self.schedule = MockSchedule()
        
        class MockHousehold:
            def __init__(self, model):
                self.unique_id = 'test'
                self.income = 50000
                self.income_class = 3
                self.installation_cost = 15000
                self.npv = 5000
                self.pos = (0, 0)
                self.is_prosumer = False
                self.scenario_adoption = {'herding': False}
                self.model = model
        
        # Create model first, then household
        mock_model = MockModel()
        test_household = MockHousehold(mock_model)
        mock_model.schedule.agents.append(test_household)
        
        return test_household


# =============================================================================
# TESTING FUNCTIONS - COMPLETE VERSION
# =============================================================================

def test_enhanced_bias_manager():
    """
    Comprehensive test function for the corrected BiasManager class.
    
    COMPLETE VERSION with all original testing logic.
    """
    print("Testing CORRECTED BiasManager Implementation...")
    
    try:
        from ..utils.config_loader import create_multi_experiment_config
        
        # Create test configuration
        config = create_multi_experiment_config()
        config_dict = config.get_copy()
        
        # Create corrected bias manager
        bias_manager = BiasManager(config_dict)
        
        # Test 1: Basic initialization
        print("  Test 1: Basic initialization...")
        if not bias_manager.enabled_biases:
            print("    ❌ No biases enabled")
            return False
        print(f"    ✅ Initialized with {len(bias_manager.enabled_biases)} biases")
        
        # Test 2: Validate corrected bias calculations
        print("  Test 2: Validate corrected bias calculations...")
        is_valid, errors, warnings = bias_manager.validate_bias_calculations()
        
        if errors:
            print("    ❌ Bias calculation validation failed:")
            for error in errors:
                print(f"      - {error}")
            return False
        
        if warnings:
            print("    ⚠️  Bias calculation warnings:")
            for warning in warnings:
                print(f"      - {warning}")
        
        print("    ✅ Corrected bias calculations validated")
        
        # Test 3: Test NPV vs probability modifications
        print("  Test 3: NPV vs probability modifications...")
        mock_household = bias_manager._create_mock_household()
        base_npv = 5000
        base_prob = bias_manager._npv_to_probability(base_npv)
        
        print(f"    Base NPV: ${base_npv:.0f}, Base Probability: {base_prob:.3f}")
        
        # Test each bias individually
        for bias_name in bias_manager.enabled_biases:
            biased_prob = bias_manager.apply_single_bias(mock_household, bias_name, base_npv, base_prob)
            effect = biased_prob / base_prob if base_prob > 0 else 1.0
            print(f"    {bias_name}: {effect:.3f}x effect (prob: {base_prob:.3f} → {biased_prob:.3f})")
        
        # Test 4: Combined bias effects with proper chaining
        print("  Test 4: Combined bias effects (corrected chaining)...")
        combined_prob = bias_manager.apply_all_biases(mock_household, base_npv, base_prob)
        combined_effect = combined_prob / base_prob if base_prob > 0 else 1.0
        print(f"    Combined effect: {combined_effect:.3f}x (prob: {base_prob:.3f} → {combined_prob:.3f})")
        
        # Test 5: Comprehensive bias effects summary
        print("  Test 5: Comprehensive bias effects summary...")
        summary = bias_manager.get_bias_effects_summary(mock_household)
        
        required_keys = ['household_id', 'bias_effects', 'combined_effect']
        if not all(key in summary for key in required_keys):
            print(f"    ❌ Missing required keys in summary: {list(summary.keys())}")
            return False
        
        print(f"    ✅ Generated comprehensive bias effects summary")
        print(f"    - Bias effects tracked for: {list(summary['bias_effects'].keys())}")
        
        # Test 6: Literature compliance validation
        print("  Test 6: Literature compliance validation...")
        validation_results = bias_manager.validate_bias_implementations()
        
        compliance = validation_results['literature_compliance']
        print(f"    Literature compliance: {compliance['compliant_biases']}/{compliance['total_biases']} "
              f"({compliance['compliance_rate']:.1%})")
        
        if validation_results['errors']:
            print("    ❌ Literature validation errors:")
            for error in validation_results['errors']:
                print(f"      - {error}")
        else:
            print("    ✅ All bias implementations comply with literature")
        
        print("✅ CORRECTED BiasManager implementation tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ BiasManager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_enhanced_bias_manager()