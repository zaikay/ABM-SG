# agents/multi_scenario_household.py V2.0 - PHASE 3 IMPLEMENTATION
"""
Enhanced multi-scenario household agent with detailed bias integration.
Implements comprehensive behavioral evaluation across all scenarios with proper herding dynamics.
"""

import numpy as np
from mesa import Agent
from .household import Household
from .bias_manager import BiasManager
from ..utils.parameters import get_all_scenarios, get_enabled_biases, NPV_SIGMOID_STEEPNESS
from .evaluation_triggers import EvaluationTriggers

class MultiScenarioHousehold(Household):
    """
    Enhanced multi-scenario household agent with comprehensive behavioral modeling.
    
    This agent extends the base Household class to:
    1. Calculate energy metrics once per timestep (efficient)
    2. Evaluate adoption decisions for all 6 scenarios (rational + 4 biases + combined)
    3. Track scenario-specific adoption status and timing with detailed metrics
    4. Support full herding bias through network interactions
    5. Provide comprehensive bias effects analysis
    """
    
    def __init__(self, unique_id, model, config, weather_patterns, grid_system):
        """
        Initialize enhanced multi-scenario household.
        
        Args:
            unique_id: Unique identifier
            model: Mesa model instance
            config: Simulation configuration
            weather_patterns: WeatherPatterns instance
            grid_system: GridSystem instance
        """
        # Initialize base household
        super().__init__(unique_id, model, config, weather_patterns, grid_system)
        
        # Get all scenarios from configuration
        self.scenarios = get_all_scenarios()
        
        # Initialize enhanced bias manager
        self.bias_manager = BiasManager(config)

        # Get evaluation triggers from model
        self.evaluation_triggers = model.evaluation_triggers
        
        # Track adoption status for each scenario
        self.scenario_adoption = {scenario: False for scenario in self.scenarios}
        self.adoption_months = {scenario: None for scenario in self.scenarios}
        
        # Track decision metrics for each scenario (for detailed analysis)
        self.scenario_npv = {scenario: None for scenario in self.scenarios}
        self.scenario_probability = {scenario: 0.0 for scenario in self.scenarios}
        self.scenario_installation_cost = {scenario: 0.0 for scenario in self.scenarios}
        
        # Track bias effects over time (for analysis)
        self.bias_effects_history = []
        self.current_bias_effects = {}
        
        # Social influence tracking for herding bias
        self.spatial_influence_history = []
        self.class_influence_history = []
        self.neighbor_adoption_rates = {scenario: 0.0 for scenario in self.scenarios}
        
        # Performance tracking
        self.decision_evaluation_count = 0
        self.last_evaluation_step = -1
        
        print(f"Enhanced MultiScenarioHousehold {unique_id} initialized for {len(self.scenarios)} scenarios")
    
    def step(self):
        """
        Execute enhanced monthly step with comprehensive multi-scenario evaluation.
        
        Process:
        1. Calculate energy metrics ONCE (inherited from base Household)
        2. Evaluate adoption for ALL scenarios with detailed bias tracking
        3. Update social influence metrics
        4. Register energy flows based on rational scenario
        """
        current_month = self.model.schedule.steps
        
        # 1. Calculate base energy metrics (inherited from Household)
        super().step()
        
        # 2. Evaluate adoption for all scenarios with detailed tracking
        self._evaluate_all_scenarios_enhanced(current_month)
        
        # 3. Update social influence tracking
        self._update_social_influence_tracking(current_month)
        
        # 4. Energy flows are already registered by parent class
        # Note: System metrics use rational scenario adoption for consistency
        
        # Update performance tracking
        self.decision_evaluation_count += 1
        self.last_evaluation_step = current_month
    
    def _evaluate_all_scenarios_enhanced(self, current_month):
        """
        Enhanced evaluation of adoption decisions with synchronized trigger system.
        
        Args:
            current_month: Current simulation month
        """
        current_step = self.model.schedule.steps
        
        # Only evaluate if household has enough data (12 months for reliable NPV calculation)
        if len(self.savings_history) < 12:
            # Store placeholder values for early months
            for scenario in self.scenarios:
                self.scenario_npv[scenario] = self.npv
                self.scenario_probability[scenario] = 0.0
                self.scenario_installation_cost[scenario] = getattr(self, 'installation_cost', 0)
            return
        
        # Base NPV and probability from rational calculation
        base_npv = self.npv
        base_probability = self._npv_to_probability(base_npv)
        installation_cost = getattr(self, 'installation_cost', 0)
        
        # Store current economic values for trigger system
        self.npv = base_npv
        self.installation_cost = installation_cost
        
        # Single household-level trigger decision for ALL scenarios
        should_evaluate = self.evaluation_triggers.should_reevaluate_household(self, current_step)

        if should_evaluate:
            # 1. DETERMINISTIC RATIONAL - Pure NPV logic
            if not self.scenario_adoption['deterministic_rational']:
                self.scenario_npv['deterministic_rational'] = base_npv
                self.scenario_probability['deterministic_rational'] = 1.0 if base_npv > 0 else 0.0
                self.scenario_installation_cost['deterministic_rational'] = installation_cost
                
                if base_npv > 0:
                    self.scenario_adoption['deterministic_rational'] = True
                    self.adoption_months['deterministic_rational'] = current_month
                    if self.evaluation_triggers.config.get('debug_evaluation_triggers', False):
                        print(f"  Household {self.unique_id}: Deterministic rational adoption in step {current_step} "
                            f"(NPV: ${base_npv:.0f})")

            # 2. NON-DETERMINISTIC RATIONAL - Sigmoid probability
            if not self.scenario_adoption['rational']:
                self.scenario_npv['rational'] = base_npv
                self.scenario_probability['rational'] = base_probability
                self.scenario_installation_cost['rational'] = installation_cost
                
                if np.random.random() < base_probability:
                    self.scenario_adoption['rational'] = True
                    self.adoption_months['rational'] = current_month
                    if self.evaluation_triggers.config.get('debug_evaluation_triggers', False):
                        print(f"  Household {self.unique_id}: Rational adoption in step {current_step} "
                            f"(prob: {base_probability:.3f}, NPV: ${base_npv:.0f})")

            # 3. BEHAVIORAL SCENARIOS - Evaluate individual biases
            enabled_biases = get_enabled_biases()
            
            for bias_name in enabled_biases:
                if not self.scenario_adoption[bias_name]:
                    adjusted_npv, biased_probability = self.bias_manager.apply_single_bias(
                        self, bias_name, base_npv, base_probability
                    )
                    
                    self.scenario_npv[bias_name] = adjusted_npv
                    self.scenario_probability[bias_name] = biased_probability
                    self.scenario_installation_cost[bias_name] = installation_cost
                    
                    if np.random.random() < biased_probability:
                        self.scenario_adoption[bias_name] = True
                        self.adoption_months[bias_name] = current_month
                        print(f"  Household {self.unique_id}: {bias_name} adoption in month {current_month} "
                            f"(prob: {biased_probability:.3f}, NPV: ${base_npv:.0f})")

            # 4. COMBINED BIAS SCENARIO - All biases applied sequentially
            if not self.scenario_adoption['all_biases']:
                adjusted_npv, combined_probability = self.bias_manager.apply_all_biases(
                    self, base_npv, base_probability
                )
                
                self.scenario_npv['all_biases'] = adjusted_npv
                self.scenario_probability['all_biases'] = combined_probability
                self.scenario_installation_cost['all_biases'] = installation_cost
                
                if np.random.random() < combined_probability:
                    self.scenario_adoption['all_biases'] = True
                    self.adoption_months['all_biases'] = current_month
                    print(f"  Household {self.unique_id}: All biases adoption in month {current_month} "
                        f"(prob: {combined_probability:.3f}, NPV: ${base_npv:.0f})")

        else:
            # Store current metrics for all scenarios without evaluating adoption decisions
            enabled_biases = get_enabled_biases()
            
            # Rational scenarios
            self.scenario_npv['deterministic_rational'] = base_npv
            self.scenario_probability['deterministic_rational'] = self.scenario_probability.get('deterministic_rational', 0.0)
            self.scenario_installation_cost['deterministic_rational'] = installation_cost
            
            self.scenario_npv['rational'] = base_npv
            self.scenario_probability['rational'] = self.scenario_probability.get('rational', 0.0)
            self.scenario_installation_cost['rational'] = installation_cost
            
            # Bias scenarios
            for bias_name in enabled_biases:
                adjusted_npv, biased_probability = self.bias_manager.apply_single_bias(
                    self, bias_name, base_npv, base_probability
                )
                self.scenario_npv[bias_name] = adjusted_npv
                self.scenario_probability[bias_name] = biased_probability
                self.scenario_installation_cost[bias_name] = installation_cost
            
            # Combined bias scenario
            adjusted_npv, combined_probability = self.bias_manager.apply_all_biases(
                self, base_npv, base_probability
            )
            self.scenario_npv['all_biases'] = adjusted_npv
            self.scenario_probability['all_biases'] = combined_probability
            self.scenario_installation_cost['all_biases'] = installation_cost

        # 5. UPDATE DETAILED BIAS EFFECTS TRACKING (PRESERVED)
        self._update_bias_effects_tracking(base_probability, current_month)
    
    def _evaluate_rational_scenarios(self, base_npv, base_probability, installation_cost, current_step, current_month):
        """Evaluate rational scenarios (always executed)."""
        # 1. DETERMINISTIC RATIONAL - Pure NPV logic
        if not self.scenario_adoption['deterministic_rational'] and base_npv > 0:
            self.scenario_adoption['deterministic_rational'] = True
            self.adoption_months['deterministic_rational'] = current_month
            self.scenario_npv['deterministic_rational'] = base_npv
            self.scenario_probability['deterministic_rational'] = 1.0  # Certain adoption
        
        # 2. NON-DETERMINISTIC RATIONAL - Sigmoid probability
        self.scenario_npv['rational'] = base_npv
        self.scenario_probability['rational'] = base_probability
        self.scenario_installation_cost['rational'] = installation_cost
        
        if not self.scenario_adoption['rational'] and np.random.random() < base_probability:
            self.scenario_adoption['rational'] = True
            self.adoption_months['rational'] = current_month

    def _evaluate_single_bias_scenario(self, bias_name, base_npv, base_probability, installation_cost, current_step, current_month):
        """Evaluate single bias scenario when triggered."""
        if not self.scenario_adoption[bias_name]:
            # Apply single bias with detailed tracking
            adjusted_npv, biased_probability = self.bias_manager.apply_single_bias(
                self, bias_name, base_npv, base_probability
            )
            
            # Store scenario metrics
            self.scenario_npv[bias_name] = adjusted_npv  # ✅ Store adjusted NPV OR original NPV for probability-based biases
            self.scenario_probability[bias_name] = biased_probability
            self.scenario_installation_cost[bias_name] = installation_cost
            
            # Make probabilistic adoption decision
            if np.random.random() < biased_probability:
                self.scenario_adoption[bias_name] = True
                self.adoption_months[bias_name] = current_month
                print(f"    ✅ Household {self.unique_id}: {bias_name} adoption in step {current_step} "
                    f"(prob: {biased_probability:.3f}, NPV: ${base_npv:.0f})")

    def _evaluate_combined_bias_scenario(self, base_npv, base_probability, installation_cost, current_step, current_month):
        """Evaluate combined bias scenario when triggered."""
        if not self.scenario_adoption['all_biases']:
            # Apply all biases sequentially
            final_adjusted_npv, combined_probability = self.bias_manager.apply_all_biases(
                self, base_npv, base_probability
            )
            
            # Store combined scenario metrics
            self.scenario_npv['all_biases'] = final_adjusted_npv
            self.scenario_probability['all_biases'] = combined_probability
            self.scenario_installation_cost['all_biases'] = installation_cost
            
            # Make probabilistic adoption decision
            if np.random.random() < combined_probability:
                self.scenario_adoption['all_biases'] = True
                self.adoption_months['all_biases'] = current_month
                print(f"    ✅ Household {self.unique_id}: All biases adoption in step {current_step} "
                    f"(prob: {combined_probability:.3f}, NPV: ${base_npv:.0f})")    
    
    def _update_bias_effects_tracking(self, base_probability, current_month):
        """
        Update detailed bias effects tracking for analysis.
        
        Args:
            base_probability: Baseline (rational) adoption probability
            current_month: Current simulation month
        """
        # Get comprehensive bias effects summary
        bias_summary = self.bias_manager.get_bias_effects_summary(self)
        
        # Add temporal information
        bias_summary['step'] = current_month
        bias_summary['year'] = (current_month // 12) + 1
        bias_summary['month_in_year'] = (current_month % 12) + 1
        
        # Store current effects
        self.current_bias_effects = bias_summary
        
        # Add to history (keep last 24 months for analysis)
        self.bias_effects_history.append(bias_summary)
        if len(self.bias_effects_history) > 24:
            self.bias_effects_history.pop(0)
    
    def _update_social_influence_tracking(self, current_month):
        """
        Update social influence tracking for herding bias analysis.
        
        Args:
            current_month: Current simulation month
        """
        # Calculate current spatial and class influences
        herding_params = self.bias_manager.bias_params.get('herding', {})
        
        if herding_params:
            # Calculate spatial influence
            spatial_influence = self.bias_manager._calculate_spatial_influence(self, herding_params)
            
            # Calculate class influence
            class_influence = self.bias_manager._calculate_class_influence(self)
            
            # Store current influences
            spatial_record = {
                'step': current_month,
                'spatial_influence': spatial_influence,
                'neighbor_count': self._count_spatial_neighbors(herding_params)
            }
            
            class_record = {
                'step': current_month,
                'class_influence': class_influence,
                'class_size': self._count_class_members()
            }
            
            # Add to history (keep last 12 months)
            self.spatial_influence_history.append(spatial_record)
            self.class_influence_history.append(class_record)
            
            if len(self.spatial_influence_history) > 12:
                self.spatial_influence_history.pop(0)
            if len(self.class_influence_history) > 12:
                self.class_influence_history.pop(0)
            
            # Update neighbor adoption rates for each scenario
            self._update_neighbor_adoption_rates()
    
    def _count_spatial_neighbors(self, herding_params):
        """Count number of spatial neighbors within influence radius."""
        if not hasattr(self, 'pos') or self.pos is None:
            return 0
        
        household_pos = np.array(self.pos)
        max_neighbors = herding_params.get('max_neighbors_considered', 10)
        
        distances = []
        for agent in self.model.schedule.agents:
            if (hasattr(agent, 'pos') and agent.pos is not None and 
                agent.unique_id != self.unique_id and
                hasattr(agent, 'scenario_adoption')):
                
                distance = np.linalg.norm(household_pos - np.array(agent.pos))
                distances.append(distance)
        
        distances.sort()
        return min(len(distances), max_neighbors)
    
    def _count_class_members(self):
        """Count number of households in same income class."""
        same_class = [agent for agent in self.model.schedule.agents 
                     if hasattr(agent, 'income_class') and 
                     agent.income_class == self.income_class and
                     agent.unique_id != self.unique_id]
        return len(same_class)
    
    def _update_neighbor_adoption_rates(self):
        """Update adoption rates of neighbors for each scenario."""
        for scenario in self.scenarios:
            # Spatial neighbors
            spatial_rate = self._calculate_spatial_adoption_rate(scenario)
            
            # Class members  
            class_rate = self._calculate_class_adoption_rate(scenario)
            
            # Store combined neighbor influence
            self.neighbor_adoption_rates[scenario] = {
                'spatial_rate': spatial_rate,
                'class_rate': class_rate,
                'combined_influence': 0.5 * spatial_rate + 0.5 * class_rate
            }
    
    def _calculate_spatial_adoption_rate(self, scenario):
        """Calculate adoption rate among spatial neighbors for specific scenario."""
        if not hasattr(self, 'pos') or self.pos is None:
            return 0.0
        
        household_pos = np.array(self.pos)
        neighbors = []
        
        # Get spatial neighbors
        for agent in self.model.schedule.agents:
            if (hasattr(agent, 'daily_consumption') and 
                agent.unique_id != self.unique_id and
                hasattr(agent, 'pos') and agent.pos is not None):
                
                distance = np.linalg.norm(household_pos - np.array(agent.pos))
                if distance <= 3.0:  # Within influence radius
                    neighbors.append(agent)
        
        if not neighbors:
            return 0.0
        
        # Count adopters in the scenario
        adopters = sum(1 for neighbor in neighbors 
                      if getattr(neighbor, 'scenario_adoption', {}).get(scenario, False))
        
        return adopters / len(neighbors)
    
    def _calculate_class_adoption_rate(self, scenario):
        """Calculate adoption rate within same income class for specific scenario."""
        if not hasattr(self, 'income_class'):
            return 0.0
        
        same_class_households = [agent for agent in self.model.schedule.agents 
                               if hasattr(agent, 'income_class') and 
                               agent.income_class == self.income_class and
                               agent.unique_id != self.unique_id]
        
        if not same_class_households:
            return 0.0
        
        # Count adopters in the scenario
        adopters = sum(1 for household in same_class_households 
                      if getattr(household, 'scenario_adoption', {}).get(scenario, False))
        
        return adopters / len(same_class_households)
    
    def _npv_to_probability(self, npv):
        """
        Convert NPV to adoption probability using sigmoid function.
        
        Args:
            npv: Net Present Value
            
        Returns:
            float: Adoption probability [0, 1]
        """
        if npv is None:
            return 0.0
        
        # Apply sigmoid transformation: P = 1 / (1 + e^(-κ × NPV))
        exponent = -NPV_SIGMOID_STEEPNESS * npv
        
        # Prevent overflow
        if exponent > 700:
            return 0.0
        elif exponent < -700:
            return 1.0
        
        probability = 1.0 / (1.0 + np.exp(exponent))
        return max(0.0, min(1.0, probability))
    
    def get_adoption_status(self, scenario):
        """
        Get adoption status for a specific scenario.
        
        Args:
            scenario: Scenario name
            
        Returns:
            bool: Whether household has adopted in this scenario
        """
        return self.scenario_adoption.get(scenario, False)
    
    def get_adoption_month(self, scenario):
        """
        Get adoption month for a specific scenario.
        
        Args:
            scenario: Scenario name
            
        Returns:
            int or None: Month of adoption, or None if not adopted
        """
        return self.adoption_months.get(scenario, None)
    
    def get_scenario_metrics(self):
        """
        Get comprehensive metrics for all scenarios (for data collection).
        
        Returns:
            dict: Comprehensive scenario metrics
        """
        metrics = {
            'household_id': self.unique_id,
            'income': self.income,
            'income_class': self.income_class,
            'daily_consumption': self.daily_consumption,
            'position': self.pos,
            'current_step': self.model.schedule.steps,
            'current_year': (self.model.schedule.steps // 12) + 1,
            
            # Base economic metrics
            'current_npv': self.npv,
            'current_installation_cost': getattr(self, 'installation_cost', 0),
            'current_annual_savings': getattr(self, 'annual_savings', 0),
            'current_payback_period': getattr(self, 'payback_period', float('inf')),
            
            # Scenario-specific metrics
            'scenarios': {},
            
            # Social influence metrics
            'neighbor_adoption_rates': self.neighbor_adoption_rates.copy(),
            'spatial_influence_current': (self.spatial_influence_history[-1]['spatial_influence'] 
                                        if self.spatial_influence_history else 0.0),
            'class_influence_current': (self.class_influence_history[-1]['class_influence'] 
                                      if self.class_influence_history else 0.0),
            
            # Performance metrics
            'decision_evaluation_count': self.decision_evaluation_count,
            'last_evaluation_step': self.last_evaluation_step
        }
        
        # Add detailed scenario metrics
        for scenario in self.scenarios:
            metrics['scenarios'][scenario] = {
                'adopted': self.scenario_adoption[scenario],
                'adoption_month': self.adoption_months[scenario],
                'npv': self.scenario_npv.get(scenario, None),
                'probability': self.scenario_probability.get(scenario, 0.0),
                'installation_cost': self.scenario_installation_cost.get(scenario, 0.0),
                'neighbor_spatial_rate': self.neighbor_adoption_rates.get(scenario, {}).get('spatial_rate', 0.0),
                'neighbor_class_rate': self.neighbor_adoption_rates.get(scenario, {}).get('class_rate', 0.0)
            }
        
        # Add current bias effects
        if self.current_bias_effects:
            metrics['current_bias_effects'] = self.current_bias_effects.copy()
        
        return metrics
    
    def get_bias_effects_time_series(self):
        """
        Get time series of bias effects for detailed analysis.
        
        Returns:
            list: Historical bias effects data
        """
        return self.bias_effects_history.copy()
    
    def get_social_influence_time_series(self):
        """
        Get time series of social influence metrics.
        
        Returns:
            dict: Historical social influence data
        """
        return {
            'spatial_influence': self.spatial_influence_history.copy(),
            'class_influence': self.class_influence_history.copy()
        }
    
    def get_comprehensive_debug_info(self):
        """
        Get comprehensive debug information for troubleshooting.
        
        Returns:
            dict: Detailed debug information
        """
        debug_info = {
            'household_id': self.unique_id,
            'income': self.income,
            'income_class': self.income_class,
            'position': self.pos,
            'current_step': self.model.schedule.steps,
            
            # Scenario tracking
            'scenarios_count': len(self.scenarios),
            'enabled_biases': get_enabled_biases(),
            'scenario_adoption': self.scenario_adoption.copy(),
            'scenario_probabilities': self.scenario_probability.copy(),
            'adoption_months': self.adoption_months.copy(),
            
            # Economic status
            'savings_history_length': len(self.savings_history),
            'current_npv': self.npv,
            'current_installation_cost': getattr(self, 'installation_cost', 0),
            'has_positive_npv': self.npv is not None and self.npv > 0,
            
            # Social influence
            'spatial_neighbors_count': (self.spatial_influence_history[-1]['neighbor_count'] 
                                      if self.spatial_influence_history else 0),
            'class_members_count': (self.class_influence_history[-1]['class_size'] 
                                  if self.class_influence_history else 0),
            'current_spatial_influence': (self.spatial_influence_history[-1]['spatial_influence'] 
                                        if self.spatial_influence_history else 0.0),
            'current_class_influence': (self.class_influence_history[-1]['class_influence'] 
                                      if self.class_influence_history else 0.0),
            
            # Performance
            'bias_effects_history_length': len(self.bias_effects_history),
            'decision_evaluations': self.decision_evaluation_count,
            'last_evaluation': self.last_evaluation_step,
            
            # Model connectivity
            'model_connected': self.model is not None,
            'bias_manager_initialized': self.bias_manager is not None,
            'total_agents_in_model': len(self.model.schedule.agents) if self.model else 0
        }
        
        return debug_info
    
    def reset_scenario_adoption(self, scenarios_to_reset=None):
        """
        Reset adoption status for specific scenarios (useful for testing).
        
        Args:
            scenarios_to_reset: List of scenario names to reset (default: all)
        """
        if scenarios_to_reset is None:
            scenarios_to_reset = self.scenarios
        
        for scenario in scenarios_to_reset:
            if scenario in self.scenario_adoption:
                self.scenario_adoption[scenario] = False
                self.adoption_months[scenario] = None
                self.scenario_probability[scenario] = 0.0
        
        print(f"Reset adoption status for household {self.unique_id} in scenarios: {scenarios_to_reset}")


# =============================================================================
# TESTING FUNCTIONS
# =============================================================================

def test_enhanced_multi_scenario_household():
    """
    Comprehensive test function for the enhanced MultiScenarioHousehold class.
    
    Returns:
        bool: True if all tests pass
    """
    print("Testing Enhanced MultiScenarioHousehold (Phase 3)...")
    
    try:
        from ..utils.config_loader import create_multi_experiment_config
        from ..environment.weather_patterns import WeatherPatterns
        from ..environment.grid_system import GridSystem
        
        # Create test configuration
        config = create_multi_experiment_config()
        config_dict = config.get_copy()
        
        # Create environment components
        weather_patterns = WeatherPatterns()
        grid_system = GridSystem(config_dict)
        
        # Create proper mock model
        class MockModel:
            def __init__(self):
                class MockSchedule:
                    def __init__(self):
                        self.steps = 0
                        self.agents = []
                self.schedule = MockSchedule()
        
        mock_model = MockModel()
        
        # Test 1: Enhanced initialization
        print("  Test 1: Enhanced initialization...")
        household = MultiScenarioHousehold(
            unique_id=1,
            model=mock_model,
            config=config_dict,
            weather_patterns=weather_patterns,
            grid_system=grid_system
        )
        
        mock_model.schedule.agents.append(household)
        
        # Verify enhanced attributes
        expected_scenarios = get_all_scenarios()
        if set(household.scenarios) != set(expected_scenarios):
            print(f"    ❌ Scenario mismatch: expected {expected_scenarios}, got {household.scenarios}")
            return False
        
        if not hasattr(household, 'bias_effects_history'):
            print(f"    ❌ Missing bias_effects_history attribute")
            return False
        
        if not hasattr(household, 'spatial_influence_history'):
            print(f"    ❌ Missing spatial_influence_history attribute")
            return False
        
        print(f"    ✅ Enhanced initialization with {len(household.scenarios)} scenarios")
        
        # Test 2: Scenario evaluation with insufficient history
        print("  Test 2: Early scenario evaluation...")
        household.step()  # Should handle insufficient history gracefully
        
        if any(household.scenario_adoption.values()):
            print(f"    ❌ Adoption should not occur with insufficient history")
            return False
        
        print(f"    ✅ Handled insufficient history correctly")
        
        # Test 3: Simulate sufficient history and test evaluation
        print("  Test 3: Full scenario evaluation...")
        
        # Simulate 12 months of savings history
        for i in range(12):
            household.savings_history.append(100 + i * 10)  # Increasing savings
        
        # Update NPV to positive value for testing
        household.npv = 5000
        household.installation_cost = 15000
        mock_model.schedule.steps = 12
        
        # Run evaluation
        household.step()
        
        # Check that scenarios were evaluated
        if not any(prob > 0 for prob in household.scenario_probability.values()):
            print(f"    ❌ No scenario probabilities calculated")
            return False
        
        print(f"    ✅ Full scenario evaluation completed")
        print(f"      Rational probability: {household.scenario_probability['rational']:.3f}")
        
        # Test 4: Comprehensive metrics collection
        print("  Test 4: Comprehensive metrics...")
        metrics = household.get_scenario_metrics()
        
        required_keys = ['household_id', 'scenarios', 'neighbor_adoption_rates', 'current_bias_effects']
        missing_keys = [key for key in required_keys if key not in metrics]
        if missing_keys:
            print(f"    ❌ Missing required keys in metrics: {missing_keys}")
            return False
        
        if len(metrics['scenarios']) != len(expected_scenarios):
            print(f"    ❌ Incomplete scenario metrics")
            return False
        
        print(f"    ✅ Comprehensive metrics collection working")
        
        # Test 5: Debug information
        print("  Test 5: Debug information...")
        debug_info = household.get_comprehensive_debug_info()
        
        debug_required_keys = ['household_id', 'scenario_adoption', 'current_npv', 'bias_effects_history_length']
        missing_debug_keys = [key for key in debug_required_keys if key not in debug_info]
        if missing_debug_keys:
            print(f"    ❌ Missing required debug keys: {missing_debug_keys}")
            return False
        
        print(f"    ✅ Debug information comprehensive")
        
        # Test 6: Social influence tracking
        print("  Test 6: Social influence tracking...")
        
        # Add another household for neighbor testing
        neighbor = MultiScenarioHousehold(
            unique_id=2,
            model=mock_model,
            config=config_dict,
            weather_patterns=weather_patterns,
            grid_system=grid_system
        )
        neighbor.pos = (1, 1)  # Close to original household
        neighbor.income_class = household.income_class  # Same income class
        mock_model.schedule.agents.append(neighbor)
        
        # Update social influence
        household._update_social_influence_tracking(12)
        
        if not household.spatial_influence_history:
            print(f"    ❌ No spatial influence history recorded")
            return False
        
        if not household.class_influence_history:
            print(f"    ❌ No class influence history recorded")
            return False
        
        print(f"    ✅ Social influence tracking working")
        
        print("✅ Enhanced MultiScenarioHousehold (Phase 3) tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Enhanced MultiScenarioHousehold test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_enhanced_multi_scenario_household()