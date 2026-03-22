# models/multi_experiment_model.py V1.0
"""
Multi-experiment model for behavioral prosumer adoption simulation.
Runs all scenarios (rational + behavioral) in a single simulation run.
"""

import time
import numpy as np
import networkx as nx
import random
from mesa import Model
from mesa.time import RandomActivation

from ..agents.multi_scenario_household import MultiScenarioHousehold
from ..agents.central_provider import CentralProvider
from ..environment.weather_patterns import WeatherPatterns
from ..environment.energy_system import EnergySystem
from ..environment.unified_metrics import UnifiedEnergyMetrics
from ..environment.grid_system import GridSystem
from ..network.builder import NetworkBuilder
from ..network.metrics import NetworkMetrics
from ..data.multi_experiment_collector import MultiExperimentCollector
from ..data.detailed_tracker import DetailedTracker
from ..data.validator import ModelValidator
from ..utils.parameters import get_all_scenarios, get_enabled_biases
from ..agents.evaluation_triggers import EvaluationTriggers
from ..data.enhanced_spatial_analyzer import IntegratedSpatialAnalyzer
from ..data.focused_spatial_metrics_calculator import FocusedSpatialMetricsCalculator

class MultiExperimentModel(Model):
    """
    Multi-experiment model that runs all behavioral scenarios simultaneously.
    
    This model extends the rational model to:
    1. Create MultiScenarioHousehold agents instead of regular households
    2. Collect data for all scenarios in parallel
    3. Track scenario-specific adoption patterns
    4. Enable cross-scenario analysis and comparison
    """
    
    def __init__(self, config):
        """
        Initialize the multi-experiment model.
        
        Args:
            config: Simulation configuration dictionary
        """
        run_seed = config.get("run_settings", {}).get("random_seed", None)
        super().__init__()
        self.config = config
        self.running = True
        self.schedule = RandomActivation(self)
        
        # Get scenarios from configuration
        self.scenarios = get_all_scenarios()
        self.enabled_biases = get_enabled_biases()
        
        # Set RNG seeds for reproducibility (NumPy, Python random, Mesa model RNG).
        if run_seed is not None:
            np.random.seed(run_seed)
            random.seed(run_seed)
            if hasattr(self, "random") and self.random is not None:
                self.random.seed(run_seed)
        
        print(f"Initializing MultiExperimentModel with {len(self.scenarios)} scenarios:")
        for scenario in self.scenarios:
            print(f"  - {scenario}")
        
        # Create environment components (same as rational model)
        self._create_environment()

        # Create evaluation triggers for behavioral decision timing
        self.evaluation_triggers = EvaluationTriggers(self)
        
        # Create agents (MultiScenarioHouseholds instead of regular Households)
        self._create_agents()
        
        # Create network (same as rational model)
        self._create_network()

        # DELAYED PROSUMER INITIALIZATION - DESACTIVATED
        #self.default_prosumer_id = None  # Will be set at step 12
        #self.default_prosumer_created = False  # Flag to track creation
        #self.delayed_prosumer_candidate = self._select_default_prosumer_candidate()
        
        # Create multi-experiment data collector
        self.data_collector = MultiExperimentCollector(self, config)
        
        # Create detailed tracker
        self.detailed_tracker = DetailedTracker(self, sample_size=5, seed=run_seed if run_seed is not None else 42)
        
        # Create network metrics analyzer
        self.network_metrics = NetworkMetrics(self)

        self.focused_spatial_metrics = FocusedSpatialMetricsCalculator(self)
        
        # Create validator
        self.validator = ModelValidator(self, enabled=True)
        
        # ❌ REMOVE THIS LINE - NO IMMEDIATE DATA COLLECTION
        # self.data_collector.collect_data()
        
        print(f"MultiExperimentModel initialized with {self.get_household_count()} households")
        print("Note: Initial data collection deferred until first step()")    

    def _create_environment(self):
        """
        Create environment components (identical to rational model).
        """
        # Create environment components
        self.weather_patterns = WeatherPatterns()
        self.grid_system = GridSystem(self.config)
        
        # Create unified energy system
        self.energy_system = EnergySystem(self.weather_patterns, self.grid_system)
        
        # Create unified metrics calculator
        self.unified_metrics = UnifiedEnergyMetrics(
            self.weather_patterns, 
            self.grid_system
        )
    
    def _create_agents(self):
        """
        Create MultiScenarioHousehold agents and central provider.
        """
        # Create multi-scenario households
        for i in range(self.config["num_households"]):
            household = MultiScenarioHousehold(
                i,
                self,
                self.config,
                self.weather_patterns,
                self.grid_system
            )
            self.schedule.add(household)
        
        # Create central provider (same as rational model)
        self.central_provider = CentralProvider(
            "central_provider",
            self,
            self.grid_system
        )
        self.schedule.add(self.central_provider)
        
        # Assign income classes
        self._assign_income_classes()
    
    def _assign_income_classes(self):
        """
        Assign income classes (quintiles) to households.
        FIXED: Handle small number of households properly.
        """
        # Get all households
        households = [agent for agent in self.schedule.agents 
                    if isinstance(agent, MultiScenarioHousehold)]
        
        if not households:
            print("Warning: No households found for income class assignment")
            return
        
        # Sort households by income
        households.sort(key=lambda h: h.income)
        
        # Calculate quintile boundaries - FIXED for small populations
        num_quintiles = self.config["income_params"]["num_quintiles"]
        num_households = len(households)
        
        if num_households < num_quintiles:
            # If we have fewer households than quintiles, assign each to a different class
            for i, household in enumerate(households):
                household.set_income_class(i + 1)
            print(f"Assigned {num_households} households to individual income classes (small population)")
        else:
            # Normal quintile assignment
            quintile_size = num_households // num_quintiles
            
            # Handle case where quintile_size is 0 (shouldn't happen with above check, but safety)
            if quintile_size == 0:
                quintile_size = 1
            
            # Assign income classes
            for i, household in enumerate(households):
                income_class = min(1 + (i // quintile_size), num_quintiles)
                household.set_income_class(income_class)
            
            print(f"Assigned income classes to {len(households)} households (quintile size: {quintile_size})")
    
    def _select_default_prosumer_candidate(self):
        """
        Select but don't activate default prosumer candidate.
        Returns household ID for delayed activation.
        """
        import random
        
        # Find upper income households (classes 4 & 5)
        upper_income_households = [
            h for h in self.schedule.agents 
            if hasattr(h, 'income_class') and h.income_class in [4, 5] and
            hasattr(h, 'scenario_adoption')  # MultiScenarioHousehold check
        ]
        
        if not upper_income_households:
            print("Warning: No upper income households found for default prosumer!")
            return None
        
        # Randomly select candidate (but don't activate yet)
        selected = random.choice(upper_income_households)
        
        print(f"Default prosumer candidate: Household {selected.unique_id} "
            f"(class {selected.income_class}, income: ${selected.income:,.0f})")
        print("Will be activated at step 12 when evaluation system becomes active")
        
        return selected.unique_id

    def _create_delayed_default_prosumer(self, step):
        """
        Create default prosumer at step 12 when evaluation system is active.
        """
        if self.delayed_prosumer_candidate is None:
            return None
        
        # Find the candidate household
        candidate_household = None
        for agent in self.schedule.agents:
            if (hasattr(agent, 'unique_id') and 
                agent.unique_id == self.delayed_prosumer_candidate):
                candidate_household = agent
                break
        
        if candidate_household is None:
            print(f"Warning: Cannot find candidate household {self.delayed_prosumer_candidate}")
            return None
        
        # Now activate as prosumer (at step 12)
        candidate_household.is_prosumer = True
        candidate_household.installation_month = step  # Current step (12)
        candidate_household.solar_capacity = 5.0
        
        # Mark as adopted in all scenarios with current step
        for scenario in candidate_household.scenarios:
            candidate_household.scenario_adoption[scenario] = True
            candidate_household.adoption_months[scenario] = step
        
        print(f"Default prosumer activated at step {step}: "
            f"Household {candidate_household.unique_id} "
            f"(class {candidate_household.income_class}, "
            f"income: ${candidate_household.income:,.0f})")
        print(f"Adopted in scenarios: {list(candidate_household.scenarios)}")
        
        self.default_prosumer_created = True
        return candidate_household.unique_id

    def _create_network(self):
        """
        Create the social network (identical to rational model).
        """
        # Get all households
        households = [agent for agent in self.schedule.agents 
                     if isinstance(agent, MultiScenarioHousehold)]
        
        # Create network builder
        network_builder = NetworkBuilder(self, self.config)
        
        # Create spatial network
        self.grid = network_builder.create_spatial_network(households)
        
        # Place agents on the grid
        for household in households:
            if household.pos is None or household.unique_id not in self.grid.G.nodes:
                self.grid.place_agent(household, household.unique_id)
        
        print(f"Created network with {len(self.grid.G.nodes)} nodes and {len(self.grid.G.edges)} edges")


    def get_central_provider(self):
        """
        Get the central provider agent.
        
        Returns:
            CentralProvider: Central provider agent
        """
        return self.central_provider
    
    def get_households(self):
        """
        Get all household agents.
        
        Returns:
            list: List of MultiScenarioHousehold agents
        """
        return [agent for agent in self.schedule.agents 
                if isinstance(agent, MultiScenarioHousehold)]
    
    def get_household_count(self):
        """
        Get total number of households.
        
        Returns:
            int: Number of households
        """
        return len(self.get_households())
    
    def get_scenario_adoption_rates(self):
        """
        Get current adoption rates for all scenarios.
        
        Returns:
            dict: Mapping scenario names to adoption rates
        """
        households = self.get_households()
        if not households:
            return {scenario: 0.0 for scenario in self.scenarios}
        
        adoption_rates = {}
        for scenario in self.scenarios:
            adopters = sum(1 for h in households if h.get_adoption_status(scenario))
            adoption_rates[scenario] = adopters / len(households)
        
        return adoption_rates
    
    def get_scenario_statistics(self):
        """
        Get comprehensive statistics for all scenarios.
        
        Returns:
            dict: Detailed scenario statistics
        """
        households = self.get_households()
        if not households:
            return {}
        
        stats = {
            'total_households': len(households),
            'current_step': self.schedule.steps,
            'current_year': (self.schedule.steps // 12) + 1,
            'scenarios': {}
        }
        
        for scenario in self.scenarios:
            adopters = [h for h in households if h.get_adoption_status(scenario)]
            
            scenario_stats = {
                'adoption_count': len(adopters),
                'adoption_rate': len(adopters) / len(households),
                'adoption_by_class': {},
                'average_adoption_month': None
            }
            
            # Adoption by income class
            for income_class in range(1, 6):
                class_households = [h for h in households if h.income_class == income_class]
                class_adopters = [h for h in adopters if h.income_class == income_class]
                
                if class_households:
                    scenario_stats['adoption_by_class'][income_class] = {
                        'total': len(class_households),
                        'adopters': len(class_adopters),
                        'rate': len(class_adopters) / len(class_households)
                    }
            
            # Average adoption month
            adoption_months = [h.get_adoption_month(scenario) for h in adopters 
                             if h.get_adoption_month(scenario) is not None]
            if adoption_months:
                scenario_stats['average_adoption_month'] = np.mean(adoption_months)
            
            stats['scenarios'][scenario] = scenario_stats
        
        return stats
    
    def step(self):
        """
        Execute one step of the model with multi-scenario tracking.
        FIXED: Proper timing for energy data collection.
        """
        current_step = self.schedule.steps
        
        # 1. Execute step for all agents (households calculate energy, central provider updates prices)
        self.schedule.step()

        # Create default prosumer at step 12 (when evaluation becomes active) DESACTIVATED - SEED FOR HERDING BIAS
        #if (current_step == 12 and 
        #    not self.default_prosumer_created and 
        #    self.delayed_prosumer_candidate is not None):
        #    
        #    self.default_prosumer_id = self._create_delayed_default_prosumer(current_step)
        
        # 2. Calculate unified system metrics if enabled
        households = self.get_households()
        
        unified_config = self.config.get("unified_metrics", {})
        if (unified_config.get("enable_unified_metrics", False) and 
            unified_config.get("metrics_granularity", "basic") in ["detailed", "full"]):
            
            system_metrics = self.unified_metrics.calculate_system_metrics(
                households, current_step
            )
            
            # Update central provider with unified metrics
            self.central_provider.update_from_unified_metrics(system_metrics)
        
        # 3. COLLECT DATA BEFORE RESET (moved before end_step)
        self.data_collector.collect_data()

        self.focused_spatial_metrics.calculate_step_metrics(current_step)
        
        # 4. Execute end-step actions for central provider (resets counters)
        self.central_provider.end_step()
        
        # 5. Track detailed data for sample households (all scenarios)
        self.detailed_tracker.track_step_data(current_step)
        self.detailed_tracker.track_hourly_data(current_step)
        
        # 6. Collect validation data
        if hasattr(self, 'validator') and self.validator.enabled:
            self.validator.collect_step_data(current_step)
        
        # 7. Print progress
        if current_step % 12 == 0:
            year = (current_step // 12) + 1
            adoption_rates = self.get_scenario_adoption_rates()
            print(f"Year {year} - Adoption rates: " + 
                ", ".join([f"{scenario}: {rate:.1%}" for scenario, rate in adoption_rates.items()]))
    
    def run(self, steps=None):
        """
        Run the model for a specified number of steps.
        
        Args:
            steps: Number of steps to run (default: from config)
        """
        if steps is None:
            steps = self.config["steps"]
        
        print(f"Starting multi-experiment simulation for {steps} steps...")
        
        for step in range(steps):
            self.step()
        
        print("Multi-experiment simulation completed!")
        
        # Print final statistics
        final_stats = self.get_scenario_statistics()
        print(f"\nFinal Results (Year {final_stats['current_year']}):")
        for scenario, stats in final_stats['scenarios'].items():
            print(f"  {scenario}: {stats['adoption_rate']:.1%} adoption rate "
                  f"({stats['adoption_count']} households)")
    
    def export_results(self, output_dir="results/multi_experiment"):
        """
        Export all results and generate visualizations.
        
        Args:
            output_dir: Directory to save results
        """
        print(f"Exporting results to {output_dir}...")
        
        # Export multi-experiment data
        self.data_collector.export_all_scenarios(output_dir)
        
        # Export detailed tracking data
        detailed_output_dir = f"{output_dir}/detailed_data"
        self.detailed_tracker.export_data(detailed_output_dir)

        # Export validation results
        if hasattr(self, 'validator') and self.validator.enabled:
            validation_output_dir = f"{output_dir}/validation"
            self.validator.create_validation_report(validation_output_dir)
        
        print(f"SIMULATION Results exported to {output_dir}")
    
    def get_model_summary(self):
        """
        Get a summary of the model for reporting.
        
        Returns:
            dict: Model summary
        """
        households = self.get_households()
        final_stats = self.get_scenario_statistics()
        
        summary = {
            'model_type': 'MultiExperimentModel',
            'scenarios': self.scenarios,
            'enabled_biases': self.enabled_biases,
            'total_households': len(households),
            'total_steps': self.schedule.steps,
            'total_years': (self.schedule.steps // 12),
            'final_adoption_rates': {
                scenario: stats['adoption_rate'] 
                for scenario, stats in final_stats['scenarios'].items()
            },
            'network_properties': {
                'nodes': len(self.grid.G.nodes),
                'edges': len(self.grid.G.edges),
                'average_degree': 2 * len(self.grid.G.edges) / len(self.grid.G.nodes) if self.grid.G.nodes else 0
            }
        }
        
        return summary


# =============================================================================
# TESTING FUNCTIONS
# =============================================================================

def test_multi_experiment_model():
    """
    Test the MultiExperimentModel class with corrected data collection timing.
    """
    print("Testing MultiExperimentModel...")
    
    try:
        from ..utils.config_loader import create_testing_config
        
        # Create test configuration (small scale)
        config = create_testing_config(num_households=10, steps=3)
        config_dict = config.get_copy()
        
        # Create model (no immediate data collection)
        model = MultiExperimentModel(config_dict)
        
        # Test initialization
        households = model.get_households()
        if len(households) != 10:
            print(f"❌ Expected 10 households, got {len(households)}")
            return False
        
        # Test scenarios
        expected_scenarios = get_all_scenarios()
        if set(model.scenarios) != set(expected_scenarios):
            print(f"❌ Scenario mismatch: expected {expected_scenarios}, got {model.scenarios}")
            return False
        
        # Check that no initial data was collected (should be empty)
        initial_data = model.data_collector.get_combined_dataframe()
        if not initial_data.empty:
            print(f"❌ Unexpected initial data collection")
            return False
        
        # Test running model (this should trigger data collection)
        model.run(steps=3)
        
        # Now there should be data
        final_data = model.data_collector.get_combined_dataframe()
        if final_data.empty:
            print(f"❌ No data collected during simulation")
            return False
        
        # Test final statistics
        final_stats = model.get_scenario_statistics()
        if final_stats['current_step'] != 3:
            print(f"❌ Expected 3 steps, got {final_stats['current_step']}")
            return False
        
        print("✅ MultiExperimentModel tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ MultiExperimentModel test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_multi_experiment_model()
