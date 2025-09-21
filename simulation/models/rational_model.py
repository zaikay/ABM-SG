# models/rational_model.py V5.3
"""
Fixed rational model with correct data collection initialization.
"""
import numpy as np
import networkx as nx
from mesa import Model
from mesa.time import RandomActivation

# Make sure you're importing the UPDATED classes
from simulation.data.validator import ModelValidator
from simulation.data.detailed_tracker import DetailedTracker

from ..agents.household import Household
from ..agents.central_provider import CentralProvider
from ..environment.weather_patterns import WeatherPatterns
from ..environment.energy_system import EnergySystem  # NEW unified system
from ..environment.unified_metrics import UnifiedEnergyMetrics
from ..environment.grid_system import GridSystem
from ..network.builder import NetworkBuilder
from ..network.metrics import NetworkMetrics
from ..data.collectors import SimulationDataCollector  # UPDATED collector
from ..utils.parameters import *

class RationalModel(Model):
    """
    Fixed rational prosumer model with correct data collection.
    """
    def __init__(self, config):
        """
        Initialize the rational model.
        
        Args:
            config: Simulation configuration dictionary
        """
        super().__init__()
        self.config = config
        self.running = True
        self.schedule = RandomActivation(self)
        
        # Set random seed
        if "run_settings" in config and "random_seed" in config["run_settings"]:
            np.random.seed(config["run_settings"]["random_seed"])
        
        # Create environment components
        self.weather_patterns = WeatherPatterns()
        self.grid_system = GridSystem(config)
        
        # Create UNIFIED energy system (replaces energy_calculator)
        self.energy_system = EnergySystem(self.weather_patterns, self.grid_system)
        
        # Create unified metrics calculator
        self.unified_metrics = UnifiedEnergyMetrics(
            self.weather_patterns, 
            self.grid_system
        )
        
        # Create agents FIRST
        self._create_agents()
        
        # Create network SECOND
        self._create_network()
        
        # Create data collector THIRD (after agents exist)
        self.data_collector = SimulationDataCollector(self, config)
        
        # Create detailed tracker
        self.detailed_tracker = DetailedTracker(self, sample_size=5)
        
        # Create network metrics analyzer
        self.network_metrics = NetworkMetrics(self)
        
        # Create validator
        self.validator = ModelValidator(self, enabled=True)
        
        # Collect initial data LAST (after everything is set up)
        self.data_collector.collect_data()
    
    def _create_agents(self):
        """
        Create agents with simplified energy system.
        """
        # Create households with unified energy system
        for i in range(self.config["num_households"]):
            household = Household(
                i,
                self,
                self.config,
                self.weather_patterns,  # Pass weather patterns directly
                self.grid_system        # Pass grid system directly
            )
            self.schedule.add(household)
        
        # Create central provider
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
        """
        # Get all households
        households = [agent for agent in self.schedule.agents if isinstance(agent, Household)]
        
        # Sort households by income
        households.sort(key=lambda h: h.income)
        
        # Calculate quintile boundaries
        num_quintiles = self.config["income_params"]["num_quintiles"]
        quintile_size = len(households) // num_quintiles
        
        # Assign income classes
        for i, household in enumerate(households):
            income_class = min(1 + (i // quintile_size), num_quintiles)
            household.set_income_class(income_class)
    
    def _create_network(self):
        """
        Create the social network.
        FIXED: Check if agents have positions before placing them.
        """
        # Get all households
        households = [agent for agent in self.schedule.agents if isinstance(agent, Household)]
        
        # Create network builder
        network_builder = NetworkBuilder(self, self.config)
        
        # Create spatial network
        self.grid = network_builder.create_spatial_network(households)
        
        # Place agents on the grid (FIXED to avoid duplicate placement)
        for household in households:
            # Only place if not already placed
            if household.pos is None or household.unique_id not in self.grid.G.nodes:
                self.grid.place_agent(household, household.unique_id)
    
    def get_central_provider(self):
        """
        Get the central provider agent.
        
        Returns:
            CentralProvider: Central provider agent
        """
        return self.central_provider
    
    def step(self):
        """Execute one step of the model."""
        current_step = self.schedule.steps
        
        # Execute step for all agents
        self.schedule.step()
        
        # Calculate unified system metrics if enabled
        households = [agent for agent in self.schedule.agents 
                    if hasattr(agent, "daily_consumption")]
        
        unified_config = self.config.get("unified_metrics", {})
        if (unified_config.get("enable_unified_metrics", False) and 
            unified_config.get("metrics_granularity", "basic") in ["detailed", "full"]):
            
            system_metrics = self.unified_metrics.calculate_system_metrics(
                households, current_step
            )
            
            # Update central provider with unified metrics
            self.central_provider.update_from_unified_metrics(system_metrics)
        
        # Execute end-step actions for central provider
        self.central_provider.end_step()
        
        # Track detailed data for sample households
        self.detailed_tracker.track_step_data(current_step)
        self.detailed_tracker.track_hourly_data(current_step)
        
        # Collect validation data
        if hasattr(self, 'validator') and self.validator.enabled:
            self.validator.collect_step_data(current_step)
        
        # Collect data
        self.data_collector.collect_data()
    
    def run(self, steps=None):
        """
        Run the model for a specified number of steps.
        
        Args:
            steps: Number of steps to run (default: from config)
        """
        if steps is None:
            steps = self.config["steps"]
        
        for _ in range(steps):
            self.step()


# DEBUGGING: Simple test to verify data collector works
def test_data_collector(config):
    """
    Test function to verify the data collector works correctly.
    
    Args:
        config: Simulation configuration
        
    Returns:
        bool: True if test passes
    """
    try:
        # Create a minimal model
        model = RationalModel(config)
        
        # Try to collect data
        model.data_collector.collect_data()
        
        # Get the data
        model_data = model.data_collector.get_model_data()
        agent_data = model.data_collector.get_agent_data()
        
        print(f"✅ Data collection test passed!")
        print(f"   Model data shape: {model_data.shape}")
        print(f"   Agent data shape: {agent_data.shape}")
        print(f"   Adoption rate: {model_data['AdoptionRate'].iloc[0]:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data collection test failed: {e}")
        return False


if __name__ == "__main__":
    # Quick test
    from ..utils.config_loader import create_rational_experiment
    
    config = create_rational_experiment().get_copy()
    config["num_households"] = 10  # Small test
    
    success = test_data_collector(config)
    if success:
        print("Data collector is working correctly!")
    else:
        print("Data collector needs fixing!")