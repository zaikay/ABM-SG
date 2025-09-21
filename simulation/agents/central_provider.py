# agents/central_provider.py V4

"""
Central energy provider agent for the prosumer simulation.
"""
from mesa import Agent
from ..utils.parameters import *

class CentralProvider(Agent):
    """
    Agent representing the central energy provider.
    """
    def __init__(self, unique_id, model, grid_system):
        """
        Initialize the central provider agent.
        
        Args:
            unique_id: Unique identifier
            model: Mesa model instance
            grid_system: GridSystem instance
        """
        super().__init__(unique_id, model)
        self.grid_system = grid_system
        
        # Current energy prices
        self.fossil_price, self.renewable_price = grid_system.get_current_prices()
        
        # System metrics
        self.monthly_consumption = 0
        self.monthly_generation = 0
        self.monthly_grid_consumption = 0

            # NEW: Additional system metrics for unified calculation
        self.monthly_peak_load = 0
        self.peak_load_breakdown = {}
        self.total_credits_earned = 0
        self.total_credits_used = 0
        self.overall_credit_utilization = 0
        self.avg_stress_index = 0
        self.seasonal_stress_breakdown = {}
        
    def step(self):
        """Central provider step - update prices only, don't reset counters yet."""
        current_month = self.model.schedule.steps
        self.grid_system.update_prices(current_month)
        self.fossil_price, self.renewable_price = self.grid_system.get_current_prices()
    
    def register_energy_flows(self, consumption, generation, grid_consumption):
        """
        Register energy flows from a household.
        
        Args:
            consumption: Total energy consumption
            generation: Total energy generation (0 for non-prosumers)
            grid_consumption: Energy consumed from the grid
        """
        self.monthly_consumption += consumption
        self.monthly_generation += generation
        self.monthly_grid_consumption += grid_consumption
    
    def end_step(self):
        """
        Actions to perform at the end of a step.
        UPDATED: Reset counters AFTER data collection.
        """
        # Update system metrics with monthly totals (existing code)
        self.grid_system.update_system_metrics(
            self.monthly_consumption,
            self.monthly_generation,
            self.monthly_grid_consumption
        )
        
        # MOVED: Reset counters AFTER data has been collected
        self.monthly_consumption = 0
        self.monthly_generation = 0
        self.monthly_grid_consumption = 0
        
        # Reset new monthly counters
        self.monthly_peak_load = 0
        self.peak_load_breakdown = {}
        self.total_credits_earned = 0
        self.total_credits_used = 0
        self.overall_credit_utilization = 0
        self.avg_stress_index = 0
        self.seasonal_stress_breakdown = {}

        # ADD new method to CentralProvider class:
    def update_from_unified_metrics(self, system_metrics):
        """
        Update central provider metrics from unified calculation.
        
        Args:
            system_metrics: Dictionary from UnifiedEnergyMetrics.calculate_system_metrics()
        """
        # Update existing metrics
        monthly = system_metrics['monthly_totals']
        self.monthly_consumption = monthly['consumption']
        self.monthly_generation = monthly['generation']
        self.monthly_grid_consumption = monthly['grid_consumption']
        
        # Update new metrics
        self.monthly_peak_load = system_metrics['monthly_peak_load']
        self.peak_load_breakdown = system_metrics['peak_loads']
        self.total_credits_earned = sum(day['earned'] for day in system_metrics['credit_metrics'].values())
        self.total_credits_used = sum(day['used'] for day in system_metrics['credit_metrics'].values())
        self.overall_credit_utilization = system_metrics['overall_credit_utilization']
        self.avg_stress_index = system_metrics['avg_stress_index']
        self.seasonal_stress_breakdown = system_metrics['seasonal_stress']