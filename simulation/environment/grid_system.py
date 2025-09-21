# environment/grid_system.py V4

"""
Grid system and energy provider functionality.
"""
import math
import numpy as np
from ..utils.parameters import *

class GridSystem:
    """
    Manages grid-related calculations and metrics.
    """
    def __init__(self, config):
        """
        Initialize the grid system.
        
        Args:
            config: Simulation configuration
        """
        self.feed_in_factor = config["grid_params"]["feed_in_factor"]
        self.fossil_price = config["grid_params"]["initial_fossil_price"]
        self.renewable_price = config["grid_params"]["initial_renewable_price"]
        self.fossil_increase_rate = config["grid_params"]["fossil_annual_increase"]
        self.renewable_decrease_rate = config["grid_params"]["renewable_annual_decrease"]
        
        # System metrics
        self.total_consumption = 0
        self.total_prosumer_generation = 0
        self.total_grid_consumption = 0
        self.grid_peak_load = 0
        self.fossil_dependency = 1.0  # Initially 100% fossil
    
    def update_prices(self, month):
        """
        Update energy prices based on the current simulation step.
        
        Args:
            month: Current month in the simulation (starts at 0)
        """
        
        
        # Update fossil price with annual increase
        self.fossil_price *= (1 + self.fossil_increase_rate/MONTHS_IN_YEAR)
        
        # Update renewable price with annual decrease
        self.renewable_price *= (1 - self.renewable_decrease_rate/MONTHS_IN_YEAR)
    
    def get_current_prices(self):
        """
        Get current energy prices.
        
        Returns:
            tuple: (fossil_price, renewable_price)
        """
        return self.fossil_price, self.renewable_price
    
    def get_solar_cost(self, month):
        """
        Calculate the current solar installation cost per kW.
        
        Args:
            month: Current month in the simulation (starts at 0)
            
        Returns:
            float: Installation cost per kW
        """
        years = math.floor(month / MONTHS_IN_YEAR)
        
        # Calculate cost reduction based on technology learning curve
        reduction_factor = (1 - SOLAR_COST_REDUCTION_ANNUAL) ** years
        
        return BASE_SOLAR_COST * reduction_factor
    
    def calculate_npv(self, installation_cost, annual_savings, lifetime=SOLAR_LIFETIME_YEARS, discount_rate=DISCOUNT_RATE):
        """
        Calculate Net Present Value for a solar installation.
        
        Args:
            installation_cost: Total installation cost
            annual_savings: Annual energy cost savings
            lifetime: System lifetime in years
            discount_rate: Discount rate for NPV calculation
            
        Returns:
            float: Net Present Value
        """
        npv = -installation_cost
        
        # Annual maintenance cost (1% of initial installation cost)
        annual_maintenance = installation_cost * ANNUAL_MAINTENANCE_PERCENTAGE
        
        for year in range(1, lifetime + 1):
            # Adjust for rising energy costs
            price_adjustment = (1 + self.fossil_increase_rate) ** (year - 1)
            adjusted_savings = annual_savings * price_adjustment
            
            # Subtract maintenance costs
            net_annual_benefit = adjusted_savings - annual_maintenance
            
            # Apply discount factor
            npv += net_annual_benefit / ((1 + discount_rate) ** year)
            
        
        return npv
    
    def update_system_metrics(self, month_consumption, month_generation, month_grid_consumption):
        """
        Update grid system metrics based on monthly data.
        
        Args:
            month_consumption: Total energy consumption for the month
            month_generation: Total prosumer generation for the month
            month_grid_consumption: Total grid consumption for the month
        """
        self.total_consumption += month_consumption
        self.total_prosumer_generation += month_generation
        self.total_grid_consumption += month_grid_consumption
        
        # Update fossil dependency
        if self.total_consumption > 0:
            self.fossil_dependency = self.total_grid_consumption / self.total_consumption
        
        # Simplified peak load calculation (based on monthly consumption)
        # This is a proxy - for detailed peak load, hourly data would be needed
        estimated_peak = month_grid_consumption / (30 * 24) * 2  # Rough estimate based on monthly average
        self.grid_peak_load = max(self.grid_peak_load, estimated_peak)
    
    def get_system_metrics(self):
        """
        Get current system metrics.
        
        Returns:
            dict: System metrics
        """
        return {
            "total_consumption": self.total_consumption,
            "total_prosumer_generation": self.total_prosumer_generation,
            "total_grid_consumption": self.total_grid_consumption,
            "fossil_dependency": self.fossil_dependency,
            "grid_peak_load": self.grid_peak_load,
            "fossil_price": self.fossil_price,
            "renewable_price": self.renewable_price
        }
    
    def get_solar_cost_with_economies(self, month, system_size_kw):
        """
        Calculate solar cost using NREL fixed + variable cost methodology.
        
        Args:
            month: Current simulation month
            system_size_kw: System size in kW
            
        Returns:
            float: Cost per kW including economies of scale
        """
        # Apply time-based cost reduction
        years = math.floor(month / MONTHS_IN_YEAR)
        time_factor = (1 - SOLAR_COST_REDUCTION_ANNUAL) ** years
        
        # Calculate cost using fixed + variable structure
        
        total_cost = (SOLAR_FIXED_COSTS + (SOLAR_VARIABLE_COST_PER_KW * system_size_kw))* time_factor
        cost_per_kw = total_cost / system_size_kw
        
        return {
            "installation_cost": total_cost,
            "cost_per_kw": cost_per_kw,
        }