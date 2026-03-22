# environment/energy_system.py V5.1

"""
CONSOLIDATED energy system that eliminates redundancies between 
energy_calculator.py and energy_metrics.py

This single class handles:
1. Low-level energy physics (hourly calculations)
2. High-level economic calculations (NPV, savings)
3. Scenario comparisons (baseline vs solar)
"""
import numpy as np
from ..utils.parameters import *

class EnergySystem:
    """
    Unified energy calculation system for households.
    Replaces both EnergyCalculator and EnergyMetrics to eliminate redundancies.
    """
    
    def __init__(self, weather_patterns, grid_system):
        """
        Initialize the unified energy system.
        
        Args:
            weather_patterns: WeatherPatterns instance
            grid_system: GridSystem instance
        """
        self.weather = weather_patterns
        self.grid_system = grid_system
        self.hourly_consumption_profile = HOURLY_CONSUMPTION_PROFILE
        self.hourly_solar_profile = HOURLY_SOLAR_PROFILE
    
    def calculate_household_metrics(self, household, current_month):
        """
        REFACTORED: Calculate household metrics with single energy scenario.
        Uses either projected capacity (for decisions) or actual capacity (for tracking).
        """
        month_in_year = (current_month % MONTHS_IN_YEAR) + 1
        
        # Determine which solar capacity to use
        solar_capacity = (household.solar_capacity if household.is_prosumer 
                        else self._calculate_projected_solar_capacity(household))
        
        # SINGLE energy scenario calculation
        energy_scenario = self._calculate_energy_scenario(
            household.daily_consumption, solar_capacity, month_in_year
        )
        
        # Calculate costs directly from the scenario
        baseline_cost = energy_scenario['total_consumption'] * self.grid_system.fossil_price
        solar_cost = energy_scenario['total_grid_consumption'] * self.grid_system.fossil_price
        monthly_savings = baseline_cost - solar_cost
        
        # Economic metrics
        installation_cost = self._calculate_installation_cost(current_month, solar_capacity)
        
        return {
            # Costs and savings
            'baseline_cost': baseline_cost,
            'solar_cost': solar_cost,
            'monthly_savings': monthly_savings,
            
            # Solar system info
            'solar_capacity': solar_capacity,
            'installation_cost': installation_cost,
            
            # Energy flows (from single calculation)
            'monthly_consumption': energy_scenario['total_consumption'],
            'monthly_generation': energy_scenario['total_generation'],
            'monthly_grid_consumption': energy_scenario['total_grid_consumption'],
            'monthly_grid_feed_in': energy_scenario['total_grid_feed_in'],
            'monthly_credits_earned': energy_scenario['total_credits_earned'],
            'monthly_credits_used': energy_scenario['total_credits_used'],
            'monthly_credits_expired': energy_scenario['total_credits_expired'],
        }
    
    def _calculate_energy_scenario(self, daily_consumption, solar_capacity, month):
        """
        Calculate complete energy scenario for given consumption and solar capacity.
        This replaces the old calculate_monthly_energy method.
        
        Args:
            daily_consumption: Daily consumption in kWh
            solar_capacity: Solar capacity in kW (0 for no solar)
            month: Month number (1-12)
            
        Returns:
            dict: Complete energy scenario metrics
        """
        # Get representative days for the month
        rep_days = self.weather.get_representative_days(month)
        
        # Initialize monthly totals
        monthly_totals = {
            'total_consumption': 0,
            'total_generation': 0,
            'total_grid_feed_in': 0,
            'total_grid_consumption': 0,
            'total_credits_earned': 0,
            'total_credits_used': 0,
            'total_credits_expired': 0,
            'total_cost': 0
        }
        
        # Process each representative day type
        for day_type, count, weather, day_of_week in rep_days:
            daily_metrics = self._calculate_representative_day(
                daily_consumption, solar_capacity, month, day_type
            )
            
            # Accumulate weighted totals
            for key in monthly_totals:
                monthly_totals[key] += daily_metrics[key] * count
        
        return monthly_totals
    
    def _calculate_representative_day(self, daily_consumption, solar_capacity, month, day_type):
        """
        Calculate energy flows for a single representative day.
        Consolidates all hourly calculations into one method.
        
        Args:
            daily_consumption: Daily consumption in kWh
            solar_capacity: Solar capacity in kW
            month: Month number (1-12)  
            day_type: Day type identifier (e.g., 'sunny_weekday')
            
        Returns:
            dict: Daily energy metrics
        """
        # Calculate hourly consumption
        consumption_multiplier = self.weather.get_consumption_multiplier(month, day_type)
        adjusted_daily_consumption = daily_consumption * consumption_multiplier
        hourly_consumption = np.array(self.hourly_consumption_profile) * adjusted_daily_consumption
        
        # Calculate hourly generation
        if solar_capacity > 0:
            generation_multiplier = self.weather.get_generation_multiplier(month, day_type)
            max_daily_generation = solar_capacity * SOLAR_PRODUCTION_RATIO
            adjusted_daily_generation = max_daily_generation * generation_multiplier
            hourly_generation = np.array(self.hourly_solar_profile) * adjusted_daily_generation
        else:
            hourly_generation = np.zeros(24)
        
        # Calculate energy balance and grid interactions
        hourly_balance = hourly_generation - hourly_consumption
        grid_metrics = self._calculate_grid_interaction(hourly_balance)
        
        # Calculate daily cost
        daily_cost = np.sum(grid_metrics["grid_consumption"]) * self.grid_system.fossil_price
        
        return {
            'total_consumption': np.sum(hourly_consumption),
            'total_generation': np.sum(hourly_generation),
            'total_grid_feed_in': grid_metrics["total_grid_feed_in"],
            'total_grid_consumption': grid_metrics["total_grid_consumption"],
            'total_credits_earned': grid_metrics["total_credits_earned"],
            'total_credits_used': grid_metrics["total_credits_used"],
            'total_credits_expired': grid_metrics["credits_expired"],
            'total_cost': daily_cost
        }
    
    def _calculate_grid_interaction(self, hourly_balance):
        """
        Calculate grid interaction for 24-hour period.
        Moved from EnergyCalculator without changes.
        """
        # Initialize tracking variables
        hourly_grid_feed_in = np.zeros(24)
        hourly_grid_consumption = np.zeros(24)
        hourly_credits_earned = np.zeros(24)
        hourly_credits_used = np.zeros(24)
        hourly_credits_available = np.zeros(24)
        
        for hour in range(24):
            balance = hourly_balance[hour]
            
            if hour > 0:
                hourly_credits_available[hour] = (hourly_credits_available[hour-1] + 
                                                hourly_credits_earned[hour-1] - 
                                                hourly_credits_used[hour-1])
            
            if balance > 0:
                # Excess generation fed to grid
                hourly_grid_feed_in[hour] = balance
                hourly_credits_earned[hour] = balance * self.grid_system.feed_in_factor
            else:
                # Deficit - use credits if available, then grid
                deficit = -balance
                credits_used = min(deficit, hourly_credits_available[hour])
                grid_needed = deficit - credits_used
                
                hourly_credits_used[hour] = credits_used
                hourly_grid_consumption[hour] = grid_needed
        
        # Calculate remaining credits at end of day (these will expire)
        final_credits = (hourly_credits_available[23] + 
                        hourly_credits_earned[23] - 
                        hourly_credits_used[23])
        
        return {
            "grid_feed_in": hourly_grid_feed_in,
            "grid_consumption": hourly_grid_consumption,
            "credits_earned": hourly_credits_earned,
            "credits_used": hourly_credits_used,
            "credits_available": hourly_credits_available,
            "credits_expired": final_credits,
            "total_grid_feed_in": np.sum(hourly_grid_feed_in),
            "total_grid_consumption": np.sum(hourly_grid_consumption),
            "total_credits_earned": np.sum(hourly_credits_earned),
            "total_credits_used": np.sum(hourly_credits_used)
        }
    
    def _calculate_projected_solar_capacity(self, household):
        """Calculate what solar capacity would be if household adopted."""
        return ((household.daily_consumption / 
                household.config["solar_params"]["solar_production_ratio"]) * 
                household.config["solar_params"]["sizing_factor"])
    
    def _calculate_installation_cost(self, current_month, solar_capacity):
        """Calculate installation cost using grid system."""
        cost_data = self.grid_system.get_solar_cost_with_economies(current_month, solar_capacity)
        return cost_data["installation_cost"]
