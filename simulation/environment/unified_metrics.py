# NEW FILE: environment/unified_metrics.py V2
"""
Unified Energy Metrics Engine - Single source of truth for all system calculations.
Calculates once, provides multiple aggregation views.
"""
import numpy as np
from ..utils.parameters import *

class UnifiedEnergyMetrics:
    """
    Single calculation engine for all system-level energy metrics.
    Computes exact values from individual household data.
    """
    
    def __init__(self, weather_patterns, grid_system):
        """
        Initialize the unified metrics engine.
        
        Args:
            weather_patterns: WeatherPatterns instance
            grid_system: GridSystem instance
        """
        self.weather = weather_patterns
        self.grid_system = grid_system
    
    def calculate_system_metrics(self, households, current_month):
        """
        Calculate complete system metrics for current month.
        Single calculation, multiple aggregation views.
        
        Args:
            households: List of household agents
            current_month: Current simulation month
            
        Returns:
            dict: Complete system metrics at multiple granularities
        """
        month_in_year = (current_month % MONTHS_IN_YEAR) + 1
        rep_days = self.weather.get_representative_days(month_in_year)
        
        # Initialize system metrics structure
        system_metrics = {
            'monthly_totals': self._init_monthly_totals(),
            'peak_loads': {},
            'credit_metrics': {},
            'seasonal_stress': {},
            'day_type_breakdown': {}
        }
        
        # Calculate metrics for each representative day
        for day_type, count, weather, day_of_week in rep_days:
            day_metrics = self._calculate_system_day_metrics(
                households, month_in_year, day_type
            )
            
            # Accumulate weighted monthly totals
            self._accumulate_monthly_totals(
                system_metrics['monthly_totals'], day_metrics, count
            )
            
            # Store day-specific metrics
            system_metrics['peak_loads'][day_type] = day_metrics['peak_load']
            system_metrics['credit_metrics'][day_type] = day_metrics['credit_utilization']
            system_metrics['seasonal_stress'][day_type] = day_metrics['stress_metrics']
            system_metrics['day_type_breakdown'][day_type] = {
                'count': count,
                'totals': day_metrics['daily_totals']
            }
        
        # Calculate derived system-level metrics
        system_metrics.update(self._calculate_derived_metrics(system_metrics))
        
        return system_metrics
    
    def _calculate_system_day_metrics(self, households, month, day_type):
        """
        Calculate exact metrics for one representative day across all households.
        
        Args:
            households: List of household agents
            month: Month number (1-12)
            day_type: Day type identifier
            
        Returns:
            dict: Complete day metrics
        """
        # Initialize system hourly arrays (exact aggregation)
        system_hourly = {
            'consumption': np.zeros(24),
            'generation': np.zeros(24),
            'grid_consumption': np.zeros(24),
            'grid_feed_in': np.zeros(24),
            'credits_earned': np.zeros(24),
            'credits_used': np.zeros(24)
        }
        
        # Sum across ALL households for each hour (exact calculation)
        for household in households:
            household_hourly = self._calculate_household_hourly_flows(
                household, month, day_type
            )
            
            # Aggregate to system totals (sum of individual values)
            for metric in system_hourly:
                system_hourly[metric] += household_hourly[metric]
        
        # Calculate exact derived metrics
        peak_load = np.max(system_hourly['grid_consumption'])
        peak_hour = np.argmax(system_hourly['grid_consumption'])
        
        total_credits_earned = np.sum(system_hourly['credits_earned'])
        total_credits_used = np.sum(system_hourly['credits_used'])
        credit_utilization_rate = (total_credits_used / total_credits_earned 
                                 if total_credits_earned > 0 else 0)
        
        stress_index = self._calculate_stress_index(system_hourly)
        
        return {
            'daily_totals': {
                'consumption': np.sum(system_hourly['consumption']),
                'generation': np.sum(system_hourly['generation']),
                'grid_consumption': np.sum(system_hourly['grid_consumption']),
                'grid_feed_in': np.sum(system_hourly['grid_feed_in']),
                'credits_earned': total_credits_earned,
                'credits_used': total_credits_used
            },
            'peak_load': peak_load,
            'peak_hour': peak_hour,
            'credit_utilization': {
                'earned': total_credits_earned,
                'used': total_credits_used,
                'utilization_rate': credit_utilization_rate
            },
            'stress_metrics': stress_index
        }
    
    def _calculate_household_hourly_flows(self, household, month, day_type):
        """
        Calculate hourly energy flows for one household on one day type.
        
        Args:
            household: Household agent
            month: Month number (1-12)
            day_type: Day type identifier
            
        Returns:
            dict: Hourly flows for all metrics
        """
        # Get multipliers for this month/day combination
        consumption_multiplier = self.weather.get_consumption_multiplier(month, day_type)
        generation_multiplier = self.weather.get_generation_multiplier(month, day_type)
        
        # Calculate hourly consumption
        adjusted_daily_consumption = household.daily_consumption * consumption_multiplier
        hourly_consumption = np.array(HOURLY_CONSUMPTION_PROFILE) * adjusted_daily_consumption
        
        # Calculate hourly generation
        if household.is_prosumer and household.solar_capacity > 0:
            max_daily_generation = household.solar_capacity * SOLAR_PRODUCTION_RATIO
            adjusted_daily_generation = max_daily_generation * generation_multiplier
            hourly_generation = np.array(HOURLY_SOLAR_PROFILE) * adjusted_daily_generation
        else:
            hourly_generation = np.zeros(24)
        
        # Calculate energy balance and grid interactions
        hourly_balance = hourly_generation - hourly_consumption
        grid_interactions = self._calculate_grid_interaction(hourly_balance)
        
        return {
            'consumption': hourly_consumption,
            'generation': hourly_generation,
            'grid_consumption': grid_interactions['grid_consumption'],
            'grid_feed_in': grid_interactions['grid_feed_in'],
            'credits_earned': grid_interactions['credits_earned'],
            'credits_used': grid_interactions['credits_used']
        }
    
    def _calculate_grid_interaction(self, hourly_balance):
        """
        Calculate grid interaction for 24-hour period.
        Reuses existing logic from energy_system.py
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
                hourly_credits_earned[hour] = balance * FEED_IN_FACTOR
            else:
                # Deficit - use credits if available, then grid
                deficit = -balance
                credits_used = min(deficit, hourly_credits_available[hour])
                grid_needed = deficit - credits_used
                
                hourly_credits_used[hour] = credits_used
                hourly_grid_consumption[hour] = grid_needed
        
        return {
            "grid_feed_in": hourly_grid_feed_in,
            "grid_consumption": hourly_grid_consumption,
            "credits_earned": hourly_credits_earned,
            "credits_used": hourly_credits_used
        }
    
    def _calculate_stress_index(self, system_hourly):
        """
        Calculate grid stress index from hourly system data.
        
        Args:
            system_hourly: Dictionary of hourly system metrics
            
        Returns:
            dict: Stress metrics
        """
        total_consumption = np.sum(system_hourly['consumption'])
        total_grid_consumption = np.sum(system_hourly['grid_consumption'])
        peak_load = np.max(system_hourly['grid_consumption'])
        avg_load = np.mean(system_hourly['grid_consumption'])
        
        # Grid dependency ratio
        grid_dependency = (total_grid_consumption / total_consumption 
                         if total_consumption > 0 else 1.0)
        
        # Peak-to-average ratio (higher = more stress)
        peak_avg_ratio = peak_load / avg_load if avg_load > 0 else 0
        
        # Load factor (closer to 1 = more efficient)
        load_factor = avg_load / peak_load if peak_load > 0 else 0
        
        return {
            'grid_dependency': grid_dependency,
            'peak_avg_ratio': peak_avg_ratio,
            'load_factor': load_factor,
            'stress_index': grid_dependency * peak_avg_ratio  # Combined stress metric
        }
    
    def _calculate_derived_metrics(self, system_metrics):
        """
        Calculate system-level derived metrics.
        
        Args:
            system_metrics: Dictionary of calculated metrics
            
        Returns:
            dict: Derived metrics
        """
        # Monthly peak load (maximum across all day types)
        monthly_peak = max(system_metrics['peak_loads'].values()) if system_metrics['peak_loads'] else 0
        
        # Overall credit utilization
        total_earned = sum(day['earned'] for day in system_metrics['credit_metrics'].values())
        total_used = sum(day['used'] for day in system_metrics['credit_metrics'].values())
        overall_credit_utilization = (total_used / total_earned if total_earned > 0 else 0)
        
        # Average stress index
        stress_values = [day['stress_index'] for day in system_metrics['seasonal_stress'].values()]
        avg_stress_index = np.mean(stress_values) if stress_values else 0
        
        return {
            'monthly_peak_load': monthly_peak,
            'overall_credit_utilization': overall_credit_utilization,
            'avg_stress_index': avg_stress_index
        }
    
    def _init_monthly_totals(self):
        """Initialize monthly totals structure."""
        return {
            'consumption': 0,
            'generation': 0,
            'grid_consumption': 0,
            'grid_feed_in': 0,
            'credits_earned': 0,
            'credits_used': 0
        }
    
    def _accumulate_monthly_totals(self, monthly_totals, day_metrics, count):
        """Accumulate daily metrics to monthly totals."""
        for key in monthly_totals:
            if key in day_metrics['daily_totals']:
                monthly_totals[key] += day_metrics['daily_totals'][key] * count