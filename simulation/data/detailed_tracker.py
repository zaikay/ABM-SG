# data/detailed_tracker.py V5.1
"""
Detailed hourly data tracking for sample households.
REFACTORED to work with unified EnergySystem and simplified household calculations.
"""
import os
import numpy as np
import pandas as pd
import random
from collections import defaultdict

class DetailedTracker:
    """
    Tracks detailed hourly data for a sample of households.
    UPDATED to work with refactored energy system.
    """
    def __init__(self, model, sample_size=5, seed=42):
        """
        Initialize the detailed tracker.
        
        Args:
            model: Mesa model instance
            sample_size: Number of households to track
            seed: Random seed for household selection
        """
        self.model = model
        self.sample_size = sample_size
        random.seed(seed)
        
        # Select sample households
        self.sample_households = self._select_sample_households()
        
        # Data storage structure
        # Format: {household_id: {step: {day_type: {hour: {metrics}}}}}
        self.hourly_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
        
        # Step data storage (monthly metrics for each household)
        # Format: {household_id: {step: {metrics}}}
        self.step_data = defaultdict(lambda: defaultdict(dict))
        
        print(f"Tracking detailed data for {len(self.sample_households)} sample households: {[h.unique_id for h in self.sample_households]}")
    
    def _select_sample_households(self):
        """
        Select sample households to track.
        
        Returns:
            list: Sample household agents
        """
        # Get all households
        all_households = [agent for agent in self.model.schedule.agents 
                         if hasattr(agent, "daily_consumption")]
        
        # Ensure we don't try to sample more households than exist
        sample_size = min(self.sample_size, len(all_households))
        
        # Randomly select sample households
        sample_households = random.sample(all_households, sample_size)
        
        return sample_households
    
    def track_step_data(self, step):
            """
            Track monthly data for the sample households.
            ENHANCED: Track all scenarios for multi-experiment analysis.
            
            Args:
                step: Current simulation step
            """
            for household in self.sample_households:
                # Check if this is a MultiScenarioHousehold
                if hasattr(household, 'scenarios') and hasattr(household, 'scenario_adoption'):
                    # Multi-scenario household - track all scenarios
                    base_data = {
                        "unique_id": household.unique_id,
                        "step": step,
                        "income": household.income,
                        "income_class": household.income_class,
                        "daily_consumption": household.daily_consumption,
                        "position_x": household.pos[0] if household.pos else 0,
                        "position_y": household.pos[1] if household.pos else 0,
                        
                        # Base economic metrics (from rational calculation)
                        "baseline_cost": getattr(household, "baseline_cost", 0),
                        "solar_cost": getattr(household, "solar_cost", 0),
                        "monthly_savings": getattr(household, "monthly_savings", 0),
                        "annual_savings": getattr(household, "annual_savings", 0),
                        "npv": getattr(household, "npv", None),
                        "installation_cost": getattr(household, "installation_cost", 0),
                        "payback_period": getattr(household, "payback_period", float('inf')),
                        
                        # Energy flows (from rational scenario)
                        "monthly_consumption": getattr(household, "monthly_consumption", 0),
                        "monthly_generation": getattr(household, "monthly_generation", 0),
                        "monthly_grid_consumption": getattr(household, "monthly_grid_consumption", 0),
                        "monthly_grid_feed_in": getattr(household, "monthly_grid_feed_in", 0),
                        "monthly_credits_earned": getattr(household, "monthly_credits_earned", 0),
                        "monthly_credits_used": getattr(household, "monthly_credits_used", 0),
                        "monthly_credits_expired": getattr(household, "monthly_credits_expired", 0),
                    }
                    
                    # Add scenario-specific adoption data
                    for scenario in household.scenarios:
                        scenario_data = base_data.copy()
                        scenario_data.update({
                            "scenario": scenario,
                            "is_prosumer": household.scenario_adoption.get(scenario, False),
                            "adoption_month": household.adoption_months.get(scenario, None),
                            "adoption_probability": household.scenario_probability.get(scenario, 0.0),
                            "adopted_this_step": (household.adoption_months.get(scenario) == step),
                        })
                        
                        # Store scenario-specific data
                        self.step_data[household.unique_id][step][scenario] = scenario_data
                    
                    # Add bias effects data if available
                    if hasattr(household, 'current_bias_effects') and household.current_bias_effects:
                        bias_effects_data = base_data.copy()
                        bias_effects_data.update({
                            "scenario": "bias_effects",
                            "bias_effects": household.current_bias_effects
                        })
                        self.step_data[household.unique_id][step]["bias_effects"] = bias_effects_data
                    
                    # Add social influence data if available
                    if hasattr(household, 'neighbor_adoption_rates') and household.neighbor_adoption_rates:
                        social_data = base_data.copy()
                        social_data.update({
                            "scenario": "social_influence",
                            "neighbor_adoption_rates": household.neighbor_adoption_rates,
                            "spatial_influence": (household.spatial_influence_history[-1]['spatial_influence'] 
                                                if household.spatial_influence_history else 0.0),
                            "class_influence": (household.class_influence_history[-1]['class_influence'] 
                                            if household.class_influence_history else 0.0)
                        })
                        self.step_data[household.unique_id][step]["social_influence"] = social_data
                        
                else:
                    # Regular household - use original tracking method
                    household_data = {
                        "unique_id": household.unique_id,
                        "step": step,
                        "income": household.income,
                        "income_class": household.income_class,
                        "daily_consumption": household.daily_consumption,
                        "is_prosumer": household.is_prosumer,
                        "solar_capacity": household.solar_capacity,
                        "installation_month": household.installation_month,
                        
                        # Economic metrics (directly from household attributes)
                        "baseline_cost": getattr(household, "baseline_cost", 0),
                        "solar_cost": getattr(household, "solar_cost", 0),
                        "monthly_savings": getattr(household, "monthly_savings", 0),
                        "annual_savings": getattr(household, "annual_savings", 0),
                        "npv": getattr(household, "npv", None),
                        "installation_cost": getattr(household, "installation_cost", 0),
                        "payback_period": getattr(household, "payback_period", float('inf')),
                        
                        # Energy flows (directly from household attributes)
                        "monthly_consumption": getattr(household, "monthly_consumption", 0),
                        "monthly_generation": getattr(household, "monthly_generation", 0),
                        "monthly_grid_consumption": getattr(household, "monthly_grid_consumption", 0),
                        "monthly_grid_feed_in": getattr(household, "monthly_grid_feed_in", 0),
                        "monthly_credits_earned": getattr(household, "monthly_credits_earned", 0),
                        "monthly_credits_used": getattr(household, "monthly_credits_used", 0),
                        "monthly_credits_expired": getattr(household, "monthly_credits_expired", 0),
                    }
                    
                    # Store the data
                    self.step_data[household.unique_id][step]["rational"] = household_data
    
    def track_hourly_data(self, step):
        """
        Track detailed hourly data for the sample households.
        UPDATED to use unified EnergySystem instead of old energy_calculator.
        
        Args:
            step: Current simulation step
        """
        # Get current month (1-12)
        current_month = (step % 12) + 1
        
        # Get weather patterns
        weather_patterns = self.model.weather_patterns
        
        # Get all representative day types for this month
        rep_days = weather_patterns.get_representative_days(current_month)
        
        # For each household and day type, calculate hourly energy flows
        for household in self.sample_households:
            for day_type, count, weather, day_of_week in rep_days:
                # Use the unified energy system for hourly calculations
                hourly_metrics = self._calculate_hourly_metrics_for_day(
                    household, current_month, day_type
                )
                
                # Store hourly data for this household and day type
                for hour in range(24):
                    hour_data = {
                        "household_id": household.unique_id,
                        "step": step,
                        "month": current_month,
                        "day_type": day_type,
                        "day_count": count,
                        "hour": hour,
                        "consumption": hourly_metrics["hourly_consumption"][hour],
                        "generation": hourly_metrics["hourly_generation"][hour],
                        "energy_balance": hourly_metrics["hourly_balance"][hour],
                        "grid_feed_in": hourly_metrics["grid_interactions"]["grid_feed_in"][hour],
                        "grid_consumption": hourly_metrics["grid_interactions"]["grid_consumption"][hour],
                        "credits_earned": hourly_metrics["grid_interactions"]["credits_earned"][hour],
                        "credits_used": hourly_metrics["grid_interactions"]["credits_used"][hour],
                        "credits_available": hourly_metrics["grid_interactions"]["credits_available"][hour],
                        "energy_cost": hourly_metrics["grid_interactions"]["grid_consumption"][hour] * self.model.grid_system.fossil_price
                    }
                    
                    self.hourly_data[household.unique_id][step][day_type][hour] = hour_data
    
    def _calculate_hourly_metrics_for_day(self, household, month, day_type):
        """
        Calculate hourly metrics for a specific household and day type.
        UPDATED to use unified EnergySystem methods.
        
        Args:
            household: Household agent
            month: Month number (1-12)
            day_type: Day type identifier
            
        Returns:
            dict: Hourly metrics for the day
        """
        # Get consumption and generation multipliers
        consumption_multiplier = self.model.weather_patterns.get_consumption_multiplier(month, day_type)
        generation_multiplier = self.model.weather_patterns.get_generation_multiplier(month, day_type)
        
        # Calculate hourly consumption
        adjusted_daily_consumption = household.daily_consumption * consumption_multiplier
        hourly_consumption = np.array(self.model.energy_system.hourly_consumption_profile) * adjusted_daily_consumption
        
        # Calculate hourly generation
        if household.is_prosumer and household.solar_capacity > 0:
            from ..utils.parameters import SOLAR_PRODUCTION_RATIO
            max_daily_generation = household.solar_capacity * SOLAR_PRODUCTION_RATIO
            adjusted_daily_generation = max_daily_generation * generation_multiplier
            hourly_generation = np.array(self.model.energy_system.hourly_solar_profile) * adjusted_daily_generation
        else:
            hourly_generation = np.zeros(24)
        
        # Calculate energy balance
        hourly_balance = hourly_generation - hourly_consumption
        
        # Calculate grid interactions using energy system method
        grid_interactions = self.model.energy_system._calculate_grid_interaction(hourly_balance)
        
        return {
            "hourly_consumption": hourly_consumption,
            "hourly_generation": hourly_generation,
            "hourly_balance": hourly_balance,
            "grid_interactions": grid_interactions
        }
    
    def track_projected_performance(self, step):
        """
        Track projected performance if households were to adopt solar.
        SIMPLIFIED to use household attributes instead of recalculating.
        
        Args:
            step: Current simulation step
        """
        # Get current month (1-12)
        current_month = (step % 12) + 1
        
        # For each non-prosumer household, get projected performance from household attributes
        for household in self.sample_households:
            if not household.is_prosumer:
                # Use the values already calculated by the household's energy_system
                # No need to recalculate - just extract from household attributes
                
                projected_data = {
                    "solar_capacity": getattr(household, "solar_capacity", 0),  # Will be 0 for non-prosumers
                    "projected_solar_capacity": getattr(household, "npv", None),  # Available from NPV calculation
                    "baseline_cost": getattr(household, "baseline_cost", 0),
                    "projected_solar_cost": getattr(household, "solar_cost", 0),  # This is projected for non-prosumers
                    "monthly_savings": getattr(household, "monthly_savings", 0),  # This is projected for non-prosumers
                    "npv": getattr(household, "npv", None),
                    "installation_cost": getattr(household, "installation_cost", 0),
                    "annual_savings": getattr(household, "annual_savings", 0),
                    "payback_period": getattr(household, "payback_period", float('inf'))
                }
                
                # Store projected data
                if "projected" not in self.step_data[household.unique_id][step]:
                    self.step_data[household.unique_id][step]["projected"] = {}
                
                self.step_data[household.unique_id][step]["projected"].update(projected_data)
                
                # Also track one representative hourly day for projected performance
                self._track_projected_hourly_data(household, step, current_month)
    
    def _track_projected_hourly_data(self, household, step, current_month):
        """
        Track projected hourly data for a non-prosumer household.
        SIMPLIFIED to use household's projected solar capacity.
        
        Args:
            household: Household agent
            step: Current step
            current_month: Current month (1-12)
        """
        # Get weather patterns
        weather_patterns = self.model.weather_patterns
        
        # Choose one representative day type (e.g., mixed weekday - typical conditions)
        rep_days = weather_patterns.get_representative_days(current_month)
        if not rep_days:
            return
            
        # Use first available day type
        day_type = rep_days[0][0]
        
        # For non-prosumers, calculate what performance would be with solar
        # Use a reasonable projected solar capacity based on their consumption
        from ..utils.parameters import SOLAR_SIZING_FACTOR, SOLAR_PRODUCTION_RATIO
        projected_solar_capacity = (household.daily_consumption / SOLAR_PRODUCTION_RATIO) * SOLAR_SIZING_FACTOR
        
        # Create a temporary household-like object for projection
        class ProjectedHousehold:
            def __init__(self, original_household, projected_capacity):
                self.daily_consumption = original_household.daily_consumption
                self.solar_capacity = projected_capacity
                self.is_prosumer = True  # For calculation purposes
        
        projected_household = ProjectedHousehold(household, projected_solar_capacity)
        
        # Calculate projected hourly metrics
        projected_metrics = self._calculate_hourly_metrics_for_day(
            projected_household, current_month, day_type
        )
        
        # Store projected hourly data
        for hour in range(24):
            projected_hour_data = {
                "household_id": household.unique_id,
                "step": step,
                "month": current_month,
                "day_type": day_type,
                "hour": hour,
                "projected_consumption": projected_metrics["hourly_consumption"][hour],
                "projected_generation": projected_metrics["hourly_generation"][hour],
                "projected_balance": projected_metrics["hourly_balance"][hour],
                "projected_grid_feed_in": projected_metrics["grid_interactions"]["grid_feed_in"][hour],
                "projected_grid_consumption": projected_metrics["grid_interactions"]["grid_consumption"][hour],
                "projected_credits_earned": projected_metrics["grid_interactions"]["credits_earned"][hour],
                "projected_credits_used": projected_metrics["grid_interactions"]["credits_used"][hour],
                "projected_credits_available": projected_metrics["grid_interactions"]["credits_available"][hour],
                "projected_energy_cost": projected_metrics["grid_interactions"]["grid_consumption"][hour] * self.model.grid_system.fossil_price
            }
            
            # Add to our hourly data structure
            if "projected" not in self.hourly_data[household.unique_id][step]:
                self.hourly_data[household.unique_id][step]["projected"] = {}
            
            self.hourly_data[household.unique_id][step]["projected"][hour] = projected_hour_data
    
    def export_data(self, output_dir="results/detailed_data"):
        """
        Export the collected data to CSV files.
        UNCHANGED - export logic remains the same.
        
        Args:
            output_dir: Directory to save the data
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Export step data including projected performance
        step_rows = []
        projected_rows = []
        
        for household_id, steps in self.step_data.items():
            for step, data in steps.items():
                # Regular data
                if isinstance(data, dict) and "projected" not in data:
                    step_rows.append(data)
                elif isinstance(data, dict) and "projected" in data:
                    # Regular data (without projected)
                    regular_data = {k: v for k, v in data.items() if k != "projected"}
                    if regular_data:
                        step_rows.append(regular_data)
                    
                    # Projected data
                    proj_data = data["projected"]
                    proj_data["unique_id"] = household_id
                    proj_data["step"] = step
                    proj_data["is_projected"] = True
                    projected_rows.append(proj_data)
        
        if step_rows:
            step_df = pd.DataFrame(step_rows)
            step_df.to_csv(f"{output_dir}/sample_households_monthly.csv", index=False)
        
        if projected_rows:
            proj_df = pd.DataFrame(projected_rows)
            proj_df.to_csv(f"{output_dir}/sample_households_projected.csv", index=False)
        
        # Export hourly data including projected
        hourly_rows = []
        projected_hourly_rows = []
        
        for household_id, steps in self.hourly_data.items():
            for step, day_types in steps.items():
                for day_type, hours in day_types.items():
                    if day_type == "projected":
                        # Projected hourly data
                        for hour, data in hours.items():
                            projected_hourly_rows.append(data)
                    else:
                        # Regular hourly data
                        for hour, data in hours.items():
                            hourly_rows.append(data)
        
        if hourly_rows:
            hourly_df = pd.DataFrame(hourly_rows)
            hourly_df.to_csv(f"{output_dir}/sample_households_hourly.csv", index=False)
        
        if projected_hourly_rows:
            proj_hourly_df = pd.DataFrame(projected_hourly_rows)
            proj_hourly_df.to_csv(f"{output_dir}/sample_households_projected_hourly.csv", index=False)
        
        print(f"Exported detailed data for {len(self.sample_households)} sample households to {output_dir}")

    def get_scenario_comparison_data(self):
        """
        Get data formatted for cross-scenario comparison.
        
        Returns:
            dict: Organized data for scenario comparison analysis
        """
        comparison_data = {
            "by_household": {},
            "by_scenario": defaultdict(list),
            "by_step": defaultdict(lambda: defaultdict(list))
        }
        
        # Organize data for easy comparison
        for household_id, steps in self.step_data.items():
            comparison_data["by_household"][household_id] = {}
            
            for step, step_data in steps.items():
                comparison_data["by_household"][household_id][step] = step_data
                
                # Organize by scenario
                for scenario, data in step_data.items():
                    if scenario not in ["bias_effects", "social_influence", "projected"]:
                        comparison_data["by_scenario"][scenario].append(data)
                        comparison_data["by_step"][step][scenario].append(data)
        
        return comparison_data
    
    def get_debug_summary(self):
        """
        Get debug summary of tracked data.
        NEW method for debugging the tracker.
        
        Returns:
            dict: Debug summary
        """
        total_step_records = sum(len(steps) for steps in self.step_data.values())
        total_hourly_records = sum(
            sum(len(hours) for day_types in steps.values() for hours in day_types.values())
            for steps in self.hourly_data.values()
        )
        
        return {
            "tracked_households": len(self.sample_households),
            "household_ids": [h.unique_id for h in self.sample_households],
            "total_step_records": total_step_records,
            "total_hourly_records": total_hourly_records,
            "latest_step": max((max(steps.keys()) for steps in self.step_data.values()), default=0)
        }