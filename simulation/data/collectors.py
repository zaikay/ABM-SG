# data/collectors.py V5.4 - COMPLETE FIX for Mesa DataCollector issues
"""
Mesa 2.4.0 compatible data collector with FIXED data structure consistency.
"""
from mesa.datacollection import DataCollector
import numpy as np
import pandas as pd

class SimulationDataCollector:
    """
    Mesa 2.4.0 compatible data collector with FIXED data consistency.
    """
    
    def __init__(self, model, config):
        """
        Initialize the data collector.
        
        Args:
            model: Mesa model instance
            config: Simulation configuration
        """
        self.model = model
        self.config = config
        
        # Mesa 2.4.0 REQUIRES ALL model reporters to be lambda functions
        self.data_collector = DataCollector(
            model_reporters={
                # Basic temporal info
                "Month": lambda m: m.schedule.steps,
                "Year": lambda m: m.schedule.steps // 12 + 1,
                
                # System-level metrics
                "AdoptionRate": lambda m: self._calculate_adoption_rate(m),
                "FossilDependency": lambda m: m.central_provider.grid_system.fossil_dependency,
                "FossilPrice": lambda m: m.central_provider.fossil_price,
                "RenewablePrice": lambda m: m.central_provider.renewable_price,
                
                # Energy totals
                "TotalConsumption": lambda m: m.central_provider.monthly_consumption,
                "TotalGeneration": lambda m: m.central_provider.monthly_generation,
                "GridConsumption": lambda m: m.central_provider.monthly_grid_consumption,
                
                # Aggregate metrics
                "MonthlyCosts": lambda m: self._calculate_monthly_costs(m),
                "IncomeClassAdoption": lambda m: self._calculate_income_class_adoption(m),
                "NetworkMetrics": lambda m: self._calculate_network_metrics(m),
                
                # NEW: Peak Load Evolution (exact calculation)
                "MonthlyPeakLoad": lambda m: getattr(m.central_provider, "monthly_peak_load", 0),
                "PeakLoadByDayType": lambda m: getattr(m.central_provider, "peak_load_breakdown", {}),
                
                # NEW: Credit System Utilization
                "TotalCreditsEarned": lambda m: getattr(m.central_provider, "total_credits_earned", 0),
                "TotalCreditsUsed": lambda m: getattr(m.central_provider, "total_credits_used", 0),
                "CreditUtilizationRate": lambda m: getattr(m.central_provider, "overall_credit_utilization", 0),
                
                # NEW: Seasonal Grid Stress Patterns
                "GridStressIndex": lambda m: getattr(m.central_provider, "avg_stress_index", 0),
                "StressByDayType": lambda m: getattr(m.central_provider, "seasonal_stress_breakdown", {}),
                "CurrentMonthInYear": lambda m: (m.schedule.steps % 12) + 1,
            },
            agent_reporters={
                # FIXED: Ensure ALL agent reporters return consistent data types
                # Use helper function to ensure consistent returns
                
                # Temporal information - FIXED to ensure consistency
                "Step": lambda a: self._safe_get_step(a),
                "Year": lambda a: self._safe_get_year(a),
                "Month": lambda a: self._safe_get_month(a),
                
                # Basic household attributes
                "AgentType": lambda a: self._safe_get_agent_type(a),
                "Income": lambda a: self._safe_get_household_attr(a, "income"),
                "IncomeClass": lambda a: self._safe_get_household_attr(a, "income_class"),
                "DailyConsumption": lambda a: self._safe_get_household_attr(a, "daily_consumption"),
                
                # Solar system status
                "IsProsumer": lambda a: self._safe_get_household_bool(a, "is_prosumer"),
                "SolarCapacity": lambda a: self._safe_get_household_attr(a, "solar_capacity", 0),
                "InstallationMonth": lambda a: self._safe_get_household_attr(a, "installation_month"),
                
                # Economic metrics
                "BaselineCost": lambda a: self._safe_get_household_attr(a, "baseline_cost", 0),
                "SolarCost": lambda a: self._safe_get_household_attr(a, "solar_cost", 0),
                "MonthlySavings": lambda a: self._safe_get_household_attr(a, "monthly_savings", 0),
                "AnnualSavings": lambda a: self._safe_get_household_attr(a, "annual_savings", 0),
                "NPV": lambda a: self._safe_get_household_attr(a, "npv"),
                "InstallationCost": lambda a: self._safe_get_household_attr(a, "installation_cost", 0),
                "PaybackPeriod": lambda a: self._safe_get_household_attr(a, "payback_period", float('inf')),
                
                # Energy flows
                "MonthlyConsumption": lambda a: self._safe_get_household_attr(a, "monthly_consumption", 0),
                "MonthlyGeneration": lambda a: self._safe_get_household_attr(a, "monthly_generation", 0),
                "MonthlyGridConsumption": lambda a: self._safe_get_household_attr(a, "monthly_grid_consumption", 0),
                "MonthlyGridFeedIn": lambda a: self._safe_get_household_attr(a, "monthly_grid_feed_in", 0),
                "MonthlyCreditsEarned": lambda a: self._safe_get_household_attr(a, "monthly_credits_earned", 0),
                "MonthlyCreditsUsed": lambda a: self._safe_get_household_attr(a, "monthly_credits_used", 0),
                "MonthlyCreditsExpired": lambda a: self._safe_get_household_attr(a, "monthly_credits_expired", 0),
                
                # Adoption potential
                "IsPotentialProsumer": lambda a: self._safe_get_potential_prosumer(a),
            }
        )
    
    # ADDED: Helper functions to ensure data consistency
    def _safe_get_step(self, agent):
        """Get step consistently for all agents."""
        if hasattr(agent, "daily_consumption"):  # Is a household
            return agent.model.schedule.steps
        return None
    
    def _safe_get_year(self, agent):
        """Get year consistently for all agents."""
        if hasattr(agent, "daily_consumption"):  # Is a household
            return (agent.model.schedule.steps // 12) + 1
        return None
    
    def _safe_get_month(self, agent):
        """Get month consistently for all agents."""
        if hasattr(agent, "daily_consumption"):  # Is a household
            return agent.model.schedule.steps
        return None
    
    def _safe_get_agent_type(self, agent):
        """Get agent type consistently."""
        if hasattr(agent, "daily_consumption"):
            return "Household"
        else:
            return "CentralProvider"
    
    def _safe_get_household_attr(self, agent, attr_name, default=None):
        """Safely get household attribute with consistent return type."""
        if hasattr(agent, "daily_consumption"):  # Is a household
            return getattr(agent, attr_name, default)
        return None
    
    def _safe_get_household_bool(self, agent, attr_name, default=False):
        """Safely get household boolean attribute."""
        if hasattr(agent, "daily_consumption"):  # Is a household
            return getattr(agent, attr_name, default)
        return None
    
    def _safe_get_potential_prosumer(self, agent):
        """Safely calculate potential prosumer status."""
        if hasattr(agent, "daily_consumption"):  # Is a household
            npv = getattr(agent, "npv", None)
            return npv is not None and npv > 0
        return None
    
    def _calculate_adoption_rate(self, model):
        """
        Calculate adoption rate.
        Mesa 2.4.0 compatible - called via lambda from model_reporters.
        """
        households = [agent for agent in model.schedule.agents if hasattr(agent, "is_prosumer")]
        if not households:
            return 0
        
        prosumers = sum(1 for h in households if h.is_prosumer)
        return prosumers / len(households)
    
    def _calculate_monthly_costs(self, model):
        """
        Calculate monthly costs.
        Mesa 2.4.0 compatible - called via lambda from model_reporters.
        """
        households = [agent for agent in model.schedule.agents if hasattr(agent, "daily_consumption")]
        if not households:
            return {"mean": 0, "median": 0, "min": 0, "max": 0, "prosumer_mean": 0, "nonprosumer_mean": 0}
        
        # Use the actual cost each household is paying
        costs = [getattr(h, "solar_cost", 0) for h in households]
        prosumer_costs = [getattr(h, "solar_cost", 0) for h in households if getattr(h, "is_prosumer", False)]
        nonprosumer_costs = [getattr(h, "solar_cost", 0) for h in households if not getattr(h, "is_prosumer", False)]
        
        return {
            "mean": np.mean(costs) if costs else 0,
            "median": np.median(costs) if costs else 0,
            "min": np.min(costs) if costs else 0,
            "max": np.max(costs) if costs else 0,
            "prosumer_mean": np.mean(prosumer_costs) if prosumer_costs else 0,
            "nonprosumer_mean": np.mean(nonprosumer_costs) if nonprosumer_costs else 0
        }
    
    def _calculate_income_class_adoption(self, model):
        """
        Calculate income class adoption.
        Mesa 2.4.0 compatible - called via lambda from model_reporters.
        """
        class_households = {}
        class_prosumers = {}
        
        for agent in model.schedule.agents:
            if hasattr(agent, "income_class") and hasattr(agent, "is_prosumer"):
                ic = getattr(agent, "income_class", 0)
                
                if ic not in class_households:
                    class_households[ic] = 0
                    class_prosumers[ic] = 0
                
                class_households[ic] += 1
                if getattr(agent, "is_prosumer", False):
                    class_prosumers[ic] += 1
        
        # Calculate adoption rates
        adoption_rates = {}
        for ic in class_households:
            if class_households[ic] > 0:
                adoption_rates[ic] = class_prosumers[ic] / class_households[ic]
            else:
                adoption_rates[ic] = 0
        
        return adoption_rates
    
    def _calculate_network_metrics(self, model):
        """
        Calculate network metrics.
        Mesa 2.4.0 compatible - called via lambda from model_reporters.
        """
        if not hasattr(model, "network_metrics"):
            return {"clusters": 0, "homophily": 0}
        
        try:
            clusters = model.network_metrics.get_adoption_clusters()
            homophily = model.network_metrics.get_adoption_homophily()
            
            return {
                "clusters": clusters.get("clusters", 0),
                "largest_cluster": clusters.get("largest_cluster", 0),
                "homophily": homophily.get("homophily_index", 0)
            }
        except Exception:
            return {"clusters": 0, "largest_cluster": 0, "homophily": 0}
    
    def collect_data(self):
        """
        Collect data for the current step.
        """
        self.data_collector.collect(self.model)
    
    def get_model_data(self):
        """
        Get collected model-level data.
        """
        return self.data_collector.get_model_vars_dataframe()
    
    def get_agent_data(self):
        """
        Get collected agent-level data with improved error handling.
        """
        try:
            return self.data_collector.get_agent_vars_dataframe()
        except ValueError as e:
            print(f"Error getting agent data: {e}")
            # Debug information
            print(f"Number of agent records: {len(self.data_collector._agent_records)}")
            if self.data_collector._agent_records:
                first_step = list(self.data_collector._agent_records.keys())[0]
                print(f"Records in first step: {len(self.data_collector._agent_records[first_step])}")
                print(f"First record sample: {self.data_collector._agent_records[first_step][0] if self.data_collector._agent_records[first_step] else 'No records'}")
            
            # Return empty DataFrame as fallback
            return pd.DataFrame()
    
    def save_data(self, output_dir="results"):
        """
        Save collected data to CSV files with improved error handling.
        """
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model data
        try:
            model_data = self.get_model_data()
            model_data.to_csv(f"{output_dir}/model_data.csv")
            print(f"Model data saved: {model_data.shape}")
        except Exception as e:
            print(f"Error saving model data: {e}")
        
        # Save agent data with error handling
        try:
            agent_data = self.get_agent_data()
            if not agent_data.empty:
                agent_data.to_csv(f"{output_dir}/agent_data.csv")
                print(f"Agent data saved: {agent_data.shape}")
            else:
                print("Warning: Agent data is empty, not saving agent_data.csv")
        except Exception as e:
            print(f"Error saving agent data: {e}")
        
        # Save configuration
        try:
            pd.Series(self.config).to_csv(f"{output_dir}/config.csv")
        except Exception as e:
            print(f"Error saving config: {e}")
        
        print(f"Data saved to {output_dir}")

    def get_summary_statistics(self):
        """
        Get basic summary statistics for debugging.
        """
        households = [agent for agent in self.model.schedule.agents if hasattr(agent, "is_prosumer")]
        
        if not households:
            return {"error": "No households found"}
        
        total_households = len(households)
        prosumers = sum(1 for h in households if getattr(h, "is_prosumer", False))
        nonprosumers = total_households - prosumers
        
        positive_npv = sum(1 for h in households 
                          if getattr(h, "npv", None) is not None and getattr(h, "npv", 0) > 0)
        
        total_consumption = sum(getattr(h, "monthly_consumption", 0) for h in households)
        total_generation = sum(getattr(h, "monthly_generation", 0) for h in households)
        
        return {
            "total_households": total_households,
            "prosumers": prosumers,
            "nonprosumers": nonprosumers,
            "adoption_rate": prosumers / total_households if total_households > 0 else 0,
            "households_with_positive_npv": positive_npv,
            "potential_adoption_rate": positive_npv / total_households if total_households > 0 else 0,
            "total_monthly_consumption": total_consumption,
            "total_monthly_generation": total_generation,
            "generation_to_consumption_ratio": total_generation / total_consumption if total_consumption > 0 else 0
        }