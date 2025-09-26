# data/multi_experiment_collector.py V2.0 - FIXED DATA COLLECTION
"""
Fixed multi-experiment data collector with comprehensive income class and system metrics.
Addresses missing data issues in visualization generation.
"""

import pandas as pd
import numpy as np
import os
from collections import defaultdict

from ..utils.parameters import get_all_scenarios, get_scenario_metadata

class MultiExperimentCollector:
    """
    FIXED: Comprehensive data collector for multi-scenario behavioral experiments.
    Now includes income class data and system metrics required for all visualizations.
    """
    
    def __init__(self, model, config):
        """
        Initialize the multi-experiment collector.
        
        Args:
            model: MultiExperimentModel instance
            config: Simulation configuration
        """
        self.model = model
        self.config = config
        self.scenarios = get_all_scenarios()
        
        # Data storage
        self.agent_data = []           # Agent-level data for all scenarios
        self.system_metrics = []       # System-level metrics
        self.bias_effects_data = []    # Bias effects analysis
        
        print(f"MultiExperimentCollector initialized for {len(self.scenarios)} scenarios")
    
    def collect_data(self):
        """
        FIXED: Collect comprehensive data including income class metrics.
        """
        current_step = self.model.schedule.steps
        current_year = (current_step // 12) + 1
        current_month = (current_step % 12) + 1
        
        households = self.model.get_households()
        
            # FOR DEBUGGING (optional):
        if current_step == 0:
            sample_household = households[0] if households else None
            if sample_household:
                npv_status = "calculated" if sample_household.npv is not None else "not calculated"
                print(f"Data collection at step {current_step}: NPV {npv_status}")
        
        # Collect agent-level data for all scenarios
        for household in households:
            base_record = {
                'Step': current_step,
                'Year': current_year,
                'Month': current_month,
                'AgentID': household.unique_id,
                'AgentType': 'Household',  # FIXED: Always include AgentType
                'Income': household.income,
                'IncomeClass': household.income_class,  # FIXED: Always include IncomeClass
                'DailyConsumption': household.daily_consumption,
                'NPV': getattr(household, 'npv', 0),
                'InstallationCost': getattr(household, 'installation_cost', 0),
                'AnnualSavings': getattr(household, 'annual_savings', 0),
                # NEW 8 energy columns after AnnualSavings
                'MonthlyConsumption': getattr(household, 'monthly_consumption', 0),
                'MonthlyGeneration': getattr(household, 'monthly_generation', 0),
                'MonthlyGridConsumption': getattr(household, 'monthly_grid_consumption', 0),
                'MonthlyGridFeedIn': getattr(household, 'monthly_grid_feed_in', 0),
                'MonthlyCreditsEarned': getattr(household, 'monthly_credits_earned', 0),
                'MonthlyCreditsUsed': getattr(household, 'monthly_credits_used', 0),
                'MonthlyCreditsExpired': getattr(household, 'monthly_credits_expired', 0),
                'HouseholdFossilDependency': self._calculate_household_fossil_dependency(household),
                'PaybackPeriod': getattr(household, 'payback_period', float('inf')),
                'Position': household.pos
            }
            
            # Add scenario-specific adoption data
            for scenario in self.scenarios:
                adoption_status = household.scenario_adoption.get(scenario, False)
                base_record[f'{scenario}_Adopted'] = adoption_status
                base_record[f'{scenario}_AdoptionMonth'] = household.adoption_months.get(scenario, None)
                base_record[f'{scenario}_NPV'] = household.scenario_npv.get(scenario, 0)
                base_record[f'{scenario}_Probability'] = household.scenario_probability.get(scenario, 0)
                
                # For compatibility with existing visualizers
                if scenario == 'rational':
                    base_record['IsProsumer'] = adoption_status  # FIXED: Add IsProsumer for compatibility
            
            # Add potential prosumer status
            base_record['IsPotentialProsumer'] = (base_record['NPV'] or 0) > 0  # FIXED: Add IsPotentialProsumer
            
            self.agent_data.append(base_record)
        
        # FIXED: Collect system-level metrics including income class data
        system_record = {
            'Step': current_step,
            'Year': current_year,
            'Month': current_month,
            'TotalHouseholds': len(households)
        }
        
        # Calculate adoption rates by scenario
        for scenario in self.scenarios:
            adopters = sum(1 for h in households if h.scenario_adoption.get(scenario, False))
            adoption_rate = adopters / len(households) if households else 0
            system_record[f'{scenario}_AdoptionRate'] = adoption_rate
            system_record[f'{scenario}_AdopterCount'] = adopters
        
        # FIXED: Calculate income class adoption rates
        for scenario in self.scenarios:
            for income_class in range(1, 6):
                class_households = [h for h in households if h.income_class == income_class]
                if class_households:
                    class_adopters = sum(1 for h in class_households if h.scenario_adoption.get(scenario, False))
                    class_rate = class_adopters / len(class_households)
                    system_record[f'{scenario}_Class{income_class}_Rate'] = class_rate
                    system_record[f'{scenario}_Class{income_class}_Count'] = class_adopters
                else:
                    system_record[f'{scenario}_Class{income_class}_Rate'] = 0
                    system_record[f'{scenario}_Class{income_class}_Count'] = 0
        
        # Add scenario-independent metrics
        if hasattr(self.model, 'central_provider'):
            cp = self.model.central_provider
            system_record.update({
                # Scenario-independent metrics (keep as-is)
                'FossilPrice': getattr(cp, 'fossil_price', 0),
                'RenewablePrice': getattr(cp, 'renewable_price', 0),
                'TotalConsumption': getattr(cp, 'monthly_consumption', 0),  # Same for all scenarios
                'CurrentMonthInYear': current_month,
                'MonthlyPeakLoad': getattr(cp, 'current_peak_load', 0)
            })
        
        # Add scenario-dependent energy metrics
        for scenario in self.scenarios:
            scenario_metrics = self._calculate_scenario_system_metrics(households, scenario)
            
            system_record.update({
                f'{scenario}_TotalGeneration': scenario_metrics['total_generation'],
                f'{scenario}_GridConsumption': scenario_metrics['grid_consumption'],
                f'{scenario}_TotalCreditsEarned': scenario_metrics['total_credits_earned'],
                f'{scenario}_TotalCreditsUsed': scenario_metrics['total_credits_used'],
                f'{scenario}_CreditUtilizationRate': scenario_metrics['credit_utilization_rate'],
                f'{scenario}_GridStressIndex': scenario_metrics['grid_stress_index'],
                f'{scenario}_FossilDependency': scenario_metrics['fossil_dependency']
            })
            
            # Calculate monthly costs structure for compatibility
            system_record['MonthlyCosts'] = {
                'mean': getattr(cp, 'avg_monthly_cost', 100),
                'prosumer_mean': getattr(cp, 'avg_prosumer_cost', 80),
                'nonprosumer_mean': getattr(cp, 'avg_nonprosumer_cost', 120)
            }
        
        # bias effects analysis
        if current_step > 0:  # Only after first step
            bias_effects = self._calculate_bias_effects(households, current_step)
            self.bias_effects_data.append(bias_effects)
        
        self.system_metrics.append(system_record)
    
    def _calculate_bias_effects(self, households, current_step):
        """
        FIXED: Calculate bias effects for this timestep.
        """
        rational_adoption_rate = sum(1 for h in households if h.scenario_adoption.get('rational', False)) / len(households)
        
        bias_effects = {
            'Step': current_step,
            'Year': (current_step // 12) + 1,
            'Month': (current_step % 12) + 1,
            'RationalBaseline': rational_adoption_rate
        }
        
        # Calculate bias effects relative to rational baseline
        for scenario in self.scenarios:
            if scenario != 'rational':
                scenario_rate = sum(1 for h in households if h.scenario_adoption.get(scenario, False)) / len(households)
                absolute_effect = scenario_rate - rational_adoption_rate
                relative_effect = (absolute_effect / rational_adoption_rate * 100) if rational_adoption_rate > 0 else 0
                
                bias_effects[f'{scenario}_AbsoluteEffect'] = absolute_effect
                bias_effects[f'{scenario}_RelativeEffect'] = relative_effect
                bias_effects[f'{scenario}_AdoptionRate'] = scenario_rate
        
        return bias_effects
    
    def _calculate_household_fossil_dependency(self, household):
        """
        NEW: Calculate household fossil dependency (grid consumption / total consumption).
        
        Args:
            household: Household agent
            
        Returns:
            float: Fossil dependency ratio (0-1, where 1 = fully dependent on grid)
        """
        monthly_consumption = getattr(household, 'monthly_consumption', household.daily_consumption * 30)
        monthly_grid_consumption = getattr(household, 'monthly_grid_consumption', monthly_consumption)
        
        # Avoid division by zero
        if monthly_consumption > 0:
            return monthly_grid_consumption / monthly_consumption
        else:
            return 1.0  # If no consumption, assume full grid dependency

    def _calculate_scenario_system_metrics(self, households, scenario):
        """
        NEW: Calculate system metrics for a specific scenario.
        Only includes contributions from households that are prosumers in THIS scenario.
        
        Args:
            households: List of household agents
            scenario: Scenario name (e.g., 'rational', 'loss_aversion')
            
        Returns:
            dict: System metrics for this scenario
        """
        total_generation = 0
        total_grid_consumption = 0
        total_credits_earned = 0
        total_credits_used = 0
        total_consumption = 0  # For dependency calculation
        prosumer_count = 0
        
        for household in households:
            # Check if household is prosumer in THIS scenario
            is_scenario_prosumer = household.scenario_adoption.get(scenario, False)
            
            # Always include consumption for dependency calculation
            household_consumption = getattr(household, 'monthly_consumption', household.daily_consumption * 30)
            total_consumption += household_consumption
            
            if is_scenario_prosumer:
                # Prosumer: include generation, credits, and prosumer grid consumption
                total_generation += getattr(household, 'monthly_generation', 0)
                total_credits_earned += getattr(household, 'monthly_credits_earned', 0)
                total_credits_used += getattr(household, 'monthly_credits_used', 0)
                total_grid_consumption += getattr(household, 'monthly_grid_consumption', 0)
                prosumer_count += 1
            else:
                # Non-prosumer: only grid consumption (= total consumption)
                total_grid_consumption += household_consumption
        
        # Calculate derived metrics
        credit_utilization_rate = (total_credits_used / total_credits_earned 
                                if total_credits_earned > 0 else 0)
        
        fossil_dependency = (total_grid_consumption / total_consumption 
                            if total_consumption > 0 else 1.0)
        
        # Simplified grid stress calculation (can be enhanced later)
        grid_stress_index = self._calculate_scenario_grid_stress(households, scenario, total_grid_consumption)
        
        return {
            'total_generation': total_generation,
            'grid_consumption': total_grid_consumption,
            'total_credits_earned': total_credits_earned,
            'total_credits_used': total_credits_used,
            'credit_utilization_rate': credit_utilization_rate,
            'grid_stress_index': grid_stress_index,
            'fossil_dependency': fossil_dependency,
            'prosumer_count': prosumer_count
        }

    def _calculate_scenario_grid_stress(self, households, scenario, total_grid_consumption):
        """
        NEW: Calculate grid stress index for a specific scenario.
        Simplified version based on grid consumption patterns.
        
        Args:
            households: List of household agents
            scenario: Scenario name
            total_grid_consumption: Total grid consumption for this scenario
            
        Returns:
            float: Grid stress index for this scenario
        """
        if total_grid_consumption == 0:
            return 0.0
        
        # Simple stress calculation based on load concentration
        # In a more sophisticated version, this would consider peak loads, timing, etc.
        total_households = len(households)
        prosumer_count = sum(1 for h in households if h.scenario_adoption.get(scenario, False))
        
        if total_households == 0:
            return 0.0
        
        # Higher stress when more households depend on grid
        non_prosumer_ratio = (total_households - prosumer_count) / total_households
        
        # Normalized stress index (0-1, where 1 = maximum stress)
        stress_index = min(1.0, non_prosumer_ratio * (total_grid_consumption / (total_households * 100)))
        
        return stress_index

    def get_scenario_dataframe(self, scenario):
        """
        Get agent data filtered for a specific scenario.
        
        Args:
            scenario: Scenario name
            
        Returns:
            pd.DataFrame: Agent data for the scenario
        """
        if not self.agent_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.agent_data)
        
        # Create scenario-specific view
        scenario_df = df.copy()
        scenario_df['Scenario'] = scenario
        scenario_df['IsProsumer'] = df[f'{scenario}_Adopted']
        scenario_df['AdoptionMonth'] = df[f'{scenario}_AdoptionMonth']
        scenario_df['ScenarioNPV'] = df[f'{scenario}_NPV']
        scenario_df['AdoptionProbability'] = df[f'{scenario}_Probability']
        
        return scenario_df
    
    def get_combined_dataframe(self):
        """
        FIXED: Get combined agent data for all scenarios in long format.
        """
        if not self.agent_data:
            return pd.DataFrame()
        
        all_scenario_data = []
        
        for scenario in self.scenarios:
            scenario_df = self.get_scenario_dataframe(scenario)
            all_scenario_data.append(scenario_df)
        
        combined_df = pd.concat(all_scenario_data, ignore_index=True)
        return combined_df
    
    def get_system_metrics_dataframe(self):
        """
        FIXED: Get system metrics DataFrame with all required columns.
        """
        if not self.system_metrics:
            return pd.DataFrame()
        
        # Convert list of dicts to DataFrame, handling nested MonthlyCosts
        system_df = pd.DataFrame(self.system_metrics)
        
        # Expand MonthlyCosts if it exists
        if 'MonthlyCosts' in system_df.columns:
            cost_df = pd.json_normalize(system_df['MonthlyCosts'])
            cost_df.columns = [f'MonthlyCosts_{col}' for col in cost_df.columns]
            system_df = pd.concat([system_df.drop('MonthlyCosts', axis=1), cost_df], axis=1)
        
        return system_df
    
    def get_bias_effects_dataframe(self):
        """
        Get bias effects DataFrame.
        """
        if not self.bias_effects_data:
            return pd.DataFrame()
        
        return pd.DataFrame(self.bias_effects_data)
    
    def export_evaluation_statistics(self, output_dir):
        """Export evaluation trigger statistics."""
        if hasattr(self.model, 'evaluation_triggers'):
            eval_stats_path = os.path.join(output_dir, "evaluation_statistics.json")
            stats = self.model.evaluation_triggers.get_evaluation_statistics()
            
            import json
            with open(eval_stats_path, 'w') as f:
                json.dump(stats, f, indent=2)
            
            print(f"  Exported evaluation statistics: {stats['total_households']} households")
            
            # Export detailed evaluation log
            eval_log_path = os.path.join(output_dir, "evaluation_log.csv")
            self.model.evaluation_triggers.export_evaluation_log(eval_log_path)
    
    def export_all_scenarios(self, output_dir="results/multi_experiment"):
        """
        FIXED: Export comprehensive data for all scenarios.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Exporting multi-experiment data to {output_dir}...")
        
        # Export individual scenario files
        for scenario in self.scenarios:
            scenario_df = self.get_scenario_dataframe(scenario)
            if not scenario_df.empty:
                filepath = os.path.join(output_dir, f"{scenario}_data.csv")
                scenario_df.to_csv(filepath, index=False)
                print(f"  Exported {scenario}: {len(scenario_df)} records")
        
        # Export combined dataset
        combined_df = self.get_combined_dataframe()
        if not combined_df.empty:
            combined_path = os.path.join(output_dir, "combined_scenarios.csv")
            combined_df.to_csv(combined_path, index=False)
            print(f"  Exported combined data: {len(combined_df)} records")
        
        # Export system metrics
        system_df = self.get_system_metrics_dataframe()
        if not system_df.empty:
            system_path = os.path.join(output_dir, "system_metrics.csv")
            system_df.to_csv(system_path, index=False)
            print(f"  Exported system metrics: {len(system_df)} records")
        
        # Export bias effects
        bias_df = self.get_bias_effects_dataframe()
        if not bias_df.empty:
            bias_path = os.path.join(output_dir, "bias_effects.csv")
            bias_df.to_csv(bias_path, index=False)
            print(f"  Exported bias effects: {len(bias_df)} records")
        
        # Export scenario metadata
        self._export_scenario_metadata(output_dir)
        
        # Export configuration
        self._export_configuration(output_dir)
        
        print(f"Multi-experiment data export completed!")
    
    def _export_scenario_metadata(self, output_dir):
        """Export scenario metadata."""
        metadata = get_scenario_metadata()
        
        metadata_records = []
        for scenario, info in metadata.items():
            record = {
                'Scenario': scenario,
                'DisplayName': info['display_name'],
                'Description': info['description'],
                'LiteratureSource': info['literature_source'],
                'Formula': info.get('formula', '')
            }
            metadata_records.append(record)
        
        metadata_df = pd.DataFrame(metadata_records)
        metadata_path = os.path.join(output_dir, "scenario_metadata.csv")
        metadata_df.to_csv(metadata_path, index=False)
        print(f"  Exported scenario metadata: {len(metadata_records)} scenarios")
    
    def _export_configuration(self, output_dir):
        """Export simulation configuration."""
        try:
            config_path = os.path.join(output_dir, "simulation_config.json")
            
            import json
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=2, default=str)
            
            print(f"  Exported configuration to {config_path}")
            
        except Exception as e:
            print(f"  Warning: Could not export configuration: {e}")
    
    def get_adoption_summary(self):
        """
        Get adoption summary across all scenarios.
        
        Returns:
            dict: Adoption summary statistics
        """
        if not self.system_metrics:
            return {}
        
        # Get final timestep data
        final_data = self.system_metrics[-1]
        
        summary = {
            'final_year': final_data.get('Year', 0),
            'total_households': final_data.get('TotalHouseholds', 0),
            'scenario_adoption_rates': {},
            'scenario_adoption_counts': {}
        }
        
        # Extract final adoption rates
        for scenario in self.scenarios:
            rate_key = f'{scenario}_AdoptionRate'
            count_key = f'{scenario}_AdopterCount'
            
            if rate_key in final_data:
                summary['scenario_adoption_rates'][scenario] = final_data[rate_key]
                summary['scenario_adoption_counts'][scenario] = final_data.get(count_key, 0)
        
        return summary
    
    def get_data_summary(self):
        """
        Get summary of collected data for debugging.
        
        Returns:
            dict: Data collection summary
        """
        summary = {
            'scenarios_tracked': len(self.scenarios),
            'scenario_names': self.scenarios,
            'combined_records': len(self.agent_data),
            'system_metrics_records': len(self.system_metrics),
            'bias_effects_records': len(self.bias_effects_data)
        }
        
        return summary


# =============================================================================
# TESTING FUNCTIONS  
# =============================================================================

def test_multi_experiment_collector():
    """
    Test the MultiExperimentCollector class.
    
    Returns:
        bool: True if tests pass
    """
    print("Testing MultiExperimentCollector...")
    
    try:
        from ..utils.config_loader import create_testing_config
        from ..models.multi_experiment_model import MultiExperimentModel
        
        # Create test configuration
        config = create_testing_config(num_households=5, steps=2)
        config_dict = config.get_copy()  # FIXED: Get dictionary from SimulationConfig
        
        # Create model with collector
        model = MultiExperimentModel(config_dict)  # FIXED: Pass dictionary
        
        # Test collector initialization
        collector = model.data_collector
        if not isinstance(collector, MultiExperimentCollector):
            print(f"❌ Expected MultiExperimentCollector, got {type(collector)}")
            return False
        
        # Test data collection
        collector.collect_data()
        
        # Test data summary
        summary = collector.get_data_summary()
        expected_scenarios = get_all_scenarios()
        
        if summary['scenarios_tracked'] != len(expected_scenarios):
            print(f"❌ Expected {len(expected_scenarios)} scenarios, got {summary['scenarios_tracked']}")
            return False
        
        # Test scenario dataframes
        for scenario in expected_scenarios:
            df = collector.get_scenario_dataframe(scenario)
            if df.empty:
                print(f"❌ No data collected for scenario {scenario}")
                return False
        
        # Test combined dataframe
        combined_df = collector.get_combined_dataframe()
        if combined_df.empty:
            print(f"❌ No combined data collected")
            return False
        
        # Test system metrics
        system_df = collector.get_system_metrics_dataframe()
        if system_df.empty:
            print(f"❌ No system metrics collected")
            return False
        
        # Run model to collect more data
        model.run(steps=2)
        
        # Test adoption summary
        adoption_summary = collector.get_adoption_summary()
        if 'scenario_adoption_rates' not in adoption_summary:
            print(f"❌ Missing adoption rates in summary")
            return False
        
        print("✅ MultiExperimentCollector tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ MultiExperimentCollector test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_multi_experiment_collector()