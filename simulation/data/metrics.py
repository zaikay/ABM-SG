# data/metrics.py V4

"""
Metrics calculation for the prosumer simulation.
"""
import numpy as np
import pandas as pd

class MetricsAnalyzer:
    """
    Analyzes simulation metrics for reporting.
    """
    def __init__(self, model_data, agent_data):
        """
        Initialize the metrics analyzer.
        
        Args:
            model_data: DataFrame with model-level data
            agent_data: DataFrame with agent-level data
        """
        self.model_data = model_data
        self.agent_data = agent_data
    
    def get_adoption_curve(self):
        """
        Extract the adoption curve data.
        
        Returns:
            DataFrame: Adoption curve data
        """
        if "AdoptionRate" not in self.model_data.columns:
            return pd.DataFrame({"Year": [], "AdoptionRate": []})
        
        # Extract year and adoption rate
        adoption_data = self.model_data[["Year", "AdoptionRate"]].copy()
        
        # Group by year and take the last value for each year
        adoption_by_year = adoption_data.groupby("Year").last().reset_index()
        
        return adoption_by_year
    
    def get_fossil_dependency(self):
        """
        Extract fossil dependency data.
        
        Returns:
            DataFrame: Fossil dependency data
        """
        if "FossilDependency" not in self.model_data.columns:
            return pd.DataFrame({"Year": [], "FossilDependency": []})
        
        # Extract year and fossil dependency
        fossil_data = self.model_data[["Year", "FossilDependency"]].copy()
        
        # Group by year and take the last value for each year
        fossil_by_year = fossil_data.groupby("Year").last().reset_index()
        
        return fossil_by_year
    
    def get_energy_costs(self):
        """
        Extract energy cost data.
        
        Returns:
            DataFrame: Energy cost data
        """
        if "MonthlyCosts" not in self.model_data.columns:
            return pd.DataFrame({"Year": [], "MeanCost": [], "ProsumerCost": [], "NonProsumerCost": []})
        
        # Extract costs data
        cost_data = pd.DataFrame({
            "Year": self.model_data["Year"],
            "MeanCost": self.model_data["MonthlyCosts"].apply(lambda x: x["mean"]),
            "ProsumerCost": self.model_data["MonthlyCosts"].apply(lambda x: x["prosumer_mean"]),
            "NonProsumerCost": self.model_data["MonthlyCosts"].apply(lambda x: x["nonprosumer_mean"])
        })
        
        # Group by year and take the mean for each year
        cost_by_year = cost_data.groupby("Year").mean().reset_index()
        
        return cost_by_year
    
    def get_income_class_adoption(self):
        """
        Extract income class adoption data.
        
        Returns:
            DataFrame: Income class adoption data
        """
        if "IncomeClassAdoption" not in self.model_data.columns:
            return pd.DataFrame({"Year": [], "Class": [], "AdoptionRate": []})
        
        # Process income class adoption data
        years = []
        classes = []
        rates = []
        
        for idx, row in self.model_data.iterrows():
            year = row["Year"]
            adoption_dict = row["IncomeClassAdoption"]
            
            for income_class, rate in adoption_dict.items():
                years.append(year)
                classes.append(f"Class {income_class}")
                rates.append(rate)
        
        class_data = pd.DataFrame({
            "Year": years,
            "Class": classes,
            "AdoptionRate": rates
        })
        
        return class_data
        
    # ADD these methods to the existing MetricsAnalyzer class:

    def get_peak_load_evolution(self):
        """
        Extract peak load evolution data (exact calculations).
        
        Returns:
            DataFrame: Peak load evolution over time
        """
        if "MonthlyPeakLoad" not in self.model_data.columns:
            return pd.DataFrame({"Year": [], "Month": [], "PeakLoad": []})
        
        peak_data = self.model_data[["Year", "Month", "MonthlyPeakLoad"]].copy()
        peak_data.columns = ["Year", "Month", "PeakLoad"]
        
        # Add seasonal indicators
        peak_data["Season"] = peak_data["Month"].apply(self._get_season)
        
        return peak_data

    def get_seasonal_stress_patterns(self):
        """
        Extract seasonal grid stress patterns.
        
        Returns:
            DataFrame: Grid stress patterns by season and day type
        """
        if "GridStressIndex" not in self.model_data.columns:
            return pd.DataFrame({"Year": [], "Month": [], "Season": [], "StressIndex": []})
        
        stress_data = self.model_data[["Year", "Month", "CurrentMonthInYear", "GridStressIndex"]].copy()
        
        # Add seasonal grouping
        stress_data["Season"] = stress_data["CurrentMonthInYear"].apply(self._get_season)
        
        # Group by season and calculate statistics
        seasonal_stats = stress_data.groupby(["Year", "Season"])["GridStressIndex"].agg([
            'mean', 'max', 'min', 'std'
        ]).reset_index()
        
        seasonal_stats.columns = ["Year", "Season", "MeanStress", "MaxStress", "MinStress", "StdStress"]
        
        return {
            "monthly_data": stress_data[["Year", "Month", "Season", "GridStressIndex"]],
            "seasonal_stats": seasonal_stats
        }

    def get_npv_statistics_by_year(self):
        """
        Extract NPV distribution evolution over time.
        
        Returns:
            DataFrame: NPV distribution statistics by year
        """
        if self.agent_data.empty or "NPV" not in self.agent_data.columns:
            return pd.DataFrame({"Year": [], "MeanNPV": [], "MedianNPV": [], 
                            "PositiveNPVRate": [], "Q1": [], "Q3": []})
        
        # Filter household data only and remove null NPVs
        household_data = self.agent_data[
            (self.agent_data['AgentType'] == 'Household') & 
            (self.agent_data['NPV'].notna())
        ].copy()
        
        if household_data.empty:
            return pd.DataFrame({"Year": [], "MeanNPV": [], "MedianNPV": [], 
                            "PositiveNPVRate": [], "Q1": [], "Q3": []})
        
        # Group by year and calculate distribution statistics
        npv_stats = household_data.groupby('Year')['NPV'].agg([
            'count', 'mean', 'median', 'std',
            lambda x: np.percentile(x, 25),  # Q1
            lambda x: np.percentile(x, 75),  # Q3
            lambda x: (x > 0).sum(),  # Positive NPV count
        ]).reset_index()
        
        npv_stats.columns = ["Year", "Count", "MeanNPV", "MedianNPV", "StdNPV", "Q1", "Q3", "PositiveCount"]
        npv_stats["PositiveNPVRate"] = npv_stats["PositiveCount"] / npv_stats["Count"]
        
        return npv_stats

    def get_payback_period_trends(self):
        """
        Extract payback period trends by income class.
        
        Returns:
            DataFrame: Payback period trends over time and by income class
        """
        if self.agent_data.empty or "PaybackPeriod" not in self.agent_data.columns:
            return pd.DataFrame({"Year": [], "IncomeClass": [], "MeanPayback": [], 
                            "MedianPayback": [], "AdoptionRate": []})
        
        # Filter household data and finite payback periods
        household_data = self.agent_data[
            (self.agent_data['AgentType'] == 'Household') & 
            (self.agent_data['PaybackPeriod'] != float('inf')) &
            (self.agent_data['PaybackPeriod'].notna()) &
            (self.agent_data['IncomeClass'].notna())
        ].copy()
        
        if household_data.empty:
            return pd.DataFrame({"Year": [], "IncomeClass": [], "MeanPayback": [], 
                            "MedianPayback": [], "AdoptionRate": []})
        
        # Group by year and income class
        payback_stats = household_data.groupby(['Year', 'IncomeClass']).agg({
            'PaybackPeriod': ['count', 'mean', 'median'],
            'IsProsumer': 'mean'  # Adoption rate
        }).reset_index()
        
        # Flatten column names
        payback_stats.columns = ['Year', 'IncomeClass', 'Count', 'MeanPayback', 'MedianPayback', 'AdoptionRate']
        
        return payback_stats

    def _get_season(self, month):
        """
        Convert month number to season name.
        
        Args:
            month: Month number (1-12)
            
        Returns:
            str: Season name
        """
        if month in [12, 1, 2]:
            return "Winter"
        elif month in [3, 4, 5]:
            return "Spring"
        elif month in [6, 7, 8]:
            return "Summer"
        else:  # [9, 10, 11]
            return "Fall"
    
    def get_key_statistics(self):
        """
        Calculate key statistics from the simulation.
        
        Returns:
            dict: Key statistics
        """
        # Initialize statistics dictionary
        stats = {}
        
        # Final adoption rate
        if "AdoptionRate" in self.model_data.columns:
            stats["final_adoption_rate"] = self.model_data["AdoptionRate"].iloc[-1]
        else:
            stats["final_adoption_rate"] = 0
        
        # Final fossil dependency
        if "FossilDependency" in self.model_data.columns:
            stats["final_fossil_dependency"] = self.model_data["FossilDependency"].iloc[-1]
        else:
            stats["final_fossil_dependency"] = 1.0
        
        # Adoption by income class (final state)
        if "IncomeClassAdoption" in self.model_data.columns:
            class_adoption = self.model_data["IncomeClassAdoption"].iloc[-1]
            stats["class_adoption"] = class_adoption
            
            # Calculate adoption inequality (ratio of highest to lowest class)
            if 1 in class_adoption and 5 in class_adoption and class_adoption[1] > 0:
                stats["adoption_inequality"] = class_adoption[5] / class_adoption[1]
            else:
                stats["adoption_inequality"] = 0
        else:
            stats["class_adoption"] = {}
            stats["adoption_inequality"] = 0
        
        # Cost differential (prosumer vs non-prosumer)
        if "MonthlyCosts" in self.model_data.columns:
            final_costs = self.model_data["MonthlyCosts"].iloc[-1]
            
            if final_costs["nonprosumer_mean"] > 0:
                stats["cost_reduction"] = 1 - (final_costs["prosumer_mean"] / final_costs["nonprosumer_mean"])
            else:
                stats["cost_reduction"] = 0
        else:
            stats["cost_reduction"] = 0
        
        return stats
    
    def get_credit_utilization_evolution(self):
        """
        Extract credit system utilization evolution.
        
        Returns:
            DataFrame: Credit utilization over time
        """
        required_cols = ["rational_TotalCreditsEarned", "rational_TotalCreditsUsed", "rational_CreditUtilizationRate"]
        if not all(col in self.model_data.columns for col in required_cols):
            print(f"⚠️ get_credit_utilization_evolution: Missing required columns {required_cols}")
            return pd.DataFrame({"Year": [], "Month": [], "CreditsEarned": [], 
                            "CreditsUsed": [], "UtilizationRate": []})
        
        credit_data = self.model_data[["Year", "Month"] + required_cols].copy()
        credit_data.columns = ["Year", "Month", "CreditsEarned", "CreditsUsed", "UtilizationRate"]

        rename_mapping = {
            'rational_TotalCreditsEarned': 'TotalCreditsEarned',
            'rational_TotalCreditsUsed': 'TotalCreditsUsed',
            'rational_CreditUtilizationRate': 'CreditUtilizationRate'
        }
        credit_data = credit_data.rename(columns=rename_mapping)

        # Add derived metrics
        credit_data["CreditsExpired"] = credit_data["CreditsEarned"] - credit_data["CreditsUsed"]
        credit_data["WasteRate"] = credit_data["CreditsExpired"] / credit_data["CreditsEarned"]
        credit_data["WasteRate"] = credit_data["WasteRate"].fillna(0)
        
        return credit_data

    def get_seasonal_stress_patterns(self):
        """
        Extract seasonal grid stress patterns.
        
        Returns:
            DataFrame: Seasonal stress patterns over time
        """
        required_cols = ["rational_GridStressIndex", "Year", "Month"]
        if not all(col in self.model_data.columns for col in required_cols):
            print(f"⚠️ get_seasonal_stress_patterns: Missing required columns {required_cols}")
            return pd.DataFrame({"Year": [], "Month": [], "Season": [], "StressIndex": []})
        
        stress_data = self.model_data[required_cols].copy()
        stress_data.columns = ["StressIndex", "Year", "Month"]

        rename_mapping = {
            'rational_GridStressIndex': 'GridStressIndex'
        }
        stress_data = stress_data.rename(columns=rename_mapping)
        # Add seasonal indicators
        stress_data["Season"] = stress_data["Month"].apply(self._get_season)
        
        return stress_data

    def get_npv_raw_data(self):
        """
        Extract NPV distribution evolution over time.
        
        Returns:
            DataFrame: NPV distribution over time
        """
        if self.agent_data.empty or "NPV" not in self.agent_data.columns:
            print("⚠️ get_npv_distribution_evolution: No NPV data in agent data")
            return pd.DataFrame({"Year": [], "NPV": [], "IncomeClass": [], "IsProsumer": []})
        
        # Filter household data with valid NPV
        household_data = self.agent_data[
            (self.agent_data['AgentType'] == 'Household') & 
            (self.agent_data['NPV'].notna())
        ].copy()
        
        if household_data.empty:
            print("⚠️ get_npv_distribution_evolution: No valid household NPV data")
            return pd.DataFrame({"Year": [], "NPV": [], "IncomeClass": [], "IsProsumer": []})
        
        # Select relevant columns
        npv_cols = ['Year', 'NPV']
        if 'IncomeClass' in household_data.columns:
            npv_cols.append('IncomeClass')
        if 'IsProsumer' in household_data.columns:
            npv_cols.append('IsProsumer')
        
        npv_data = household_data[npv_cols].copy()
        
        return npv_data
    
    def get_npv_by_income_class(self, year=None):
        """NPV stats grouped by income class and year"""
        raw_data = self.get_npv_raw_data()
        return raw_data.groupby(['Year', 'IncomeClass'])['NPV'].agg(['mean', 'count'])
    
    def get_positive_npv_households(self):
        """Count of households with positive NPV by year"""
        raw_data = self.get_npv_raw_data()
        return raw_data[raw_data['NPV'] > 0].groupby('Year').size()

    def get_energy_cost_evolution(self):
        """Energy cost evolution with NPV positive households"""
        # Check for price columns
        price_cols = [col for col in self.model_data.columns if 'Price' in col]
        if not price_cols:
            return pd.DataFrame()
            
        cost_data = self.model_data[['Year']].copy()
        
        # Add available price columns
        for col in price_cols:
            cost_data[col] = self.model_data[col]
        
        # Add NPV positive household percentage 
        if not self.agent_data.empty and 'NPV' in self.agent_data.columns:
            yearly_npv = self.agent_data.groupby('Year').agg({
                'NPV': lambda x: (x > 0).mean() * 100
            }).rename(columns={'NPV': 'NPVPositivePercent'})
            cost_data = cost_data.merge(yearly_npv, on='Year', how='left')
        
        return cost_data.drop_duplicates('Year')


    def get_energy_mix_evolution(self, scenario=None):
        """Energy mix data for stacked area charts using REAL energy data - YEARLY AGGREGATED"""
        # Get real energy data columns including temporal info
        energy_cols = ['Year', 'Month', 'TotalConsumption', 'TotalGeneration', 'GridConsumption', 
                    'TotalCreditsEarned', 'TotalCreditsUsed']
        
        if scenario:
            # Look for scenario-specific columns first
            scenario_cols = []
            for col in energy_cols:
                scenario_col = f'{scenario}_{col}'
                if scenario_col in self.model_data.columns:
                    scenario_cols.append(scenario_col)
                elif col in self.model_data.columns:
                    scenario_cols.append(col)
        else:
            scenario_cols = [col for col in energy_cols if col in self.model_data.columns]
        
        # Get the data
        available_cols = [col for col in scenario_cols if col in self.model_data.columns]
        if len(available_cols) < 4:  # Need at least Year + Month + 2 energy columns
            print(f"Warning: Insufficient energy data columns. Available: {available_cols}")
            return pd.DataFrame()
            
        mix_data = self.model_data[available_cols].copy()
        
        # Standardize column names (remove scenario prefix if present)
        if scenario:
            mix_data.columns = [col.replace(f'{scenario}_', '') for col in mix_data.columns]
        
        # Show raw data structure for debugging
        print(f"Available initial columns: {list(self.model_data.columns)}")
        print(f"Raw monthly data shape: {mix_data.shape}")
        print(f"Available columns: {list(mix_data.columns)}")
        if 'Year' in mix_data.columns:
            print(f"Year range: {mix_data['Year'].min()}-{mix_data['Year'].max()}")
            year_counts = mix_data['Year'].value_counts().sort_index()
            print(f"Records per year: {year_counts.iloc[0] if len(year_counts) > 0 else 0}")
            print(f"First few years: {year_counts.head(3).to_dict()}")
        
        # YEARLY AGGREGATION - Sum monthly values to get annual totals
        if 'Year' not in mix_data.columns:
            print("Warning: Year column missing for aggregation")
            return pd.DataFrame()
        
        # Define columns to aggregate by summation (energy values)
        energy_columns = ['TotalConsumption', 'TotalGeneration', 'GridConsumption', 
                        'TotalCreditsEarned', 'TotalCreditsUsed']
        
        agg_dict = {}
        for col in energy_columns:
            if col in mix_data.columns:
                agg_dict[col] = 'sum'  # Sum monthly values to get annual totals
        
        if not agg_dict:
            print("Warning: No energy columns found for aggregation")
            return pd.DataFrame()
        
        # Perform yearly aggregation
        yearly_mix = mix_data.groupby('Year').agg(agg_dict).reset_index()
        
        print(f"Yearly aggregated shape: {yearly_mix.shape}")
        print(f"Sample yearly totals (first 3 years):")
        if len(yearly_mix) >= 3:
            for i in range(3):
                year = yearly_mix.iloc[i]['Year']
                consumption = yearly_mix.iloc[i]['TotalConsumption']
                generation = yearly_mix.iloc[i].get('TotalGeneration', 0)
                credits = yearly_mix.iloc[i].get('TotalCreditsUsed', 0)
                print(f"  Year {year}: Total={consumption:.0f}, Generation={generation:.0f}, Credits={credits:.0f}")
        
        # Calculate actual energy mix components using REAL aggregated data
        if all(col in yearly_mix.columns for col in ['TotalConsumption', 'GridConsumption', 'TotalCreditsUsed']):
            
            # REAL CALCULATIONS using yearly totals:
            # 1. Grid Consumption (pure grid power)
            yearly_mix['GridConsumption'] = yearly_mix['GridConsumption']
            
            # 2. Credit Consumption (credits used for demand)
            yearly_mix['CreditConsumption'] = yearly_mix['TotalCreditsUsed']
            
            # 3. Direct Solar Consumption (solar used directly when generated)
            yearly_mix['DirectSolarConsumption'] = (yearly_mix['TotalConsumption'] - 
                                                yearly_mix['GridConsumption'] - 
                                                yearly_mix['CreditConsumption']).clip(lower=0)
            
            # 4. Cumulative layers for stacked area chart
            yearly_mix['Layer1_DirectSolar'] = yearly_mix['DirectSolarConsumption']
            yearly_mix['Layer2_DirectPlusCredit'] = (yearly_mix['DirectSolarConsumption'] + 
                                                    yearly_mix['CreditConsumption'])
            yearly_mix['Layer3_Total'] = yearly_mix['TotalConsumption']
            
            print(f"Energy mix layers calculated:")
            print(f"  Direct Solar: {yearly_mix['DirectSolarConsumption'].sum():.0f} total")
            print(f"  Credit Use: {yearly_mix['CreditConsumption'].sum():.0f} total")
            print(f"  Grid Use: {yearly_mix['GridConsumption'].sum():.0f} total")
            
        else:
            print(f"Warning: Missing required columns for real energy mix calculation")
            print(f"Available: {list(yearly_mix.columns)}")
            print(f"Required: TotalConsumption, GridConsumption, TotalCreditsUsed")
            return pd.DataFrame()
                
        return yearly_mix

    def get_income_class_system_metrics(self, scenario=None):
        """System metrics stratified by income class and scenario"""
        if self.agent_data.empty or 'IncomeClass' not in self.agent_data.columns:
            return pd.DataFrame()
        
        agent_data = self.agent_data.copy()
        
        # Determine which columns to use based on scenario
        if scenario:
            adopted_col = f'{scenario}_Adopted' if f'{scenario}_Adopted' in agent_data.columns else None
            npv_col = f'{scenario}_NPV' if f'{scenario}_NPV' in agent_data.columns else 'NPV'
        else:
            # When scenario=None, default to 'rational' as baseline for consistency
            adopted_col = 'IsProsumer' if 'IsProsumer' in agent_data.columns else 'rational_Adopted'
            npv_col = 'NPV'
            print("ℹ️  Using rational scenario as baseline for scenario-independent income class metrics")
        
        if not adopted_col or adopted_col not in agent_data.columns:
            print(f"Warning: Column {adopted_col} not found for scenario {scenario}")
            print(f"Available columns with 'Adopted': {[col for col in agent_data.columns if 'Adopted' in col]}")
            return pd.DataFrame()
        
        # Group by year and income class
        group_cols = ['Year', 'IncomeClass']
        agg_dict = {npv_col: 'mean', adopted_col: 'mean'}
        
        # Add consumption columns if available
        if 'DailyConsumption' in agent_data.columns:
            agg_dict['DailyConsumption'] = 'sum'
        
        class_metrics = agent_data.groupby(group_cols).agg(agg_dict).reset_index()
        
        # Rename columns for consistency
        class_metrics.columns = ['Year', 'IncomeClass', 'MeanNPV', 'AdoptionRate'] + \
                            list(class_metrics.columns[4:])
        
        return class_metrics

    def get_adoption_velocity_metrics(self):
        """Calculate adoption velocity and timing metrics"""
        from ..utils.parameters import get_all_scenarios
        
        velocity_data = []
        scenarios = get_all_scenarios()
        
        for scenario in scenarios:
            rate_col = f'{scenario}_AdoptionRate'
            if rate_col not in self.model_data.columns:
                continue
                
            adoption_curve = self.model_data[rate_col].values
            
            if len(adoption_curve) <= 1:
                continue
            
            # Calculate velocity (first derivative) - monthly to annual
            velocity = np.diff(adoption_curve) * 12
            
            # Find key metrics
            peak_velocity = velocity.max() if len(velocity) > 0 else 0
            peak_velocity_year = np.argmax(velocity) / 12 + 1 if len(velocity) > 0 else None
            
            # Find 50% threshold year
            threshold_50_idx = np.where(adoption_curve >= 0.5)[0]
            threshold_50_year = (threshold_50_idx[0] / 12 + 1) if len(threshold_50_idx) > 0 else None
            
            velocity_data.append({
                'Scenario': scenario,
                'PeakVelocity': peak_velocity,
                'PeakVelocityYear': peak_velocity_year,
                'Threshold50Year': threshold_50_year,
                'VelocityCurve': velocity,
                'AdoptionCurve': adoption_curve
            })
        
        return pd.DataFrame(velocity_data)

    def export_all_system_metrics(self, output_dir="data"):
        """Export all system-level metrics to CSV files following project structure"""
        import os
        from ..utils.parameters import get_all_scenarios
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Get available scenarios from data
        available_scenarios = []
        for col in self.model_data.columns:
            if col.endswith('_AdoptionRate'):
                scenario = col.replace('_AdoptionRate', '')
                available_scenarios.append(scenario)
        
        # Find main scenarios to export (rational + combined scenario)
        main_scenarios = ['rational'] if 'rational' in available_scenarios else []
        combined_scenarios = [s for s in available_scenarios if 'all' in s.lower() or 'combined' in s.lower()]
        if combined_scenarios:
            main_scenarios.extend(combined_scenarios[:1])  # Add first combined scenario
        
        # Export each metric dataset
        try:
            cost_data = self.get_energy_cost_evolution()
            if not cost_data.empty:
                cost_data.to_csv(f"{output_dir}/energy_cost_evolution.csv", index=False)
                print(f"✅ Exported energy_cost_evolution.csv ({len(cost_data)} records)")
        except Exception as e:
            print(f"⚠️ Could not export energy_cost_evolution.csv: {e}")
        
        try:
            velocity_data = self.get_adoption_velocity_metrics()
            if not velocity_data.empty:
                # Export summary metrics only (not numpy arrays)
                summary_data = velocity_data[['Scenario', 'PeakVelocity', 'PeakVelocityYear', 'Threshold50Year']].copy()
                summary_data.to_csv(f"{output_dir}/adoption_velocity_metrics.csv", index=False)
                print(f"✅ Exported adoption_velocity_metrics.csv ({len(summary_data)} scenarios)")
        except Exception as e:
            print(f"⚠️ Could not export adoption_velocity_metrics.csv: {e}")
        
        # Export income class metrics for main scenarios
        for scenario in main_scenarios:
            try:
                income_data = self.get_income_class_system_metrics(scenario)
                if not income_data.empty:
                    income_data.to_csv(f"{output_dir}/income_class_metrics_{scenario}.csv", index=False)
                    print(f"✅ Exported income_class_metrics_{scenario}.csv ({len(income_data)} records)")
            except Exception as e:
                print(f"⚠️ Could not export income_class_metrics_{scenario}.csv: {e}")
        
        # Export energy mix data
        for scenario in main_scenarios:
            try:
                mix_data = self.get_energy_mix_evolution(scenario)
                if not mix_data.empty:
                    mix_data.to_csv(f"{output_dir}/energy_mix_evolution_{scenario}.csv", index=False)
                    print(f"✅ Exported energy_mix_evolution_{scenario}.csv ({len(mix_data)} records)")
            except Exception as e:
                print(f"⚠️ Could not export energy_mix_evolution_{scenario}.csv: {e}")
        
        print(f"📊 System metrics exported to {output_dir}/")

    def get_payback_period_trends(self):
        """
        Extract payback period trends over time and by income class.
        
        Returns:
            DataFrame: Payback period trends over time and by income class
        """
        if self.agent_data.empty or "PaybackPeriod" not in self.agent_data.columns:
            print("⚠️ get_payback_period_trends: No PaybackPeriod data in agent data")
            return pd.DataFrame({"Year": [], "IncomeClass": [], "MeanPayback": [], 
                            "MedianPayback": [], "AdoptionRate": []})
        
        # Filter household data and finite payback periods
        household_data = self.agent_data[
            (self.agent_data['AgentType'] == 'Household') & 
            (self.agent_data['PaybackPeriod'] != float('inf')) &
            (self.agent_data['PaybackPeriod'].notna()) &
            (self.agent_data['IncomeClass'].notna())
        ].copy()
        
        if household_data.empty:
            print("⚠️ get_payback_period_trends: No valid payback period data")
            return pd.DataFrame({"Year": [], "IncomeClass": [], "MeanPayback": [], 
                            "MedianPayback": [], "AdoptionRate": []})
        
        # Group by year and income class
        payback_stats = household_data.groupby(['Year', 'IncomeClass']).agg({
            'PaybackPeriod': ['count', 'mean', 'median'],
            'IsProsumer': 'mean'  # Adoption rate
        }).reset_index()
        
        # Flatten column names
        payback_stats.columns = ['Year', 'IncomeClass', 'Count', 'MeanPayback', 'MedianPayback', 'AdoptionRate']
        
        return payback_stats