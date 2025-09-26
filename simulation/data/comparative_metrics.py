# data/comparative_metrics.py
"""
Pure metrics calculation and CSV export layer.
Clean separation from visualization - only calculates and exports data.
"""

import pandas as pd
import numpy as np
import os
from ..utils.parameters import get_all_scenarios, get_scenario_metadata

class ComparativeMetrics:
    """
    Pure metrics calculation and export class.
    Calculates adoption metrics and exports to standardized CSV files.
    """
    
    def __init__(self, model_data, agent_data):
        """Initialize with simulation data."""
        self.model_data = model_data
        self.agent_data = agent_data
        self.scenarios = get_all_scenarios()
        self.metadata = get_scenario_metadata()
        
        # Validate data and extract base datasets
        self._validate_and_prepare_data()
        
        # Pre-extract system metrics for visualization layer
        self.system_metrics = self._extract_system_metrics()
    
    def _validate_and_prepare_data(self):
        """Validate data and prepare base datasets."""
        self.adoption_data = self._extract_adoption_time_series()
        self.income_data = self._extract_income_class_time_series()
        
        if self.adoption_data.empty:
            raise ValueError("No adoption time series data found")
        
        print(f"ComparativeMetrics initialized with {len(self.adoption_data)} time points")
        print(f"Available scenarios: {list(self.adoption_data.columns)}")
    
    def _extract_adoption_time_series(self):
        """Extract adoption time series for all scenarios."""
        if self.model_data.empty:
            return pd.DataFrame()
        
        adoption_data = pd.DataFrame()
        
        for scenario in self.scenarios:
            rate_col = f'{scenario}_AdoptionRate'
            if rate_col in self.model_data.columns:
                adoption_data[scenario] = self.model_data[rate_col]
        
        if not adoption_data.empty and 'Year' in self.model_data.columns:
            adoption_data.index = self.model_data['Year']
            adoption_data = adoption_data.groupby(adoption_data.index).last()
        
        return adoption_data
    
    def _extract_income_class_time_series(self):
        """Extract income class time series data."""
        if self.model_data.empty:
            return pd.DataFrame()
        
        class_cols = [col for col in self.model_data.columns 
                     if 'Class' in col and 'Rate' in col]
        
        if not class_cols:
            return pd.DataFrame()
        
        class_data = self.model_data[['Year'] + class_cols].copy()
        if 'Year' in class_data.columns:
            class_data = class_data.groupby('Year')[class_cols].last()
        
        return class_data
    
    def calculate_adoption_time_series(self):
        """
        Calculate master adoption time series using raw step-level data as foundation.
        This should be 240 rows (TOTAL_STEPS) showing monthly progression.
        
        Returns:
            pd.DataFrame: Step-level time series (240 rows for 20 years)
        """
        if self.model_data.empty:
            return pd.DataFrame()
        
        # Import population parameter correctly
        from ..utils.parameters import NUM_HOUSEHOLDS, TOTAL_STEPS
        population = NUM_HOUSEHOLDS
        
        print(f"Creating step-level time series with {population} households over {TOTAL_STEPS} steps")
        
        # Use raw step-level data (no aggregation)
        step_data = self.model_data.copy()
        
        # Verify we have the expected number of steps
        expected_steps = TOTAL_STEPS  # Should be 240
        actual_steps = len(step_data)
        print(f"Expected steps: {expected_steps}, Actual steps: {actual_steps}")
        
        # Create time series indexed by step (0-239)
        time_series = pd.DataFrame()
        time_series.index = step_data.index
        time_series.index.name = 'Step'
        
        # Add time conversion columns
        time_series['Year'] = (step_data.index // 12) + 1  # 1-based years (steps 0-11 = year 1)
        time_series['Month'] = (step_data.index % 12) + 1  # 1-based months
        time_series['DecimalYear'] = (step_data.index / 12.0) + 1  # 1.0, 1.08, 1.17, etc.
        
        # Add adoption data for each scenario
        for scenario in self.scenarios:
            rate_col = f'{scenario}_AdoptionRate'
            if rate_col in step_data.columns:
                # Adoption rates (as percentages) - raw step data
                adoption_rates = step_data[rate_col] * 100
                time_series[f'{scenario}_adoption_rate_pct'] = adoption_rates
                
                # Total adopters using correct population
                total_adopters = (step_data[rate_col] * population).round().astype(int)
                time_series[f'{scenario}_total_adopters'] = total_adopters
                
                # Debug early adoption issue
                early_steps = step_data.index < 13  # Steps 0-12
                early_adopters = total_adopters[early_steps]
                if early_adopters.sum() > 0:
                    print(f"Warning: {scenario} has {early_adopters.sum()} early adopters in steps 0-12")
        
        # Reset index to have Step as a column
        time_series = time_series.reset_index()
        
        print(f"Time series shape: {time_series.shape} (should be {TOTAL_STEPS} rows)")
        
        return time_series
    
    def calculate_critical_mass_timing(self, thresholds=[10, 30, 50, 90]):
        """
        Calculate when each scenario reaches critical mass thresholds using step-level precision.
        
        Args:
            thresholds: List of percentage thresholds to analyze
            
        Returns:
            pd.DataFrame: Critical mass timing analysis with step precision
        """
        results = []
        
        if self.model_data.empty:
            print("Warning: No model data available for timing analysis")
            return pd.DataFrame()
        
        # Work with step-level data for precision (before yearly aggregation)
        step_data = self.model_data.copy()
        
        # Extract rational baseline at step level
        rational_col = None
        
        for scenario in self.scenarios:
            rate_col = f'{scenario}_AdoptionRate'
            if rate_col in step_data.columns and 'rational' in scenario:
                rational_col = rate_col
                break
        print(f'cols {step_data.columns}')
        if not rational_col:
            print("Warning: No rational baseline found for timing differences")
            return pd.DataFrame()
        
        rational_data = step_data[rational_col] * 100  # Convert to percentage
        
                        # Calculate timing for each scenario
        for scenario in self.scenarios:
            rate_col = f'{scenario}_AdoptionRate'
            if rate_col not in step_data.columns:
                continue
                
            scenario_data = step_data[rate_col] * 100  # Convert to percentage
            
            for threshold in thresholds:
                # Find step when threshold is reached
                threshold_indices = np.where(scenario_data >= threshold)[0]
                
                if len(threshold_indices) > 0:
                    threshold_step = int(threshold_indices[0])
                    threshold_value = round(float(scenario_data.iloc[threshold_step]), 1)
                    reaches_threshold = True
                    
                    # Convert step to time
                    time_detail = self._step_to_time_detailed(threshold_step)
                    threshold_year = time_detail['decimal_years']
                else:
                    threshold_step = None
                    threshold_year = None
                    threshold_value = None
                    reaches_threshold = False
                    time_detail = None
                
                # Calculate rational baseline timing at same threshold
                rational_threshold_indices = np.where(rational_data >= threshold)[0]
                if len(rational_threshold_indices) > 0:
                    rational_threshold_step = int(rational_threshold_indices[0])
                    rational_time_detail = self._step_to_time_detailed(rational_threshold_step)
                    rational_threshold_year = rational_time_detail['decimal_years']
                else:
                    rational_threshold_step = None
                    rational_threshold_year = None
                    rational_time_detail = None
                
                # Calculate time difference in steps, then convert
                if threshold_step is not None and rational_threshold_step is not None:
                    step_difference = threshold_step - rational_threshold_step
                    time_difference_years = round(step_difference / 12.0, 1)
                    time_difference_months = step_difference
                else:
                    step_difference = None
                    time_difference_years = None
                    time_difference_months = None
                
                # Build result entry with detailed timing info
                result = {
                    'scenario': scenario,
                    'threshold_pct': threshold,
                    'threshold_step': threshold_step,
                    'threshold_year': threshold_year,
                    'threshold_value': threshold_value,
                    'rational_threshold_step': rational_threshold_step,
                    'rational_threshold_year': rational_threshold_year,
                    'step_difference': step_difference,
                    'time_difference_years': time_difference_years,
                    'time_difference_months': time_difference_months,
                    'reaches_threshold': reaches_threshold
                }
                
                # Add detailed time breakdowns if available
                if time_detail:
                    result.update({
                        'threshold_years_whole': time_detail['years'],
                        'threshold_months_remainder': time_detail['months']
                    })
                
                if rational_time_detail:
                    result.update({
                        'rational_years_whole': rational_time_detail['years'],
                        'rational_months_remainder': rational_time_detail['months']
                    })
                
                results.append(result)
        
        return pd.DataFrame(results)
    
    def _step_to_time_detailed(self, step):
        """
        Convert step number to detailed time format.
        
        Args:
            step: Step number (0-based)
            
        Returns:
            dict: Detailed time breakdown
        """
        if step is None:
            return None
            
        years = step // 12
        months = step % 12
        
        return {
            'step': step,
            'total_months': step,
            'years': years,
            'months': months,
            'decimal_years': round(step / 12.0, 1)
        }
    
    def calculate_area_analysis(self):
        """
        Calculate area between curves analysis using step-level precision.
        
        Returns:
            pd.DataFrame: Area analysis with different interpretations
        """
        # Work with step-level data for precision
        if self.model_data.empty:
            return pd.DataFrame()
        
        # Find rational baseline column
        rational_col = None
        for scenario in self.scenarios:
            rate_col = f'{scenario}_AdoptionRate'
            if rate_col in self.model_data.columns and 'rational' in scenario and 'deterministic' not in scenario:
                rational_col = rate_col
                break
        
        if not rational_col:
            print("Warning: No rational baseline found for area analysis")
            return pd.DataFrame()
        
        baseline = self.model_data[rational_col].values
        results = []
        
        # Get simulation parameters
        total_steps = len(self.model_data)
        simulation_years = total_steps / 12  # Monthly steps to years
        population_size = 1000  # From parameters
        
        for scenario in self.scenarios:
            if scenario == 'rational':
                continue
                
            rate_col = f'{scenario}_AdoptionRate'
            if rate_col not in self.model_data.columns:
                continue
                
            scenario_curve = self.model_data[rate_col].values
            
            # Calculate raw area difference using step-level data
            # dx = 1/12 for monthly resolution (12 months per year)
            cumulative_adoption_gap = np.trapz(scenario_curve - baseline, dx=1/12)
            
            # Multiple interpretations of the area difference
            average_annual_difference = cumulative_adoption_gap / simulation_years
            total_additional_adoptions = cumulative_adoption_gap * population_size
            
            # Final adoption difference
            final_adoption_difference_pct = (scenario_curve[-1] - baseline[-1]) * 100
            
            results.append({
                'scenario': scenario,
                'cumulative_adoption_gap': cumulative_adoption_gap,
                'average_annual_difference': average_annual_difference,
                'total_additional_adoptions': total_additional_adoptions,
                'final_adoption_difference_pct': final_adoption_difference_pct,
                'simulation_years': simulation_years,
                'total_steps': total_steps
            })
        
        return pd.DataFrame(results)
    
    def calculate_scenario_comparison_summary(self):
        """
        Calculate statistical summary of scenario comparisons.
        
        Returns:
            pd.DataFrame: Statistical summary of all scenarios
        """
        if 'rational' not in self.adoption_data.columns:
            print("Warning: No rational baseline found for comparison summary")
            return pd.DataFrame()
        
        baseline = self.adoption_data['rational'] * 100  # Convert to percentage
        results = []
        
        for scenario in self.adoption_data.columns:
            scenario_data = self.adoption_data[scenario] * 100
            
            if scenario == 'rational':
                # Baseline statistics
                results.append({
                    'scenario': scenario,
                    'final_adoption_rate_pct': scenario_data.iloc[-1],
                    'avg_difference_vs_rational': 0.0,
                    'min_difference_vs_rational': 0.0,
                    'max_difference_vs_rational': 0.0,
                    'std_difference_vs_rational': 0.0
                })
            else:
                # Calculate differences
                differences = scenario_data - baseline
                
                results.append({
                    'scenario': scenario,
                    'final_adoption_rate_pct': scenario_data.iloc[-1],
                    'avg_difference_vs_rational': differences.mean(),
                    'min_difference_vs_rational': differences.min(),
                    'max_difference_vs_rational': differences.max(),
                    'std_difference_vs_rational': differences.std()
                })
        
        return pd.DataFrame(results)
    
    def calculate_adoption_snapshots(self, time_fractions=[0.1, 0.3, 0.5, 0.9]):
        """
        Calculate adoption rates at specific time points.
        
        Args:
            time_fractions: List of time fractions (0.1 = 10% of simulation time)
            
        Returns:
            pd.DataFrame: Adoption snapshots table
        """
        if self.adoption_data.empty:
            return pd.DataFrame()
        
        total_steps = len(self.adoption_data)
        results = []
        
        for scenario in self.adoption_data.columns:
            scenario_data = self.adoption_data[scenario] * 100  # Convert to percentage
            display_name = self.metadata.get(scenario, {}).get('display_name', scenario)
            
            row = {'scenario': scenario, 'display_name': display_name}
            
            for frac in time_fractions:
                idx = min(int(frac * (total_steps - 1)), total_steps - 1)
                row[f'{int(frac*100)}pct_time'] = scenario_data.iloc[idx]
            
            results.append(row)
        
        return pd.DataFrame(results)
    
    def _extract_system_metrics(self):
        """Extract system-level metrics for visualization layer."""
        if self.model_data.empty:
            return pd.DataFrame()
        
        # Look for system metric columns (UPDATED to use scenario-specific names)
        system_cols = ['Year', 'FossilPrice', 'RenewablePrice', 'rational_TotalCreditsEarned', 
                    'rational_TotalCreditsUsed', 'rational_CreditUtilizationRate', 'rational_GridStressIndex',
                    'MonthlyPeakLoad', 'rational_FossilDependency']
        
        available_cols = [col for col in system_cols if col in self.model_data.columns]
        
        if len(available_cols) <= 1:  # Only Year column
            return pd.DataFrame()
        
        system_data = self.model_data[available_cols].copy()
        
        if 'Year' in system_data.columns:
            system_data = system_data.groupby('Year').last().reset_index()
        
        # IMPORTANT: Rename columns back for compatibility with visualization code
        rename_mapping = {
            'rational_TotalCreditsEarned': 'TotalCreditsEarned',
            'rational_TotalCreditsUsed': 'TotalCreditsUsed', 
            'rational_CreditUtilizationRate': 'CreditUtilizationRate',
            'rational_GridStressIndex': 'GridStressIndex',
            'rational_FossilDependency': 'FossilDependency'
        }
        system_data = system_data.rename(columns=rename_mapping)
        
        return system_data
    
    def export_all_metrics(self, output_dir="results/metrics"):
        """
        Export all calculated metrics to CSV files.
        
        Args:
            output_dir: Directory to save CSV files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Exporting comparative metrics to {output_dir}...")
        
        # 1. Master adoption time series
        adoption_ts = self.calculate_adoption_time_series()
        if not adoption_ts.empty:
            adoption_ts.to_csv(os.path.join(output_dir, 'adoption_time_series.csv'))
            print(f"  ✓ adoption_time_series.csv: {len(adoption_ts)} time points")
        
        # 2. Critical mass timing analysis
        critical_mass = self.calculate_critical_mass_timing()
        if not critical_mass.empty:
            critical_mass.to_csv(os.path.join(output_dir, 'critical_mass_timing.csv'), index=False)
            print(f"  ✓ critical_mass_timing.csv: {len(critical_mass)} threshold analyses")
        
        # 3. Area analysis with multiple interpretations
        area_analysis = self.calculate_area_analysis()
        if not area_analysis.empty:
            area_analysis.to_csv(os.path.join(output_dir, 'area_analysis.csv'), index=False)
            print(f"  ✓ area_analysis.csv: {len(area_analysis)} scenario analyses")
        
        # 4. Scenario comparison summary
        comparison_summary = self.calculate_scenario_comparison_summary()
        if not comparison_summary.empty:
            comparison_summary.to_csv(os.path.join(output_dir, 'scenario_comparison_summary.csv'), index=False)
            print(f"  ✓ scenario_comparison_summary.csv: {len(comparison_summary)} scenarios")
        
        # 5. Adoption snapshots table
        snapshots = self.calculate_adoption_snapshots()
        if not snapshots.empty:
            snapshots.to_csv(os.path.join(output_dir, 'adoption_snapshots_table.csv'), index=False)
            print(f"  ✓ adoption_snapshots_table.csv: {len(snapshots)} scenarios")
        
        print(f"✅ All comparative metrics exported to {output_dir}")
        return True
    
    def get_metrics_summary(self):
        """Get a summary of available metrics."""
        summary = {
            'data_points': len(self.adoption_data),
            'scenarios': list(self.adoption_data.columns),
            'time_range': (self.adoption_data.index.min(), self.adoption_data.index.max()) if not self.adoption_data.empty else None,
            'has_income_data': not self.income_data.empty
        }
        
        return summary