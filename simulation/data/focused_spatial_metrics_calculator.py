# data/focused_spatial_metrics_calculator.py V2.0 - CORRECTED FOCUSED BEHAVIORAL METRICS
"""
CORRECTED: Focused metrics calculator with proper neighbor access and time series data.
Fixes: neighbor structure access, temporal classification, and time series storage.
"""

import pandas as pd
import numpy as np
import os
from collections import defaultdict

from ..utils.parameters import get_all_scenarios, get_scenario_colors

class FocusedSpatialMetricsCalculator:
    """
    CORRECTED: Focused metrics calculator for behavioral prosumer analysis.
    """
    
    def __init__(self, model):
        """Initialize the focused metrics calculator."""
        self.model = model
        self.scenarios = get_all_scenarios()
        
        # CORRECTED: Store time series data for income adoption rates
        self.income_adoption_timeseries = defaultdict(lambda: defaultdict(list))  # {scenario: {income_class: [rates_over_time]}}
        self.income_adoption_years = []  # Track years for time series
        
        # CORRECTED: Store time series data for neighbor evolution
        self.neighbor_evolution_data = []   # Time series with proper calculations
        
        # CORRECTED: Store adoption context with proper neighbor calculations
        self.adoption_context_data = []     # Context when households adopt
        
        # CORRECTED: Store network snapshots with temporal classification
        self.network_snapshots = {}         # Network state at key timepoints
        
        # Track adoption steps for temporal classification
        self.household_adoption_steps = defaultdict(dict)  # {household_id: {scenario: step}}
        
        # Track collection steps
        self.current_step = 0
        self.snapshot_years = [2, 5, 10, 20]  # Years for network snapshots
        self.snapshot_steps = [y * 12 for y in self.snapshot_years]  # Convert to steps
        
        print(f"FocusedSpatialMetricsCalculator initialized for {len(self.scenarios)} scenarios")
    
    def calculate_step_metrics(self, current_step):
        """Calculate focused metrics for current simulation step."""
        self.current_step = current_step
        
        # Track adoption timing for all households
        self._track_adoption_timing()
        
        # Calculate metrics at each step
        self.calculate_income_adoption_timeseries()
        self.calculate_neighbor_evolution()
        
        # Capture adoption context for new adopters
        self.calculate_adoption_timing_context()
        
        # Capture network snapshots at key timepoints
        if current_step in self.snapshot_steps:
            self.calculate_network_snapshots()
    
    def _track_adoption_timing(self):
        """Track when each household adopts in each scenario."""
        households = self.model.get_households()
        
        for household in households:
            for scenario in self.scenarios:
                current_adoption = household.scenario_adoption.get(scenario, False)
                household_id = household.unique_id
                
                # If adopted and not previously recorded, record adoption step
                if (current_adoption and 
                    household_id not in self.household_adoption_steps or
                    scenario not in self.household_adoption_steps[household_id]):
                    self.household_adoption_steps[household_id][scenario] = self.current_step
    
    def calculate_income_adoption_timeseries(self):
        """CORRECTED: Calculate time series of adoption rates by income class."""
        households = self.model.get_households()
        current_year = (self.current_step // 12) + 1
        
        # Only calculate at year boundaries (every 12 steps)
        if self.current_step % 12 != 0:
            return
        
        # Track years for time series
        if current_year not in self.income_adoption_years:
            self.income_adoption_years.append(current_year)
        
        for scenario in self.scenarios:
            # Group households by income class
            income_groups = defaultdict(list)
            for household in households:
                income_class = getattr(household, 'income_class', 1)
                income_groups[income_class].append(household)
            
            # Calculate adoption rate for each income class
            for income_class, class_households in income_groups.items():
                if class_households:
                    adopted_count = sum(1 for h in class_households 
                                      if h.scenario_adoption.get(scenario, False))
                    adoption_rate = adopted_count / len(class_households)
                    
                    # Store in time series
                    self.income_adoption_timeseries[scenario][income_class].append({
                        'year': current_year,
                        'step': self.current_step,
                        'adoption_rate': adoption_rate
                    })
    
    def calculate_neighbor_evolution(self):
        """CORRECTED: Track prosumer neighbors evolution using proper neighbor access."""
        households = self.model.get_households()
        current_year = (self.current_step // 12) + 1
        
        neighbor_data = {
            'step': self.current_step,
            'year': current_year
        }
        
        for scenario in self.scenarios:
            # Calculate average prosumer neighbors for all households
            all_household_prosumer_counts = []
            nonprosumer_household_prosumer_counts = []
            
            for household in households:
                # CORRECTED: Use spatial_neighbors which is [(neighbor_household, distance), ...]
                if hasattr(household, 'spatial_neighbors') and household.spatial_neighbors:
                    prosumer_neighbors = 0
                    total_neighbors = len(household.spatial_neighbors)
                    
                    # Count prosumer neighbors for this scenario
                    for neighbor_household, distance in household.spatial_neighbors:
                        if neighbor_household.scenario_adoption.get(scenario, False):
                            prosumer_neighbors += 1
                    
                    all_household_prosumer_counts.append(prosumer_neighbors)
                    
                    # For non-prosumers in this scenario, track their prosumer neighbors
                    if not household.scenario_adoption.get(scenario, False):
                        nonprosumer_household_prosumer_counts.append(prosumer_neighbors)
            
            # Store averages
            neighbor_data[f'{scenario}_avg_prosumer_neighbors'] = (
                np.mean(all_household_prosumer_counts) if all_household_prosumer_counts else 0
            )
            neighbor_data[f'{scenario}_avg_nonprosumer_prosumer_neighbors'] = (
                np.mean(nonprosumer_household_prosumer_counts) if nonprosumer_household_prosumer_counts else 0
            )
        
        self.neighbor_evolution_data.append(neighbor_data)
    
    def calculate_adoption_timing_context(self):
        """CORRECTED: Capture context when households adopt (only for new adopters)."""
        households = self.model.get_households()
        
        for household in households:
            household_id = household.unique_id
            
            for scenario in self.scenarios:
                # Check if this household just adopted in this scenario (current step)
                if (household_id in self.household_adoption_steps and
                    scenario in self.household_adoption_steps[household_id] and
                    self.household_adoption_steps[household_id][scenario] == self.current_step):
                    
                    # Calculate spatial adoption rate in neighborhood
                    spatial_adopters = 0
                    total_neighbors = 0
                    
                    # CORRECTED: Use spatial_neighbors properly
                    if hasattr(household, 'spatial_neighbors') and household.spatial_neighbors:
                        total_neighbors = len(household.spatial_neighbors)
                        
                        for neighbor_household, distance in household.spatial_neighbors:
                            if neighbor_household.scenario_adoption.get(scenario, False):
                                spatial_adopters += 1
                    
                    spatial_rate = spatial_adopters / total_neighbors if total_neighbors > 0 else 0
                    
                    # Calculate class adoption rate
                    household_income_class = getattr(household, 'income_class', 1)
                    class_adopters = 0
                    class_total = 0
                    
                    for other in households:
                        if getattr(other, 'income_class', 1) == household_income_class:
                            class_total += 1
                            if other.scenario_adoption.get(scenario, False):
                                class_adopters += 1
                    
                    class_rate = class_adopters / class_total if class_total > 0 else 0
                    
                    # Store adoption context
                    context_record = {
                        'household_id': household_id,
                        'scenario': scenario,
                        'adoption_step': self.current_step,
                        'adoption_year': (self.current_step // 12) + 1,
                        'spatial_adoption_rate': spatial_rate,
                        'class_adoption_rate': class_rate,
                        'income_class': household_income_class,
                        'total_neighbors': total_neighbors
                    }
                    
                    self.adoption_context_data.append(context_record)
    
    def calculate_network_snapshots(self):
        """CORRECTED: Capture network state with temporal adoption classification."""
        current_year = (self.current_step // 12) + 1
        households = self.model.get_households()
        
        if current_year not in self.network_snapshots:
            self.network_snapshots[current_year] = {}
        
        for scenario in self.scenarios:
            # CORRECTED: Classify nodes by temporal adoption status
            node_classification = {
                'non_prosumers': [],          # Never adopted
                'new_prosumers': [],          # Adopted in current month (current step)
                'old_prosumers': []           # Adopted in previous month (step-1)
            }
            
            for household in households:
                household_id = household.unique_id
                is_prosumer = household.scenario_adoption.get(scenario, False)
                
                if not is_prosumer:
                    node_classification['non_prosumers'].append(household_id)
                else:
                    # Get adoption step for this scenario
                    adoption_step = self.household_adoption_steps.get(household_id, {}).get(scenario, 0)
                    
                    if adoption_step == self.current_step:  # Adopted this month
                        node_classification['new_prosumers'].append(household_id)
                    elif adoption_step == self.current_step - 1:  # Adopted last month
                        node_classification['old_prosumers'].append(household_id)
                    # Note: Households that adopted before step-1 are not classified
            
            self.network_snapshots[current_year][scenario] = node_classification
    
    def export_metrics(self, output_dir="results/focused_spatial_metrics"):
        """Export all focused metrics to CSV files."""
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Exporting corrected focused metrics to {output_dir}...")
        
        # 1. Export income adoption time series
        self._export_income_adoption_timeseries(output_dir)
        
        # 2. Export neighbor evolution
        self._export_neighbor_evolution(output_dir)
        
        # 3. Export adoption context
        self._export_adoption_context(output_dir)
        
        # 4. Export network snapshots
        self._export_network_snapshots(output_dir)
        
        print("Corrected focused metrics export completed!")
    
    def _export_income_adoption_timeseries(self, output_dir):
        """CORRECTED: Export income adoption time series to CSV."""
        records = []
        
        for scenario, scenario_data in self.income_adoption_timeseries.items():
            for income_class, timeseries in scenario_data.items():
                for datapoint in timeseries:
                    records.append({
                        'Year': datapoint['year'],
                        'Step': datapoint['step'],
                        'Scenario': scenario,
                        'IncomeClass': income_class,
                        'AdoptionRate': datapoint['adoption_rate']
                    })
        
        if records:
            income_df = pd.DataFrame(records)
            income_path = os.path.join(output_dir, "focused_spatial_income_adoption_timeseries.csv")
            income_df.to_csv(income_path, index=False)
            print(f"  Exported income adoption time series: {len(records)} records")
    
    def _export_neighbor_evolution(self, output_dir):
        """Export neighbor evolution to CSV."""
        if self.neighbor_evolution_data:
            neighbor_df = pd.DataFrame(self.neighbor_evolution_data)
            neighbor_path = os.path.join(output_dir, "focused_spatial_neighbor_evolution.csv")
            neighbor_df.to_csv(neighbor_path, index=False)
            print(f"  Exported neighbor evolution: {len(self.neighbor_evolution_data)} records")
    
    def _export_adoption_context(self, output_dir):
        """Export adoption context to CSV."""
        if self.adoption_context_data:
            context_df = pd.DataFrame(self.adoption_context_data)
            context_path = os.path.join(output_dir, "focused_spatial_adoption_context.csv")
            context_df.to_csv(context_path, index=False)
            print(f"  Exported adoption context: {len(self.adoption_context_data)} records")
    
    def _export_network_snapshots(self, output_dir):
        """CORRECTED: Export network snapshots with temporal classification."""
        records = []
        
        for year, year_data in self.network_snapshots.items():
            for scenario, node_types in year_data.items():
                for node_type, household_ids in node_types.items():
                    for household_id in household_ids:
                        records.append({
                            'Year': year,
                            'Scenario': scenario,
                            'NodeType': node_type,
                            'HouseholdId': household_id
                        })
        
        if records:
            snapshot_df = pd.DataFrame(records)
            snapshot_path = os.path.join(output_dir, "focused_spatial_network_snapshots.csv")
            snapshot_df.to_csv(snapshot_path, index=False)
            print(f"  Exported network snapshots: {len(records)} records")
    
    def get_data_summary(self):
        """Get summary of collected focused metrics."""
        summary = {
            'scenarios_tracked': len(self.scenarios),
            'current_step': self.current_step,
            'income_timeseries_scenarios': len(self.income_adoption_timeseries),
            'neighbor_evolution_records': len(self.neighbor_evolution_data),
            'adoption_context_records': len(self.adoption_context_data),
            'network_snapshot_years': len(self.network_snapshots),
            'tracked_adoptions': len(self.household_adoption_steps)
        }
        
        return summary