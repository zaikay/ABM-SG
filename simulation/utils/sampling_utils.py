# utils/sampling_utils.py V1
"""
Sampling utilities for visualization of large-scale agent-based models.
Provides stratified sampling to maintain demographic and spatial patterns while reducing visual complexity.
"""

import numpy as np
import pandas as pd
import random
from collections import defaultdict
from ..utils.parameters import SPATIAL_VISUALIZATION_CONFIG

class VisualizationSampler:
    """
    Handles sampling of households for visualization purposes.
    
    Maintains representativeness while reducing visual complexity for large populations.
    """
    
    def __init__(self, sample_size=None, random_seed=42):
        """
        Initialize the visualization sampler.
        
        Args:
            sample_size: Number of households to sample (default from config)
            random_seed: Random seed for reproducible sampling
        """
        self.sample_size = sample_size or SPATIAL_VISUALIZATION_CONFIG['sample_size']
        self.random_seed = random_seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        print(f"VisualizationSampler initialized with sample_size={self.sample_size}")
    
    def stratified_spatial_sample(self, households, maintain_adoption_rates=True):
        """
        Create stratified sample maintaining spatial and demographic patterns.
        
        Strategy:
        1. Stratify by income class (maintain class distribution)
        2. Within each class, stratify by spatial quadrants
        3. Maintain prosumer/non-prosumer ratios within each stratum
        4. Use systematic sampling within strata for spatial distribution
        
        Args:
            households: List of household agents
            maintain_adoption_rates: Whether to maintain adoption rate proportions
            
        Returns:
            list: Sampled household agents
        """
        if len(households) <= self.sample_size:
            return households.copy()
        
        # Convert to DataFrame for easier manipulation
        household_data = []
        for h in households:
            household_data.append({
                'agent': h,
                'unique_id': h.unique_id,
                'income_class': getattr(h, 'income_class', 1),
                'is_prosumer': getattr(h, 'is_prosumer', False),
                'pos_x': h.pos[0] if hasattr(h, 'pos') and h.pos else 0,
                'pos_y': h.pos[1] if hasattr(h, 'pos') and h.pos else 0,
                'income': getattr(h, 'income', 0)
            })
        
        df = pd.DataFrame(household_data)
        
        # Calculate target sample sizes per income class
        class_counts = df['income_class'].value_counts().sort_index()
        class_proportions = class_counts / len(df)
        
        sampled_households = []
        
        # Sample from each income class
        for income_class in sorted(class_counts.index):
            class_df = df[df['income_class'] == income_class].copy()
            
            # Target sample size for this class
            target_class_size = max(1, int(self.sample_size * class_proportions[income_class]))
            
            if len(class_df) <= target_class_size:
                # Include all households from this class
                sampled_households.extend(class_df['agent'].tolist())
                continue
            
            # Spatial stratification within income class
            sampled_from_class = self._spatial_stratified_sample(
                class_df, target_class_size, maintain_adoption_rates
            )
            
            sampled_households.extend(sampled_from_class)
        
        # Adjust to exact sample size if needed
        if len(sampled_households) > self.sample_size:
            # Randomly remove excess
            excess = len(sampled_households) - self.sample_size
            to_remove = random.sample(range(len(sampled_households)), excess)
            sampled_households = [h for i, h in enumerate(sampled_households) if i not in to_remove]
        elif len(sampled_households) < self.sample_size:
            # Add more households randomly
            remaining_households = [h for h in households if h not in sampled_households]
            needed = self.sample_size - len(sampled_households)
            additional = random.sample(remaining_households, min(needed, len(remaining_households)))
            sampled_households.extend(additional)
        
        print(f"Sampled {len(sampled_households)} households from {len(households)} total")
        self._print_sample_statistics(sampled_households, households)
        
        return sampled_households
    
    def _spatial_stratified_sample(self, class_df, target_size, maintain_adoption_rates):
        """
        Perform spatial stratification within an income class.
        
        Args:
            class_df: DataFrame of households in one income class
            target_size: Target number of households to sample
            maintain_adoption_rates: Whether to maintain prosumer ratios
            
        Returns:
            list: Sampled household agents
        """
        # Create spatial quadrants
        x_median = class_df['pos_x'].median()
        y_median = class_df['pos_y'].median()
        
        quadrants = {
            'NE': class_df[(class_df['pos_x'] >= x_median) & (class_df['pos_y'] >= y_median)],
            'NW': class_df[(class_df['pos_x'] < x_median) & (class_df['pos_y'] >= y_median)],
            'SE': class_df[(class_df['pos_x'] >= x_median) & (class_df['pos_y'] < y_median)],
            'SW': class_df[(class_df['pos_x'] < x_median) & (class_df['pos_y'] < y_median)]
        }
        
        # Remove empty quadrants
        quadrants = {k: v for k, v in quadrants.items() if len(v) > 0}
        
        if len(quadrants) == 0:
            return []
        
        # Distribute target size across quadrants proportionally
        total_in_quadrants = sum(len(q) for q in quadrants.values())
        quadrant_targets = {}
        
        for quad_name, quad_df in quadrants.items():
            proportion = len(quad_df) / total_in_quadrants
            quadrant_targets[quad_name] = max(1, int(target_size * proportion))
        
        # Adjust to match target size exactly
        current_total = sum(quadrant_targets.values())
        if current_total != target_size:
            # Adjust the largest quadrant
            largest_quad = max(quadrant_targets.keys(), key=lambda k: quadrant_targets[k])
            quadrant_targets[largest_quad] += target_size - current_total
        
        # Sample from each quadrant
        sampled_agents = []
        
        for quad_name, quad_df in quadrants.items():
            quad_target = quadrant_targets[quad_name]
            
            if maintain_adoption_rates and len(quad_df) > 0:
                # Maintain prosumer ratio within quadrant
                sampled_from_quad = self._sample_with_adoption_ratio(quad_df, quad_target)
            else:
                # Simple random sample from quadrant
                if len(quad_df) <= quad_target:
                    sampled_from_quad = quad_df['agent'].tolist()
                else:
                    sampled_indices = random.sample(range(len(quad_df)), quad_target)
                    sampled_from_quad = [quad_df.iloc[i]['agent'] for i in sampled_indices]
            
            sampled_agents.extend(sampled_from_quad)
        
        return sampled_agents
    
    def _sample_with_adoption_ratio(self, df, target_size):
        """
        Sample households while maintaining prosumer/non-prosumer ratios.
        
        Args:
            df: DataFrame of households
            target_size: Target number of households to sample
            
        Returns:
            list: Sampled household agents
        """
        if len(df) <= target_size:
            return df['agent'].tolist()
        
        # Calculate current adoption rate
        prosumer_count = df['is_prosumer'].sum()
        adoption_rate = prosumer_count / len(df) if len(df) > 0 else 0
        
        # Target prosumer count in sample
        target_prosumers = max(0, int(target_size * adoption_rate))
        target_nonprosumers = target_size - target_prosumers
        
        # Sample prosumers and non-prosumers separately
        prosumers = df[df['is_prosumer'] == True]
        nonprosumers = df[df['is_prosumer'] == False]
        
        sampled_agents = []
        
        # Sample prosumers
        if len(prosumers) <= target_prosumers:
            sampled_agents.extend(prosumers['agent'].tolist())
        elif target_prosumers > 0:
            sampled_indices = random.sample(range(len(prosumers)), target_prosumers)
            sampled_agents.extend([prosumers.iloc[i]['agent'] for i in sampled_indices])
        
        # Sample non-prosumers
        if len(nonprosumers) <= target_nonprosumers:
            sampled_agents.extend(nonprosumers['agent'].tolist())
        elif target_nonprosumers > 0:
            sampled_indices = random.sample(range(len(nonprosumers)), target_nonprosumers)
            sampled_agents.extend([nonprosumers.iloc[i]['agent'] for i in sampled_indices])
        
        # If we still don't have enough, sample randomly from the remainder
        if len(sampled_agents) < target_size:
            remaining = [agent for agent in df['agent'] if agent not in sampled_agents]
            needed = target_size - len(sampled_agents)
            if len(remaining) > 0:
                additional = random.sample(remaining, min(needed, len(remaining)))
                sampled_agents.extend(additional)
        
        return sampled_agents
    
    def _print_sample_statistics(self, sampled_households, all_households):
        """
        Print statistics comparing sample to full population.
        
        Args:
            sampled_households: List of sampled household agents
            all_households: List of all household agents
        """
        def get_stats(households):
            stats = {
                'total': len(households),
                'income_classes': defaultdict(int),
                'prosumers': 0,
                'adoption_rate': 0
            }
            
            for h in households:
                stats['income_classes'][getattr(h, 'income_class', 1)] += 1
                if getattr(h, 'is_prosumer', False):
                    stats['prosumers'] += 1
            
            stats['adoption_rate'] = stats['prosumers'] / stats['total'] if stats['total'] > 0 else 0
            return stats
        
        sample_stats = get_stats(sampled_households)
        population_stats = get_stats(all_households)
        
        print(f"\n=== Sampling Statistics ===")
        print(f"Sample size: {sample_stats['total']} / {population_stats['total']} "
              f"({100 * sample_stats['total'] / population_stats['total']:.1f}%)")
        
        print(f"Adoption rates:")
        print(f"  Population: {100 * population_stats['adoption_rate']:.1f}%")
        print(f"  Sample:     {100 * sample_stats['adoption_rate']:.1f}%")
        
        print(f"Income class distribution:")
        for income_class in sorted(set(list(sample_stats['income_classes'].keys()) + 
                                     list(population_stats['income_classes'].keys()))):
            pop_pct = 100 * population_stats['income_classes'][income_class] / population_stats['total']
            sample_pct = 100 * sample_stats['income_classes'][income_class] / sample_stats['total']
            print(f"  Class {income_class}: Population {pop_pct:.1f}%, Sample {sample_pct:.1f}%")
    
    def sample_for_scenario_comparison(self, households_by_scenario, timepoint=None):
        """
        Sample households for cross-scenario comparison at a specific timepoint.
        
        Maintains the same set of households across all scenarios for valid comparison.
        
        Args:
            households_by_scenario: Dict mapping scenario names to household lists
            timepoint: Specific timepoint to sample for (optional)
            
        Returns:
            dict: Mapping scenario names to sampled household lists
        """
        # Use the rational scenario as the baseline for sampling
        if 'rational' in households_by_scenario:
            baseline_households = households_by_scenario['rational']
        else:
            baseline_households = list(households_by_scenario.values())[0]
        
        # Get stratified sample from baseline
        sampled_household_ids = set()
        sampled_baseline = self.stratified_spatial_sample(baseline_households)
        sampled_household_ids = {h.unique_id for h in sampled_baseline}
        
        # Apply same sampling to all scenarios
        sampled_by_scenario = {}
        
        for scenario_name, scenario_households in households_by_scenario.items():
            # Find households with matching IDs
            sampled_scenario = [h for h in scenario_households 
                              if h.unique_id in sampled_household_ids]
            sampled_by_scenario[scenario_name] = sampled_scenario
        
        print(f"Sampled {len(sampled_household_ids)} households for {len(households_by_scenario)} scenarios")
        
        return sampled_by_scenario
    
    def get_network_layout_positions(self, households, layout_algorithm='spring'):
        """
        Calculate layout positions for network visualization.
        
        Args:
            households: List of household agents
            layout_algorithm: Layout algorithm ('spring', 'circular', 'random')
            
        Returns:
            dict: Mapping household IDs to (x, y) positions
        """
        positions = {}
        
        if layout_algorithm == 'spring':
            # Use existing positions as starting point, apply spring layout adjustment
            for h in households:
                if hasattr(h, 'pos') and h.pos is not None:
                    # Add small random perturbation for better visualization
                    x, y = h.pos
                    x += np.random.normal(0, 0.1)
                    y += np.random.normal(0, 0.1)
                    positions[h.unique_id] = (x, y)
                else:
                    # Random position if no position available
                    positions[h.unique_id] = (np.random.uniform(-1, 1), np.random.uniform(-1, 1))
        
        elif layout_algorithm == 'circular':
            # Arrange households in a circle
            n = len(households)
            for i, h in enumerate(households):
                angle = 2 * np.pi * i / n
                x = np.cos(angle)
                y = np.sin(angle)
                positions[h.unique_id] = (x, y)
        
        elif layout_algorithm == 'random':
            # Random positions
            for h in households:
                positions[h.unique_id] = (np.random.uniform(-1, 1), np.random.uniform(-1, 1))
        
        else:
            # Default: use existing positions
            for h in households:
                if hasattr(h, 'pos') and h.pos is not None:
                    positions[h.unique_id] = h.pos
                else:
                    positions[h.unique_id] = (0, 0)
        
        return positions

    def advanced_multi_scenario_sample(self, households_by_scenario, sample_size=None):
        """
        Advanced sampling for multi-scenario comparison with enhanced representativeness.
        
        Ensures the same households are tracked across all scenarios while maintaining
        demographic and spatial patterns.
        
        Args:
            households_by_scenario: Dict mapping scenario names to household lists
            sample_size: Sample size (default: self.sample_size)
            
        Returns:
            dict: Mapping scenario names to sampled household lists
        """
        if sample_size is None:
            sample_size = self.sample_size
        
        # Use rational scenario as baseline for sampling
        if 'rational' in households_by_scenario:
            baseline_households = households_by_scenario['rational']
        else:
            baseline_households = list(households_by_scenario.values())[0]
        
        # Get total population if available
        total_households = len(baseline_households)
        actual_sample_size = min(sample_size, total_households)
        
        print(f"Advanced multi-scenario sampling: {actual_sample_size} from {total_households} households")
        
        # Create enhanced stratified sample
        sampled_baseline = self._enhanced_stratified_sample(baseline_households, actual_sample_size)
        sampled_household_ids = {h.unique_id for h in sampled_baseline}
        
        # Apply same sampling to all scenarios
        sampled_by_scenario = {}
        
        for scenario_name, scenario_households in households_by_scenario.items():
            # Find households with matching IDs
            sampled_scenario = []
            
            for household in scenario_households:
                if household.unique_id in sampled_household_ids:
                    # Create a copy with scenario-specific attributes
                    sampled_household = self._create_scenario_specific_household(household, scenario_name)
                    sampled_scenario.append(sampled_household)
            
            sampled_by_scenario[scenario_name] = sampled_scenario
            print(f"  {scenario_name}: {len(sampled_scenario)} households sampled")
        
        return sampled_by_scenario
    
    def _enhanced_stratified_sample(self, households, target_size):
        """
        Enhanced stratified sampling with improved representativeness.
        
        Args:
            households: List of household agents
            target_size: Target sample size
            
        Returns:
            list: Sampled households
        """
        if len(households) <= target_size:
            return households.copy()
        
        # Convert to DataFrame for analysis
        household_data = []
        for h in households:
            household_data.append({
                'agent': h,
                'unique_id': h.unique_id,
                'income_class': getattr(h, 'income_class', 1),
                'is_prosumer': getattr(h, 'is_prosumer', False),
                'pos_x': h.pos[0] if hasattr(h, 'pos') and h.pos else np.random.uniform(-5, 5),
                'pos_y': h.pos[1] if hasattr(h, 'pos') and h.pos else np.random.uniform(-5, 5),
                'income': getattr(h, 'income', 50000),
                'daily_consumption': getattr(h, 'daily_consumption', 20)
            })
        
        df = pd.DataFrame(household_data)
        
        # Multi-stage stratification
        sampled_households = []
        
        # Stage 1: Income class stratification
        for income_class in sorted(df['income_class'].unique()):
            class_df = df[df['income_class'] == income_class].copy()
            class_proportion = len(class_df) / len(df)
            class_target = max(1, int(target_size * class_proportion))
            
            if len(class_df) <= class_target:
                sampled_households.extend(class_df['agent'].tolist())
                continue
            
            # Stage 2: Within class, stratify by prosumer status
            prosumer_df = class_df[class_df['is_prosumer'] == True]
            nonprosumer_df = class_df[class_df['is_prosumer'] == False]
            
            prosumer_proportion = len(prosumer_df) / len(class_df) if len(class_df) > 0 else 0
            prosumer_target = int(class_target * prosumer_proportion)
            nonprosumer_target = class_target - prosumer_target
            
            # Sample prosumers
            if len(prosumer_df) > 0 and prosumer_target > 0:
                if len(prosumer_df) <= prosumer_target:
                    sampled_households.extend(prosumer_df['agent'].tolist())
                else:
                    # Spatial stratification within prosumers
                    prosumer_sample = self._spatial_stratified_sample_enhanced(prosumer_df, prosumer_target)
                    sampled_households.extend(prosumer_sample)
            
            # Sample non-prosumers
            if len(nonprosumer_df) > 0 and nonprosumer_target > 0:
                if len(nonprosumer_df) <= nonprosumer_target:
                    sampled_households.extend(nonprosumer_df['agent'].tolist())
                else:
                    # Spatial stratification within non-prosumers
                    nonprosumer_sample = self._spatial_stratified_sample_enhanced(nonprosumer_df, nonprosumer_target)
                    sampled_households.extend(nonprosumer_sample)
        
        # Ensure exact sample size
        if len(sampled_households) > target_size:
            sampled_households = random.sample(sampled_households, target_size)
        elif len(sampled_households) < target_size:
            # Add random households to reach target
            remaining = [h for h in households if h not in sampled_households]
            needed = target_size - len(sampled_households)
            if remaining:
                additional = random.sample(remaining, min(needed, len(remaining)))
                sampled_households.extend(additional)
        
        return sampled_households
    
    def _spatial_stratified_sample_enhanced(self, df, target_size):
        """
        Enhanced spatial stratification with better coverage.
        
        Args:
            df: DataFrame with household data
            target_size: Target sample size
            
        Returns:
            list: Sampled household agents
        """
        if len(df) <= target_size:
            return df['agent'].tolist()
        
        # Create spatial grid
        x_min, x_max = df['pos_x'].min(), df['pos_x'].max()
        y_min, y_max = df['pos_y'].min(), df['pos_y'].max()
        
        # Determine grid size based on target sample size
        grid_size = max(2, int(np.sqrt(target_size)))
        
        x_bins = np.linspace(x_min, x_max, grid_size + 1)
        y_bins = np.linspace(y_min, y_max, grid_size + 1)
        
        # FIX: Create a proper copy to avoid pandas warnings
        df_copy = df.copy()
        
        # Assign grid cells
        df_copy['grid_x'] = pd.cut(df_copy['pos_x'], x_bins, labels=False, include_lowest=True)
        df_copy['grid_y'] = pd.cut(df_copy['pos_y'], y_bins, labels=False, include_lowest=True)
        df_copy['grid_cell'] = df_copy['grid_x'].astype(str) + '_' + df_copy['grid_y'].astype(str)
        
        # Sample from each grid cell proportionally
        cells_with_data = df_copy.groupby('grid_cell').size()
        total_cells = len(cells_with_data)
        samples_per_cell = target_size // total_cells
        remaining_samples = target_size % total_cells
        
        sampled_agents = []
        
        for cell, cell_df in df_copy.groupby('grid_cell'):
            cell_target = samples_per_cell
            if remaining_samples > 0:
                cell_target += 1
                remaining_samples -= 1
            
            cell_target = min(cell_target, len(cell_df))
            
            if cell_target > 0:
                if len(cell_df) <= cell_target:
                    sampled_agents.extend(cell_df['agent'].tolist())
                else:
                    # Random sample within cell
                    cell_sample = cell_df.sample(n=cell_target, random_state=self.random_seed)
                    sampled_agents.extend(cell_sample['agent'].tolist())
        
        return sampled_agents
    
    def _create_scenario_specific_household(self, household, scenario_name):
        """
        Create a household copy with scenario-specific attributes.
        
        Args:
            household: Original household agent
            scenario_name: Name of the scenario
            
        Returns:
            household: Household with scenario-specific adoption status
        """
        # Create a copy of the household
        scenario_household = household
        
        # Set scenario-specific adoption status
        if hasattr(household, 'scenario_adoption'):
            scenario_household.is_prosumer = household.scenario_adoption.get(scenario_name, False)
            scenario_household.adoption_month = household.adoption_months.get(scenario_name, None)
        else:
            # Fallback for non-multi-scenario households
            scenario_household.is_prosumer = getattr(household, 'is_prosumer', False)
            scenario_household.adoption_month = None
        
        # Add scenario identifier
        scenario_household.current_scenario = scenario_name
        
        return scenario_household


# =============================================================================
# TESTING FUNCTIONS
# =============================================================================

def test_visualization_sampler():
    """
    Test the VisualizationSampler class.
    
    Returns:
        bool: True if tests pass
    """
    print("Testing VisualizationSampler...")
    
    try:
        # Create mock households
        class MockHousehold:
            def __init__(self, unique_id, income_class, is_prosumer, pos):
                self.unique_id = unique_id
                self.income_class = income_class
                self.is_prosumer = is_prosumer
                self.pos = pos
                self.income = 30000 + income_class * 15000  # Mock income
        
        # Create test population
        households = []
        for i in range(100):
            income_class = (i % 5) + 1  # Classes 1-5
            is_prosumer = i % 4 == 0    # 25% adoption rate
            pos = (np.random.uniform(-5, 5), np.random.uniform(-5, 5))
            households.append(MockHousehold(i, income_class, is_prosumer, pos))
        
        # Test sampler
        sampler = VisualizationSampler(sample_size=25)
        
        # Test stratified sampling
        sampled = sampler.stratified_spatial_sample(households)
        
        if len(sampled) != 25:
            print(f"❌ Expected 25 sampled households, got {len(sampled)}")
            return False
        
        # Check that sample maintains diversity
        sampled_classes = set(h.income_class for h in sampled)
        if len(sampled_classes) < 3:
            print(f"❌ Sample should maintain income class diversity, got {sampled_classes}")
            return False
        
        # Test scenario comparison sampling
        scenarios = {
            'rational': households,
            'loss_aversion': households.copy(),  # Would be different in real use
            'all_biases': households.copy()
        }
        
        sampled_scenarios = sampler.sample_for_scenario_comparison(scenarios)
        
        if len(sampled_scenarios) != 3:
            print(f"❌ Expected 3 scenarios, got {len(sampled_scenarios)}")
            return False
        
        # Check that all scenarios have same household IDs
        rational_ids = {h.unique_id for h in sampled_scenarios['rational']}
        for scenario_name, scenario_households in sampled_scenarios.items():
            scenario_ids = {h.unique_id for h in scenario_households}
            if scenario_ids != rational_ids:
                print(f"❌ Scenario {scenario_name} has different household IDs")
                return False
        
        # Test layout positions
        positions = sampler.get_network_layout_positions(sampled, 'spring')
        
        if len(positions) != len(sampled):
            print(f"❌ Expected {len(sampled)} positions, got {len(positions)}")
            return False
        
        print("✅ VisualizationSampler tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ VisualizationSampler test failed: {e}")
        return False

if __name__ == "__main__":
    test_visualization_sampler()