# agents/evaluation_triggers.py
"""
Evaluation trigger system for behavioral biases.
Implements semi-annual staggered evaluation with economic change triggers.
"""

import numpy as np
from ..utils.parameters import get_evaluation_trigger_config

class EvaluationTriggers:
    """
    Semi-annual evaluation system for all behavioral biases.
    
    Features:
    - 12-month initial delay for consumption history building
    - Staggered semi-annual evaluations (randomized monthly assignment)
    - Economic change triggers (NPV threshold-based)
    - Annual re-randomization of evaluation months
    - Maximum 2 evaluations per household per year
    - Absorbing state for adopted scenarios
    """
    
    def __init__(self, model):
        """Initialize evaluation trigger system."""
        self.model = model
        self.config = get_evaluation_trigger_config()
        
        # Initialize random number generator
        seed = self.config['random_assignment_seed_base'] + model.random.randint(0, 1000)
        self.rng = np.random.RandomState(seed)
        
        # Household evaluation tracking
        self.household_schedules = {}  # household_id -> schedule info
        self.evaluation_counts = {}    # household_id -> {year: count}
        
        if self.config['debug_evaluation_triggers']:
            print(f"EvaluationTriggers initialized with config: {self.config}")
    
    def should_reevaluate(self, household, scenario, current_step):
        """
        Legacy scenario-specific method - delegates to household-level decision.
        """
        # Rule 1: Never re-evaluate if already adopted (absorbing state)
        if household.scenario_adoption.get(scenario, False):
            return False
        
        # Rule 2: Check if this scenario type should use triggers
        rational_scenarios = ['deterministic_rational', 'rational']
        if scenario in rational_scenarios and not self.config.get('apply_to_rational_scenarios', False):
            return True  # Always evaluate if triggers disabled for rational scenarios
        
        # Delegate to household-level decision
        return self.should_reevaluate_household(household, current_step)
            
    
    def should_reevaluate_household(self, household, current_step):
        """
        Household-level trigger decision for ALL scenarios synchronously.
        
        Args:
            household: MultiScenarioHousehold instance
            current_step: Current simulation step
            
        Returns:
            bool: True if household should evaluate ALL scenarios, False otherwise
        """
        
        # Rule 1: No evaluation before initial delay period
        if current_step <= self.config['initial_evaluation_delay']:
            return False
        
        # Continue with existing trigger logic...
        household_id = household.unique_id
        current_year = self._get_year_from_step(current_step)
        current_month = self._get_month_from_step(current_step)
        
        # Rule 2: Maximum evaluations per year check
        if self._has_reached_max_evaluations(household_id, current_year):
            return False
        
        # Rule 3: Initialize or update household schedule
        self._ensure_household_schedule(household_id, current_year)
        
        # Rule 4: Check for economic change trigger (overrides timing)
        if self._check_economic_trigger(household, current_step):
            self._record_evaluation(household_id, current_year, current_step, 'economic_trigger')
            return True
        
        # Rule 5: Check for life event trigger (rare random events)
        if self._check_life_event_trigger():
            self._record_evaluation(household_id, current_year, current_step, 'life_event_trigger')
            return True
        
        # Rule 6: Check scheduled evaluation
        if self._check_scheduled_evaluation(household_id, current_year, current_month, current_step):
            self._record_evaluation(household_id, current_year, current_step, 'scheduled_evaluation')
            return True
        
        return False

    def _ensure_household_schedule(self, household_id, current_year):
        """Ensure household has evaluation schedule for current year."""
        if household_id not in self.household_schedules:
            self.household_schedules[household_id] = {}
        
        if current_year not in self.household_schedules[household_id]:
            # Assign new random evaluation months for this year
            self._assign_evaluation_months(household_id, current_year)
    
    def _assign_evaluation_months(self, household_id, year):
        """Assign evaluation months based on cycle configuration."""
        cycle_months = self.config['evaluation_cycle_months']
        
        if cycle_months == 1:
            # Monthly evaluations: all months 1-12
            evaluation_months = list(range(1, 13))
            
            self.household_schedules[household_id][year] = {
                'evaluation_months': evaluation_months,
                'evaluations_done': [],
                'last_npv': None
            }
        else:
            # Semi-annual evaluations: random month in each half
            first_half_month = self.rng.randint(1, 7)    # 1-6
            second_half_month = self.rng.randint(7, 13)  # 7-12
            
            self.household_schedules[household_id][year] = {
                'first_half_month': first_half_month,
                'second_half_month': second_half_month,
                'evaluations_done': [],
                'last_npv': None
            }
        
        if self.config['debug_evaluation_triggers']:
            if cycle_months == 1:
                print(f"  Household {household_id} Year {year}: Monthly evaluation schedule")
            else:
                print(f"  Household {household_id} Year {year}: "
                    f"Evaluation months {first_half_month}, {second_half_month}")
    
    def _check_scheduled_evaluation(self, household_id, current_year, current_month, current_step):
        """Check if it's time for scheduled evaluation (monthly or semi-annual)."""
        schedule = self.household_schedules[household_id][current_year]
        
        # Handle both monthly and semi-annual schedules
        if 'evaluation_months' in schedule:
            # Monthly schedule
            target_months = schedule['evaluation_months']
        else:
            # Semi-annual schedule
            target_months = [schedule['first_half_month'], schedule['second_half_month']]
        
        if current_month in target_months:
            # Check if we haven't already evaluated this month this year
            if current_month not in schedule['evaluations_done']:
                return True
        
        return False
    
    def _check_economic_trigger(self, household, current_step):
        """Check if NPV changed significantly since last evaluation."""
        household_id = household.unique_id
        current_year = self._get_year_from_step(current_step)
        
        # Get current NPV
        current_npv = getattr(household, 'npv', 0)
        
        # Get stored NPV from last evaluation
        if (household_id in self.household_schedules and 
            current_year in self.household_schedules[household_id]):
            
            last_npv = self.household_schedules[household_id][current_year]['last_npv']
            
            if last_npv is not None:
                # Calculate percentage change
                if abs(last_npv) > 0:
                    change_pct = abs(current_npv - last_npv) / abs(last_npv)
                else:
                    change_pct = 1.0 if abs(current_npv) > 0 else 0.0
                
                if change_pct >= self.config['economic_trigger_threshold']:
                    if self.config['debug_evaluation_triggers']:
                        print(f"  Economic trigger: Household {household_id} NPV changed "
                              f"{change_pct:.1%} (${last_npv:.0f} -> ${current_npv:.0f})")
                    return True
        
        # Store current NPV for future comparisons
        self._store_current_npv(household_id, current_year, current_npv)
        return False
    
    def _check_life_event_trigger(self):
        """Check for rare life event trigger."""
        return self.rng.random() < self.config['life_event_probability']
    
    def _has_reached_max_evaluations(self, household_id, current_year):
        """Check if household has reached maximum evaluations for current year."""
        if household_id not in self.evaluation_counts:
            return False
        
        year_counts = self.evaluation_counts[household_id].get(current_year, 0)
        return year_counts >= self.config['max_evaluations_per_year']
    
    def _record_evaluation(self, household_id, current_year, current_step, trigger_type):
        """Record that an evaluation occurred."""
        # Update evaluation count
        if household_id not in self.evaluation_counts:
            self.evaluation_counts[household_id] = {}
        
        if current_year not in self.evaluation_counts[household_id]:
            self.evaluation_counts[household_id][current_year] = 0
        
        self.evaluation_counts[household_id][current_year] += 1
        
        # Update schedule tracking
        current_month = self._get_month_from_step(current_step)
        if (household_id in self.household_schedules and 
            current_year in self.household_schedules[household_id]):
            
            schedule = self.household_schedules[household_id][current_year]
            if current_month not in schedule['evaluations_done']:
                schedule['evaluations_done'].append(current_month)
        
        if self.config['debug_evaluation_triggers']:
            count = self.evaluation_counts[household_id][current_year]
            print(f"  Evaluation recorded: Household {household_id} Year {current_year} "
                  f"({trigger_type}) - Count: {count}/{self.config['max_evaluations_per_year']}")
    
    def _store_current_npv(self, household_id, current_year, npv):
        """Store current NPV for future economic change detection."""
        if (household_id in self.household_schedules and 
            current_year in self.household_schedules[household_id]):
            self.household_schedules[household_id][current_year]['last_npv'] = npv
    
    def _get_year_from_step(self, step):
        """Convert step to year (1-based)."""
        return ((step - 1) // 12) + 1
    
    def _get_month_from_step(self, step):
        """Convert step to month (1-12)."""
        return ((step - 1) % 12) + 1
    
    def get_evaluation_statistics(self):
        """Get statistics on evaluation patterns for analysis."""
        stats = {
            'total_households': len(self.household_schedules),
            'evaluations_by_year': {},
            'evaluations_by_trigger_type': {},
            'average_evaluations_per_household': 0
        }
        
        total_evaluations = 0
        
        for household_id, year_counts in self.evaluation_counts.items():
            for year, count in year_counts.items():
                if year not in stats['evaluations_by_year']:
                    stats['evaluations_by_year'][year] = 0
                stats['evaluations_by_year'][year] += count
                total_evaluations += count
        
        if len(self.household_schedules) > 0:
            stats['average_evaluations_per_household'] = total_evaluations / len(self.household_schedules)
        
        return stats
    
    def export_evaluation_log(self, output_path):
        """Export evaluation log for analysis."""
        import pandas as pd
        
        records = []
        for household_id, years in self.household_schedules.items():
            for year, schedule in years.items():
                record = {
                    'HouseholdID': household_id,
                    'Year': year,
                    'FirstHalfMonth': schedule['first_half_month'],
                    'SecondHalfMonth': schedule['second_half_month'],
                    'EvaluationsDone': len(schedule['evaluations_done']),
                    'TotalEvaluations': self.evaluation_counts.get(household_id, {}).get(year, 0)
                }
                records.append(record)
        
        df = pd.DataFrame(records)
        df.to_csv(output_path, index=False)
        print(f"Evaluation log exported to {output_path}")