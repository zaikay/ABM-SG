# data/validator.py V5.1
"""
Model validation components for the prosumer simulation.
REFACTORED to work with unified EnergySystem and simplified household calculations.
"""
import numpy as np
import pandas as pd
import os
from collections import defaultdict

class ModelValidator:
    """
    Validates simulation results and checks for consistency.
    UPDATED to work with refactored energy system.
    """
    
    def __init__(self, model, enabled=True):
        """
        Initialize the model validator.
        
        Args:
            model: Mesa model instance
            enabled: Whether validation is enabled
        """
        self.model = model
        self.enabled = enabled
        
        # Validation data storage
        self.validation_data = defaultdict(list)
        self.step_validations = defaultdict(dict)
        
        # Validation thresholds
        self.thresholds = {
            "npv_magnitude": 100000,  # NPV shouldn't exceed $100k
            "savings_rate": 0.8,      # Savings shouldn't exceed 80% of baseline cost
            "payback_period": 50,     # Payback shouldn't exceed 50 years
            "consumption_range": (5, 50),  # Daily consumption range (kWh)
            "generation_efficiency": 10,   # Generation shouldn't exceed 10x capacity
        }
        
        print(f"Model validator {'enabled' if enabled else 'disabled'}")
    
    def collect_step_data(self, step):
        """
        Collect validation data for the current step.
        SIMPLIFIED to use household attributes instead of recalculating.
        
        Args:
            step: Current simulation step
        """
        if not self.enabled:
            return
        
        # Get all households
        households = [agent for agent in self.model.schedule.agents 
                     if hasattr(agent, "daily_consumption")]
        
        step_validation = {
            "step": step,
            "total_households": len(households),
            "prosumers": 0,
            "adoption_rate": 0,
            "validation_errors": [],
            "validation_warnings": []
        }
        
        for household in households:
            # Validate household using pre-calculated attributes (no recalculation!)
            household_validation = self._validate_household_attributes(household, step)
            
            # Accumulate validation results
            if household_validation["is_prosumer"]:
                step_validation["prosumers"] += 1
            
            step_validation["validation_errors"].extend(household_validation["errors"])
            step_validation["validation_warnings"].extend(household_validation["warnings"])
        
        # Calculate adoption rate
        if step_validation["total_households"] > 0:
            step_validation["adoption_rate"] = step_validation["prosumers"] / step_validation["total_households"]
        
        # Store step validation
        self.step_validations[step] = step_validation
        
        # Log critical errors
        if step_validation["validation_errors"]:
            print(f"⚠️  Step {step}: {len(step_validation['validation_errors'])} validation errors detected")
    
    def _validate_household_attributes(self, household, step):
        """
        Validate household attributes using already-calculated values.
        NO recalculation - just validates existing attributes.
        
        Args:
            household: Household agent
            step: Current step
            
        Returns:
            dict: Validation results
        """
        validation = {
            "household_id": household.unique_id,
            "step": step,
            "is_prosumer": getattr(household, "is_prosumer", False),
            "errors": [],
            "warnings": []
        }
        
        # Basic attribute validation
        daily_consumption = getattr(household, "daily_consumption", 0)
        if not (self.thresholds["consumption_range"][0] <= daily_consumption <= self.thresholds["consumption_range"][1]):
            validation["errors"].append(f"Household {household.unique_id}: Invalid daily consumption: {daily_consumption:.2f} kWh")
        
        # Economic validation (using pre-calculated attributes)
        npv = getattr(household, "npv", None)
        if npv is not None:
            if abs(npv) > self.thresholds["npv_magnitude"]:
                validation["warnings"].append(f"Household {household.unique_id}: Large NPV magnitude: ${npv:.0f}")
        
        annual_savings = getattr(household, "annual_savings", 0)
        baseline_cost = getattr(household, "baseline_cost", 0)
        if baseline_cost > 0:
            savings_rate = annual_savings / (baseline_cost * 12)  # Annual baseline cost
            if savings_rate > self.thresholds["savings_rate"]:
                validation["warnings"].append(f"Household {household.unique_id}: High savings rate: {savings_rate:.1%}")
        
        payback_period = getattr(household, "payback_period", float('inf'))
        if payback_period > self.thresholds["payback_period"] and payback_period != float('inf'):
            validation["warnings"].append(f"Household {household.unique_id}: Long payback period: {payback_period:.1f} years")
        
        # Prosumer-specific validation
        if household.is_prosumer:
            solar_capacity = getattr(household, "solar_capacity", 0)
            monthly_generation = getattr(household, "monthly_generation", 0)
            
            # Validate generation efficiency
            if solar_capacity > 0:
                from ..utils.parameters import SOLAR_PRODUCTION_RATIO
                max_monthly_generation = solar_capacity * SOLAR_PRODUCTION_RATIO * 30  # Rough monthly estimate
                if monthly_generation > max_monthly_generation * 1.5:  # Allow 50% variance
                    validation["errors"].append(f"Household {household.unique_id}: Excessive generation: {monthly_generation:.1f} kWh/month")
        
        # Energy balance validation
        monthly_consumption = getattr(household, "monthly_consumption", 0)
        expected_monthly = daily_consumption * 30
        if abs(monthly_consumption - expected_monthly) > expected_monthly * 0.5:  # Allow 50% variance
            validation["warnings"].append(f"Household {household.unique_id}: Monthly consumption variance: {monthly_consumption:.1f} vs expected {expected_monthly:.1f}")
        
        return validation
    
    def validate_system_consistency(self, step):
        """
        Validate system-level consistency.
        SIMPLIFIED to use model attributes instead of recalculating.
        
        Args:
            step: Current simulation step
            
        Returns:
            dict: System validation results
        """
        if not self.enabled:
            return {"errors": [], "warnings": []}
        
        system_validation = {
            "step": step,
            "errors": [],
            "warnings": []
        }
        
        # Get system metrics from central provider
        total_consumption = getattr(self.model.central_provider, "monthly_consumption", 0)
        total_generation = getattr(self.model.central_provider, "monthly_generation", 0)
        total_grid_consumption = getattr(self.model.central_provider, "monthly_grid_consumption", 0)
        
        # Energy balance validation
        if total_consumption > 0:
            generation_ratio = total_generation / total_consumption
            if generation_ratio > 1.5:  # Generation shouldn't exceed 150% of consumption
                system_validation["warnings"].append(f"High generation ratio: {generation_ratio:.2f}")
        
        # Grid dependency validation
        if total_consumption > 0:
            grid_dependency = total_grid_consumption / total_consumption
            if grid_dependency > 1.1:  # Grid consumption shouldn't exceed 110% of total consumption
                system_validation["errors"].append(f"Grid dependency exceeds consumption: {grid_dependency:.2f}")
        
        # Adoption rate validation
        households = [agent for agent in self.model.schedule.agents if hasattr(agent, "is_prosumer")]
        if households:
            adoption_rate = sum(1 for h in households if h.is_prosumer) / len(households)
            if adoption_rate > 0.95:  # Unlikely to have >95% adoption
                system_validation["warnings"].append(f"Very high adoption rate: {adoption_rate:.1%}")
        
        return system_validation
    
    def print_validation_results(self):
        """
        Print validation results summary.
        """
        if not self.enabled:
            print("Validation disabled")
            return
        
        if not self.step_validations:
            print("No validation data collected")
            return
        
        total_errors = sum(len(v["validation_errors"]) for v in self.step_validations.values())
        total_warnings = sum(len(v["validation_warnings"]) for v in self.step_validations.values())
        
        print(f"\n=== VALIDATION SUMMARY ===")
        print(f"Steps validated: {len(self.step_validations)}")
        print(f"Total errors: {total_errors}")
        print(f"Total warnings: {total_warnings}")
        
        if total_errors > 0:
            print(f"\n⚠️  {total_errors} VALIDATION ERRORS FOUND")
            
            # Show most recent errors
            latest_step = max(self.step_validations.keys())
            latest_errors = self.step_validations[latest_step]["validation_errors"]
            if latest_errors:
                print(f"Latest errors (step {latest_step}):")
                for error in latest_errors[:5]:  # Show first 5 errors
                    print(f"  - {error}")
        
        if total_warnings > 0:
            print(f"\n⚠️  {total_warnings} VALIDATION WARNINGS")
        
        if total_errors == 0 and total_warnings == 0:
            print("✅ All validations passed!")
    
    def create_validation_report(self, output_dir="results/validation"):
        """
        Create detailed validation report.
        
        Args:
            output_dir: Directory to save validation report
        """
        if not self.enabled or not self.step_validations:
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Create validation summary DataFrame
        validation_summary = []
        for step, validation in self.step_validations.items():
            validation_summary.append({
                "Step": step,
                "TotalHouseholds": validation["total_households"],
                "Prosumers": validation["prosumers"],
                "AdoptionRate": validation["adoption_rate"],
                "ErrorCount": len(validation["validation_errors"]),
                "WarningCount": len(validation["validation_warnings"])
            })
        
        if validation_summary:
            summary_df = pd.DataFrame(validation_summary)
            summary_df.to_csv(f"{output_dir}/validation_summary.csv", index=False)
        
        # Create detailed error log
        all_errors = []
        all_warnings = []
        
        for step, validation in self.step_validations.items():
            for error in validation["validation_errors"]:
                all_errors.append({"Step": step, "Type": "Error", "Message": error})
            for warning in validation["validation_warnings"]:
                all_warnings.append({"Step": step, "Type": "Warning", "Message": warning})
        
        if all_errors:
            errors_df = pd.DataFrame(all_errors)
            errors_df.to_csv(f"{output_dir}/validation_errors.csv", index=False)
        
        if all_warnings:
            warnings_df = pd.DataFrame(all_warnings)
            warnings_df.to_csv(f"{output_dir}/validation_warnings.csv", index=False)
        
        print(f"Validation report saved to {output_dir}")
    
    def get_validation_summary(self):
        """
        Get validation summary for debugging.
        
        Returns:
            dict: Validation summary
        """
        if not self.enabled:
            return {"validation_disabled": True}
        
        if not self.step_validations:
            return {"no_data": True}
        
        total_errors = sum(len(v["validation_errors"]) for v in self.step_validations.values())
        total_warnings = sum(len(v["validation_warnings"]) for v in self.step_validations.values())
        
        latest_step = max(self.step_validations.keys())
        latest_validation = self.step_validations[latest_step]
        
        return {
            "steps_validated": len(self.step_validations),
            "total_errors": total_errors,
            "total_warnings": total_warnings,
            "latest_step": latest_step,
            "latest_adoption_rate": latest_validation["adoption_rate"],
            "latest_prosumers": latest_validation["prosumers"],
            "latest_households": latest_validation["total_households"],
            "validation_enabled": True
        }