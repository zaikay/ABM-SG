# utils/config_loader.py V2

"""
Utilities for loading and managing simulation configurations.
"""
import json
import os
import copy
from .parameters import *

class SimulationConfig:
    """
    Configuration manager for simulation experiments.
    """
    def __init__(self, experiment_name="rational_baseline"):
        """
        Initialize a configuration with default parameters.
        
        Args:
            experiment_name: Name of the experiment
        """
        self.experiment_name = experiment_name
        
        # Build default configuration
        self.config = {
            "experiment_name": experiment_name,
            "num_households": NUM_HOUSEHOLDS,
            "neighbors_per_household": NEIGHBORS_PER_HOUSEHOLD,
            "steps": TOTAL_STEPS,
            "income_params": {
                "lognormal_mean": INCOME_LOGNORMAL_MEAN,
                "lognormal_sd": INCOME_LOGNORMAL_SD,
                "num_quintiles": INCOME_QUINTILES
            },
            "consumption_params": {
                "income_elasticity": CONSUMPTION_INCOME_ELASTICITY,
                "variation_sd": CONSUMPTION_VARIATION_SD,
                "base_consumption": BASE_CONSUMPTION
            },
            "solar_params": {
                "sizing_factor": SOLAR_SIZING_FACTOR,
                "lifetime_years": SOLAR_LIFETIME_YEARS,
                "discount_rate": DISCOUNT_RATE,
                "base_cost": BASE_SOLAR_COST,
                "annual_cost_reduction": SOLAR_COST_REDUCTION_ANNUAL,
                "solar_production_ratio": SOLAR_PRODUCTION_RATIO
            },
            "grid_params": {
                "feed_in_factor": FEED_IN_FACTOR,
                "initial_fossil_price": INITIAL_FOSSIL_PRICE,
                "fossil_annual_increase": FOSSIL_ANNUAL_INCREASE,
                "initial_renewable_price": INITIAL_RENEWABLE_PRICE,
                "renewable_annual_decrease": RENEWABLE_ANNUAL_DECREASE
            },
            "weather_params": {
                "reference_heating_temp": REFERENCE_HEATING_TEMP,
                "comfort_cooling_temp": COMFORT_COOLING_TEMP,
                "heating_sensitivity": HEATING_SENSITIVITY,
                "cooling_sensitivity": COOLING_SENSITIVITY
            },
            "run_settings": {
                "random_seed": 42,
                "collect_data": True,
                "data_collection_interval": 1  # Every month
            }
        }
    
    def update(self, updates):
        """
        Update configuration with new values.
        
        Args:
            updates: Dictionary with configuration updates
        """
        def update_dict(original, updates):
            for key, value in updates.items():
                if key in original and isinstance(value, dict) and isinstance(original[key], dict):
                    update_dict(original[key], value)
                else:
                    original[key] = value
        
        update_dict(self.config, updates)
    
    def save(self, filepath=None):
        """
        Save configuration to a JSON file.
        
        Args:
            filepath: Path to save configuration (default: results/configs/{experiment_name}.json)
        """
        if filepath is None:
            os.makedirs("results/configs", exist_ok=True)
            filepath = f"results/configs/{self.experiment_name}.json"
        
        with open(filepath, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def load(self, filepath):
        """
        Load configuration from a JSON file.
        
        Args:
            filepath: Path to the configuration file
        """
        with open(filepath, 'r') as f:
            loaded_config = json.load(f)
            self.config = loaded_config
            self.experiment_name = loaded_config.get("experiment_name", "loaded_experiment")
    
    def get_copy(self):
        """
        Get a deep copy of the configuration.
        
        Returns:
            dict: Copy of the configuration dictionary
        """
        return copy.deepcopy(self.config)


def create_experiment_config(experiment_name, overrides=None):
    """
    Create a configuration for an experiment.
    
    Args:
        experiment_name: Name of the experiment
        overrides: Dictionary with configuration overrides
        
    Returns:
        SimulationConfig: Configuration object
    """
    config = SimulationConfig(experiment_name)
    if overrides:
        config.update(overrides)
    return config


def create_rational_experiment():
    """
    Create configuration for the rational experiment with unified metrics.
    
    Returns:
        SimulationConfig: Configuration for rational experiment
    """
    config = create_experiment_config("rational_baseline")
    
    # Add unified metrics configuration
    unified_metrics_config = {
        "unified_metrics": {
            "enable_unified_metrics": ENABLE_UNIFIED_METRICS,
            "metrics_granularity": METRICS_GRANULARITY,
            "peak_load_method": PEAK_LOAD_CALCULATION_METHOD,
            "track_credit_utilization": TRACK_CREDIT_UTILIZATION,
            "track_seasonal_stress": TRACK_SEASONAL_STRESS,
            "stress_weights": {
                "grid_dependency_weight": STRESS_GRID_DEPENDENCY_WEIGHT,
                "peak_avg_ratio_weight": STRESS_PEAK_AVG_RATIO_WEIGHT
            }
        }
    }
    
    config.update(unified_metrics_config)
    return config

# =============================================================================
# NEW: BEHAVIORAL EXPERIMENT CONFIGURATIONS
# =============================================================================

# Fix 1: create_multi_experiment_config function
def create_multi_experiment_config():
    """
    Create configuration for multi-experiment behavioral study.
    
    This configuration enables all scenarios (rational + behavioral) to run
    in a single simulation for direct comparison.
    
    Returns:
        SimulationConfig: Configuration for multi-experiment behavioral study
    """
    # Start with rational baseline
    base_config = create_rational_experiment().get_copy()
    
    # Add behavioral configuration
    behavioral_config = {
        "experiment_type": "multi_scenario_behavioral",
        "scenarios": get_all_scenarios(),
        
        # Behavioral bias parameters (extracted from BEHAVIORAL_BIASES)
        "behavioral_params": {
            bias_name: BEHAVIORAL_BIASES[bias_name]['parameters']
            for bias_name in get_enabled_biases()
        },
        
        # Behavioral bias metadata (for documentation and analysis)
        "bias_metadata": {
            bias_name: {
                'display_name': BEHAVIORAL_BIASES[bias_name]['display_name'],
                'description': BEHAVIORAL_BIASES[bias_name]['description'],
                'literature_source': BEHAVIORAL_BIASES[bias_name]['literature_source'],
                'formula': BEHAVIORAL_BIASES[bias_name].get('formula', ''),
                'application_order': BEHAVIORAL_BIASES[bias_name]['application_order']
            }
            for bias_name in get_enabled_biases()
        },
        
        # Visualization configuration
        "visualization_params": SPATIAL_VISUALIZATION_CONFIG,
        
        # Data collection configuration
        "data_collection": {
            "track_all_scenarios": True,
            "detailed_tracking_sample_size": 5,
            "export_individual_scenarios": True,
            "export_combined_dataset": True,
            "export_scenario_metadata": True,
            "collect_spatial_data": True,  # For network visualizations
            "collect_bias_effects": True   # For bias impact analysis
        },
        
        # Multi-experiment specific settings
        "multi_experiment": {
            "run_all_scenarios_simultaneously": True,
            "maintain_same_random_conditions": True,  # Same weather, income, etc.
            "export_comparative_analysis": True,
            "create_spatial_visualizations": True
        }
    }
    
    # Update base configuration
    base_config.update(behavioral_config)
    
    # FIXED: Create SimulationConfig with correct parameters
    config = SimulationConfig("multi_experiment_behavioral")
    config.config = base_config
    return base_config


# Fix 2: create_single_bias_experiment function
def create_single_bias_experiment(bias_name):
    """
    Create configuration for testing a single bias in isolation.
    
    Args:
        bias_name: Name of the bias to test (e.g., 'loss_aversion')
        
    Returns:
        SimulationConfig: Configuration for single bias experiment
    """
    if bias_name not in BEHAVIORAL_BIASES:
        raise ValueError(f"Unknown bias: {bias_name}. Available: {list(BEHAVIORAL_BIASES.keys())}")
    
    if not BEHAVIORAL_BIASES[bias_name].get('enabled', False):
        raise ValueError(f"Bias '{bias_name}' is not enabled in configuration")
    
    # Start with rational baseline
    base_config = create_rational_experiment().get_copy()
    
    # Configure for single bias testing
    single_bias_config = {
        "experiment_type": "single_bias_behavioral",
        "target_bias": bias_name,
        "scenarios": ['rational', bias_name],  # Only rational and this bias
        
        # Only this bias's parameters
        "behavioral_params": {
            bias_name: BEHAVIORAL_BIASES[bias_name]['parameters']
        },
        
        # Metadata for this bias
        "bias_metadata": {
            bias_name: {
                'display_name': BEHAVIORAL_BIASES[bias_name]['display_name'],
                'description': BEHAVIORAL_BIASES[bias_name]['description'],
                'literature_source': BEHAVIORAL_BIASES[bias_name]['literature_source'],
                'formula': BEHAVIORAL_BIASES[bias_name].get('formula', ''),
                'application_order': BEHAVIORAL_BIASES[bias_name]['application_order']
            }
        },
        
        # Simplified data collection
        "data_collection": {
            "track_all_scenarios": True,
            "detailed_tracking_sample_size": 10,  # More detailed for single bias
            "export_individual_scenarios": True,
            "export_combined_dataset": True,
            "focus_on_bias_effects": True
        }
    }
    
    base_config.update(single_bias_config)
    
    # FIXED: Create SimulationConfig with correct parameters
    #experiment_name = f"single_bias_{bias_name}"
    #config = SimulationConfig(experiment_name)
    #config.config = base_config
    return base_config


# Fix 3: create_testing_config function
def create_testing_config(num_households=50, steps=12):
    """
    Create a lightweight configuration for testing purposes.
    
    Args:
        num_households: Number of households for testing (default: 50)
        steps: Number of simulation steps (default: 12 = 1 year)
        
    Returns:
        SimulationConfig: Configuration for testing
    """
    # FIXED: Don't call create_multi_experiment_config() directly
    # Instead, create a base config and add behavioral features
    base_config = create_rational_experiment().get_copy()
    
    # Add behavioral configuration
    behavioral_config = {
        "experiment_type": "multi_scenario_behavioral",
        "scenarios": get_all_scenarios(),
        "behavioral_params": {
            bias_name: BEHAVIORAL_BIASES[bias_name]['parameters']
            for bias_name in get_enabled_biases()
        },
        "bias_metadata": {
            bias_name: {
                'display_name': BEHAVIORAL_BIASES[bias_name]['display_name'],
                'description': BEHAVIORAL_BIASES[bias_name]['description'],
                'literature_source': BEHAVIORAL_BIASES[bias_name]['literature_source'],
                'formula': BEHAVIORAL_BIASES[bias_name].get('formula', ''),
                'application_order': BEHAVIORAL_BIASES[bias_name]['application_order']
            }
            for bias_name in get_enabled_biases()
        },
        "visualization_params": SPATIAL_VISUALIZATION_CONFIG,
        "data_collection": {
            "track_all_scenarios": True,
            "detailed_tracking_sample_size": min(5, num_households // 10),
            "export_individual_scenarios": False,  # Skip for testing
            "export_combined_dataset": True,
            "export_scenario_metadata": True
        }
    }
    
    # Override for fast testing
    testing_overrides = {
        "experiment_name": "testing_behavioral",
        "num_households": num_households,
        "steps": steps,
        "run_settings": {
            "random_seed": 12345,
            "collect_data": True,
            "data_collection_interval": 1
        },
        "visualization_params": {
            **SPATIAL_VISUALIZATION_CONFIG,
            "sample_size": min(25, num_households)  # Smaller sample for testing
        }
    }
    
    # Combine all configurations
    base_config.update(behavioral_config)
    base_config.update(testing_overrides)
    
    # FIXED: Create SimulationConfig correctly
    config = SimulationConfig("testing_behavioral")
    config.config = base_config
    return config


# Fix 4: create_sensitivity_analysis_config function
def create_sensitivity_analysis_config():
    """
    Create configuration for sensitivity analysis of behavioral parameters.
    
    Returns:
        SimulationConfig: Configuration for sensitivity analysis
    """
    # Start with multi-experiment config
    base_config = create_rational_experiment().get_copy()
    
    # Add behavioral configuration
    behavioral_config = {
        "experiment_type": "multi_scenario_behavioral",
        "scenarios": get_all_scenarios(),
        "behavioral_params": {
            bias_name: BEHAVIORAL_BIASES[bias_name]['parameters']
            for bias_name in get_enabled_biases()
        }
    }
    
    # Sensitivity analysis specific settings
    sensitivity_config = {
        "experiment_type": "sensitivity_analysis",
        "sensitivity_analysis": {
            "method": "morris",  # Morris elementary effects screening
            "parameters_to_vary": [
                "loss_aversion.baseline_coefficient",
                "loss_aversion.income_sensitivity", 
                "present_bias.beta_min",
                "present_bias.beta_max",
                "status_quo.baseline_strength",
                "herding.spatial_beta_shape_a",
                "herding.class_beta_shape_a"
            ],
            "parameter_ranges": {
                "loss_aversion.baseline_coefficient": [1.5, 3.0],
                "loss_aversion.income_sensitivity": [0.1, 0.4],
                "present_bias.beta_min": [0.5, 0.7],
                "present_bias.beta_max": [0.7, 0.9],
                "status_quo.baseline_strength": [0.6, 0.9],  # CORRECTED: Updated for new range
                "herding.spatial_beta_shape_a": [2.0, 4.0],
                "herding.class_beta_shape_a": [1.5, 3.0],
                "herding.bandwagon_beta_shape_a": [1.0, 2.0]  # NEW: Added bandwagon parameter
            },
            "num_trajectories": 20,
            "output_metrics": [
                "final_adoption_rate",
                "adoption_by_income_class",
                "fossil_dependency_reduction",
                "peak_load_reduction"
            ]
        },
        "run_settings": {
            "random_seed": None,  # Will be varied for sensitivity analysis
            "collect_data": True
        }
    }
    
    # Combine configurations
    base_config.update(behavioral_config)
    base_config.update(sensitivity_config)
    
    # FIXED: Create SimulationConfig correctly
    config = SimulationConfig("sensitivity_analysis")
    config.config = base_config
    return config

# Add this function to utils/config_loader.py (around line 400)

def validate_experiment_config(config):
    """
    Validate experiment configuration for completeness and consistency.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        tuple: (is_valid, errors, warnings)
    """
    errors = []
    warnings = []
    
    # Check required fields
    required_fields = [
        "experiment_name", "num_households", "steps",
        "income_params", "consumption_params", "solar_params", "grid_params"
    ]
    
    for field in required_fields:
        if field not in config:
            errors.append(f"Missing required field: {field}")
    
    # Check household count
    if config.get("num_households", 0) < 1:
        errors.append("num_households must be at least 1")
    elif config.get("num_households", 0) < 10:
        warnings.append("Very low household count may not be representative")
    elif config.get("num_households", 0) > 10000:
        warnings.append("Large household count may be computationally expensive")
    
    # Check simulation steps
    if config.get("steps", 0) < 1:
        errors.append("steps must be at least 1")
    elif config.get("steps", 0) < 12:
        warnings.append("Less than 1 year of simulation may not show adoption patterns")
    elif config.get("steps", 0) > 600:  # 50 years
        warnings.append("Very long simulation may be computationally expensive")
    
    # Check behavioral configuration if present
    if config.get("experiment_type") in ["multi_scenario_behavioral", "single_bias_behavioral"]:
        if "behavioral_params" not in config:
            errors.append("Behavioral experiment missing behavioral_params")
        
        if "scenarios" not in config:
            errors.append("Behavioral experiment missing scenarios definition")
        
        # Check that enabled biases have parameters
        scenarios = config.get("scenarios", [])
        behavioral_params = config.get("behavioral_params", {})
        
        for scenario in scenarios:
            if scenario not in ['rational', 'all_biases'] and scenario not in behavioral_params:
                errors.append(f"Scenario '{scenario}' missing behavioral parameters")
    
    # Check visualization parameters
    if "visualization_params" in config:
        viz_params = config["visualization_params"]
        sample_size = viz_params.get("sample_size", 0)
        total_households = config.get("num_households", 0)
        
        if sample_size > total_households:
            errors.append(f"Visualization sample size ({sample_size}) exceeds total households ({total_households})")
        elif sample_size < 1 and total_households > 0:
            warnings.append("Very small visualization sample size may not be representative")
    
    # Check data collection settings
    if "data_collection" in config:
        data_config = config["data_collection"]
        if data_config.get("detailed_tracking_sample_size", 0) > config.get("num_households", 0):
            errors.append("Detailed tracking sample size exceeds total households")
    
    is_valid = len(errors) == 0
    return is_valid, errors, warnings