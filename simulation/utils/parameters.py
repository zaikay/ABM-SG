# utils/parameters.py V5.1 - PHASE 3 COMPLETE BEHAVIORAL CONFIGURATION
"""
Enhanced configuration parameters for the behavioral prosumer simulation model.
Includes comprehensive behavioral biases configuration with literature-grounded parameters.
"""

# Simulation parameters
MONTHS_IN_YEAR = 12
YEARS_TO_SIMULATE = 20
TOTAL_STEPS = YEARS_TO_SIMULATE * MONTHS_IN_YEAR

# Population parameters
NUM_HOUSEHOLDS = 5000
NEIGHBORS_PER_HOUSEHOLD = 10

# Income parameters - ENHANCED for better heterogeneity
INCOME_LOGNORMAL_MEAN = 10.8   # Median ~$49K (was 10.46 for ~$35K)
INCOME_LOGNORMAL_SD = 0.8      # Wider spread (was 0.62)
INCOME_QUINTILES = 5

# Consumption parameters - ENHANCED for stronger income relationship
CONSUMPTION_INCOME_ELASTICITY = 0.5  # Stronger relationship (was 0.3)
CONSUMPTION_VARIATION_SD = 0.25      # More individual variation (was 0.15)  
BASE_CONSUMPTION = 15.0              # Lower baseline favors high income (was 20.0)

# Solar parameters - CORRECTED to match manuscript
SOLAR_SIZING_FACTOR = 1.2            # α_solar from manuscript (was 1.0)
SOLAR_PRODUCTION_RATIO = 4.5         # 
SOLAR_LIFETIME_YEARS = 20            # Restore standard (was 20)
DISCOUNT_RATE = 0.04                 # r for NPV calculations

# Grid parameters - ALIGNED with manuscript
FEED_IN_FACTOR = 0                 # α_feed from manuscript (was 0.7)

# Solar costs - REALISTIC current values
BASE_SOLAR_COST = 3000               # Realistic 2024 cost (was 2500)
SOLAR_COST_REDUCTION_ANNUAL = 0.05   # 

# Energy prices
INITIAL_FOSSIL_PRICE = 0.15
FOSSIL_ANNUAL_INCREASE = 0.02
INITIAL_RENEWABLE_PRICE = 0.15
RENEWABLE_ANNUAL_DECREASE = 0.03      

ANNUAL_MAINTENANCE_PERCENTAGE = 0     # 1% of initial installation cost per year
ANNUAL_DEGRADATION_RATE = 0.005       # 0.5% reduction in generation per year

# Hourly profiles
HOURLY_CONSUMPTION_PROFILE = [
    0.02, 0.015, 0.01, 0.01, 0.015, 0.02,     # 00:00-06:00
    0.04, 0.06, 0.05, 0.04, 0.035, 0.04,      # 06:00-12:00
    0.045, 0.04, 0.035, 0.04, 0.06, 0.08,     # 12:00-18:00
    0.09, 0.08, 0.06, 0.05, 0.04, 0.03        # 18:00-00:00
]

HOURLY_SOLAR_PROFILE = [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.01,            # 00:00-06:00
    0.03, 0.07, 0.1, 0.12, 0.13, 0.14,        # 06:00-12:00
    0.13, 0.12, 0.1, 0.07, 0.03, 0.01,        # 12:00-18:00
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0              # 18:00-00:00
]

# Temperature parameters
REFERENCE_HEATING_TEMP = 18.0  # Tref in °C
COMFORT_COOLING_TEMP = 22.0    # Tcomfort in °C
HEATING_SENSITIVITY = 0.04     # αheat
COOLING_SENSITIVITY = 0.03     # αcool

# Day type variations (Table 4 in manuscript)
DAY_TYPE_VARIATIONS = {
    "sunny_weekday": {"consumption": 0.95, "generation": 1.15},
    "sunny_weekend": {"consumption": 1.10, "generation": 1.15},
    "mixed_weekday": {"consumption": 1.00, "generation": 1.00},
    "mixed_weekend": {"consumption": 1.15, "generation": 1.00},
    "cloudy_weekday": {"consumption": 1.05, "generation": 0.70},
    "cloudy_weekend": {"consumption": 1.20, "generation": 0.70}
}

# NPV sigmoid parameters
NPV_SIGMOID_STEEPNESS = 0.01  # κ for adoption probability sigmoid

# Economies of scale parameters (NREL/DOE methodology)
SOLAR_FIXED_COSTS = 5500      # $ per installation
SOLAR_VARIABLE_COST_PER_KW = 2000  # $ per kW

# Metrics calculation configuration
ENABLE_UNIFIED_METRICS = True  # Enable unified metrics calculation
METRICS_GRANULARITY = "detailed"  # Options: "basic", "detailed", "full"

# Peak load calculation settings
PEAK_LOAD_CALCULATION_METHOD = "exact"  # Options: "exact", "estimated"

# Credit system analysis settings
TRACK_CREDIT_UTILIZATION = True  # Track credit earning/usage patterns
TRACK_SEASONAL_STRESS = True     # Track seasonal grid stress patterns

# Stress index calculation weights
STRESS_GRID_DEPENDENCY_WEIGHT = 0.6    # Weight for grid dependency in stress calculation
STRESS_PEAK_AVG_RATIO_WEIGHT = 0.4     # Weight for peak-to-average ratio

# =============================================================================
# BEHAVIORAL BIASES CONFIGURATION - SINGLE SOURCE OF TRUTH
# =============================================================================

BEHAVIORAL_BIASES = {
    'loss_aversion': {
        'enabled': True,
        'display_name': 'Loss Aversion',
        'description': 'Losses weighted more heavily than gains (Prospect Theory)',
        'literature_source': 'Kahneman & Tversky (1979), Gächter et al. (2022)',
        'parameters': {
            'baseline_coefficient': 2.25,
            'variation_std': 0.1,           
            'min_coefficient': 1.5,         
            'max_coefficient': 3.0  
        },
        'application_order': 2,  # Stage 1: Individual utility transformation
        'formula': 'NPV_adjusted = NPV - (λ_i - 1) × C_install'
    },
    'present_bias': {
        'enabled': True,
        'display_name': 'Present Bias',
        'description': 'Disproportionate valuation of immediate vs future rewards',
        'literature_source': 'Laibson (1997), Newell & Siikamäki (2015), Heutel (2019)',
        'parameters': {
            'beta_min': 0.6,                # Minimum β_i value
            'beta_max': 0.7,                # Maximum β_i value
            'individual_variation': True,    # Whether to add individual variation
            'discount_all_future': True,     # Apply to all future periods
            'variation_std': 0.05           # Standard deviation for individual variation
        },
        'application_order': 3,
        'formula': 'NPV_adjusted = NPV - (1-β_i) × (NPV + C_install)'
    },
    'status_quo': {
        'enabled': True,
        'display_name': 'Status Quo Bias',
        'description': 'Switching cost representing inertia and resistance to change',
        'literature_source': 'Samuelson & Zeckhauser (1988), Devine & Waddell (2023), Gerarden et al. (2017)',
        'parameters': {
            'baseline_strength': 0.60,       # 30% switching cost (literature mean)
            'individual_variation': 0.08,    # Individual heterogeneity (std dev)
            'min_strength': 0.30,           # 20% minimum hurdle (literature range)
            'max_strength': 0.90,           # 50% maximum hurdle (literature range)
        },
        'application_order': 4,
        'formula': 'NPV_adjusted = NPV - σᵢ × C_install (σᵢ ∈ [0.20, 0.50])'
    },
    'herding': {
        'enabled': True,
        'display_name': 'herding Bias',
        'description': 'Social influence from neighbors and income class',
        'literature_source': 'Bollinger & Gillingham (2012), Palm (2017), Graziano & Gillingham',
        'parameters': {
            'spatial_weight': 0.75,                    # Spatial dominance (literature)
            'class_weight': 0.25,                      # Class homophily (secondary)
            'variation_std': 0.1, # Individual susceptibility distribution
            # Calibration parameters
            'target_effect_per_neighbor': 0.1,       # 7.8 pp per neighbor (scaled from B&G)
            'max_total_effect': 0.30,                  # 15 pp maximum (safety cap)
            'spatial_beta_shape_a': 2.5,    # Beta distribution for spatial influence TODELETE
            'spatial_beta_shape_b': 5.0,    # β_i ~ Beta(2.5, 5) TODELETE
            'class_beta_shape_a': 2.0,      # Beta distribution for class influence TODELETE
            'class_beta_shape_b': 4.5,      # γ_i ~ Beta(2, 4.5) TODELETE
            # Network parameters  
            'distance_normalization': 1.0,  # d_0 for distance weighting TOCHECK
            'max_neighbors_considered': 10,
            'spatial_radius': 5.0,
        },
        'application_order': 5,  # Stage 2: Social utility integration
        'formula': 'f_herd(P) = P + ρ_combined'
    },
    'optimism_bias': {
        'enabled': True,
        'display_name': 'Optimism Bias',
        'description': 'Symmetric optimism affecting both benefits and costs equally - Symmetric effects to avoid confounding with loss aversion',
        'literature_source': 'Planning fallacy (Kahneman & Tversky 1979), Energy efficiency programs (Allcott & Greenstone 2017)',
        'parameters': {
        'base_optimism': 0.1,  # 32%/2 for dual application
        'individual_variation': 0.075,
        'effect_variation_min': 0.05,
        'effect_variation_max': 0.20,
        },
        'application_order': 1,
        'formula': 'NPV_opt = (1 + ω_i) x NPV + Costs × 2 x ω_i'
    }
}

# =============================================================================
# DERIVED CONFIGURATIONS (automatically generated)
# =============================================================================

def get_enabled_biases():
    """Get list of enabled bias names in application order"""
    enabled = [(bias_name, config) for bias_name, config in BEHAVIORAL_BIASES.items() 
               if config.get('enabled', False)]
    # Sort by application order
    enabled.sort(key=lambda x: x[1].get('application_order', 999))
    return [bias_name for bias_name, _ in enabled]

def get_all_scenarios():
    """Get all available scenarios including both rational variants"""
    enabled_biases = get_enabled_biases()
    scenarios = [
        'deterministic_rational',    # NEW: Pure NPV > 0 logic
        'rational',                  # RENAMED: Non-deterministic (sigmoid)
        *enabled_biases,            # Individual biases
        'all_biases'                # Combined biases
    ]
    return scenarios

def get_scenario_colors():
    """Generate consistent colors for scenarios"""
    scenarios = get_all_scenarios()
    # Professional color palette for publications
    color_palette = [
        '#000080',  # Dark Blue - Deterministic Rational
        '#1f77b4',  # Light Blue - Non-Deterministic Rational
        '#ff7f0e',  # Orange - Loss Aversion
        '#2ca02c',  # Green - Present Bias
        '#d62728',  # Red - Status Quo
        '#9467bd',  # Purple - Herding
        '#8c564b',  # Brown - Optimism Bias
        '#000000'   # Black - All Biases Combined
    ]
    return {scenario: color_palette[i % len(color_palette)] 
            for i, scenario in enumerate(scenarios)}

def get_scenario_metadata():
    """Get metadata for all scenarios including literature sources"""
    scenarios = get_all_scenarios()
    metadata = {}
    
    # Rational baseline
    metadata['deterministic_rational'] = {
        'display_name': 'Deterministic rational',
        'description': 'Pure economic decision-making (NPV > 0)',
        'literature_source': 'Standard economic theory',
        'color': get_scenario_colors()['rational'],
        'formula': 'Adopt if NPV > 0'
    }
    metadata['rational']={
        'display_name': 'Non-Deterministic Rational', 
        'description': 'Economic decision with uncertainty (sigmoid)',
        'literature_source': 'Standard economic economics',
        'color':get_scenario_colors()['deterministic_rational'],
        'formula': 'P(adopt) = 1/(1 + e^(-κ×NPV))'
    }

    # Individual bias scenarios
    for bias_name in get_enabled_biases():
        if bias_name in BEHAVIORAL_BIASES:
            bias_config = BEHAVIORAL_BIASES[bias_name]
            metadata[bias_name] = {
                'display_name': bias_config['display_name'],
                'description': bias_config['description'],
                'literature_source': bias_config['literature_source'],
                'formula': bias_config.get('formula', ''),
                'color': get_scenario_colors()[bias_name]
            }
    
    # Combined scenario
    metadata['all_biases'] = {
        'display_name': 'All Biases Combined',
        'description': 'Sequential application of all enabled biases',
        'literature_source': 'Integrated behavioral framework',
        'color': get_scenario_colors()['all_biases'],
        'formula': 'Sequential application of all bias transformations'
    }
    
    return metadata

# =============================================================================
# VISUALIZATION CONFIGURATION
# =============================================================================

SPATIAL_VISUALIZATION_CONFIG = {
    'sample_size': 250,  # Households to show in spatial plots
    'binary_colors': {
        'prosumer': '#2ca02c',      # Green
        'nonprosumer': '#1f77b4',   # Blue
        'background': '#f8f9fa',    # Light background
        'grid': '#dee2e6'           # Light grid
    },
    'grid_layouts': {
        '6x2': {'timepoints': [5, 20], 'figure_size': (12, 18)},
        '6x4': {'timepoints': [1, 5, 10, 20], 'figure_size': (16, 18)}
    }
}

# =============================================================================
# EVALUATION TRIGGER CONFIGURATION
# =============================================================================

EVALUATION_TRIGGER_CONFIG = {
    'initial_evaluation_delay': 12,        # No evaluation before step 13 (build consumption history)
    'evaluation_cycle_months': 6,          # Semi-annual evaluation cycles
    'economic_trigger_threshold': 0.20,    # 20% NPV change triggers re-evaluation
    'max_evaluations_per_year': 2,         # Maximum evaluations per household per year
    'random_assignment_seed_base': 42,     # Base seed for evaluation month assignment
    'life_event_probability': 0.005,      # 0.5% monthly chance of life event trigger
    'enable_social_triggers': False,       # Future extension for social triggers
    'debug_evaluation_triggers': True,      # Print debug info for trigger events
    'apply_to_rational_scenarios': True    # NEW: Apply triggers to rational scenarios too
}

def get_evaluation_trigger_config():
    """Get evaluation trigger configuration."""
    return EVALUATION_TRIGGER_CONFIG.copy()


# =============================================================================
# SPATIAL ANALYSIS CONFIGURATION - ADD TO parameters.py
# =============================================================================

# Enhanced Spatial Analyzer Parameters
SPATIAL_ANALYSIS_PARAMS = {
    # Statistical Analysis Parameters
    'statistical_tests': {
        'significance_level': 0.05,              # Alpha for statistical tests (p-value threshold)
        'confidence_level': 0.95,                # Confidence level for intervals
        'bootstrap_samples': 1000,               # Bootstrap samples for confidence intervals
        'min_sample_size': 10,                   # Minimum sample size for statistical tests
    },
    
    # Spatial Autocorrelation (Moran's I) Parameters
    'spatial_autocorrelation': {
        'max_distance_percentile': 75,           # Use 75th percentile of distances as cutoff
        'distance_decay_function': 'inverse',    # 'inverse', 'exponential', or 'linear'
        'min_neighbors_for_analysis': 3,         # Minimum neighbors required for analysis
        'weight_normalization': 'row',           # 'row', 'global', or 'none'
    },
    
    # Spatial Distance Thresholds
    'distance_thresholds': {
        'cascade_analysis': 3.0,                 # Max distance for adoption cascade membership
        'neighbor_influence': 2.5,               # Distance for neighbor influence calculation
        'clustering_analysis': 4.0,              # Distance for spatial clustering analysis
        'visualization_edges': 5.0,              # Max distance to show edges in visualizations
    },
    
    # Clustering Analysis Parameters (DBSCAN)
    'clustering': {
        'eps': 0.5,                             # DBSCAN eps parameter (neighborhood size)
        'min_samples': 5,                        # DBSCAN min_samples (minimum cluster size)
        'eps_range': [0.3, 2.0],                # Range for automatic eps optimization
        'eps_step': 0.1,                        # Step size for eps optimization
        'min_cluster_size': 10,                  # Minimum households per cluster for analysis
        'adoption_threshold': 0.3,               # Threshold for hotspot/coldspot classification
    },
    
    # Temporal Analysis Parameters
    'temporal_analysis': {
        'default_analysis_points': [12, 24, 60, 120, 180, 240],  # Years 1, 2, 5, 10, 15, 20 for 240-step simulation
        'early_years_detailed': [12, 24, 36, 48, 60],           # Detailed analysis for first 5 years
        'milestone_years': [60, 120, 180, 240],                  # Key milestone analysis points (5, 10, 15, 20 years)
        'biennial_analysis': [24, 48, 72, 96, 120, 144, 168, 192, 216, 240],  # Every 2 years
        'velocity_window': 5,                    # Time window for velocity calculations (steps)
        'cascade_time_window': 5,                # Max time between cascade adoptions (steps)
        'min_cascade_size': 3,                   # Minimum households in cascade
    },
    
    # Grid-Based Analysis Parameters
    'spatial_grid': {
        'resolution': 1.0,                       # Spatial grid resolution for velocity mapping
        'min_households_per_cell': 2,           # Minimum households per grid cell
        'smoothing_kernel': 'gaussian',          # 'gaussian', 'uniform', or 'none'
        'smoothing_bandwidth': 1.5,              # Kernel bandwidth for smoothing
    },
    
    # Network Analysis Parameters
    'network_analysis': {
        'k_neighbors': 10,                       # Number of neighbors for network analysis (uses existing neighbors_per_household)
        'max_network_size': 1000,               # Maximum network size for analysis
        'edge_weight_threshold': 0.1,           # Minimum edge weight to consider
        'centrality_measures': ['degree', 'betweenness', 'closeness'],  # Centrality measures to calculate
    },
    
    # Homophily Analysis Parameters
    'homophily_analysis': {
        'ei_index_threshold': 0.1,              # Threshold for significant homophily
        'income_class_groups': 'quintiles',     # 'quintiles', 'terciles', or 'custom'
        'custom_income_thresholds': None,       # Custom income thresholds if needed
        'chi_square_min_expected': 5,           # Minimum expected frequency for chi-square test
    },
    
    # Visualization Parameters
    'visualization': {
        'default_figure_size': (15, 10),        # Default figure size for spatial plots
        'node_size_range': [30, 100],           # Node size range for network plots
        'edge_alpha': 0.4,                      # Edge transparency
        'color_scheme': 'viridis',               # Color scheme for income classes
        'dpi': 300,                             # Resolution for saved figures
        'spatial_plot_aspect': 'equal',         # Aspect ratio for spatial plots
        'grid_alpha': 0.3,                      # Grid transparency
    },
    
    # Data Processing Parameters
    'data_processing': {
        'max_households_sample': 1000,          # Maximum households for large dataset sampling
        'missing_data_strategy': 'exclude',      # 'exclude', 'interpolate', or 'impute'
        'outlier_detection_method': 'iqr',      # 'iqr', 'zscore', or 'none'
        'outlier_threshold': 1.5,               # IQR multiplier for outlier detection
    }
}

# Literature-based parameter validation ranges
SPATIAL_ANALYSIS_VALIDATION = {
    'morans_i_range': [-1.0, 1.0],             # Theoretical Moran's I range
    'clustering_coefficient_range': [0.0, 1.0], # Clustering coefficient range
    'homophily_ei_range': [-1.0, 1.0],         # EI index theoretical range
    'cramers_v_range': [0.0, 1.0],             # Cramer's V range
    'spatial_radius_max': 20.0,                 # Maximum reasonable spatial radius
}

# =============================================================================
# HELPER FUNCTIONS FOR SPATIAL ANALYSIS PARAMETERS
# =============================================================================

def get_spatial_analysis_params():
    """
    Get spatial analysis parameters.
    
    Returns:
        dict: Spatial analysis configuration parameters
    """
    return SPATIAL_ANALYSIS_PARAMS.copy()

def get_spatial_distance_threshold(analysis_type):
    """
    Get distance threshold for specific analysis type.
    
    Args:
        analysis_type: Type of analysis ('cascade', 'neighbor', 'clustering', 'visualization')
        
    Returns:
        float: Distance threshold for the analysis type
    """
    thresholds = SPATIAL_ANALYSIS_PARAMS['distance_thresholds']
    
    threshold_map = {
        'cascade': thresholds['cascade_analysis'],
        'neighbor': thresholds['neighbor_influence'],
        'clustering': thresholds['clustering_analysis'],
        'visualization': thresholds['visualization_edges']
    }
    
    return threshold_map.get(analysis_type, thresholds['neighbor_influence'])

def get_dbscan_params():
    """
    Get DBSCAN clustering parameters.
    
    Returns:
        dict: DBSCAN parameters (eps, min_samples, etc.)
    """
    return SPATIAL_ANALYSIS_PARAMS['clustering'].copy()

def get_statistical_thresholds():
    """
    Get statistical analysis thresholds.
    
    Returns:
        dict: Statistical test parameters
    """
    return SPATIAL_ANALYSIS_PARAMS['statistical_tests'].copy()

def get_temporal_analysis_params():
    """
    Get temporal analysis parameters.
    
    Returns:
        dict: Temporal analysis configuration
    """
    return SPATIAL_ANALYSIS_PARAMS['temporal_analysis'].copy()

def validate_spatial_analysis_params():
    """
    Validate spatial analysis parameters against literature ranges.
    
    Returns:
        tuple: (errors, warnings) lists
    """
    errors = []
    warnings = []
    
    # Check statistical parameters
    stats_params = SPATIAL_ANALYSIS_PARAMS['statistical_tests']
    if not 0 < stats_params['significance_level'] < 1:
        errors.append(f"Significance level should be in (0,1), got {stats_params['significance_level']}")
    
    if stats_params['bootstrap_samples'] < 100:
        warnings.append(f"Low bootstrap samples ({stats_params['bootstrap_samples']}) may affect reliability")
    
    # Check clustering parameters
    cluster_params = SPATIAL_ANALYSIS_PARAMS['clustering']
    if cluster_params['eps'] <= 0:
        errors.append(f"DBSCAN eps should be > 0, got {cluster_params['eps']}")
    
    if cluster_params['min_samples'] < 2:
        errors.append(f"DBSCAN min_samples should be >= 2, got {cluster_params['min_samples']}")
    
    # Check distance thresholds
    distances = SPATIAL_ANALYSIS_PARAMS['distance_thresholds']
    max_reasonable = SPATIAL_ANALYSIS_VALIDATION['spatial_radius_max']
    for dist_type, dist_value in distances.items():
        if dist_value > max_reasonable:
            warnings.append(f"Large {dist_type} distance threshold: {dist_value}")
    
    # Check temporal parameters
    temporal_params = SPATIAL_ANALYSIS_PARAMS['temporal_analysis']
    if temporal_params['velocity_window'] < 1:
        errors.append("Velocity window should be >= 1")
    
    if temporal_params['min_cascade_size'] < 2:
        errors.append("Minimum cascade size should be >= 2")
    
    return errors, warnings

# Add to existing validate_configuration() function
def validate_spatial_configuration():
    """Extended validation including spatial analysis parameters."""
    # Get existing validation
    errors, warnings = validate_configuration()  # Existing function
    
    # Add spatial analysis validation
    spatial_errors, spatial_warnings = validate_spatial_analysis_params()
    errors.extend(spatial_errors)
    warnings.extend(spatial_warnings)
    
    return errors, warnings

# =============================================================================
# UPDATE TO EXISTING BEHAVIORAL_BIASES - INTEGRATE SPATIAL PARAMS
# =============================================================================

# Update herding bias to use spatial analysis parameters
def get_herding_spatial_params():
    """
    Get herding bias spatial parameters integrated with spatial analysis config.
    
    Returns:
        dict: Integrated herding and spatial analysis parameters
    """
    herding_params = BEHAVIORAL_BIASES['herding']['parameters'].copy()
    spatial_params = SPATIAL_ANALYSIS_PARAMS
    
    # Integrate spatial distance parameters
    herding_params.update({
        'neighbor_distance_threshold': get_spatial_distance_threshold('neighbor'),
        'clustering_distance_threshold': get_spatial_distance_threshold('clustering'),
        'statistical_significance_level': spatial_params['statistical_tests']['significance_level'],
    })
    
    return herding_params


# =============================================================================
# TESTING AND VALIDATION CONFIGURATION
# =============================================================================

TESTING_CONFIG = {
    'unit_tests': {
        'bias_calculation_tolerance': 1e-6,  # Numerical tolerance for bias calculations
        'probability_bounds': [0.0, 1.0],   # Valid probability range
        'npv_sanity_bounds': [-100000, 100000]  # Reasonable NPV range
    },
    'integration_tests': {
        'min_households_for_test': 10,       # Minimum households for integration tests
        'max_simulation_steps': 12,          # Steps for quick integration tests
        'expected_data_columns': [           # Required columns in output data
            'Step', 'Year', 'AgentType', 'Scenario', 'IsProsumer', 
            'Income', 'IncomeClass', 'NPV', 'InstallationCost'
        ]
    },
    'validation_checks': {
        'adoption_rate_bounds': [0.0, 1.0],  # Valid adoption rate range
        'bias_effect_bounds': [0.1, 10.0],   # Reasonable bias effect multipliers
        'energy_balance_tolerance': 0.01,    # Energy conservation tolerance
        'spatial_influence_max': 1.0,        # Maximum spatial influence
        'class_influence_max': 1.0           # Maximum class influence
    },
    'literature_validation': {
        'loss_aversion_coefficient_range': [1.5, 4.0],     # Typical λ range
        'present_bias_beta_range': [0.5, 0.9],             # Typical β range
        'status_quo_strength_range': [0.6, 0.9],           # Updated σ range (was [0.1, 0.5])
        'herding_multiplier_range': [0.5, 3.0]             # Typical influence range
    }
}

# =============================================================================
# EXPERIMENT CONFIGURATIONS
# =============================================================================

DEFAULT_EXPERIMENT_CONFIG = {
    'rational_baseline': {
        'scenarios': ['rational'],
        'description': 'Pure economic decision-making baseline',
        'duration_years': 20,
        'households': 1000
    },
    'individual_biases': {
        'scenarios': ['rational'] + get_enabled_biases(),
        'description': 'Rational baseline plus individual bias effects',
        'duration_years': 20,
        'households': 1000
    },
    'full_behavioral': {
        'scenarios': get_all_scenarios(),
        'description': 'Complete behavioral study with all scenarios',
        'duration_years': 20,
        'households': 1000
    },
    'sensitivity_analysis': {
        'scenarios': get_all_scenarios(),
        'description': 'Parameter sensitivity analysis',
        'duration_years': 10,
        'households': 500,
        'parameter_variations': {
            'loss_aversion_coefficient': [1.5, 2.0, 2.25, 2.5, 3.0],
            'present_bias_beta_min': [0.5, 0.6, 0.7],
            'status_quo_strength': [0.6, 0.7, 0.75, 0.8, 0.85],  # Updated range
            'herding_spatial_strength': [1.0, 2.0, 2.5, 3.0, 4.0]
        }
    }
}

# =============================================================================
# HELPER FUNCTIONS FOR VALIDATION AND CONFIGURATION
# =============================================================================

def validate_configuration():
    """Validate the configuration for consistency and completeness"""
    errors = []
    warnings = []
    
    # Check enabled biases
    enabled_biases = get_enabled_biases()
    if not enabled_biases:
        warnings.append("No behavioral biases enabled - will run rational only")
    
    # Check scenario count
    scenarios = get_all_scenarios()
    if len(scenarios) < 2:
        warnings.append("Only rational scenario available")
    elif len(scenarios) > 10:
        warnings.append(f"Large number of scenarios ({len(scenarios)}) may be slow")
    
    # Check color mapping
    colors = get_scenario_colors()
    if len(colors) != len(scenarios):
        errors.append("Mismatch between scenarios and colors")
    
    # Check parameter validity for each enabled bias
    for bias_name, config in BEHAVIORAL_BIASES.items():
        if config.get('enabled', False):
            params = config.get('parameters', {})
            
            # Check loss aversion parameters
            if bias_name == 'loss_aversion':
                baseline_coeff = params.get('baseline_coefficient', 0)
                if baseline_coeff <= 1:
                    errors.append(f"Loss aversion coefficient should be > 1, got {baseline_coeff}")
                if baseline_coeff > 5:
                    warnings.append(f"Very high loss aversion coefficient: {baseline_coeff}")
            
            # Check present bias parameters
            elif bias_name == 'present_bias':
                beta_min = params.get('beta_min', 0)
                beta_max = params.get('beta_max', 1)
                if not (0 < beta_min < beta_max < 1):
                    errors.append(f"Present bias beta range invalid: {beta_min} to {beta_max}")
            
            # Check status quo parameters
            elif bias_name == 'status_quo':
                strength = params.get('baseline_strength', 0)
                if not (0 <= strength <= 1):
                    errors.append(f"Status quo strength should be [0,1], got {strength}")
            
            # Check herding parameters
            elif bias_name == 'herding':
                max_mult = params.get('max_influence_multiplier', 1)
                if max_mult < 1:
                    errors.append(f"Herding max multiplier should be >= 1, got {max_mult}")
                if max_mult > 5:
                    warnings.append(f"Very high herding multiplier: {max_mult}")
    
    # Check visualization parameters
    viz_config = SPATIAL_VISUALIZATION_CONFIG
    sample_size = viz_config.get('sample_size', 0)
    if sample_size > NUM_HOUSEHOLDS:
        errors.append(f"Visualization sample size ({sample_size}) exceeds total households ({NUM_HOUSEHOLDS})")
    
    return errors, warnings

def print_configuration_summary():
    """Print a summary of the current configuration"""
    print("=" * 80)
    print("BEHAVIORAL PROSUMER SIMULATION CONFIGURATION")
    print("=" * 80)
    
    print(f"Population: {NUM_HOUSEHOLDS} households")
    print(f"Duration: {TOTAL_STEPS} steps ({YEARS_TO_SIMULATE} years)")
    print(f"Network: {NEIGHBORS_PER_HOUSEHOLD} neighbors per household")
    
    print(f"\nBehavioral Biases Configuration:")
    enabled_biases = get_enabled_biases()
    if enabled_biases:
        for bias_name in enabled_biases:
            config = BEHAVIORAL_BIASES[bias_name]
            print(f"  ✅ {config['display_name']}: {config['description']}")
            print(f"     Literature: {config['literature_source']}")
    else:
        print("  ❌ No biases enabled (rational only)")
    
    print(f"\nSimulation Scenarios ({len(get_all_scenarios())}):")
    colors = get_scenario_colors()
    for scenario in get_all_scenarios():
        color = colors[scenario]
        print(f"  - {scenario} ({color})")
    
    print(f"\nVisualization: Sample {SPATIAL_VISUALIZATION_CONFIG['sample_size']} households")
    
    # Validation
    errors, warnings = validate_configuration()
    if errors:
        print(f"\n❌ Configuration Errors:")
        for error in errors:
            print(f"  - {error}")
    if warnings:
        print(f"\n⚠️  Configuration Warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    
    if not errors:
        print(f"\n✅ Configuration is valid!")
    
    print("=" * 80)

def get_bias_parameter_summary():
    """Get summary of all bias parameters for documentation"""
    summary = {}
    
    for bias_name, config in BEHAVIORAL_BIASES.items():
        if config.get('enabled', False):
            summary[bias_name] = {
                'display_name': config['display_name'],
                'description': config['description'],
                'literature_source': config['literature_source'],
                'formula': config.get('formula', ''),
                'parameters': config['parameters'].copy(),
                'application_order': config.get('application_order', 999)
            }
    
    return summary

# Run validation on import (optional)
if __name__ == "__main__":
    print_configuration_summary()