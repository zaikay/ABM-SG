# agents/household.py V5.1

import numpy as np
from mesa import Agent
from ..environment.energy_system import EnergySystem
from ..utils.parameters import *

class Household(Agent):
    """
    Household agent using unified energy system (no redundant calculations).
    """
    
    def __init__(self, unique_id, model, config, weather_patterns, grid_system):
        """Initialize household with unified energy system."""
        super().__init__(unique_id, model)
        
        self.config = config
        self.grid_system = grid_system
        
        # Initialize income and consumption
        self._initialize_income_consumption()

        # Initialize behavioral coefficients once
        self._initialize_behavioral_coefficients()

        # SPATIAL NETWORK ATTRIBUTES (populated during network building)
        self.spatial_neighbors = []          # List of (neighbor, distance) tuples
        self.spatial_neighbor_ids = set()    # Quick lookup set of neighbor IDs
        
        # Solar system status
        self.is_prosumer = False
        self.solar_capacity = 0
        self.installation_month = None
        
        # Create unified energy system (replaces both calculator and metrics)
        self.energy_system = EnergySystem(weather_patterns, grid_system)
        
        # Current metrics (calculated once per month)
        self.baseline_cost = 0
        self.solar_cost = 0
        self.monthly_savings = 0
        self.annual_savings = 0
        self.installation_cost = 0
        self.npv = None
        self.payback_period = float('inf')
        
        # Energy flows
        self.monthly_consumption = 0
        self.monthly_generation = 0
        self.monthly_grid_consumption = 0
        self.monthly_grid_feed_in = 0
        
        # Historical tracking
        self.savings_history = []
        
        # Basic attributes
        self.pos = None
        self.income_class = 0
    
    def _initialize_income_consumption(self):
        """Initialize income and consumption (unchanged)."""
        params = self.config["income_params"]
        self.income = np.random.lognormal(
            params["lognormal_mean"],
            params["lognormal_sd"]
        )
        
        median_income = np.exp(params["lognormal_mean"])
        beta = self.config["consumption_params"]["income_elasticity"]
        base_consumption = self.config["consumption_params"]["base_consumption"]
        alpha = np.log(base_consumption) - beta * np.log(median_income)
        
        epsilon = np.random.normal(0, self.config["consumption_params"]["variation_sd"])
        log_consumption = alpha + beta * np.log(self.income) + epsilon
        self.daily_consumption = np.exp(log_consumption)
    
    def set_income_class(self, income_class):
        """Set income class."""
        self.income_class = income_class
    
    def _initialize_behavioral_coefficients(self):
        """
        Initialize ALL behavioral coefficients once and store them.
        Simple, clean approach that works for any behavioral bias.
        """
        from ..utils.parameters import BEHAVIORAL_BIASES
        import numpy as np
        
        # Create household-specific seed for reproducibility
        household_seed = hash(self.unique_id) % (2**32)
        np.random.seed(household_seed)
        
        # Store all behavioral coefficients in a dictionary
        self.behavioral_coefficients = {}
        
        # Initialize coefficients for each enabled bias
        for bias_name, bias_config in BEHAVIORAL_BIASES.items():
            if not bias_config.get('enabled', False):
                continue
                
            params = bias_config.get('parameters', {})
            bias_coeffs = {}
            
            # Loss Aversion coefficients
            if bias_name == 'loss_aversion':
                bias_coeffs['variation_std'] = np.random.normal(0, params['variation_std']
                   )
            
            # Present Bias coefficients
            elif bias_name == 'present_bias':
                bias_coeffs['beta_i'] = np.random.uniform(
                    params.get('beta_min', 0.6), 
                    params.get('beta_max', 0.8)
                )
            
            # Status Quo coefficients
            elif bias_name == 'status_quo':
                bias_coeffs['individual_variation'] = np.random.normal(
                    0, params.get('individual_variation', 0.08)
                )
            
            # Herding coefficients
            elif bias_name == 'herding':
                bias_coeffs['variation_std'] = np.random.normal(0, params['variation_std']
                   )
            # optimism coefficients
            elif bias_name == 'optimism_bias':
                bias_coeffs['individual_variation'] = np.random.normal(0, params.get('individual_variation', 0.05)
                )
            
            # Store coefficients for this bias
            self.behavioral_coefficients[bias_name] = bias_coeffs
        
        # Reset random seed
        np.random.seed(None)
    
    def get_behavioral_coefficient(self, bias_name, coeff_name, default_value=0.5):
        """
        Simple method to get any behavioral coefficient.
        
        Args:
            bias_name: Name of the bias (e.g., 'herding', 'loss_aversion')
            coeff_name: Name of the coefficient (e.g., 'beta_i', 'gamma_i')
            default_value: Default value if coefficient not found
        
        Returns:
            float: The coefficient value
        """
        return self.behavioral_coefficients.get(bias_name, {}).get(coeff_name, default_value)

    def step(self):
        """Execute monthly step with SINGLE calculation call."""
        current_month = self.model.schedule.steps
        
        # 1. Calculate energy metrics (no NPV)
        metrics = self.energy_system.calculate_household_metrics(self, current_month)
        
        # 2. Update from metrics (except annual_savings and NPV)
        self._update_from_metrics(metrics)
        
        # 3. Update annual savings with complete history
        self._update_annual_savings()
        
        # 4. Calculate NPV separately using updated annual_savings
        self.npv = self.grid_system.calculate_npv(
            self.installation_cost,
            self.annual_savings,
            self.config["solar_params"]["lifetime_years"],
            self.config["solar_params"]["discount_rate"]
        )
        
        # 5. Calculate payback period
        self.payback_period = (self.installation_cost / self.annual_savings 
                            if self.annual_savings > 0 else float('inf'))
        metrics['annual_savings'] = self.annual_savings
        metrics['npv'] = self.npv
        metrics['payback_period'] = self.payback_period

        # 6. Consider adoption with updated NPV
        if not self.is_prosumer and len(self.savings_history) >= 12:
            self.consider_adoption(current_month, metrics)
        
        # 7. Register energy flows
        self._register_energy_flows()
    
    def _update_from_metrics(self, metrics):
        """
        SIMPLIFIED: Update household attributes from single set of metrics.
        """
        # Costs (same for all households)
        self.baseline_cost = metrics['baseline_cost']
        self.solar_cost = metrics['solar_cost']
        self.monthly_savings = metrics['monthly_savings']
        
        # Solar system
        self.installation_cost = metrics['installation_cost']
        self.solar_capacity = metrics['solar_capacity']
        
        # Energy flows (ENHANCED with credit attributes)
        self.monthly_consumption = metrics['monthly_consumption']
        self.monthly_generation = metrics['monthly_generation']
        self.monthly_grid_consumption = metrics['monthly_grid_consumption']
        self.monthly_grid_feed_in = metrics['monthly_grid_feed_in']
        
        # NEW STEP 1: Store credit metrics that were previously missing
        self.monthly_credits_earned = metrics['monthly_credits_earned']
        self.monthly_credits_used = metrics['monthly_credits_used']
        self.monthly_credits_expired = metrics['monthly_credits_expired']
        
        # Add to history
        self.savings_history.append(self.monthly_savings)
    
    def _update_annual_savings(self):
        """
        Calculate 12-month rolling savings.
        For households with < 12 months of data, extrapolate or use conservative estimate.
        """
        if len(self.savings_history) >= 12:
            self.annual_savings = sum(self.savings_history[-12:])
        elif len(self.savings_history) > 0:
            # Conservative extrapolation for early months
            avg_monthly = sum(self.savings_history) / len(self.savings_history)
            self.annual_savings = avg_monthly * 12
        else:
            self.annual_savings = 0
    
    def consider_adoption(self, current_month, metrics):
        """Consider adoption using pre-calculated metrics."""
        if self.npv is not None and self.npv > 0:
            self.is_prosumer = True
            self.installation_month = current_month
            
            print(f"Household {self.unique_id} adopted solar in month {current_month} with NPV {self.npv:.2f}")
            return True
        return False
    
    def _register_energy_flows(self):
        """Register energy flows with central provider."""
        central_provider = self.model.get_central_provider()
        if central_provider:
            central_provider.register_energy_flows(
                self.monthly_consumption,
                self.monthly_generation, 
                self.monthly_grid_consumption
            )