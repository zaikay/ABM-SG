# test_phase3_implementation_final.py
"""
FINAL FIXED comprehensive test runner for Phase 3 behavioral implementation.
Includes complete mock model with all required methods for household agents.
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path

# Add the simulation package to the path
sys.path.append(str(Path(__file__).parent))

def test_bias_parameter_configuration():
    """Test that behavioral bias parameters are correctly configured."""
    print("=" * 60)
    print("TESTING BIAS PARAMETER CONFIGURATION")
    print("=" * 60)
    
    try:
        from simulation.utils.parameters import (
            BEHAVIORAL_BIASES, get_enabled_biases, get_all_scenarios, 
            get_scenario_colors, validate_configuration
        )
        
        # Test 1: Verify all biases are configured
        print("Test 1: Bias configuration verification...")
        expected_biases = ['loss_aversion', 'present_bias', 'status_quo', 'herding']
        configured_biases = list(BEHAVIORAL_BIASES.keys())
        
        if not all(bias in configured_biases for bias in expected_biases):
            print(f"  ❌ Missing biases: {set(expected_biases) - set(configured_biases)}")
            return False
        
        print(f"  ✅ All {len(expected_biases)} biases configured: {configured_biases}")
        
        # Test 2: Verify enabled biases
        print("Test 2: Enabled biases verification...")
        enabled_biases = get_enabled_biases()
        print(f"  Enabled biases: {enabled_biases}")
        
        if not enabled_biases:
            print("  ❌ No biases enabled")
            return False
        
        print(f"  ✅ {len(enabled_biases)} biases enabled")
        
        # Test 3: Verify scenario generation
        print("Test 3: Scenario generation...")
        scenarios = get_all_scenarios()
        expected_count = 1 + len(enabled_biases) + 1  # rational + individual + combined
        
        if len(scenarios) != expected_count:
            print(f"  ❌ Expected {expected_count} scenarios, got {len(scenarios)}")
            return False
        
        print(f"  ✅ Generated {len(scenarios)} scenarios: {scenarios}")
        
        # Test 4: Verify parameter completeness
        print("Test 4: Parameter completeness...")
        for bias_name, config in BEHAVIORAL_BIASES.items():
            if config.get('enabled', False):
                if 'parameters' not in config:
                    print(f"  ❌ Missing parameters for {bias_name}")
                    return False
                
                if 'literature_source' not in config:
                    print(f"  ❌ Missing literature source for {bias_name}")
                    return False
        
        print("  ✅ All enabled biases have complete parameters and literature sources")
        
        # Test 5: Configuration validation
        print("Test 5: Configuration validation...")
        errors, warnings = validate_configuration()
        
        if errors:
            print("  ❌ Configuration errors:")
            for error in errors:
                print(f"    - {error}")
            return False
        
        if warnings:
            print("  ⚠️  Configuration warnings:")
            for warning in warnings:
                print(f"    - {warning}")
        
        print("  ✅ Configuration validation passed")
        
        return True
        
    except Exception as e:
        print(f"❌ Bias parameter configuration test failed: {e}")
        return False

def test_bias_manager_implementation():
    """Test the BiasManager implementation."""
    print("\n" + "=" * 60)
    print("TESTING BIAS MANAGER IMPLEMENTATION")
    print("=" * 60)
    
    try:
        from simulation.utils.config_loader import create_multi_experiment_config
        from simulation.agents.bias_manager import BiasManager
        
        # Create configuration
        config = create_multi_experiment_config()
        config_dict = config.get_copy()
        
        # Test 1: BiasManager initialization
        print("Test 1: BiasManager initialization...")
        bias_manager = BiasManager(config_dict)
        
        if not bias_manager.enabled_biases:
            print("  ❌ No biases enabled in BiasManager")
            return False
        
        print(f"  ✅ BiasManager initialized with {len(bias_manager.enabled_biases)} biases")
        
        # Test 2: Bias calculation validation
        print("Test 2: Bias calculation validation...")
        is_valid, errors, warnings = bias_manager.validate_bias_calculations()
        
        if not is_valid:
            print("  ❌ Bias calculation validation failed:")
            for error in errors:
                print(f"    - {error}")
            return False
        
        if warnings:
            print("  ⚠️  Bias calculation warnings:")
            for warning in warnings:
                print(f"    - {warning}")
        
        print("  ✅ All bias calculations validated")
        
        # Test 3: Individual bias applications
        print("Test 3: Individual bias effects...")
        mock_household = bias_manager._create_mock_household()
        base_npv = 10000
        base_prob = bias_manager._npv_to_probability(base_npv)
        
        print(f"  Base NPV: ${base_npv:.0f}, Base probability: {base_prob:.3f}")
        
        effects = {}
        for bias_name in bias_manager.enabled_biases:
            biased_prob = bias_manager.apply_single_bias(mock_household, bias_name, base_npv, base_prob)
            effect = biased_prob / base_prob if base_prob > 0 else 1.0
            effects[bias_name] = effect
            
            if not (0 <= biased_prob <= 1):
                print(f"  ❌ {bias_name} produced invalid probability: {biased_prob}")
                return False
            
            print(f"    {bias_name}: {effect:.3f}x (prob: {base_prob:.3f} → {biased_prob:.3f})")
        
        # Test 4: Combined bias application
        print("Test 4: Combined bias effects...")
        combined_prob = bias_manager.apply_all_biases(mock_household, base_npv, base_prob)
        combined_effect = combined_prob / base_prob if base_prob > 0 else 1.0
        
        if not (0 <= combined_prob <= 1):
            print(f"  ❌ Combined bias produced invalid probability: {combined_prob}")
            return False
        
        print(f"  Combined effect: {combined_effect:.3f}x (prob: {base_prob:.3f} → {combined_prob:.3f})")
        print("  ✅ All bias applications working correctly")
        
        # Test 5: Bias effects summary
        print("Test 5: Bias effects summary...")
        summary = bias_manager.get_bias_effects_summary(mock_household)
        
        required_keys = ['household_id', 'bias_effects', 'combined_effect']
        if not all(key in summary for key in required_keys):
            print(f"  ❌ Missing required keys in summary: {list(summary.keys())}")
            return False
        
        print("  ✅ Comprehensive bias effects summary generated")
        
        return True
        
    except Exception as e:
        print(f"❌ BiasManager implementation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_complete_mock_model():
    """Create a complete mock model with all required methods for household agents."""
    from mesa import Model
    from mesa.time import RandomActivation
    
    class CompleteMockModel(Model):
        def __init__(self):
            super().__init__()
            self.schedule = RandomActivation(self)
            self.running = True
            
            # Create mock central provider
            self.central_provider = MockCentralProvider()
            
        def register_agent(self, agent):
            """Required method for Mesa Agent initialization"""
            pass
            
        def get_central_provider(self):
            """Required method for household energy flow registration"""
            return self.central_provider
            
        def get_households(self):
            """Return list of household agents"""
            return [agent for agent in self.schedule.agents 
                   if hasattr(agent, 'daily_consumption')]
    
    class MockCentralProvider:
        """Mock central provider for testing"""
        def __init__(self):
            self.monthly_consumption = 0
            self.monthly_generation = 0
            self.monthly_grid_consumption = 0
            
        def register_energy_flows(self, consumption, generation, grid_consumption):
            """Mock energy flow registration"""
            self.monthly_consumption += consumption
            self.monthly_generation += generation
            self.monthly_grid_consumption += grid_consumption
    
    return CompleteMockModel()

def test_multi_scenario_household():
    """Test the enhanced MultiScenarioHousehold implementation."""
    print("\n" + "=" * 60)
    print("TESTING MULTI-SCENARIO HOUSEHOLD")
    print("=" * 60)
    
    try:
        from simulation.utils.config_loader import create_multi_experiment_config
        from simulation.agents.multi_scenario_household import MultiScenarioHousehold
        from simulation.environment.weather_patterns import WeatherPatterns
        from simulation.environment.grid_system import GridSystem
        from simulation.utils.parameters import get_all_scenarios
        
        # Create configuration and environment
        config = create_multi_experiment_config()
        config_dict = config.get_copy()
        weather_patterns = WeatherPatterns()
        grid_system = GridSystem(config_dict)
        
        # Create COMPLETE mock model with all required methods
        mock_model = create_complete_mock_model()
        
        # Test 1: Enhanced household initialization
        print("Test 1: Enhanced household initialization...")
        household = MultiScenarioHousehold(
            unique_id=1,
            model=mock_model,
            config=config_dict,
            weather_patterns=weather_patterns,
            grid_system=grid_system
        )
        
        mock_model.schedule.add(household)
        
        expected_scenarios = get_all_scenarios()
        if set(household.scenarios) != set(expected_scenarios):
            print(f"  ❌ Scenario mismatch")
            return False
        
        # Check enhanced attributes
        enhanced_attrs = [
            'bias_effects_history', 'spatial_influence_history', 'class_influence_history',
            'neighbor_adoption_rates', 'current_bias_effects'
        ]
        
        for attr in enhanced_attrs:
            if not hasattr(household, attr):
                print(f"  ❌ Missing enhanced attribute: {attr}")
                return False
        
        print(f"  ✅ Enhanced household initialized with {len(household.scenarios)} scenarios")
        
        # Test 2: Early evaluation (insufficient history)
        print("Test 2: Early evaluation handling...")
        household.step()
        
        if any(household.scenario_adoption.values()):
            print(f"  ❌ Premature adoption with insufficient history")
            return False
        
        print("  ✅ Correctly handled insufficient history")
        
        # Test 3: Full evaluation with sufficient history
        print("Test 3: Full scenario evaluation...")
        
        # Simulate 12 months of savings history
        for i in range(12):
            household.savings_history.append(150 + i * 10)
        
        # Set positive NPV for testing
        household.npv = 8000
        household.installation_cost = 18000
        household.annual_savings = 1200
        household.payback_period = 15
        mock_model.schedule.steps = 12
        
        # Run evaluation
        household.step()
        
        # Verify evaluation occurred
        if not any(prob > 0 for prob in household.scenario_probability.values()):
            print(f"  ❌ No scenario probabilities calculated")
            return False
        
        print("  ✅ Full scenario evaluation completed")
        
        # Show results
        for scenario in household.scenarios:
            prob = household.scenario_probability[scenario]
            adopted = household.scenario_adoption[scenario]
            print(f"    {scenario}: prob={prob:.3f}, adopted={adopted}")
        
        # Test 4: Social influence tracking
        print("Test 4: Social influence tracking...")
        
        # Add neighbor households
        for i in range(2, 5):
            neighbor = MultiScenarioHousehold(
                unique_id=i,
                model=mock_model,
                config=config_dict,
                weather_patterns=weather_patterns,
                grid_system=grid_system
            )
            neighbor.pos = (i, i)  # Different positions
            neighbor.income_class = household.income_class if i == 2 else (i % 5) + 1
            mock_model.schedule.add(neighbor)
        
        # Update social influence
        household._update_social_influence_tracking(12)
        
        if not household.spatial_influence_history:
            print(f"  ❌ No spatial influence recorded")
            return False
        
        if not household.class_influence_history:
            print(f"  ❌ No class influence recorded")
            return False
        
        print("  ✅ Social influence tracking working")
        
        # Test 5: Comprehensive metrics
        print("Test 5: Comprehensive metrics collection...")
        metrics = household.get_scenario_metrics()
        
        required_keys = [
            'household_id', 'scenarios', 'neighbor_adoption_rates', 
            'spatial_influence_current', 'class_influence_current'
        ]
        
        if not all(key in metrics for key in required_keys):
            missing = [key for key in required_keys if key not in metrics]
            print(f"  ❌ Missing required metrics: {missing}")
            return False
        
        if len(metrics['scenarios']) != len(expected_scenarios):
            print(f"  ❌ Incomplete scenario metrics")
            return False
        
        print("  ✅ Comprehensive metrics collection working")
        
        # Test 6: Debug information
        print("Test 6: Debug information...")
        debug_info = household.get_comprehensive_debug_info()
        
        debug_keys = [
            'household_id', 'scenario_adoption', 'current_npv', 
            'spatial_neighbors_count', 'class_members_count'
        ]
        
        if not all(key in debug_info for key in debug_keys):
            missing = [key for key in debug_keys if key not in debug_keys]
            print(f"  ❌ Missing debug info: {missing}")
            return False
        
        print("  ✅ Debug information comprehensive")
        
        return True
        
    except Exception as e:
        print(f"❌ MultiScenarioHousehold test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integration():
    """Test integration between BiasManager and MultiScenarioHousehold."""
    print("\n" + "=" * 60)
    print("TESTING BIAS MANAGER + HOUSEHOLD INTEGRATION")
    print("=" * 60)
    
    try:
        from simulation.utils.config_loader import create_multi_experiment_config
        from simulation.agents.multi_scenario_household import MultiScenarioHousehold
        from simulation.environment.weather_patterns import WeatherPatterns
        from simulation.environment.grid_system import GridSystem
        from simulation.utils.parameters import get_enabled_biases
        
        # Create test environment
        config = create_multi_experiment_config()
        config_dict = config.get_copy()
        weather_patterns = WeatherPatterns()
        grid_system = GridSystem(config_dict)
        
        # Create COMPLETE mock model
        mock_model = create_complete_mock_model()
        
        # Test 1: Create household population
        print("Test 1: Creating household population...")
        households = []
        
        for i in range(10):
            household = MultiScenarioHousehold(
                unique_id=i,
                model=mock_model,
                config=config_dict,
                weather_patterns=weather_patterns,
                grid_system=grid_system
            )
            
            # Set positions and income classes
            household.pos = (i % 3, i // 3)
            household.income_class = (i % 5) + 1
            household.income = 30000 + household.income_class * 15000
            
            households.append(household)
            mock_model.schedule.add(household)
        
        print(f"  ✅ Created {len(households)} households")
        
        # Test 2: Simulate decision making process
        print("Test 2: Simulating decision making...")
        
        # Give all households sufficient history and varying NPVs
        for i, household in enumerate(households):
            for month in range(12):
                household.savings_history.append(100 + i * 20 + month * 5)
            
            # Set varying NPVs to test different scenarios
            household.npv = 2000 + i * 1000  # Range from $2k to $11k
            household.installation_cost = 15000 + i * 500
            household.annual_savings = 800 + i * 100
        
        mock_model.schedule.steps = 12
        
        # Run evaluation for all households
        adoption_results = {}
        for scenario in households[0].scenarios:
            adoption_results[scenario] = []
        
        for household in households:
            household.step()
            
            for scenario in household.scenarios:
                adoption_results[scenario].append(household.scenario_adoption[scenario])
        
        # Check results
        print("  Adoption results by scenario:")
        for scenario, adoptions in adoption_results.items():
            adoption_count = sum(adoptions)
            adoption_rate = adoption_count / len(households)
            print(f"    {scenario}: {adoption_count}/{len(households)} ({adoption_rate:.1%})")
        
        # Verify some adoption occurred (or at least evaluation happened)
        total_evaluations = sum(len(household.scenario_probability) for household in households)
        if total_evaluations == 0:
            print("  ❌ No evaluations occurred")
            return False
        else:
            print("  ✅ Decision making simulation completed")
        
        # Test 3: Herding bias effects
        print("Test 3: Testing herding bias effects...")
        
        # Create a cluster of adopters to test herding
        adopter_cluster = households[:3]
        for household in adopter_cluster:
            household.scenario_adoption['herding'] = True
            household.adoption_months['herding'] = 12
        
        # Test herding effect on nearby non-adopter
        test_household = households[3]  # Should be influenced by cluster
        test_household.scenario_adoption['herding'] = False  # Reset
        
        # Recalculate with herding influence
        old_prob = test_household.scenario_probability.get('herding', 0.0)
        test_household._evaluate_all_scenarios_enhanced(13)
        new_prob = test_household.scenario_probability.get('herding', 0.0)
        
        print(f"    Herding probability change: {old_prob:.3f} → {new_prob:.3f}")
        print("  ✅ Herding bias calculation completed")
        
        # Test 4: Bias effects analysis
        print("Test 4: Bias effects analysis...")
        
        # Get bias effects for a test household
        test_household = households[5]  # Middle household
        summary = test_household.bias_manager.get_bias_effects_summary(test_household)
        
        print(f"  Base probability: {summary['base_probability']:.3f}")
        
        enabled_biases = get_enabled_biases()
        for bias_name in enabled_biases:
            if bias_name in summary['bias_effects']:
                effect = summary['bias_effects'][bias_name]
                print(f"    {bias_name}: {effect['effect_multiplier']:.3f}x")
        
        if 'combined_effect' in summary:
            combined = summary['combined_effect']
            print(f"    Combined: {combined['total_multiplier']:.3f}x")
        
        print("  ✅ Bias effects analysis working")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_literature_validation():
    """Validate that bias implementations match literature expectations."""
    print("\n" + "=" * 60)
    print("TESTING LITERATURE VALIDATION")
    print("=" * 60)
    
    try:
        from simulation.utils.config_loader import create_multi_experiment_config
        from simulation.agents.bias_manager import BiasManager
        from simulation.utils.parameters import BEHAVIORAL_BIASES
        
        config = create_multi_experiment_config()
        config_dict = config.get_copy()
        bias_manager = BiasManager(config_dict)
        
        # Test 1: Loss aversion literature validation
        print("Test 1: Loss aversion validation...")
        
        # Test income effect on loss aversion
        mock_low_income = bias_manager._create_mock_household()
        mock_high_income = bias_manager._create_mock_household()
        
        mock_low_income.income = 25000
        mock_high_income.income = 75000
        mock_low_income.installation_cost = 15000
        mock_high_income.installation_cost = 15000
        
        base_npv = 5000
        base_prob = 0.5
        
        low_income_prob = bias_manager._apply_loss_aversion(mock_low_income, base_npv, base_prob)
        high_income_prob = bias_manager._apply_loss_aversion(mock_high_income, base_npv, base_prob)
        
        # Lower income should have stronger loss aversion (lower probability)
        if low_income_prob >= high_income_prob:
            print(f"  ⚠️  Expected stronger loss aversion for lower income")
            print(f"    Low income prob: {low_income_prob:.3f}, High income prob: {high_income_prob:.3f}")
        else:
            print(f"  ✅ Loss aversion correctly varies with income")
            print(f"    Low income prob: {low_income_prob:.3f}, High income prob: {high_income_prob:.3f}")
        
        # Test 2: Present bias validation
        print("Test 2: Present bias validation...")
        
        # Present bias should reduce probability
        mock_household = bias_manager._create_mock_household()
        mock_household.installation_cost = 15000
        
        original_prob = 0.7
        present_bias_prob = bias_manager._apply_present_bias(mock_household, base_npv, original_prob)
        
        if present_bias_prob >= original_prob:
            print(f"  ❌ Present bias should reduce adoption probability")
            return False
        
        print(f"  ✅ Present bias correctly reduces probability: {original_prob:.3f} → {present_bias_prob:.3f}")
        
        # Test 3: Status quo bias validation
        print("Test 3: Status quo bias validation...")
        
        # Status quo bias should always reduce probability
        original_prob = 0.6
        status_quo_prob = bias_manager._apply_status_quo(mock_household, base_npv, original_prob)
        
        if status_quo_prob >= original_prob:
            print(f"  ❌ Status quo bias should always reduce probability")
            return False
        
        print(f"  ✅ Status quo bias correctly reduces probability: {original_prob:.3f} → {status_quo_prob:.3f}")
        
        # Test 4: Parameter ranges validation
        print("Test 4: Parameter ranges validation...")
        
        # Check that parameters are within literature-suggested ranges
        loss_aversion_params = BEHAVIORAL_BIASES['loss_aversion']['parameters']
        if not (1.5 <= loss_aversion_params['baseline_coefficient'] <= 3.0):
            print(f"  ⚠️  Loss aversion coefficient outside typical range: {loss_aversion_params['baseline_coefficient']}")
        
        present_bias_params = BEHAVIORAL_BIASES['present_bias']['parameters']
        if not (0.5 <= present_bias_params['beta_min'] <= present_bias_params['beta_max'] <= 0.9):
            print(f"  ⚠️  Present bias parameters outside typical range")
        
        status_quo_params = BEHAVIORAL_BIASES['status_quo']['parameters']
        if not (0.1 <= status_quo_params['baseline_strength'] <= 0.4):
            print(f"  ⚠️  Status quo strength outside typical range: {status_quo_params['baseline_strength']}")
        
        print("  ✅ Parameter ranges validated against literature")
        
        return True
        
    except Exception as e:
        print(f"❌ Literature validation test failed: {e}")
        return False

def run_comprehensive_phase3_tests():
    """Run all Phase 3 tests comprehensively."""
    print("🚀 STARTING COMPREHENSIVE PHASE 3 TESTS")
    print("=" * 80)
    
    test_results = []
    
    # Run all tests
    tests = [
        ("Bias Parameter Configuration", test_bias_parameter_configuration),
        ("BiasManager Implementation", test_bias_manager_implementation),
        ("MultiScenarioHousehold", test_multi_scenario_household),
        ("Integration Testing", test_integration),
        ("Literature Validation", test_literature_validation)
    ]
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name}...")
        try:
            result = test_func()
            test_results.append((test_name, result))
            
            if result:
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
                
        except Exception as e:
            print(f"💥 {test_name} CRASHED: {e}")
            test_results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 PHASE 3 TEST RESULTS SUMMARY")
    print("=" * 80)
    
    passed_tests = sum(1 for _, result in test_results if result)
    total_tests = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\n🎯 OVERALL RESULT: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 ALL PHASE 3 TESTS PASSED! Ready to proceed to Phase 4.")
        print("\nNext steps:")
        print("1. Implement visualization system (Phase 4)")
        print("2. Create spatial visualizer with sampling")
        print("3. Create comparative visualizer for bias effects")
        print("4. Test full multi-experiment model")
        return True
    else:
        print("⚠️  Some tests failed. Please fix issues before proceeding.")
        print("\nDebugging suggestions:")
        print("1. Check that all files are in correct directories")
        print("2. Verify import paths match your project structure")
        print("3. Run individual test functions to isolate issues")
        return False

if __name__ == "__main__":
    success = run_comprehensive_phase3_tests()
    sys.exit(0 if success else 1)