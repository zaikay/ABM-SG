# test_phase2.py
"""
Phase 2 testing script for multi-scenario household and model implementation.
Tests the core behavioral agent and multi-experiment model functionality.
"""

import os
import sys
import traceback

# Add simulation package to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all Phase 2 components can be imported."""
    print("Testing Phase 2 imports...")
    
    try:
        # Test multi-scenario household import
        from simulation.agents.multi_scenario_household import MultiScenarioHousehold
        print("✅ MultiScenarioHousehold imported successfully")
        
        # Test multi-experiment model import
        from simulation.models.multi_experiment_model import MultiExperimentModel
        print("✅ MultiExperimentModel imported successfully")
        
        # Test multi-experiment collector import
        from simulation.data.multi_experiment_collector import MultiExperimentCollector
        print("✅ MultiExperimentCollector imported successfully")
        
        # Test enhanced config loader
        from simulation.utils.config_loader import create_multi_experiment_config, create_testing_config
        print("✅ Enhanced config loader imported successfully")
        
        # Test parameter functions
        from simulation.utils.parameters import get_all_scenarios, get_enabled_biases
        scenarios = get_all_scenarios()
        biases = get_enabled_biases()
        print(f"✅ Parameters loaded: {len(scenarios)} scenarios, {len(biases)} biases")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        traceback.print_exc()
        return False

def test_multi_scenario_household():
    """Test MultiScenarioHousehold functionality."""
    print("\nTesting MultiScenarioHousehold...")
    
    try:
        from simulation.agents.multi_scenario_household import test_multi_scenario_household
        return test_multi_scenario_household()
        
    except Exception as e:
        print(f"❌ MultiScenarioHousehold test failed: {e}")
        traceback.print_exc()
        return False

def test_multi_experiment_model():
    """Test MultiExperimentModel functionality."""
    print("\nTesting MultiExperimentModel...")
    
    try:
        from simulation.models.multi_experiment_model import test_multi_experiment_model
        return test_multi_experiment_model()
        
    except Exception as e:
        print(f"❌ MultiExperimentModel test failed: {e}")
        traceback.print_exc()
        return False

def test_multi_experiment_collector():
    """Test MultiExperimentCollector functionality."""
    print("\nTesting MultiExperimentCollector...")
    
    try:
        from simulation.data.multi_experiment_collector import test_multi_experiment_collector
        return test_multi_experiment_collector()
        
    except Exception as e:
        print(f"❌ MultiExperimentCollector test failed: {e}")
        traceback.print_exc()
        return False

def test_configuration_integration():
    """Test configuration integration with new components."""
    print("\nTesting configuration integration...")
    
    try:
        from simulation.utils.config_loader import create_multi_experiment_config, validate_experiment_config
        from simulation.utils.parameters import get_all_scenarios
        
        # Create configuration
        config = create_multi_experiment_config()
        config_dict = config.get_copy()  # FIXED: Get dictionary from SimulationConfig
        
        # Validate configuration
        is_valid, errors, warnings = validate_experiment_config(config_dict)  # FIXED: Pass dictionary
        
        if not is_valid:
            print(f"❌ Configuration validation failed: {errors}")
            return False
        
        if warnings:
            print(f"⚠️  Configuration warnings: {warnings}")
        
        # Check key components
        required_keys = ['scenarios', 'behavioral_params', 'experiment_type']
        for key in required_keys:
            if key not in config_dict:  # FIXED: Check dictionary
                print(f"❌ Missing required config key: {key}")
                return False
        
        # Check scenarios match
        config_scenarios = set(config_dict['scenarios'])  # FIXED: Use dictionary
        param_scenarios = set(get_all_scenarios())
        
        if config_scenarios != param_scenarios:
            print(f"❌ Scenario mismatch: config={config_scenarios}, params={param_scenarios}")
            return False
        
        print("✅ Configuration integration test passed")
        return True
        
    except Exception as e:
        print(f"❌ Configuration integration test failed: {e}")
        traceback.print_exc()
        return False

def test_end_to_end_mini_simulation():
    """Test end-to-end mini simulation."""
    print("\nTesting end-to-end mini simulation...")
    
    try:
        from simulation.utils.config_loader import create_testing_config
        from simulation.models.multi_experiment_model import MultiExperimentModel
        from simulation.utils.parameters import get_all_scenarios
        
        # Create minimal configuration
        config = create_testing_config(num_households=5, steps=3)
        config_dict = config.get_copy()
        
        print(f"  Creating model with {config_dict['num_households']} households for {config_dict['steps']} steps...")
        
        # Create model
        model = MultiExperimentModel(config_dict)
        
        # Verify initialization
        households = model.get_households()
        if len(households) != 5:
            print(f"❌ Expected 5 households, got {len(households)}")
            return False
        
        # Check initial state
        initial_rates = model.get_scenario_adoption_rates()
        if not all(rate == 0.0 for rate in initial_rates.values()):
            print(f"❌ Expected all initial adoption rates to be 0.0, got {initial_rates}")
            return False
        
        print("  Running 3 simulation steps...")
        
        # Run simulation
        for step in range(3):
            model.step()
            
            # Check data collection
            collector_summary = model.data_collector.get_data_summary()
            if collector_summary['scenarios_tracked'] != len(get_all_scenarios()):
                print(f"❌ Expected {len(get_all_scenarios())} scenarios tracked, got {collector_summary['scenarios_tracked']}")
                return False
        
        # Check final state
        final_stats = model.get_scenario_statistics()
        if final_stats['current_step'] != 3:
            print(f"❌ Expected final step 3, got {final_stats['current_step']}")
            return False
        
        # Check that all scenarios are tracked
        expected_scenarios = get_all_scenarios()
        for scenario in expected_scenarios:
            if scenario not in final_stats['scenarios']:
                print(f"❌ Missing scenario in final stats: {scenario}")
                return False
        
        # Test data export
        print("  Testing data export...")
        test_output_dir = "results/test_phase2"
        model.export_results(test_output_dir)
        
        # Check that key files exist
        expected_files = ["combined_scenarios.csv", "system_metrics.csv", "scenario_metadata.csv"]
        for filename in expected_files:
            filepath = os.path.join(test_output_dir, filename)
            if not os.path.exists(filepath):
                print(f"❌ Missing expected output file: {filepath}")
                return False
        
        print("✅ End-to-end mini simulation test passed")
        return True
        
    except Exception as e:
        print(f"❌ End-to-end test failed: {e}")
        traceback.print_exc()
        return False

def test_bias_effects():
    """Test that bias effects are being calculated."""
    print("\nTesting bias effects calculation...")
    
    try:
        from simulation.utils.config_loader import create_testing_config
        from simulation.models.multi_experiment_model import MultiExperimentModel
        from simulation.utils.parameters import get_enabled_biases
        
        # Create configuration
        config = create_testing_config(num_households=3, steps=12)  # 1 year to get NPV
        config_dict = config.get_copy()
        
        # Create model
        model = MultiExperimentModel(config_dict)
        
        # Run for 12 steps to get NPV calculations
        for step in range(12):
            model.step()
        
        # Check that households have bias effects
        households = model.get_households()
        households_with_bias_effects = 0
        
        for household in households:
            if hasattr(household, 'bias_effects') and household.bias_effects:
                households_with_bias_effects += 1
                
                # Check that bias effects have expected structure
                required_keys = ['base_probability', 'base_npv', 'bias_multipliers']
                for key in required_keys:
                    if key not in household.bias_effects:
                        print(f"❌ Missing bias effects key '{key}' for household {household.unique_id}")
                        return False
        
        print(f"  {households_with_bias_effects}/{len(households)} households have bias effects")
        
        # Check bias effects data collection
        bias_df = model.data_collector.get_bias_effects_dataframe()
        if bias_df.empty:
            print("⚠️  No bias effects data collected (may be normal if no households have positive NPV)")
        else:
            print(f"  Collected {len(bias_df)} bias effects records")
        
        print("✅ Bias effects test passed")
        return True
        
    except Exception as e:
        print(f"❌ Bias effects test failed: {e}")
        traceback.print_exc()
        return False

def run_all_phase2_tests():
    """Run all Phase 2 tests."""
    print("="*60)
    print("PHASE 2 TESTING: Multi-Scenario Household & Model")
    print("="*60)
    
    tests = [
        ("Imports", test_imports),
        ("MultiScenarioHousehold", test_multi_scenario_household),
        ("MultiExperimentModel", test_multi_experiment_model),
        ("MultiExperimentCollector", test_multi_experiment_collector),
        ("Configuration Integration", test_configuration_integration),
        ("End-to-End Mini Simulation", test_end_to_end_mini_simulation),
        ("Bias Effects", test_bias_effects)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = test_func()
            results.append((test_name, success))
            
            if success:
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
                
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("PHASE 2 TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:<35} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL PHASE 2 TESTS PASSED!")
        print("\nPhase 2 is ready. You can proceed to Phase 3: Bias Implementation Details")
        return True
    else:
        print("❌ Some tests failed. Please fix issues before proceeding.")
        return False

def main():
    """Main function for running Phase 2 tests."""
    success = run_all_phase2_tests()
    
    if success:
        print("\n🚀 NEXT STEPS:")
        print("1. Review the test results above")
        print("2. Run a quick multi-experiment simulation:")
        print("   python run_multi_experiment.py --quick-test")
        print("3. If successful, proceed to Phase 3: Bias Implementation Details")
        return True
    else:
        print("\n🔧 TROUBLESHOOTING:")
        print("1. Check that all Phase 1 components are working")
        print("2. Verify that all new files are in the correct locations")
        print("3. Check imports and dependencies")
        print("4. Run individual tests to isolate issues")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)