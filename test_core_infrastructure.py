#!/usr/bin/env python3
"""
Test runner for Core Infrastructure (Phase 1) of behavioral prosumer simulation.

Tests:
1. Enhanced parameters configuration
2. BiasManager functionality
3. VisualizationSampler
4. SpatialVisualizer basic functionality
5. Network initialization and positioning

Run this script to validate Phase 1 implementation before proceeding to Phase 2.
"""

import os
import sys
import traceback
import numpy as np

# Add the project root to Python path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def test_enhanced_parameters():
    """Test the enhanced parameters configuration."""
    print("\n" + "="*60)
    print("TESTING: Enhanced Parameters Configuration")
    print("="*60)
    
    try:
        # Import and test parameters
        from simulation.utils.parameters import (
            BEHAVIORAL_BIASES, get_enabled_biases, get_all_scenarios,
            get_scenario_colors, get_scenario_metadata, validate_configuration,
            print_configuration_summary
        )
        
        print("✓ Successfully imported enhanced parameters")
        
        # Test basic configuration functions
        enabled_biases = get_enabled_biases()
        all_scenarios = get_all_scenarios()
        scenario_colors = get_scenario_colors()
        scenario_metadata = get_scenario_metadata()
        
        print(f"✓ Enabled biases: {enabled_biases}")
        print(f"✓ All scenarios: {all_scenarios}")
        print(f"✓ Scenario colors: {len(scenario_colors)} colors mapped")
        print(f"✓ Scenario metadata: {len(scenario_metadata)} entries")
        
        # Test configuration validation
        errors, warnings = validate_configuration()
        
        if errors:
            print(f"❌ Configuration validation failed:")
            for error in errors:
                print(f"   - {error}")
            return False
        
        if warnings:
            print(f"⚠️  Configuration warnings:")
            for warning in warnings:
                print(f"   - {warning}")
        
        # Test configuration summary
        print("\n--- Configuration Summary ---")
        print_configuration_summary()
        
        print("\n✅ Enhanced parameters configuration test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Enhanced parameters test FAILED: {e}")
        traceback.print_exc()
        return False

def test_bias_manager():
    """Test the BiasManager functionality."""
    print("\n" + "="*60)
    print("TESTING: BiasManager Functionality")
    print("="*60)
    
    try:
        from simulation.agents.bias_manager import BiasManager, test_bias_manager
        from simulation.utils.config_loader import create_rational_experiment
        
        print("✓ Successfully imported BiasManager")
        
        # Create test configuration
        config = create_rational_experiment().get_copy()
        
        # Test BiasManager initialization
        bias_manager = BiasManager(config)
        print(f"✓ BiasManager initialized with biases: {bias_manager.enabled_biases}")
        
        # Test individual methods with mock household
        class MockModel:
            def __init__(self):
                class MockSchedule:
                    def __init__(self):
                        self.agents = []  # Start empty, add households later
                self.schedule = MockSchedule()
        
        class MockHousehold:
            def __init__(self, model):
                self.unique_id = 'test_household'
                self.income = 50000
                self.income_class = 3
                self.installation_cost = 15000
                self.npv = 5000
                self.pos = (2.0, 3.0)
                self.is_prosumer = False
                self.scenario_adoption = {'herding': False}
                self.model = model  # Use the provided model
        
        # Create model first, then household
        mock_model = MockModel() 
        test_household = MockHousehold(mock_model)
        
        # Add household to model's schedule (for herding calculations)
        mock_model.schedule.agents.append(test_household)
        
        # Test NPV to probability conversion
        base_prob = bias_manager._npv_to_probability(5000)
        print(f"✓ NPV to probability conversion: NPV=5000 → P={base_prob:.3f}")
        
        if not (0 <= base_prob <= 1):
            print(f"❌ Invalid probability: {base_prob}")
            return False
        
        # Test individual bias applications
        for bias_name in bias_manager.enabled_biases:
            try:
                biased_prob = bias_manager.apply_single_bias(
                    test_household, bias_name, 5000, base_prob
                )
                effect = biased_prob / base_prob if base_prob > 0 else 1.0
                print(f"✓ {bias_name}: P={biased_prob:.3f} (effect: {effect:.2f}x)")
                
                if not (0 <= biased_prob <= 1):
                    print(f"❌ Invalid biased probability for {bias_name}: {biased_prob}")
                    return False
                    
            except Exception as e:
                print(f"❌ Error testing {bias_name}: {e}")
                return False
        
        # Test combined bias application
        combined_prob = bias_manager.apply_all_biases(test_household, 5000, base_prob)
        combined_effect = combined_prob / base_prob if base_prob > 0 else 1.0
        print(f"✓ Combined biases: P={combined_prob:.3f} (total effect: {combined_effect:.2f}x)")
        
        # Test bias summary
        summary = bias_manager.get_bias_summary(test_household)
        print(f"✓ Bias summary generated with {len(summary['bias_effects'])} bias effects")
        
        # Run built-in validation test
        print("\n--- Running BiasManager validation test ---")
        success = test_bias_manager()
        
        if success:
            print("\n✅ BiasManager test PASSED")
            return True
        else:
            print("\n❌ BiasManager validation test FAILED")
            return False
        
    except Exception as e:
        print(f"❌ BiasManager test FAILED: {e}")
        traceback.print_exc()
        return False

def test_visualization_sampler():
    """Test the VisualizationSampler functionality."""
    print("\n" + "="*60)
    print("TESTING: VisualizationSampler Functionality")
    print("="*60)
    
    try:
        from simulation.utils.sampling_utils import (
            VisualizationSampler, test_visualization_sampler
        )
        
        print("✓ Successfully imported VisualizationSampler")
        
        # Run built-in test
        print("\n--- Running VisualizationSampler validation test ---")
        success = test_visualization_sampler()
        
        if success:
            print("\n✅ VisualizationSampler test PASSED")
            return True
        else:
            print("\n❌ VisualizationSampler test FAILED")
            return False
        
    except Exception as e:
        print(f"❌ VisualizationSampler test FAILED: {e}")
        traceback.print_exc()
        return False

def test_spatial_visualizer():
    """Test the SpatialVisualizer functionality."""
    print("\n" + "="*60)
    print("TESTING: SpatialVisualizer Functionality")
    print("="*60)
    
    try:
        from simulation.data.spatial_visualizer import (
            SpatialVisualizer, test_spatial_visualizer
        )
        
        print("✓ Successfully imported SpatialVisualizer")
        
        # Run built-in test
        print("\n--- Running SpatialVisualizer validation test ---")
        success = test_spatial_visualizer()
        
        if success:
            print("\n✅ SpatialVisualizer test PASSED")
            return True
        else:
            print("\n❌ SpatialVisualizer test FAILED")
            return False
        
    except Exception as e:
        print(f"❌ SpatialVisualizer test FAILED: {e}")
        traceback.print_exc()
        return False

def test_network_initialization():
    """Test network initialization and household positioning."""
    print("\n" + "="*60)
    print("TESTING: Network Initialization and Positioning")
    print("="*60)
    
    try:
        # Test with existing rational model to verify network setup
        from simulation.models.rational_model import RationalModel
        from simulation.utils.config_loader import create_rational_experiment
        
        print("✓ Successfully imported network components")
        
        # Create small test configuration
        config = create_rational_experiment().get_copy()
        config["num_households"] = 25  # Small test population
        config["steps"] = 1  # Just initialization
        
        print(f"✓ Created test configuration: {config['num_households']} households")
        
        # Initialize model
        model = RationalModel(config)
        
        print("✓ RationalModel initialized successfully")
        
        # Check household creation
        households = [agent for agent in model.schedule.agents 
                     if hasattr(agent, "daily_consumption")]
        
        print(f"✓ Created {len(households)} household agents")
        
        if len(households) != config["num_households"]:
            print(f"❌ Expected {config['num_households']} households, got {len(households)}")
            return False
        
        # Check household positioning
        positioned_households = 0
        for household in households:
            if hasattr(household, 'pos') and household.pos is not None:
                positioned_households += 1
        
        print(f"✓ {positioned_households} households have spatial positions")
        
        if positioned_households != len(households):
            print(f"❌ Not all households have positions: {positioned_households}/{len(households)}")
            return False
        
        # Check network structure
        if hasattr(model, 'grid') and hasattr(model.grid, 'G'):
            network = model.grid.G
            print(f"✓ Network created with {network.number_of_nodes()} nodes and {network.number_of_edges()} edges")
            
            # Check connectivity
            avg_degree = 2 * network.number_of_edges() / network.number_of_nodes() if network.number_of_nodes() > 0 else 0
            print(f"✓ Average network degree: {avg_degree:.2f}")
            
            if avg_degree < 5:  # Expect roughly 10 neighbors per household
                print(f"⚠️  Low network connectivity: {avg_degree:.2f} (expected ~10)")
        
        # Check household attributes
        income_classes = set()
        prosumers = 0
        
        for household in households:
            income_classes.add(getattr(household, 'income_class', 0))
            if getattr(household, 'is_prosumer', False):
                prosumers += 1
        
        print(f"✓ Income classes present: {sorted(income_classes)}")
        print(f"✓ Initial prosumers: {prosumers} ({100*prosumers/len(households):.1f}%)")
        
        # Test one simulation step
        print("\n--- Testing one simulation step ---")
        model.step()
        print("✓ Successfully executed one simulation step")
        
        # Check data collection
        if hasattr(model, 'data_collector'):
            try:
                model_data = model.data_collector.get_model_data()
                agent_data = model.data_collector.get_agent_data()
                print(f"✓ Data collection working: {len(model_data)} model records, {len(agent_data)} agent records")
            except Exception as e:
                print(f"⚠️  Data collection issue: {e}")
        
        print("\n✅ Network initialization test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Network initialization test FAILED: {e}")
        traceback.print_exc()
        return False

def test_integration():
    """Test integration between components."""
    print("\n" + "="*60)
    print("TESTING: Component Integration")
    print("="*60)
    
    try:
        from simulation.utils.config_loader import create_rational_experiment
        from simulation.agents.bias_manager import BiasManager
        from simulation.utils.sampling_utils import VisualizationSampler
        from simulation.data.spatial_visualizer import SpatialVisualizer
        
        print("✓ All components imported successfully")
        
        # Create configuration
        config = create_rational_experiment().get_copy()
        
        # Test BiasManager with configuration
        bias_manager = BiasManager(config)
        print(f"✓ BiasManager integrated with configuration: {len(bias_manager.enabled_biases)} biases")
        
        # Test VisualizationSampler with spatial configuration
        sampler = VisualizationSampler()
        print(f"✓ VisualizationSampler initialized with sample size: {sampler.sample_size}")
        
        # Test SpatialVisualizer with configuration
        visualizer = SpatialVisualizer(config=config)
        print(f"✓ SpatialVisualizer integrated with {len(visualizer.scenarios)} scenarios")
        
        # Test scenario consistency across components
        from simulation.utils.parameters import get_all_scenarios, get_scenario_colors
        
        scenarios = get_all_scenarios()
        colors = get_scenario_colors()
        
        print(f"✓ Consistent scenarios across components: {scenarios}")
        print(f"✓ Scenario colors defined: {len(colors)} mappings")
        
        if len(scenarios) != len(colors):
            print(f"❌ Scenario-color mismatch: {len(scenarios)} scenarios vs {len(colors)} colors")
            return False
        
        # Test that all enabled biases have scenarios
        enabled_biases = bias_manager.enabled_biases
        missing_scenarios = set(enabled_biases) - set(scenarios)
        
        if missing_scenarios:
            print(f"❌ Missing scenarios for biases: {missing_scenarios}")
            return False
        
        print("✓ All enabled biases have corresponding scenarios")
        
        print("\n✅ Component integration test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Integration test FAILED: {e}")
        traceback.print_exc()
        return False

def create_test_visualization():
    """Create a test visualization to verify everything works."""
    print("\n" + "="*60)
    print("CREATING: Test Visualization")
    print("="*60)
    
    try:
        import matplotlib.pyplot as plt
        from simulation.utils.parameters import SPATIAL_VISUALIZATION_CONFIG, get_all_scenarios
        from simulation.data.spatial_visualizer import MockHouseholdForVisualization
        
        print("✓ Imports successful")
        
        # Create test output directory
        test_dir = "test_results/phase1_validation"
        os.makedirs(test_dir, exist_ok=True)
        print(f"✓ Created test directory: {test_dir}")
        
        # Create mock households with different scenarios
        scenarios = get_all_scenarios()
        n_households = 100
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, scenario in enumerate(scenarios[:6]):  # Limit to 6 for visualization
            ax = axes[i]
            
            # Create mock households for this scenario
            households = []
            for j in range(n_households):
                # Vary adoption rate by scenario
                base_adoption_rate = 0.1 + i * 0.1  # 10% to 60%
                is_prosumer = np.random.random() < base_adoption_rate
                
                household = MockHouseholdForVisualization({
                    'AgentID': j,
                    'IncomeClass': (j % 5) + 1,
                    'IsProsumer': is_prosumer,
                    'Income': 30000 + (j % 5) * 15000,
                    'PosX': np.random.uniform(-5, 5),
                    'PosY': np.random.uniform(-5, 5)
                })
                households.append(household)
            
            # Plot households
            positions = np.array([h.pos for h in households])
            income_classes = np.array([h.income_class for h in households])
            is_prosumer = np.array([h.is_prosumer for h in households])
            
            # Plot non-prosumers
            nonprosumer_mask = ~is_prosumer
            if np.any(nonprosumer_mask):
                nonprosumer_pos = positions[nonprosumer_mask]
                nonprosumer_classes = income_classes[nonprosumer_mask]
                
                for income_class in np.unique(nonprosumer_classes):
                    class_mask = nonprosumer_classes == income_class
                    class_pos = nonprosumer_pos[class_mask]
                    
                    if len(class_pos) > 0:
                        color = SPATIAL_VISUALIZATION_CONFIG['income_class_colors'][income_class]
                        ax.scatter(class_pos[:, 0], class_pos[:, 1],
                                 c=color, marker='o', s=30, alpha=0.4, edgecolors='white')
            
            # Plot prosumers
            prosumer_mask = is_prosumer
            if np.any(prosumer_mask):
                prosumer_pos = positions[prosumer_mask]
                prosumer_classes = income_classes[prosumer_mask]
                
                for income_class in np.unique(prosumer_classes):
                    class_mask = prosumer_classes == income_class
                    class_pos = prosumer_pos[class_mask]
                    
                    if len(class_pos) > 0:
                        color = SPATIAL_VISUALIZATION_CONFIG['income_class_colors'][income_class]
                        ax.scatter(class_pos[:, 0], class_pos[:, 1],
                                 c=color, marker='o', s=60, alpha=0.8, 
                                 edgecolors='black', linewidths=1)
            
            # Format plot
            adoption_rate = np.mean(is_prosumer) * 100
            ax.set_title(f'{scenario.replace("_", " ").title()}\n{adoption_rate:.1f}% Adoption')
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.set_xticks([])
            ax.set_yticks([])
        
        # Hide unused subplots
        for i in range(len(scenarios), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Phase 1 Test: Spatial Visualization by Scenario', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save test visualization
        output_path = os.path.join(test_dir, 'phase1_spatial_test.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Test visualization saved: {output_path}")
        
        # Create configuration summary plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Plot bias configuration summary
        from simulation.utils.parameters import BEHAVIORAL_BIASES
        
        bias_names = []
        enabled_status = []
        colors = []
        
        for bias_name, config in BEHAVIORAL_BIASES.items():
            bias_names.append(bias_name.replace('_', ' ').title())
            enabled_status.append(1 if config['enabled'] else 0)
            colors.append('green' if config['enabled'] else 'red')
        
        bars = ax.bar(bias_names, enabled_status, color=colors, alpha=0.7)
        ax.set_title('Behavioral Biases Configuration Status', fontsize=14, fontweight='bold')
        ax.set_ylabel('Enabled (1) / Disabled (0)')
        ax.set_ylim(0, 1.2)
        
        # Add labels
        for bar, enabled in zip(bars, enabled_status):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                   'ENABLED' if enabled else 'DISABLED',
                   ha='center', va='bottom', fontweight='bold')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        config_path = os.path.join(test_dir, 'bias_configuration_status.png')
        plt.savefig(config_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Configuration summary saved: {config_path}")
        
        print("\n✅ Test visualizations created successfully")
        return True
        
    except Exception as e:
        print(f"❌ Test visualization creation FAILED: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all Phase 1 tests."""
    print("🚀 STARTING PHASE 1 CORE INFRASTRUCTURE TESTS")
    print("=" * 80)
    
    # Track test results
    test_results = {}
    
    # Run individual tests
    test_results['parameters'] = test_enhanced_parameters()
    test_results['bias_manager'] = test_bias_manager()
    test_results['sampler'] = test_visualization_sampler()
    test_results['spatial_viz'] = test_spatial_visualizer()
    test_results['network_init'] = test_network_initialization()
    test_results['integration'] = test_integration()
    test_results['visualization'] = create_test_visualization()
    
    # Summary
    print("\n" + "="*80)
    print("PHASE 1 TEST RESULTS SUMMARY")
    print("="*80)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:20s}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed ({100*passed/total:.1f}%)")
    
    if passed == total:
        print("\n🎉 ALL PHASE 1 TESTS PASSED!")
        print("✅ Core infrastructure is ready for Phase 2 implementation")
        print("\nNext steps:")
        print("1. Review test outputs in test_results/phase1_validation/")
        print("2. Proceed with Phase 2: Multi-Scenario Household implementation")
        print("3. Keep existing rational model as baseline reference")
        return True
    else:
        print(f"\n❌ {total-passed} test(s) failed. Please fix issues before proceeding to Phase 2.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)