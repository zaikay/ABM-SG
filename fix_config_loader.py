#!/usr/bin/env python3
"""
Quick fix script for config_loader.py SimulationConfig constructor issue.
"""

import os
import re

def fix_config_loader():
    """Fix the SimulationConfig constructor calls in config_loader.py"""
    
    config_file = "simulation/utils/config_loader.py"
    
    if not os.path.exists(config_file):
        print(f"❌ File not found: {config_file}")
        return False
    
    print(f"🔧 Fixing {config_file}...")
    
    # Read the file
    with open(config_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix 1: create_multi_experiment_config
    old_pattern1 = r'return SimulationConfig\("multi_experiment_behavioral", base_config\)'
    new_pattern1 = '''config = SimulationConfig("multi_experiment_behavioral")
    config.config = base_config
    return config'''
    
    content = re.sub(old_pattern1, new_pattern1, content)
    
    # Fix 2: create_single_bias_experiment  
    old_pattern2 = r'return SimulationConfig\(experiment_name, base_config\)'
    new_pattern2 = '''config = SimulationConfig(experiment_name)
    config.config = base_config
    return config'''
    
    content = re.sub(old_pattern2, new_pattern2, content)
    
    # Fix 3: create_sensitivity_analysis_config
    old_pattern3 = r'return SimulationConfig\("sensitivity_analysis", base_config\)'
    new_pattern3 = '''config = SimulationConfig("sensitivity_analysis")
    config.config = base_config
    return config'''
    
    content = re.sub(old_pattern3, new_pattern3, content)
    
    # Fix 4: Update create_testing_config to not call create_multi_experiment_config
    # This is more complex, so we'll replace the entire function
    testing_config_old = r'def create_testing_config\(num_households=50, steps=12\):.*?return config'
    testing_config_new = '''def create_testing_config(num_households=50, steps=12):
    """
    Create a lightweight configuration for testing purposes.
    
    Args:
        num_households: Number of households for testing (default: 50)
        steps: Number of simulation steps (default: 12 = 1 year)
        
    Returns:
        SimulationConfig: Configuration for testing
    """
    # Create base config without calling create_multi_experiment_config
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
            "export_individual_scenarios": False,
            "export_combined_dataset": True,
            "export_scenario_metadata": True
        }
    }
    
    # Testing overrides
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
            "sample_size": min(25, num_households)
        }
    }
    
    # Combine configurations
    base_config.update(behavioral_config)
    base_config.update(testing_overrides)
    
    # Create config correctly
    config = SimulationConfig("testing_behavioral")
    config.config = base_config
    return config'''
    
    content = re.sub(testing_config_old, testing_config_new, content, flags=re.DOTALL)
    
    # Write the fixed file
    with open(config_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ Fixed {config_file}")
    return True

if __name__ == "__main__":
    success = fix_config_loader()
    if success:
        print("🎉 Config loader fixed! Run the tests again:")
        print("python test_phase2.py")
    else:
        print("❌ Fix failed. Please apply manual fixes.")