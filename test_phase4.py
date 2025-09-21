#!/usr/bin/env python3
# test_phase4_fixes.py - Quick fixes for Phase 4 testing issues
"""
Quick fixes for the Phase 4 testing issues identified.
This script can be run to verify the fixes work.
"""

import os
import sys
import pandas as pd
import numpy as np

# Add the simulation package to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_comparative_visualizer_fix():
    """Test the fixed ComparativeVisualizer."""
    print("Testing ComparativeVisualizer fixes...")
    
    try:
        from simulation.data.comparative_visualizer import ComparativeVisualizer
        from simulation.utils.parameters import get_all_scenarios
        
        # Create properly structured mock data
        scenarios = get_all_scenarios()
        years = list(range(1, 6))  # 5 years for faster testing
        
        # Create system data with correct structure
        system_records = []
        for year in years:
            record = {
                'Year': year,
                'Month': 12,
                'TotalHouseholds': 100
            }
            
            # Add scenario-specific columns
            for i, scenario in enumerate(scenarios):
                base_rate = min(0.8, year * 0.05)  # Growth over time
                
                # Different scenarios have different rates
                if scenario == 'rational':
                    rate = base_rate
                elif scenario == 'loss_aversion':
                    rate = base_rate * 0.8
                elif scenario == 'all_biases':
                    rate = base_rate * 0.5
                else:
                    rate = base_rate * 0.9
                
                record[f'{scenario}_AdoptionRate'] = rate
                record[f'{scenario}_AdopterCount'] = int(rate * 100)
            
            system_records.append(record)
        
        system_df = pd.DataFrame(system_records)
        
        # Create agent data with matching structure
        agent_records = []
        for year in years:
            for household_id in range(1, 21):  # 20 households
                for scenario in scenarios:
                    year_system_data = system_df[system_df['Year'] == year]
                    if not year_system_data.empty:
                        adoption_rate = year_system_data[f'{scenario}_AdoptionRate'].iloc[0]
                        is_prosumer = np.random.random() < adoption_rate
                    else:
                        is_prosumer = False
                    
                    record = {
                        'Year': year,
                        'HouseholdID': household_id,
                        'Scenario': scenario,
                        'IsProsumer': is_prosumer,
                        'IncomeClass': ((household_id - 1) % 5) + 1
                    }
                    agent_records.append(record)
        
        agent_df = pd.DataFrame(agent_records)
        
        # Test ComparativeVisualizer
        os.makedirs('test_results/fixes', exist_ok=True)
        
        visualizer = ComparativeVisualizer(
            model_data=system_df,
            agent_data=agent_df
        )
        
        # Test data extraction
        adoption_data = visualizer._extract_adoption_time_series()
        if adoption_data.empty:
            print("❌ Still no adoption data extracted")
            return False
        
        print(f"✅ Adoption data extracted: {adoption_data.shape}")
        
        # Test visualization creation
        visualizer.create_adoption_comparison('test_results/fixes')
        print("✅ Adoption comparison created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ ComparativeVisualizer fix test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_detailed_tracker_fix():
    """Test the fixed DetailedTracker."""
    print("\nTesting DetailedTracker fixes...")
    
    try:
        # Import after adding the method
        import importlib
        import simulation.data.detailed_tracker
        importlib.reload(simulation.data.detailed_tracker)
        
        from simulation.data.detailed_tracker import DetailedTracker
        
        class MockModel:
            def __init__(self):
                class MockSchedule:
                    def __init__(self):
                        self.agents = []
                        self.steps = 1
                self.schedule = MockSchedule()
        
        class MockHousehold:
            def __init__(self, unique_id):
                self.unique_id = unique_id
                self.daily_consumption = 20
                self.income = 50000
                self.income_class = 3
                self.pos = (0, 0)
        
        mock_model = MockModel()
        for i in range(5):
            mock_model.schedule.agents.append(MockHousehold(i))
        
        tracker = DetailedTracker(mock_model, sample_size=3)
        
        # Test the new method
        comparison_data = tracker.get_scenario_comparison_data()
        print(f"✅ get_scenario_comparison_data method works: {type(comparison_data)}")
        
        return True
        
    except Exception as e:
        print(f"❌ DetailedTracker fix test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_quick_fixes_test():
    """Run all quick fix tests."""
    print("PHASE 4 QUICK FIXES TEST")
    print("=" * 40)
    
    tests = [
        ("ComparativeVisualizer Fix", test_comparative_visualizer_fix),
        ("DetailedTracker Fix", test_detailed_tracker_fix)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 40)
    print("QUICK FIXES SUMMARY")
    print("=" * 40)
    
    for test_name, success in results:
        status = "✅ FIXED" if success else "❌ STILL BROKEN"
        print(f"{test_name:25s}: {status}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    if passed == total:
        print(f"\n🎉 All fixes successful! ({passed}/{total})")
        print("You can now re-run the full Phase 4 tests.")
    else:
        print(f"\n⚠️  {total-passed} issues still need attention.")
    
    return passed == total

if __name__ == "__main__":
    run_quick_fixes_test()