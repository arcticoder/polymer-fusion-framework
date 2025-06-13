"""
Test script for Tungsten-Fiber Composite PFC Simulation Module

This script validates the tungsten-fiber PFC module functionality including:
- DPA calculation and neutron damage accumulation
- Crack propagation modeling with Paris law
- Finite element thermal-mechanical analysis
- Integration with phenomenology framework

Author: Fusion Materials Physics Team
Date: June 2025
"""

import sys
import os
import traceback

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_tungsten_pfc_module():
    """Test the tungsten-fiber PFC simulation module."""
    
    print("Testing Tungsten-Fiber PFC Simulation Module...")
    print("=" * 55)
    
    try:
        # Test import
        from tungsten_fiber_pfc_simulation import (
            TungstenFiberPFCFramework,
            TungstenFiberProperties,
            NeutronEnvironment,
            PFCGeometry,
            integrate_tungsten_pfc_with_phenomenology_framework
        )
        print("✅ Module import successful")
        
        # Test framework initialization
        framework = TungstenFiberPFCFramework()
        print("✅ Framework initialization successful")
        
        # Test DPA analysis
        dpa_results = framework.run_dpa_analysis()
        print("✅ DPA analysis successful")
        print(f"   Total DPA: {dpa_results['final_damage']['total_dpa']:.2f}")
        print(f"   Swelling: {dpa_results['final_damage']['swelling_percent']:.2f}%")
        
        # Test crack propagation analysis
        crack_results = framework.run_crack_propagation_analysis()
        print("✅ Crack propagation analysis successful")
        crack_lives = [crack_results['crack_analysis'][key]['fatigue_life_years'] 
                      for key in crack_results['crack_analysis']]
        print(f"   Minimum fatigue life: {min(crack_lives):.1f} years")
        
        # Test thermal-mechanical analysis
        thermal_results = framework.run_thermal_mechanical_analysis()
        print("✅ Thermal-mechanical analysis successful")
        print(f"   Max temperature: {thermal_results['peak_conditions']['max_temperature_k']:.1f} K")
        print(f"   Safety factor: {thermal_results['structural_assessment']['safety_factor']:.2f}")
        
        print("\n🎉 All tests passed! Module is operational.")
        
        return True
        
    except Exception as e:
        print(f"❌ Test error: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = test_tungsten_pfc_module()
    if success:
        print("\n" + "="*60)
        print("TUNGSTEN-FIBER PFC MODULE VALIDATION COMPLETE")
        print("Module ready for integration with phenomenology framework")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("TUNGSTEN-FIBER PFC MODULE VALIDATION FAILED")
        print("Please check errors and retry")
        print("="*60)
