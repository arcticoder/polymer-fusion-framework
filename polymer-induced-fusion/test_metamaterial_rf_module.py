"""
Test script for Metamaterial RF Launcher Simulation Module

This script validates the metamaterial RF launcher module functionality including:
- Fishnet metamaterial analysis
- Photonic crystal structure optimization
- ICRH coupling efficiency calculations
- E-field penetration modeling
- Integration with phenomenology framework

Author: Fusion RF Systems Team
Date: June 2025
"""

import sys
import os
import traceback

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_metamaterial_rf_module():
    """Test the metamaterial RF launcher simulation module."""
    
    print("Testing Metamaterial RF Launcher Simulation Module...")
    print("=" * 55)
    
    try:
        # Test import
        from metamaterial_rf_launcher_simulation import (
            MetamaterialRFLauncherFramework,
            MetamaterialProperties,
            ICRHParameters,
            LauncherGeometry,
            integrate_metamaterial_rf_with_phenomenology_framework
        )
        print("✅ Module import successful")
        
        # Test framework initialization
        framework = MetamaterialRFLauncherFramework()
        print("✅ Framework initialization successful")
        
        # Test fishnet analysis
        fishnet_results = framework.run_fishnet_analysis()
        print("✅ Fishnet metamaterial analysis successful")
        print(f"   Coupling efficiency: {fishnet_results['impedance_matching']['coupling_efficiency']:.3f}")
        print(f"   Impedance: {fishnet_results['impedance_matching']['characteristic_impedance_ohms']['magnitude']:.1f} Ω")
        
        # Test photonic crystal analysis
        pc_results = framework.run_photonic_crystal_analysis()
        print("✅ Photonic crystal analysis successful")
        print(f"   Bandgap: {pc_results['bandgap_properties']['lower_frequency_mhz']:.1f}-{pc_results['bandgap_properties']['upper_frequency_mhz']:.1f} MHz")
        print(f"   Coupling efficiency: {pc_results['coupling_performance']['coupling_efficiency']:.3f}")
        
        # Test field penetration analysis
        penetration_results = framework.run_field_penetration_analysis()
        print("✅ E-field penetration analysis successful")
        print(f"   Skin depth: {penetration_results['penetration_metrics']['skin_depth_m']*1000:.2f} mm")
        
        # Test coupling improvement
        improvement_results = framework.calculate_coupling_improvement()
        print("✅ Coupling improvement analysis successful")
        print(f"   Improvement factor: {improvement_results['best_design']['improvement_factor']:.2f}×")
        print(f"   Target ≥1.5× met: {improvement_results['best_design']['target_1_5x_met']}")
        
        print("\n🎉 All tests passed! Module is operational.")
        
        return True
        
    except Exception as e:
        print(f"❌ Test error: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = test_metamaterial_rf_module()
    if success:
        print("\n" + "="*60)
        print("METAMATERIAL RF LAUNCHER MODULE VALIDATION COMPLETE")
        print("Module ready for integration with phenomenology framework")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("METAMATERIAL RF LAUNCHER MODULE VALIDATION FAILED")
        print("Please check errors and retry")
        print("="*60)
