"""
Quick test script for liquid-metal divertor simulation module
"""

import sys
import os

def test_liquid_metal_module():
    """Test the liquid-metal divertor simulation module."""
    
    print("Testing Liquid-Metal Divertor Simulation Module...")
    print("=" * 55)
    
    try:
        # Import the module
        from liquid_metal_divertor_simulation import LiquidMetalDivertorFramework
        print("✅ Module import successful")
        
        # Initialize framework
        framework = LiquidMetalDivertorFramework()
        print("✅ Framework initialization successful")
        
        # Test MHD analysis
        mhd_results = framework.run_mhd_analysis()
        print("✅ MHD analysis successful")
        print(f"   Hartmann number: {mhd_results['mhd_parameters']['hartmann_number']:.1f}")
        print(f"   Peak heat flux: {mhd_results['heat_flux_profile']['peak_heat_flux_mw_m2']:.1f} MW/m²")
        
        # Test erosion analysis
        erosion_results = framework.run_erosion_analysis()
        print("✅ Erosion analysis successful")
        print(f"   Equilibrium achieved: {erosion_results['equilibrium_assessment']['equilibrium_achieved']}")
        print(f"   Minimum lifetime: {erosion_results['lifetime_analysis']['minimum_lifetime_hours']:.1f} hours")
        
        print("\n🎉 All tests passed! Module is operational.")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False

if __name__ == "__main__":
    success = test_liquid_metal_module()
    sys.exit(0 if success else 1)
