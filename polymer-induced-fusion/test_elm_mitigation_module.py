"""
Test script for Dynamic ELM Mitigation Simulation Module

This script validates the functionality of the dynamic ELM mitigation
simulation module including RMP optimization, ELM dynamics modeling,
and heat pulse load minimization.
"""

import sys
import os
import numpy as np

# Add the current directory to Python path for imports
sys.path.append(os.getcwd())

def test_dynamic_elm_mitigation_module():
    """Test the dynamic ELM mitigation simulation module."""
    
    print("Testing Dynamic ELM Mitigation Simulation Module")
    print("=" * 50)
    
    try:
        # Import the module
        from dynamic_elm_mitigation_simulation import (
            RMPCoilSystem,
            ELMDynamicsModel,
            DynamicELMMitigationFramework,
            integrate_elm_mitigation_with_phenomenology_framework
        )
        print("✓ Module import successful")
        
        # Test 1: RMP Coil System
        print("\n1. Testing RMP Coil System...")
        rmp_system = RMPCoilSystem(n_coils=18, coil_current_max=5000.0, toroidal_mode_n=3)
        assert rmp_system.n_coils == 18
        assert rmp_system.I_max == 5000.0
        assert rmp_system.n_mode == 3
        assert len(rmp_system.coil_angles) == 18
        print("  ✓ RMP coil system initialization successful")
        
        # Test coil phase optimization
        coil_currents = rmp_system.optimize_coil_phases(target_phase=np.pi/4)
        assert len(coil_currents) == 18
        assert np.max(np.abs(coil_currents)) <= rmp_system.I_max
        print(f"  ✓ Coil phase optimization: max current {np.max(np.abs(coil_currents)):.0f} A")
        
        # Test RMP field calculation
        rmp_field = rmp_system.calculate_rmp_field(coil_currents, 0.0, 0.0)
        assert isinstance(rmp_field, complex)
        print(f"  ✓ RMP field calculation: |B| = {abs(rmp_field):.2e}")
        
        # Test 2: ELM Dynamics Model
        print("\n2. Testing ELM Dynamics Model...")
        elm_model = ELMDynamicsModel()
        
        assert 'pedestal_pressure' in elm_model.plasma_params
        assert elm_model.elm_frequency_natural > 0
        print(f"  ✓ ELM model initialization: f_ELM = {elm_model.elm_frequency_natural:.0f} Hz")
        
        # Test ballooning stability
        stability = elm_model.calculate_ballooning_stability(
            pressure_gradient=10000.0, rmp_amplitude=0.002)
        assert isinstance(stability, (int, float))
        assert stability >= 0
        print(f"  ✓ Ballooning stability calculation: α = {stability:.3f}")
        
        # Test ELM heat pulse
        heat_pulse = elm_model.calculate_elm_heat_pulse(
            elm_amplitude=1.0, mitigation_factor=0.3)
        
        required_keys = ['peak_heat_flux', 'pulse_duration', 'energy_density', 'mitigation_effectiveness']
        for key in required_keys:
            assert key in heat_pulse
        
        assert heat_pulse['peak_heat_flux'] >= 0
        assert heat_pulse['pulse_duration'] > 0
        print(f"  ✓ Heat pulse calculation: q = {heat_pulse['peak_heat_flux']:.1f} MW/m²")
        
        # Test 3: Dynamic ELM Mitigation Framework
        print("\n3. Testing Dynamic ELM Mitigation Framework...")
        elm_framework = DynamicELMMitigationFramework(
            optimization_target='minimize_heat_flux'
        )
        
        assert elm_framework.optimization_target == 'minimize_heat_flux'
        assert hasattr(elm_framework, 'rmp_system')
        assert hasattr(elm_framework, 'elm_model')
        print("  ✓ ELM mitigation framework initialization successful")
        
        # Test ELM cycle simulation
        print("    Running ELM cycle simulation...")
        elm_cycle = elm_framework.simulate_elm_cycle(
            rmp_phase=np.pi/3, rmp_timing=0.005, duration=0.05)
        
        required_cycle_keys = ['time', 'pedestal_pressure', 'rmp_amplitude', 
                              'elm_activity', 'heat_flux', 'elm_frequency', 
                              'max_heat_flux', 'average_heat_flux']
        for key in required_cycle_keys:
            assert key in elm_cycle
        
        assert len(elm_cycle['time']) > 0
        assert elm_cycle['max_heat_flux'] >= 0
        assert elm_cycle['elm_frequency'] >= 0
        print(f"  ✓ ELM cycle simulation: f = {elm_cycle['elm_frequency']:.1f} Hz, q_max = {elm_cycle['max_heat_flux']:.1f} MW/m²")
        
        # Test objective function
        test_params = [np.pi/4, 0.01]  # phase, timing
        objective_value = elm_framework.objective_function(test_params)
        assert isinstance(objective_value, (int, float))
        assert objective_value >= 0
        print(f"  ✓ Objective function evaluation: {objective_value:.2f}")
        
        # Test 4: Quick optimization run (reduced iterations for testing)
        print("\n4. Testing optimization algorithm...")
        print("  Running optimization (this may take a moment)...")
        
        # Temporarily reduce optimization parameters for testing
        original_maxiter = 50
        elm_framework.optimization_target = 'minimize_heat_flux'
        
        optimization_results = elm_framework.optimize_rmp_parameters()
        
        # Validate optimization results structure
        required_opt_keys = ['optimal_parameters', 'performance_metrics', 'simulation_results']
        for key in required_opt_keys:
            assert key in optimization_results
        
        # Validate parameter ranges
        params = optimization_results['optimal_parameters']
        assert 0 <= params['rmp_phase_deg'] <= 360
        assert params['rmp_timing_ms'] >= 0
        
        performance = optimization_results['performance_metrics']
        assert 'heat_flux_reduction_percent' in performance
        assert 'optimization_success' in performance
        
        print(f"  ✓ Optimization completed successfully")
        print(f"    Optimal phase: {params['rmp_phase_deg']:.1f}°")
        print(f"    Optimal timing: {params['rmp_timing_ms']:.1f} ms")
        print(f"    Heat flux reduction: {performance['heat_flux_reduction_percent']:.1f}%")
        
        # Test 5: Integration function
        print("\n5. Testing phenomenology framework integration...")
        print("  Running integration test (this may take a moment)...")
        
        integration_results = integrate_elm_mitigation_with_phenomenology_framework()
        
        # Validate integration results
        assert 'simulation_parameters' in integration_results
        assert 'optimization_results' in integration_results
        assert 'phenomenology_summary' in integration_results
        
        pheno_summary = integration_results['phenomenology_summary']
        assert pheno_summary['integration_status'] == 'SUCCESS'
        
        print("  ✓ Integration test successful")
        print(f"    Status: {pheno_summary['integration_status']}")
        print(f"    Heat flux reduction: {pheno_summary['elm_mitigation']['heat_flux_reduction']:.1f}%")
        print(f"    Optimization success: {pheno_summary['elm_mitigation']['optimization_success']}")
        
        print("\n" + "=" * 50)
        print("ALL DYNAMIC ELM MITIGATION MODULE TESTS PASSED! ✓")
        print("=" * 50)
        
        # Print final summary
        final_reduction = integration_results['optimization_results']['performance_metrics']['heat_flux_reduction_percent']
        final_phase = integration_results['optimization_results']['optimal_parameters']['rmp_phase_deg']
        final_timing = integration_results['optimization_results']['optimal_parameters']['rmp_timing_ms']
        
        print(f"\nFINAL PERFORMANCE SUMMARY:")
        print(f"  Heat Flux Reduction: {final_reduction:.1f}%")
        print(f"  Optimal RMP Phase: {final_phase:.1f}°")
        print(f"  Optimal RMP Timing: {final_timing:.1f} ms")
        print(f"  Optimization Success: {integration_results['optimization_results']['performance_metrics']['optimization_success']}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except AssertionError as e:
        print(f"❌ Assertion error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_dynamic_elm_mitigation_module()
    if success:
        print("\n🎉 Dynamic ELM Mitigation Module validation complete!")
    else:
        print("\n💥 Dynamic ELM Mitigation Module validation failed!")
        sys.exit(1)
