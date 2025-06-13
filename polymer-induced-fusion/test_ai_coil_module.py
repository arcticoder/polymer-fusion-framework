"""
Test script for AI-Optimized Coil Geometry Simulation Module

This script validates the functionality of the AI-optimized coil geometry
simulation module including genetic algorithm optimization, field ripple
analysis, and bootstrap current enhancement.
"""

import sys
import os
import numpy as np

# Add the current directory to Python path for imports
sys.path.append(os.getcwd())

def test_ai_optimized_coil_module():
    """Test the AI-optimized coil geometry simulation module."""
    
    print("Testing AI-Optimized Coil Geometry Simulation Module")
    print("=" * 55)
    
    try:
        # Import the module
        from ai_optimized_coil_geometry_simulation import (
            GeneticCoilOptimizer,
            FieldRippleAnalyzer, 
            BootstrapCurrentAnalyzer,
            AIOptimizedCoilFramework,
            integrate_ai_coil_with_phenomenology_framework
        )
        print("✓ Module import successful")
        
        # Test 1: Genetic Coil Optimizer
        print("\n1. Testing Genetic Coil Optimizer...")
        optimizer = GeneticCoilOptimizer(population_size=20, generations=10)
        assert optimizer.population_size == 20
        assert optimizer.generations == 10
        assert len(optimizer.bounds) == 8  # 8 design parameters
        print("  ✓ Genetic optimizer initialization successful")
        
        # Test 2: Field Ripple Analyzer
        print("\n2. Testing Field Ripple Analyzer...")
        ripple_analyzer = FieldRippleAnalyzer(major_radius=6.0, minor_radius=2.0)
        
        # Test with sample coil parameters
        test_params = np.array([0.5, 1.5, 0.4, 4, 15.0, 10.0, 1.2, 0.3])
        ripple = ripple_analyzer.calculate_field_ripple(test_params)
        
        assert isinstance(ripple, (int, float))
        assert ripple >= 0
        print(f"  ✓ Field ripple calculation: {ripple:.2f}%")
        
        # Test 3: Bootstrap Current Analyzer
        print("\n3. Testing Bootstrap Current Analyzer...")
        bootstrap_analyzer = BootstrapCurrentAnalyzer(plasma_beta=0.05, temperature_keV=20.0)
        
        bootstrap_fraction = bootstrap_analyzer.calculate_bootstrap_current(test_params, ripple)
        
        assert isinstance(bootstrap_fraction, (int, float))
        assert 0 <= bootstrap_fraction <= 1
        print(f"  ✓ Bootstrap current calculation: {bootstrap_fraction:.1%}")
        
        # Test 4: AI-Optimized Coil Framework
        print("\n4. Testing AI-Optimized Coil Framework...")
        coil_framework = AIOptimizedCoilFramework(
            major_radius=6.0,
            minor_radius=2.0,
            target_ripple=1.0,
            target_bootstrap=0.8
        )
        
        # Test fitness function
        fitness = coil_framework.fitness_function(test_params)
        assert isinstance(fitness, (int, float))
        assert fitness >= 0
        print(f"  ✓ Fitness function evaluation: {fitness:.4f}")
        
        # Test 5: Quick optimization run (reduced iterations for testing)
        print("\n5. Testing optimization algorithm...")
        coil_framework.optimizer.generations = 5  # Reduce for testing
        coil_framework.optimizer.population_size = 20
        
        optimization_results = coil_framework.run_genetic_optimization()
        
        # Validate optimization results structure
        required_keys = ['best_parameters', 'performance_metrics', 'target_achievement']
        for key in required_keys:
            assert key in optimization_results
        
        # Validate parameter ranges
        params = optimization_results['best_parameters']
        assert 0.1 <= params['inner_radius_ratio'] <= 0.9
        assert 1.1 <= params['outer_radius_ratio'] <= 2.5
        assert 0.1 <= params['height_ratio'] <= 0.8
        assert 2 <= params['number_of_turns'] <= 8
        
        print(f"  ✓ Optimization completed successfully")
        print(f"    Field ripple: {optimization_results['performance_metrics']['field_ripple_percent']:.2f}%")
        print(f"    Bootstrap current: {optimization_results['performance_metrics']['bootstrap_current_fraction']:.1%}")
        
        # Test 6: Integration function
        print("\n6. Testing phenomenology framework integration...")
        print("  Running integration test (this may take a moment)...")
        
        # Run integration with reduced parameters for testing
        original_framework = AIOptimizedCoilFramework()
        original_framework.optimizer.generations = 10  # Reduce for testing
        original_framework.optimizer.population_size = 30
        
        integration_results = integrate_ai_coil_with_phenomenology_framework()
        
        # Validate integration results
        assert 'simulation_parameters' in integration_results
        assert 'optimization_results' in integration_results
        assert 'phenomenology_summary' in integration_results
        
        pheno_summary = integration_results['phenomenology_summary']
        assert pheno_summary['integration_status'] == 'SUCCESS'
        
        print("  ✓ Integration test successful")
        print(f"    Status: {pheno_summary['integration_status']}")
        print(f"    Ripple target met: {pheno_summary['coil_optimization']['ripple_target_met']}")
        print(f"    Bootstrap target met: {pheno_summary['coil_optimization']['bootstrap_target_met']}")
        
        print("\n" + "=" * 55)
        print("ALL AI-OPTIMIZED COIL MODULE TESTS PASSED! ✓")
        print("=" * 55)
        
        # Print final summary
        final_ripple = integration_results['optimization_results']['performance_metrics']['field_ripple_percent']
        final_bootstrap = integration_results['optimization_results']['performance_metrics']['bootstrap_current_fraction']
        
        print(f"\nFINAL PERFORMANCE SUMMARY:")
        print(f"  Field Ripple: {final_ripple:.2f}%")
        print(f"  Bootstrap Current: {final_bootstrap:.1%}")
        print(f"  Optimization Convergence: {integration_results['optimization_results']['performance_metrics']['success']}")
        
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
    success = test_ai_optimized_coil_module()
    if success:
        print("\n🎉 AI-Optimized Coil Geometry Module validation complete!")
    else:
        print("\n💥 AI-Optimized Coil Geometry Module validation failed!")
        sys.exit(1)
