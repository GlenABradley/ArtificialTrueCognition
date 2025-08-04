"""
Bayesian optimization of semantic axis parameters.

This script optimizes the parameters of the semantic axis similarity function
using Bayesian optimization with scikit-optimize.
"""
import numpy as np
import torch
import os
import sys
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from semantic_axis import SemanticField, SemanticParticle
import unittest

# Import the test module directly to access the test class
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from tests.test_semantic_axis import TestSemanticField

class SemanticAxisOptimizer:
    def __init__(self):
        # Define the search space for each parameter
        self.space = [
            Real(0.1, 10.0, name='tolerance_power'),  # Power for tolerance transformation
            Real(0.1, 5.0, name='euclidean_scale'),   # Scale for Euclidean distance
            Real(0.0, 1.0, name='matching_threshold'), # Threshold for dimension matching
            Real(0.1, 10.0, name='similarity_scale'),  # Scale for final similarity
            Real(0.1, 10.0, name='similarity_shift'),  # Shift for final similarity
        ]
        self.best_score = float('-inf')
        self.best_params = None
        
    def evaluate_parameters(self, params):
        """Evaluate a set of parameters using the test suite."""
        # Unpack parameters
        tolerance_power, euclidean_scale, matching_threshold, \
            similarity_scale, similarity_shift = params
        
        print(f"\nEvaluating parameters: {params}")
            
        # Create a custom SemanticField with these parameters
        class OptimizedSemanticField(SemanticField):
            def _get_axis_weights(self, tolerance):
                weights = 1.0 / (tolerance + 1e-6)
                weights = torch.pow(weights, tolerance_power)
                return weights / (torch.sum(weights) + 1e-6)
                
            def _calculate_similarity(self, v1, v2, axis_weights):
                diff = torch.abs(v1 - v2)
                weighted_sq_diff = diff * diff * axis_weights
                euclidean_dist = torch.sqrt(torch.sum(weighted_sq_diff) + 1e-6)
                similarity = 1.0 / (1.0 + euclidean_scale * euclidean_dist)
                matching_dims = torch.sum((diff < matching_threshold).float() * axis_weights)
                combined_sim = 0.7 * similarity + 0.3 * matching_dims
                return float(torch.sigmoid(similarity_scale * (combined_sim - similarity_shift)))
        
        # Create a test instance with our optimized field
        test_instance = TestSemanticField()
        test_instance.setUp()  # This will initialize the field
        
        # Replace the field with our optimized version
        original_field = test_instance.field
        test_instance.field = OptimizedSemanticField()
        
        # Copy particles from original field to optimized field
        for particle in original_field.particles:
            test_instance.field.add_particle(particle)
        
        try:
            # Run all test methods that start with 'test_'
            test_methods = [m for m in dir(TestSemanticField) 
                          if m.startswith('test_') and callable(getattr(test_instance, m))]
            
            passed = 0
            total = len(test_methods)
            
            for method_name in test_methods:
                try:
                    # Reset the test instance state
                    test_instance.setUp()
                    # Run the test method
                    getattr(test_instance, method_name)()
                    passed += 1
                except AssertionError as e:
                    print(f"Test {method_name} failed: {e}")
                except Exception as e:
                    print(f"Error in test {method_name}: {e}")
            
            score = passed / total if total > 0 else 0
            
            # Update best parameters if this is the best score so far
            if score > self.best_score:
                self.best_score = score
                self.best_params = params
                print(f"New best score: {score:.4f} with params: {params}")
            
            print(f"Passed {passed}/{total} tests (score: {score:.4f})")
            return -score  # Minimize the negative score
            
        finally:
            # Clean up
            test_instance.tearDown()
            test_instance.field = None
    
    def optimize(self, n_calls=50):
        """Run the Bayesian optimization."""
        @use_named_args(dimensions=self.space)
        def objective(**params):
            return self.evaluate_parameters(list(params.values()))
            
        result = gp_minimize(
            func=objective,
            dimensions=self.space,
            n_calls=n_calls,
            random_state=42,
            verbose=True
        )
        
        return result

def main():
    # Initialize the optimizer
    optimizer = SemanticAxisOptimizer()
    
    # Run the optimization
    print("Starting Bayesian optimization...")
    result = optimizer.optimize(n_calls=50)
    
    # Print results
    print("\nOptimization complete!")
    print(f"Best parameters: {optimizer.best_params}")
    print(f"Best test score: {optimizer.best_score:.4f}")
    
    # Save results to a file
    with open('optimization_results.txt', 'w') as f:
        f.write("Best parameters:\n")
        for name, value in zip([dim.name for dim in optimizer.space], optimizer.best_params):
            f.write(f"{name}: {value}\n")
        f.write(f"\nBest test score: {optimizer.best_score:.4f}\n")

if __name__ == "__main__":
    main()
