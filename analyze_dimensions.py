"""Dimensional Analysis Runner

This script runs a comprehensive dimensional analysis on a semantic field
and generates visualizations of the results.
"""
import os
import torch
import numpy as np
from semantic_axis import SemanticField, SemanticParticle
from dimensional_analysis import run_analysis

def create_test_field():
    """Create a test semantic field with sample data."""
    field = SemanticField()
    
    # Add some test particles with controlled patterns
    concepts = {
        'science': [0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.1, 0.9, 0.2, 0.8, 0.3, 0.7],
        'art': [0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.9, 0.1, 0.8, 0.2, 0.7, 0.3],
        'technology': [0.8, 0.2, 0.9, 0.1, 0.6, 0.4, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6],
        'history': [0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4],
        'mathematics': [0.85, 0.15, 0.75, 0.25, 0.65, 0.35, 0.15, 0.85, 0.25, 0.75, 0.35, 0.65],
        'music': [0.15, 0.85, 0.25, 0.75, 0.35, 0.65, 0.85, 0.15, 0.75, 0.25, 0.65, 0.35],
    }
    
    # Add some random variations
    for concept, vector in concepts.items():
        for _ in range(5):  # 5 variations per concept
            # Add some noise to create variations
            noise = np.random.normal(0, 0.05, 12)
            noisy_vector = np.clip(np.array(vector) + noise, 0, 1)
            # Convert numpy array to list before creating tensor
            field.add_particle(SemanticParticle(concept, noisy_vector.tolist()))
    
    return field

def main():
    print("Creating test semantic field...")
    field = create_test_field()
    
    print("Running dimensional analysis...")
    results = run_analysis(
        field,
        num_samples=1000,
        visualize=True
    )
    
    # Print summary statistics
    print("\n=== Dimensional Analysis Summary ===")
    print(f"Number of particles: {len(field.particles)}")
    print(f"Dimensionality: {len(field.particles[0].vector) if field.particles else 0}")
    print("\nVariance by dimension:", np.round(results['variance'], 4))
    print("\nAverage impact on similarity:", np.round(results['impact'], 4))
    print("\nAverage sensitivity:", np.round(results['sensitivity'], 4))
    
    # Save results to file
    output_dir = "analysis_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save visualizations
    import matplotlib.pyplot as plt
    plt.savefig(os.path.join(output_dir, "dimensional_analysis.png"))
    
    # Save numerical results
    np.savez_compressed(
        os.path.join(output_dir, "dimensional_analysis.npz"),
        variance=results['variance'],
        correlation=results['correlation'],
        impact=results['impact'],
        sensitivity=results['sensitivity']
    )
    
    print(f"\nAnalysis complete. Results saved to '{output_dir}/'")

if __name__ == "__main__":
    main()
