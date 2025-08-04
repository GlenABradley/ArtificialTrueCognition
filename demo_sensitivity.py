"""Demo script for adaptive sensitivity tuning in semantic space."""
import numpy as np
import torch
from semantic_axis import SemanticField, SemanticParticle
from visualize_semantics import SemanticVisualizer
import matplotlib.pyplot as plt

def create_test_field() -> SemanticField:
    """Create a test semantic field with sample concepts."""
    field = SemanticField(target_contrast=0.3)  # Moderate contrast target
    
    # Define some test concepts with semantic relationships
    concepts = {
        # Basic emotions
        'joy': [0.9, 0.8, 0.1, 0.2, 0.9, 0.1, 0.1, 0.2, 0.3, 0.1, 0.4, 0.2],
        'sadness': [0.8, 0.7, 0.2, 0.3, 0.1, 0.9, 0.2, 0.8, 0.4, 0.2, 0.3, 0.3],
        'anger': [0.85, 0.75, 0.3, 0.4, 0.8, 0.7, 0.4, 0.6, 0.6, 0.3, 0.5, 0.4],
        
        # Time-related concepts
        'past': [0.6, 0.5, 0.7, 0.6, 0.3, 0.1, 0.5, 0.7, 0.2, 0.1, 0.6, 0.5],
        'present': [0.5, 0.4, 0.6, 0.5, 0.4, 0.5, 0.6, 0.6, 0.3, 0.2, 0.5, 0.6],
        'future': [0.4, 0.3, 0.5, 0.4, 0.5, 0.9, 0.7, 0.5, 0.4, 0.3, 0.4, 0.7],
        
        # Abstract concepts
        'freedom': [0.7, 0.6, 0.8, 0.9, 0.7, 0.6, 0.8, 0.4, 0.8, 0.9, 0.7, 0.8],
        'justice': [0.6, 0.5, 0.9, 0.8, 0.5, 0.4, 0.7, 0.8, 0.9, 0.8, 0.8, 0.9],
        'beauty': [0.8, 0.7, 0.6, 0.7, 0.9, 0.3, 0.6, 0.3, 0.7, 0.7, 0.9, 0.6]
    }
    
    # Add some noise to create variations of each concept
    for concept, base_vector in concepts.items():
        for i in range(5):  # 5 variations per concept
            noise = np.random.normal(0, 0.05, 12)  # Small noise
            vector = np.clip(base_vector + noise, 0, 1)  # Keep in [0,1] range
            particle = SemanticParticle(f"{concept}_{i}", torch.tensor(vector, dtype=torch.float32))
            field.add_particle(particle)
    
    return field

def run_demo():
    # Create and visualize initial field
    print("Creating test semantic field...")
    field = create_test_field()
    
    print("Initial state:")
    print(f"Number of particles: {len(field.particles)}")
    print(f"Initial sensitivities: {field.sensitivities.numpy().round(3)}")
    
    # Visualize before tuning
    visualizer = SemanticVisualizer(field)
    print("\nVisualizing initial state...")
    visualizer.plot_semantic_space("initial_semantic_space.png")
    
    # Tune sensitivities and visualize
    print("\nTuning sensitivities...")
    field.tune_sensitivities()
    print(f"Adjusted sensitivities: {field.sensitivities.numpy().round(3)}")
    
    # Visualize after tuning
    print("\nVisualizing after sensitivity tuning...")
    visualizer.plot_semantic_space("tuned_semantic_space.png")
    
    # Show the most and least sensitive dimensions
    sorted_dims = np.argsort(field.sensitivities.numpy())[::-1]
    print("\nMost sensitive dimensions:")
    for i in sorted_dims[:3]:
        print(f"  {field.axis_names[i]}: {field.sensitivities[i]:.3f}")
    
    print("\nLeast sensitive dimensions:")
    for i in sorted_dims[-3:]:
        print(f"  {field.axis_names[i]}: {field.sensitivities[i]:.3f}")
    
    print("\nDemo complete. Check the generated images:")
    print("- initial_semantic_space.png: Initial state")
    print("- tuned_semantic_space.png: After sensitivity tuning")

if __name__ == "__main__":
    run_demo()
