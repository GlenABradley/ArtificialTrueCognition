"""Test script for hierarchical semantic clustering and context-aware similarity."""
import torch
import numpy as np
from semantic_axis import SemanticField, SemanticParticle
from visualize_semantics import SemanticVisualizer

def create_test_field():
    """Create a test semantic field with sample concepts."""
    field = SemanticField(min_cluster_size=3)
    
    # Create some test concepts with variations
    concepts = {
        # Basic emotions
        'joy': [0.9, 0.8, 0.1, 0.2, 0.9, 0.1, 0.1, 0.2, 0.3, 0.1, 0.4, 0.2],
        'sadness': [0.8, 0.7, 0.2, 0.3, 0.1, 0.9, 0.2, 0.8, 0.4, 0.2, 0.3, 0.3],
        'anger': [0.85, 0.75, 0.3, 0.4, 0.8, 0.7, 0.4, 0.6, 0.6, 0.3, 0.5, 0.4],
        
        # Time-related concepts
        'past': [0.6, 0.5, 0.7, 0.6, 0.3, 0.1, 0.5, 0.7, 0.2, 0.1, 0.6, 0.5],
        'present': [0.5, 0.4, 0.6, 0.5, 0.4, 0.5, 0.6, 0.6, 0.3, 0.2, 0.5, 0.6],
        'future': [0.4, 0.3, 0.5, 0.4, 0.5, 0.9, 0.7, 0.5, 0.4, 0.3, 0.4, 0.7]
    }
    
    # Add variations of each concept
    for concept, base_vector in concepts.items():
        for i in range(3):  # 3 variations per concept
            # Add some noise to create variations
            noise = np.random.normal(0, 0.03, 12)
            vector = np.clip(base_vector + noise, 0, 1)
            particle = SemanticParticle(f"{concept}_{i}", torch.tensor(vector, dtype=torch.float32))
            field.add_particle(particle)
    
    return field

def demo():
    """Demonstrate hierarchical clustering and context-aware similarity."""
    print("Creating test semantic field...")
    field = create_test_field()
    
    # Visualize the semantic space
    print("\nVisualizing semantic space...")
    visualizer = SemanticVisualizer(field)
    visualizer.plot_semantic_space("semantic_space.png")
    
    # Tune sensitivities
    print("\nTuning sensitivities...")
    field.tune_sensitivities()
    print(f"Adjusted sensitivities: {field.sensitivities.numpy().round(3)}")
    
    # Rebuild hierarchy with tuned sensitivities
    print("\nRebuilding hierarchy with tuned sensitivities...")
    field.rebuild_hierarchy()
    
    # Demonstrate context-aware similarity
    print("\nTesting context-aware similarity:")
    
    # Get two particles to compare
    particle1 = field.particles[0]  # First joy variation
    particle2 = field.particles[3]  # First sadness variation
    
    # Compare at different resolutions
    for res in [0.2, 0.5, 0.8]:
        context = field.get_cluster_context(0, resolution=res)
        sim = field.calculate_similarity(
            particle1.vector, 
            particle2.vector,
            context=context
        )
        print(f"\nResolution: {res:.1f}")
        print(f"Focus dimensions: {context.get('focus_dimensions', [])}")
        print(f"Similarity between '{particle1.concept}' and '{particle2.concept}': {sim:.3f}")
    
    print("\nDemo complete! Check 'semantic_space.png' for visualization.")

if __name__ == "__main__":
    demo()
