"""Test script for dynamic semantic attention mechanisms."""
import torch
import numpy as np
from semantic_attention import SemanticAttention, MultiScaleAttention, visualize_attention
from semantic_axis import SemanticField, SemanticParticle

def create_test_field():
    """Create a test semantic field with sample concepts."""
    field = SemanticField()
    
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

def test_semantic_attention():
    """Test the SemanticAttention mechanism."""
    print("Testing SemanticAttention...")
    
    # Create test input
    batch_size = 2
    seq_len = 5
    num_dimensions = 12
    x = torch.randn(batch_size, seq_len, num_dimensions)
    
    # Create attention layer
    attention = SemanticAttention(num_dimensions=num_dimensions)
    
    # Forward pass
    output, weights = attention(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {weights.shape}")
    
    # Visualize attention weights
    dimension_names = [f"Dim {i}" for i in range(num_dimensions)]
    visualize_attention(weights, dimension_names, "Semantic Attention Weights")

def test_multi_scale_attention():
    """Test the MultiScaleAttention mechanism."""
    print("\nTesting MultiScaleAttention...")
    
    # Create test input
    batch_size = 2
    seq_len = 5
    num_dimensions = 12
    num_scales = 3
    x = torch.randn(batch_size, seq_len, num_dimensions)
    
    # Create multi-scale attention layer
    multi_scale_attn = MultiScaleAttention(
        num_dimensions=num_dimensions,
        num_scales=num_scales,
        hidden_size=64,
        num_heads=4
    )
    
    # Forward pass
    output, weights_list = multi_scale_attn(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of scales: {len(weights_list)}")
    print(f"Weights shapes: {[w.shape for w in weights_list]}")
    
    # Visualize attention weights for each scale
    dimension_names = [f"Dim {i}" for i in range(num_dimensions)]
    for i, weights in enumerate(weights_list):
        visualize_attention(weights, dimension_names, f"Scale {i+1} Attention Weights")

def test_with_semantic_field():
    """Test attention mechanisms with actual semantic particles."""
    print("\nTesting with SemanticField...")
    
    # Create test semantic field
    field = create_test_field()
    
    # Get particle vectors
    particles = field.particles
    vectors = torch.stack([p.vector for p in particles])
    
    # Add batch and sequence dimensions
    x = vectors.unsqueeze(0)  # [1, num_particles, num_dimensions]
    
    print(f"Number of particles: {len(particles)}")
    print(f"Input shape: {x.shape}")
    
    # Create attention layer
    attention = SemanticAttention(num_dimensions=12)
    
    # Forward pass
    output, weights = attention(x)
    
    # Visualize attention weights with concept names
    concept_names = [p.concept for p in particles]
    visualize_attention(weights[0], concept_names, "Attention Between Concepts")
    
    # Visualize attention to each dimension
    dim_attention = weights.mean(dim=1)  # Average over query positions
    dimension_names = [f"Dim {i}" for i in range(12)]
    visualize_attention(dim_attention, concept_names, "Attention to Each Dimension")

if __name__ == "__main__":
    # Run tests
    test_semantic_attention()
    test_multi_scale_attention()
    test_with_semantic_field()
