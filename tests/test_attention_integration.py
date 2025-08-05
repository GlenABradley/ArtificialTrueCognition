"""Test integration of attention mechanism in SemanticField."""
import torch
import numpy as np
from syncopation_engine import SemanticField, SemanticParticle

def test_attention_similarity():
    """Test that attention-based similarity works as expected."""
    # Create field with attention
    field = SemanticField(use_attention=True)
    
    # Add some test particles
    particle1 = SemanticParticle("test1", torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2]))
    particle2 = SemanticParticle("test2", torch.tensor([0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 0.05, 0.15, 0.25]))
    particle3 = SemanticParticle("test3", torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, 0.9, 0.8]))
    
    for p in [particle1, particle2, particle3]:
        field.add_particle(p)
    
    # Test similarity with default context
    sim12 = field.calculate_similarity(particle1.vector, particle2.vector)
    sim13 = field.calculate_similarity(particle1.vector, particle3.vector)
    
    print(f"Similarity between test1 and test2 (should be high): {sim12:.4f}")
    print(f"Similarity between test1 and test3 (should be low): {sim13:.4f}")
    
    # Test batch processing
    vectors = torch.stack([particle1.vector, particle2.vector, particle3.vector])
    sim_matrix = field.calculate_similarity(vectors.unsqueeze(1), vectors.unsqueeze(0))
    
    print("\nSimilarity matrix:")
    print(sim_matrix.numpy().round(2))
    
    # Test attention weights
    print("\nAttention weights shape:", field.attention_weights[0].shape)
    
    # Test disabling attention
    sim12_no_attn = field.calculate_similarity(
        particle1.vector, 
        particle2.vector,
        context={"use_attention": False}
    )
    print(f"\nSimilarity without attention: {sim12_no_attn:.4f}")
    
    # Test resolution adjustment
    sim12_low_res = field.calculate_similarity(
        particle1.vector,
        particle2.vector,
        context={"resolution": 0.1}  # Very coarse resolution
    )
    print(f"Similarity at low resolution: {sim12_low_res:.4f}")

if __name__ == "__main__":
    test_attention_similarity()
