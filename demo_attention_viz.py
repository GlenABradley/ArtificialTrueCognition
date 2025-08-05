"""Demo script for visualizing attention weights in the semantic field."""
import torch
import numpy as np
from syncopation_engine import SemanticField, SemanticParticle
from visualize_attention import visualize_all_attention_perspectives

def create_test_field():
    """Create a test semantic field with sample particles."""
    field = SemanticField(use_attention=True)
    
    # Sample concepts with variations
    concepts = {
        'joy': [0.9, 0.8, 0.1, 0.2, 0.9, 0.1, 0.1, 0.2, 0.3, 0.1, 0.4, 0.2],
        'sadness': [0.8, 0.7, 0.2, 0.3, 0.1, 0.9, 0.2, 0.8, 0.4, 0.2, 0.3, 0.3],
        'anger': [0.85, 0.75, 0.3, 0.4, 0.8, 0.7, 0.4, 0.6, 0.6, 0.3, 0.5, 0.4],
        'fear': [0.7, 0.6, 0.4, 0.5, 0.7, 0.8, 0.5, 0.7, 0.5, 0.4, 0.6, 0.5],
        'surprise': [0.6, 0.5, 0.5, 0.6, 0.9, 0.6, 0.3, 0.4, 0.7, 0.5, 0.7, 0.6],
        'disgust': [0.8, 0.7, 0.2, 0.3, 0.2, 0.8, 0.6, 0.5, 0.3, 0.2, 0.4, 0.3],
        'trust': [0.9, 0.8, 0.2, 0.3, 0.8, 0.2, 0.1, 0.3, 0.8, 0.7, 0.5, 0.6],
        'anticipation': [0.7, 0.6, 0.3, 0.4, 0.7, 0.5, 0.2, 0.3, 0.6, 0.5, 0.7, 0.5]
    }
    
    # Add particles with slight variations
    for concept, base_vector in concepts.items():
        for i in range(3):  # 3 variations per concept
            # Add some noise
            noise = np.random.normal(0, 0.03, 12)
            vector = np.clip(base_vector + noise, 0, 1)
            particle = SemanticParticle(f"{concept}_{i}", torch.tensor(vector, dtype=torch.float32))
            field.add_particle(particle)
    
    return field

def main():
    print("Creating test semantic field...")
    field = create_test_field()
    
    # Get some particles to compare
    particles = field.particles[:8]  # Use first 8 particles
    vectors = torch.stack([p.vector for p in particles])
    
    print("Computing attention weights...")
    # Compute attention for all particles
    with torch.no_grad():
        # Get attention weights for all particles
        batch = vectors.unsqueeze(1)  # [N, 1, D]
        
        # Get attention weights for the batch
        _, all_attention_weights = field.attention(batch)
        
        print(f"Number of attention scales: {len(all_attention_weights)}")
        for i, w in enumerate(all_attention_weights):
            print(f"Scale {i} weights shape: {w.shape}")
        
        # Use the first scale's attention weights
        attention_weights = all_attention_weights[0]  # [batch_size, num_heads, seq_len, seq_len]
        print(f"Selected attention weights shape: {attention_weights.shape}")
        
        # Get number of heads and sequence length
        batch_size, num_heads, seq_len, _ = attention_weights.shape
        num_dims = vectors.size(1)
        
        print(f"Batch size: {batch_size}, Num heads: {num_heads}, Seq len: {seq_len}")
        
        # Reshape to expected format for visualization
        # We'll visualize the attention from the first token to all others
        # Take only the first row of each attention matrix (attention from first token)
        attention_weights = attention_weights[:, :, 0, :]  # [batch_size, num_heads, seq_len]
        
        # Add an extra dimension to make it compatible with visualization functions
        attention_weights = attention_weights.unsqueeze(2)  # [batch_size, num_heads, 1, seq_len]
    
    # Get dimension names
    dim_names = field.axis_names
    
    print("Generating visualizations...")
    # Visualize attention for the first scale (change index for different scales)
    scale_idx = 0
    viz_weights = attention_weights[scale_idx]
    
    # Visualize all perspectives
    visualizations = visualize_all_attention_perspectives(
        viz_weights, 
        dim_names=dim_names,
        particle_names=[p.concept for p in particles]
    )
    
    # Save visualizations
    print("Saving visualizations...")
    for name, fig in visualizations.items():
        if isinstance(fig, go.Figure):
            fig.write_html(f"attention_{name}.html")
        else:
            plt.figure(fig.number)
            plt.savefig(f"attention_{name}.png", bbox_inches='tight', dpi=300)
    
    print("Done! Check the generated visualization files.")

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    
    main()
