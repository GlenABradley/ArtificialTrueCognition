"""Visualization tools for attention weights in the semantic field."""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch

def plot_attention_heatmap(weights, dim_names=None, particle_names=None, title="Attention Heatmap"):
    """Create a heatmap of attention weights.
    
    Args:
        weights: Tensor of shape [batch_size, num_heads, 1, 1] or [batch_size, num_heads, seq_len]
        dim_names: List of dimension names
        particle_names: List of particle names
        title: Plot title
    """
    if isinstance(weights, torch.Tensor):
        weights = weights.detach().cpu().numpy()
    
    # Handle different input shapes
    if weights.ndim == 4:
        batch_size, num_heads, seq_len, _ = weights.shape
        # Convert to [batch_size, num_heads]
        weights = weights.squeeze(-1).squeeze(-1)
    elif weights.ndim == 3:
        batch_size, num_heads, seq_len = weights.shape
        weights = weights.squeeze(-1)  # Remove the last dimension if it's 1
    else:
        raise ValueError(f"Unexpected weights shape: {weights.shape}")
    
    # Create a figure with subplots for each head
    fig, axes = plt.subplots(1, num_heads, figsize=(6*num_heads, 5))
    if num_heads == 1:
        axes = [axes]
    
    for head in range(num_heads):
        # Get attention weights for this head across all particles
        head_weights = weights[:, head]
        
        # Create a matrix where each row is a particle's attention weights
        # For now, we'll just plot the attention weights as a bar chart
        ax = axes[head]
        ax.bar(range(batch_size), head_weights)
        
        # Set labels and title
        ax.set_title(f"Head {head+1}")
        ax.set_xlabel("Particle Index")
        ax.set_ylabel("Attention Weight")
        ax.set_ylim(0, 1)
        
        # Rotate x-axis labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    
    plt.suptitle(title, y=1.05)
    plt.tight_layout()
    return fig
    
    plt.suptitle(title, y=1.05)
    plt.tight_layout()
    return fig

def plot_attention_radar(weights, dim_names=None, particle_names=None, title="Attention Radar"):
    """Create a radar chart of attention weights.
    
    Args:
        weights: Tensor of shape [batch_size, num_heads, 1, 1] or [batch_size, num_heads, seq_len]
        dim_names: Optional list of dimension names
        particle_names: Optional list of particle names
        title: Plot title
    """
    if isinstance(weights, torch.Tensor):
        weights_np = weights.detach().cpu().numpy()
    else:
        weights_np = np.array(weights)
    
    # Handle different input shapes
    if weights_np.ndim == 4:
        batch_size, num_heads, seq_len, _ = weights_np.shape
        weights_np = weights_np.squeeze(-1).squeeze(-1)  # [batch_size, num_heads]
    elif weights_np.ndim == 3:
        batch_size, num_heads, seq_len = weights_np.shape
        weights_np = weights_np.squeeze(-1)  # Remove the last dimension if it's 1
    else:
        raise ValueError(f"Unexpected weights shape: {weights_np.shape}")
    
    if particle_names is None:
        particle_names = [f"Particle {i}" for i in range(batch_size)]
    
    # Create a radar chart for each head
    fig = make_subplots(
        rows=1, 
        cols=num_heads,
        specs=[[{'type': 'polar'}] * num_heads],
        subplot_titles=[f'Head {i+1}' for i in range(num_heads)]
    )
    
    for head in range(num_heads):
        # Get attention weights for this head across all particles
        head_weights = weights_np[:, head]
        
        # Create a radar chart for this head
        # We'll show attention weights for each particle
        for i in range(batch_size):
            # Create a simple triangle shape for each particle
            r = [head_weights[i], head_weights[i], 0]  # Third point at 0 to create a wedge
            theta = [0, 120, 240]  # Three points for a triangle
            
            fig.add_trace(
                go.Scatterpolar(
                    r=r,
                    theta=theta,
                    fill='toself',
                    name=particle_names[i],
                    showlegend=(head == 0),  # Only show legend for first head
                    opacity=0.7
                ),
                row=1,
                col=head+1
            )
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1])
        ),
        title=title,
        showlegend=True,
        height=400,
        width=300*num_heads,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.1,
            xanchor="center",
            x=0.5
        )
    )
    
    return fig

def plot_attention_3d(weights, dim_names=None, particle_names=None, title="3D Attention"):
    """Create a 3D visualization of attention weights.
    
    Args:
        weights: Tensor of shape [batch_size, num_heads, 1, 1] or [batch_size, num_heads, seq_len]
        dim_names: Optional list of dimension names
        particle_names: Optional list of particle names
        title: Plot title
    """
    if isinstance(weights, torch.Tensor):
        weights_np = weights.detach().cpu().numpy()
    else:
        weights_np = np.array(weights)
    
    # Handle different input shapes
    if weights_np.ndim == 4:
        batch_size, num_heads, seq_len, _ = weights_np.shape
        weights_np = weights_np.squeeze(-1).squeeze(-1)  # [batch_size, num_heads]
    elif weights_np.ndim == 3:
        batch_size, num_heads, seq_len = weights_np.shape
        weights_np = weights_np.squeeze(-1)  # Remove the last dimension if it's 1
    else:
        raise ValueError(f"Unexpected weights shape: {weights_np.shape}")
    
    if particle_names is None:
        particle_names = [f"Particle {i}" for i in range(batch_size)]
    
    # Create a 3D plot for each head
    fig = make_subplots(
        rows=1, 
        cols=num_heads,
        specs=[[{'type': 'scatter3d'}] * num_heads],
        subplot_titles=[f'Head {i+1}' for i in range(num_heads)]
    )
    
    for head in range(num_heads):
        # Get attention weights for this head across all particles
        head_weights = weights_np[:, head]
        
        # Create a 3D scatter plot for this head
        x = np.arange(batch_size)  # Particle index
        y = np.ones_like(x) * head  # Head index
        z = head_weights  # Attention weights
        
        # Add text labels for particles
        text = [f"{name}<br>Weight: {w:.3f}" for name, w in zip(particle_names, head_weights)]
        
        fig.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode='markers+text',
                text=text,
                textposition="top center",
                marker=dict(
                    size=8,
                    color=z,  # Color by attention weight
                    colorscale='Viridis',
                    opacity=0.8,
                    colorbar=dict(title='Attention')
                ),
                name=f'Head {head+1}'
            ),
            row=1,
            col=head+1
        )
        
        # Update scene for each subplot
        fig.update_scenes(
            xaxis_title="Particle",
            yaxis_title="Head",
            zaxis_title="Attention Weight",
            row=1,
            col=head+1
        )
    
    fig.update_layout(
        title=title,
        height=600,
        width=400*num_heads,
        showlegend=False
    )
    
    return fig

def plot_attention_streamgraph(weights, dim_names=None, particle_names=None, title="Attention Streamgraph"):
    """Create a streamgraph of attention weights across particles.
    
    Args:
        weights: Tensor of shape [batch_size, num_heads, 1, 1] or [batch_size, num_heads, seq_len]
        dim_names: Not used, kept for API compatibility
        particle_names: Optional list of particle names
        title: Plot title
    """
    if isinstance(weights, torch.Tensor):
        weights = weights.detach().cpu().numpy()
    
    # Handle different input shapes
    if weights.ndim == 4:
        batch_size, num_heads, seq_len, _ = weights.shape
        weights = weights.squeeze(-1).squeeze(-1)  # [batch_size, num_heads]
    elif weights.ndim == 3:
        batch_size, num_heads, seq_len = weights.shape
        weights = weights.squeeze(-1)  # Remove the last dimension if it's 1
    else:
        raise ValueError(f"Unexpected weights shape: {weights.shape}")
    
    if particle_names is None:
        particle_names = [f"Particle {i}" for i in range(batch_size)]
    
    fig = go.Figure()
    
    # For each head, create a streamgraph
    for head in range(num_heads):
        # Get attention weights for this head across all particles
        head_weights = weights[:, head]  # [batch_size]
        
        # Create a simple area chart for this head
        fig.add_trace(go.Scatter(
            x=np.arange(batch_size),
            y=head_weights,
            mode='lines+markers',
            name=f'Head {head+1}',
            hoverinfo='text',
            hovertext=[f"{name}<br>Weight: {weight:.3f}" 
                      for name, weight in zip(particle_names, head_weights)],
            line=dict(width=2),
            marker=dict(size=8)
        ))
    
    fig.update_layout(
        title=title,
        xaxis=dict(
            title="Particle",
            ticktext=particle_names,
            tickvals=np.arange(batch_size)
        ),
        yaxis_title="Attention Weight",
        showlegend=True,
        height=500,
        width=800,
        hovermode='x unified'
    )
    
    return fig

def visualize_all_attention_perspectives(weights, dim_names=None, particle_names=None):
    """Generate all available attention visualizations."""
    if dim_names is None:
        num_dims = weights.shape[2]
        dim_names = [f"Dim {i}" for i in range(num_dims)]
    
    # Generate all visualizations
    heatmap = plot_attention_heatmap(weights, dim_names, particle_names)
    radar = plot_attention_radar(weights, dim_names)
    surface_3d = plot_attention_3d(weights, dim_names)
    streamgraph = plot_attention_streamgraph(weights, dim_names)
    
    return {
        'heatmap': heatmap,
        'radar': radar,
        '3d_surface': surface_3d,
        'streamgraph': streamgraph
    }

# Example usage
if __name__ == "__main__":
    # Generate some example data
    num_particles = 5
    num_heads = 4
    num_dims = 12
    
    # Create random attention weights
    weights = torch.rand(num_particles, num_heads, num_dims, num_dims)
    dim_names = [f"Dim {i}" for i in range(num_dims)]
    
    # Generate visualizations
    visualizations = visualize_all_attention_perspectives(weights, dim_names)
    
    # Display the visualizations
    for name, fig in visualizations.items():
        if isinstance(fig, go.Figure):
            fig.show()
        else:
            plt.figure(fig.number)
            plt.show()
