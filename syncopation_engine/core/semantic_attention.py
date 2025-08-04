"""Dynamic semantic attention mechanism for adaptive focus on semantic dimensions.

This module implements attention mechanisms that can dynamically focus on different
semantic dimensions based on context, enabling adaptive resolution and focus.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Union
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class SemanticAttention(nn.Module):
    """Implements a dynamic attention mechanism for semantic dimensions."""
    
    def __init__(self, num_dimensions: int = 12, hidden_size: int = 32, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_dimensions = num_dimensions
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        assert self.head_dim * num_heads == hidden_size, "Hidden size must be divisible by number of heads"
        
        # Projection layers
        self.query_proj = nn.Linear(num_dimensions, hidden_size)
        self.key_proj = nn.Linear(num_dimensions, hidden_size)
        self.value_proj = nn.Linear(num_dimensions, hidden_size)
        self.output_proj = nn.Linear(hidden_size, num_dimensions)
        self.layer_norm = nn.LayerNorm(num_dimensions)
        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize parameters using Xavier/Glorot initialization."""
        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.xavier_uniform_(self.key_proj.weight)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.xavier_uniform_(self.output_proj.weight)
        
        if self.query_proj.bias is not None:
            nn.init.constant_(self.query_proj.bias, 0.)
        if self.key_proj.bias is not None:
            nn.init.constant_(self.key_proj.bias, 0.)
        if self.value_proj.bias is not None:
            nn.init.constant_(self.value_proj.bias, 0.)
        if self.output_proj.bias is not None:
            nn.init.constant_(self.output_proj.bias, 0.)
    
    def forward(self, query: torch.Tensor, key: Optional[torch.Tensor] = None,
               value: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if key is None:
            key = query
        if value is None:
            value = query
            
        batch_size = query.size(0)
        seq_len = query.size(1) if query.dim() > 2 else 1
        
        # Project inputs
        Q = self.query_proj(query)
        K = self.key_proj(key)
        V = self.value_proj(value)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_size)
        
        # Project and normalize
        output = self.output_proj(context)
        output = self.layer_norm(query + self.dropout(output))
        
        return output, attention_weights


class MultiScaleAttention(nn.Module):
    """Multi-scale attention mechanism that operates at different resolution levels.
    
    This module applies multiple attention mechanisms at different scales and
    combines their outputs adaptively.
    """
    
    def __init__(self, num_dimensions: int = 12, num_scales: int = 3, 
                 hidden_size: int = 64, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_dimensions = num_dimensions
        self.num_scales = num_scales
        self.hidden_size = hidden_size
        
        # Create attention modules for each scale
        self.attention_layers = nn.ModuleList([
            SemanticAttention(
                num_dimensions=num_dimensions,
                hidden_size=hidden_size,
                num_heads=num_heads,
                dropout=dropout
            )
            for _ in range(num_scales)
        ])
        
        # Learnable scale weights
        self.scale_weights = nn.Parameter(torch.ones(num_scales) / num_scales)
        
        # Output projection
        self.output_proj = nn.Linear(num_dimensions * num_scales, num_dimensions)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(num_dimensions)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        batch_size, seq_len, _ = x.shape
        
        # Apply each attention scale
        scale_outputs = []
        attention_weights_list = []
        
        for attn_layer in self.attention_layers:
            scale_output, attn_weights = attn_layer(x, mask=mask)
            scale_outputs.append(scale_output)
            attention_weights_list.append(attn_weights)
        
        # Concatenate scale outputs with learned weights
        weighted_outputs = []
        for i, out in enumerate(scale_outputs):
            weight = self.scale_weights[i].view(1, 1, 1)
            weighted_outputs.append(out * weight)
        
        concatenated = torch.cat(weighted_outputs, dim=-1)
        
        # Project back to original dimension size
        output = self.output_proj(concatenated)
        
        # Residual connection and layer normalization
        output = self.layer_norm(x + self.dropout(output))
        
        return output, attention_weights_list


def visualize_attention(attention_weights: Union[torch.Tensor, List[torch.Tensor]], 
                      dimension_names: Optional[List[str]] = None,
                      title: str = "Attention Weights") -> None:
    """Visualize attention weights across dimensions.
    
    Args:
        attention_weights: Tensor of shape [num_heads, seq_len, seq_len] or list of such tensors
        dimension_names: Optional list of dimension names for labeling
        title: Title for the plot
    """
    if dimension_names is None:
        dimension_names = [f"Dim {i}" for i in range(attention_weights[0].size(-1))]
    
    if isinstance(attention_weights, list):
        num_plots = len(attention_weights)
        fig, axes = plt.subplots(1, num_plots, figsize=(5*num_plots, 5))
        
        for i, weights in enumerate(attention_weights):
            if weights.dim() == 4:  # [batch, num_heads, seq_len, seq_len]
                weights = weights[0]  # Take first in batch
            if weights.dim() == 3:  # [num_heads, seq_len, seq_len]
                # Average across heads for visualization
                weights = weights.mean(dim=0)
            
            sns.heatmap(
                weights.detach().cpu().numpy(),
                ax=axes[i] if num_plots > 1 else axes,
                cmap='viridis',
                xticklabels=dimension_names,
                yticklabels=dimension_names,
                vmin=0,
                vmax=1
            )
            if num_plots > 1:
                axes[i].set_title(f'Scale {i+1}')
    else:
        if attention_weights.dim() == 4:
            attention_weights = attention_weights[0]
        if attention_weights.dim() == 3:
            attention_weights = attention_weights.mean(dim=0)
            
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            attention_weights.detach().cpu().numpy(),
            cmap='viridis',
            xticklabels=dimension_names,
            yticklabels=dimension_names,
            vmin=0,
            vmax=1
        )
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


# Test the attention mechanisms
if __name__ == "__main__":
    # Create test input
    num_dimensions = 12
    batch_size = 2
    seq_len = 5
    x = torch.randn(batch_size, seq_len, num_dimensions)
    
    # Test SemanticAttention
    print("Testing SemanticAttention...")
    attn = SemanticAttention(num_dimensions=num_dimensions)
    output, weights = attn(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {weights.shape}")
    
    # Visualize attention weights
    print("\nVisualizing attention weights...")
    dimension_names = [f"Dim {i}" for i in range(num_dimensions)]
    visualize_attention(weights, dimension_names, "Single-Scale Attention")
    
    # Test MultiScaleAttention
    print("\nTesting MultiScaleAttention...")
    multi_attn = MultiScaleAttention(num_dimensions=num_dimensions, num_scales=3)
    output, weights_list = multi_attn(x)
    print(f"Output shape: {output.shape}")
    print(f"Number of attention scales: {len(weights_list)}")
    print(f"Weights shapes: {[w.shape for w in weights_list]}")
    
    # Visualize multi-scale attention
    print("\nVisualizing multi-scale attention...")
    visualize_attention(weights_list, dimension_names, "Multi-Scale Attention")
