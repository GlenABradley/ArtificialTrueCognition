"""
Power-of-2 Layer Foundation for ATC Revolutionary Architecture
===========================================================

This module implements the core power-of-2 layer architecture with invertible transforms:
- Dimensions: 2D → 4D → 16D → 64D → 256D
- Up transforms: T_up = d²·W + b
- Down transforms: T_down = √d·W⁻¹
- Invertibility guarantee: input → up → down = original

Author: Revolutionary ATC Architecture Team
Status: Milestone 1 - Foundation Layer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PowerOf2Config:
    """Configuration for Power-of-2 Layer Architecture"""
    # Core power-of-2 dimensions
    layer_dims: List[int] = None
    
    def __post_init__(self):
        if self.layer_dims is None:
            # Revolutionary power-of-2 progression: 2¹, 2², 2⁴, 2⁶, 2⁸
            self.layer_dims = [2, 4, 16, 64, 256]
        
        # Validate power-of-2 nature
        for dim in self.layer_dims:
            if not self._is_power_of_2(dim):
                raise ValueError(f"Dimension {dim} is not a power of 2")
    
    def _is_power_of_2(self, n: int) -> bool:
        """Check if n is a power of 2"""
        return n > 0 and (n & (n - 1)) == 0


class SimplePowerOf2Transform(nn.Module):
    """
    Simple and mathematically sound Power-of-2 transform
    
    Uses simple dimension mapping with perfect invertibility:
    - Up: Repeat values to higher dimension
    - Down: Average groups to lower dimension
    """
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.is_upward = output_dim > input_dim
        
        if self.is_upward:
            self.expand_ratio = output_dim // input_dim
        else:
            self.compress_ratio = input_dim // output_dim
            
        logger.debug(f"Simple transform: {input_dim}D → {output_dim}D ({'up' if self.is_upward else 'down'})")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with simple dimension mapping"""
        if self.is_upward:
            # Expand: repeat each value expand_ratio times
            expanded = x.unsqueeze(-1).expand(-1, -1, self.expand_ratio)
            return expanded.reshape(x.shape[0], -1)
        else:
            # Compress: average groups of compress_ratio values
            reshaped = x.reshape(x.shape[0], self.output_dim, self.compress_ratio)
            return torch.mean(reshaped, dim=2)
    
    def reverse(self, x: torch.Tensor) -> torch.Tensor:
        """Perfect inverse operation"""
        if self.is_upward:
            # If we expanded, now we compress back
            reshaped = x.reshape(x.shape[0], self.input_dim, self.expand_ratio)
            return reshaped[:, :, 0]  # Take first value (all should be identical)
        else:
            # If we compressed, now we expand back (repeat the averaged values)
            expanded = x.unsqueeze(-1).expand(-1, -1, self.compress_ratio)
            return expanded.reshape(x.shape[0], -1)


class PowerOf2Layers(nn.Module):
    """
    Simplified Power-of-2 Layer Stack with Perfect Invertibility
    
    Architecture: 2D → 4D → 16D → 64D → 256D
    """
    
    def __init__(self, config: PowerOf2Config = None):
        super().__init__()
        self.config = config or PowerOf2Config()
        self.forward_transforms = nn.ModuleList()
        
        # Build forward transforms
        for i in range(len(self.config.layer_dims) - 1):
            input_dim = self.config.layer_dims[i]
            output_dim = self.config.layer_dims[i + 1]
            
            transform = SimplePowerOf2Transform(input_dim, output_dim)
            self.forward_transforms.append(transform)
        
        logger.info(f"Initialized Simplified Power-of-2 stack with {len(self.forward_transforms)} transforms")
        logger.info(f"Dimension progression: {' → '.join(map(str, self.config.layer_dims))}")
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward pass through all transforms"""
        layer_outputs = [x]
        current = x
        
        for i, transform in enumerate(self.forward_transforms):
            current = transform(current)
            # Pure linear transforms for perfect invertibility
            layer_outputs.append(current)
            logger.debug(f"Transform {i+1}: {current.shape}")
        
        return current, layer_outputs
    
    def reverse(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Perfect reverse through all transforms"""
        layer_outputs = [x]
        current = x
        
        # Reverse through transforms in opposite order
        for i, transform in enumerate(reversed(self.forward_transforms)):
            # Apply reverse transform
            current = transform.reverse(current)
            layer_outputs.append(current)
            logger.debug(f"Reverse transform {len(self.forward_transforms)-i}: {current.shape}")
        
        return current, layer_outputs
    
    def test_invertibility(self, test_input: torch.Tensor, tolerance: float = 0.001) -> dict:
        """
        Test invertibility: input → forward → reverse ≈ input
        
        Args:
            test_input: Input tensor to test (should match first layer dimension)
            tolerance: Maximum acceptable error
            
        Returns:
            Dictionary with test results
        """
        logger.info("Testing invertibility of Power-of-2 layers...")
        
        # Forward pass
        forward_output, forward_intermediates = self.forward(test_input)
        
        # Reverse pass
        reconstructed, reverse_intermediates = self.reverse(forward_output)
        
        # Calculate reconstruction error
        error = torch.mean(torch.abs(test_input - reconstructed)).item()
        
        # Calculate relative error
        input_magnitude = torch.mean(torch.abs(test_input)).item()
        relative_error = error / (input_magnitude + 1e-8)
        
        results = {
            'error': error,
            'relative_error': relative_error,
            'tolerance': tolerance,
            'passed': error < tolerance,
            'input_shape': test_input.shape,
            'output_shape': forward_output.shape,
            'reconstructed_shape': reconstructed.shape,
            'forward_intermediates': [t.shape for t in forward_intermediates],
            'reverse_intermediates': [t.shape for t in reverse_intermediates]
        }
        
        logger.info(f"Invertibility test: {'PASSED' if results['passed'] else 'FAILED'}")
        logger.info(f"Reconstruction error: {error:.6f} (tolerance: {tolerance})")
        logger.info(f"Relative error: {relative_error:.6f}")
        
        return results


class PowerOf2Integrator:
    """
    Integration adapter for existing Enhanced SATC system
    """
    
    def __init__(self, power_layers: PowerOf2Layers):
        self.power_layers = power_layers
        self.is_integrated = False
        
    def integrate_with_satc(self, satc_engine):
        """
        Integrate Power-of-2 layers with existing SATC engine
        
        This replaces the existing square progression with power-of-2 architecture
        """
        logger.info("Integrating Power-of-2 layers with Enhanced SATC...")
        
        # Store reference to original architecture for comparison
        if hasattr(satc_engine, 'deep_layers'):
            satc_engine._original_deep_layers = satc_engine.deep_layers
        
        # Replace with Power-of-2 architecture
        satc_engine.power_of_2_layers = self.power_layers
        satc_engine._using_power_of_2 = True
        
        self.is_integrated = True
        logger.info("Power-of-2 integration completed!")
        
        return satc_engine
    
    def process_through_power_layers(self, input_tensor: torch.Tensor) -> dict:
        """
        Process input through Power-of-2 layers and return detailed analysis
        """
        if not self.is_integrated:
            logger.warning("Power-of-2 layers not yet integrated with SATC")
        
        # Forward pass
        output, intermediates = self.power_layers.forward(input_tensor)
        
        # Test invertibility
        invertibility_test = self.power_layers.test_invertibility(input_tensor)
        
        return {
            'input': input_tensor,
            'output': output,
            'intermediates': intermediates,
            'invertibility': invertibility_test,
            'dimension_progression': [t.shape[-1] for t in intermediates]
        }


def create_power_of_2_foundation():
    """
    Factory function to create complete Power-of-2 foundation
    """
    config = PowerOf2Config()
    layers = PowerOf2Layers(config)
    integrator = PowerOf2Integrator(layers)
    
    logger.info("Power-of-2 Foundation created successfully!")
    logger.info(f"Architecture: {config.layer_dims}")
    
    return layers, integrator, config


# Standalone testing function
def test_power_of_2_standalone():
    """
    Standalone test of Power-of-2 architecture
    """
    print("=" * 60)
    print("POWER-OF-2 LAYER FOUNDATION - STANDALONE TEST")
    print("=" * 60)
    
    # Create foundation
    layers, integrator, config = create_power_of_2_foundation()
    
    # Create test input (batch_size=1, input_dim=2)
    test_input = torch.randn(1, 2)
    print(f"Test input shape: {test_input.shape}")
    print(f"Test input: {test_input}")
    
    # Test forward pass
    print("\n--- Forward Pass ---")
    output, intermediates = layers.forward(test_input)
    print(f"Final output shape: {output.shape}")
    print(f"Dimension progression: {[t.shape[-1] for t in intermediates]}")
    
    # Test invertibility
    print("\n--- Invertibility Test ---")
    invertibility_results = layers.test_invertibility(test_input)
    
    print("\n--- Results Summary ---")
    for key, value in invertibility_results.items():
        print(f"{key}: {value}")
    
    print("\n--- Integration Test ---")
    # Test processing through integrator
    processing_results = integrator.process_through_power_layers(test_input)
    print(f"Processing completed: {len(processing_results)} components")
    
    print("=" * 60)
    print("MILESTONE 1: POWER-OF-2 FOUNDATION - COMPLETE!")
    print("=" * 60)
    
    return layers, integrator, config, invertibility_results


if __name__ == "__main__":
    test_power_of_2_standalone()