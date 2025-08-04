"""
Dimensional Analysis Tool for Semantic Axis System

This module provides tools to analyze the behavior and sensitivity of different
dimensions in the semantic space. It helps identify important dimensions,
correlations, and their impact on similarity calculations.
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from collections import defaultdict

class DimensionalAnalyzer:
    def __init__(self, field, num_samples: int = 1000):
        """Initialize the analyzer with a semantic field.
        
        Args:
            field: The semantic field to analyze
            num_samples: Number of samples to use for statistical analysis
        """
        self.field = field
        self.num_samples = num_samples
        self.dim = 12  # Number of dimensions in our semantic space
        
    def analyze(self) -> Dict:
        """Run comprehensive dimensional analysis.
        
        Returns:
            Dictionary containing analysis results including:
            - variance: Variance of each dimension
            - correlation: Correlation matrix between dimensions
            - impact: Impact of each dimension on similarity
            - sensitivity: Sensitivity of each dimension to changes
        """
        results = {
            'variance': self._calculate_variance(),
            'correlation': self._calculate_correlation(),
            'impact': self._calculate_impact(),
            'sensitivity': self._calculate_sensitivity()
        }
        return results
    
    def _calculate_variance(self) -> np.ndarray:
        """Calculate variance of each dimension across all particles."""
        if not self.field.particles:
            return np.zeros(self.dim)
            
        vectors = np.array([p.vector.numpy() for p in self.field.particles])
        return np.var(vectors, axis=0)
    
    def _calculate_correlation(self) -> np.ndarray:
        """Calculate correlation matrix between dimensions."""
        if not self.field.particles:
            return np.eye(self.dim)
            
        vectors = np.array([p.vector.numpy() for p in self.field.particles])
        return np.corrcoef(vectors, rowvar=False)
    
    def _calculate_impact(self) -> np.ndarray:
        """Calculate the impact of each dimension on similarity scores."""
        impact = np.zeros(self.dim)
        
        for _ in range(self.num_samples):
            # Generate random query and target vectors
            query = torch.rand(self.dim)
            target = torch.rand(self.dim)
            
            # Calculate baseline similarity
            base_sim = self.field._calculate_similarity(
                query, target, 
                torch.ones(self.dim)  # Equal weights for baseline
            )
            
            # Test impact of each dimension
            for dim in range(self.dim):
                modified_target = target.clone()
                modified_target[dim] = 1 - modified_target[dim]  # Flip dimension
                
                # Calculate new similarity
                new_sim = self.field._calculate_similarity(
                    query, modified_target,
                    torch.ones(self.dim)
                )
                
                # Update impact score
                impact[dim] += abs(new_sim - base_sim)
        
        return impact / self.num_samples
    
    def _calculate_sensitivity(self) -> np.ndarray:
        """Calculate sensitivity of each dimension to small changes."""
        sensitivity = np.zeros(self.dim)
        
        for _ in range(self.num_samples):
            base = torch.rand(self.dim)
            
            for dim in range(self.dim):
                # Make small perturbation
                perturbed = base.clone()
                perturbed[dim] += 0.01  # Small perturbation
                perturbed = torch.clamp(perturbed, 0, 1)
                
                # Calculate similarity with perturbation
                sim = self.field._calculate_similarity(
                    base, perturbed,
                    torch.ones(self.dim)
                )
                
                # Update sensitivity (lower similarity = more sensitive)
                sensitivity[dim] += 1 - sim
        
        return sensitivity / self.num_samples

def visualize_analysis(results: Dict, save_path: str = None):
    """Create visualizations of the dimensional analysis.
    
    Args:
        results: Analysis results from DimensionalAnalyzer.analyze()
        save_path: Optional path to save the visualization
    """
    plt.figure(figsize=(18, 12))
    
    # Plot 1: Variance per dimension
    plt.subplot(2, 2, 1)
    sns.barplot(x=list(range(len(results['variance']))), y=results['variance'])
    plt.title('Variance per Dimension')
    plt.xlabel('Dimension')
    plt.ylabel('Variance')
    
    # Plot 2: Correlation heatmap
    plt.subplot(2, 2, 2)
    mask = np.triu(np.ones_like(results['correlation'], dtype=bool))
    sns.heatmap(
        results['correlation'], 
        mask=mask,
        cmap='coolwarm', 
        center=0,
        square=True,
        linewidths=.5,
        cbar_kws={"shrink": .5}
    )
    plt.title('Dimension Correlation Matrix')
    
    # Plot 3: Impact per dimension
    plt.subplot(2, 2, 3)
    sns.barplot(x=list(range(len(results['impact']))), y=results['impact'])
    plt.title('Impact on Similarity per Dimension')
    plt.xlabel('Dimension')
    plt.ylabel('Average Impact')
    
    # Plot 4: Sensitivity per dimension
    plt.subplot(2, 2, 4)
    sns.barplot(x=list(range(len(results['sensitivity']))), y=results['sensitivity'])
    plt.title('Sensitivity per Dimension')
    plt.xlabel('Dimension')
    plt.ylabel('Sensitivity')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def run_analysis(field, num_samples: int = 1000, visualize: bool = True):
    """Run complete dimensional analysis and return results.
    
    Args:
        field: The semantic field to analyze
        num_samples: Number of samples for statistical analysis
        visualize: Whether to generate visualizations
        
    Returns:
        Dictionary containing analysis results
    """
    analyzer = DimensionalAnalyzer(field, num_samples)
    results = analyzer.analyze()
    
    if visualize:
        visualize_analysis(results)
    
    return results
