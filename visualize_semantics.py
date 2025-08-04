"""Visualization tools for semantic space analysis."""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional
import torch

class SemanticVisualizer:
    """Visualization tools for semantic space analysis."""
    
    def __init__(self, field):
        """Initialize with a SemanticField instance."""
        self.field = field
        self.fig = None
        
    def plot_semantic_space(self, save_path: Optional[str] = None) -> None:
        """Create a comprehensive visualization of the semantic space."""
        if not self.field.particles:
            print("No particles to visualize")
            return
            
        vectors = torch.stack([p.vector for p in self.field.particles]).numpy()
        
        # Create figure with subplots
        self.fig = plt.figure(figsize=(18, 12))
        gs = self.fig.add_gridspec(3, 2, width_ratios=[3, 1])
        
        # 1. Correlation matrix
        ax1 = self.fig.add_subplot(gs[0, 0])
        self._plot_correlation_matrix(ax1)
        
        # 2. Sensitivity plot
        ax2 = self.fig.add_subplot(gs[1, 0])
        self._plot_sensitivities(ax2)
        
        # 3. Dimensional variance
        ax3 = self.fig.add_subplot(gs[2, 0])
        self._plot_dimensional_variance(ax3, vectors)
        
        # 4. 2D projection
        ax4 = self.fig.add_subplot(gs[:, 1])
        self._plot_2d_projection(ax4, vectors)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    def _plot_correlation_matrix(self, ax) -> None:
        """Plot the correlation matrix with adaptive smoothing."""
        if not hasattr(self.field, 'correlation_history') or not self.field.correlation_history:
            return
            
        corr_matrix = self.field.correlation_history[-1]  # Most recent
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        
        sns.heatmap(
            corr_matrix,
            mask=mask,
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            ax=ax
        )
        ax.set_title('Dimensional Correlation Matrix')
        ax.set_xticks(np.arange(len(self.field.axis_names)) + 0.5)
        ax.set_xticklabels(self.field.axis_names, rotation=45, ha='right')
        ax.set_yticks(np.arange(len(self.field.axis_names)) + 0.5)
        ax.set_yticklabels(self.field.axis_names, rotation=0)
    
    def _plot_sensitivities(self, ax) -> None:
        """Plot the current sensitivity values for each dimension."""
        if not hasattr(self.field, 'sensitivities'):
            return
            
        sensitivities = self.field.sensitivities.numpy()
        x = np.arange(len(sensitivities))
        
        bars = ax.bar(x, sensitivities)
        ax.set_title('Dimensional Sensitivities')
        ax.set_xticks(x)
        ax.set_xticklabels(self.field.axis_names, rotation=45, ha='right')
        ax.set_ylim(0, np.max(sensitivities) * 1.2)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                height + 0.01,
                f'{height:.2f}',
                ha='center',
                va='bottom',
                fontsize=8
            )
    
    def _plot_dimensional_variance(self, ax, vectors: np.ndarray) -> None:
        """Plot the variance of each dimension."""
        variances = np.var(vectors, axis=0)
        x = np.arange(len(variances))
        
        ax.bar(x, variances)
        ax.set_title('Dimensional Variance')
        ax.set_xticks(x)
        ax.set_xticklabels(self.field.axis_names, rotation=45, ha='right')
        ax.set_ylim(0, np.max(variances) * 1.2)
    
    def _plot_2d_projection(self, ax, vectors: np.ndarray) -> None:
        """Plot a 2D projection of the semantic space using PCA."""
        from sklearn.decomposition import PCA
        
        if len(vectors) < 2:
            return
            
        # Use PCA for dimensionality reduction
        pca = PCA(n_components=2)
        projected = pca.fit_transform(vectors)
        
        # Get unique concepts
        concepts = list(set(p.concept for p in self.field.particles))
        concept_to_idx = {c: i for i, c in enumerate(concepts)}
        colors = plt.cm.tab20(np.linspace(0, 1, len(concepts)))
        
        # Plot each concept
        for i, p in enumerate(self.field.particles):
            idx = concept_to_idx[p.concept]
            ax.scatter(
                projected[i, 0],
                projected[i, 1],
                color=colors[idx],
                label=p.concept if i == 0 else "",
                alpha=0.7
            )
        
        ax.set_title('2D Projection of Semantic Space')
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Add variance explained
        var_exp = pca.explained_variance_ratio_
        ax.text(
            0.05, 0.95,
            f'Explained variance: {var_exp[0]:.1%}, {var_exp[1]:.1%}',
            transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=0.8)
        )
