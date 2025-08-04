"""Hierarchical semantic clustering for multi-resolution analysis."""
import numpy as np
import torch
import hdbscan
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class SemanticCluster:
    """Represents a cluster in semantic space."""
    id: int
    particles: List[int]  # Indices of particles in this cluster
    level: int  # Hierarchical level (0 = root, increases with depth)
    parent: Optional[int]  # Parent cluster ID
    children: List[int]  # Child cluster IDs
    center: torch.Tensor  # Center of the cluster
    
class SemanticHierarchy:
    """Manages hierarchical clustering of semantic particles."""
    
    def __init__(self, min_cluster_size: int = 5, min_samples: int = 2):
        """Initialize the hierarchy.
        
        Args:
            min_cluster_size: Minimum number of points in a cluster
            min_samples: Number of samples in a neighborhood for a point to be a core point
        """
        self.clusters: Dict[int, SemanticCluster] = {}
        self.next_cluster_id = 0
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
    
    def build_hierarchy(self, particles: List['SemanticParticle']) -> None:
        """Build hierarchical clusters from particles.
        
        Args:
            particles: List of SemanticParticle objects to cluster
        """
        if not particles:
            return
            
        # Convert particles to feature matrix
        X = torch.stack([p.vector for p in particles]).numpy()
        
        # Run HDBSCAN clustering
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            gen_min_span_tree=True,
            cluster_selection_method='eom'  # Excess of mass
        )
        
        # Fit and get cluster labels
        labels = clusterer.fit_predict(X)
        
        # Build cluster hierarchy
        self._process_cluster_tree(clusterer, particles, labels)
    
    def _process_cluster_tree(self, clusterer, particles, labels):
        """Process the HDBSCAN cluster tree into our hierarchy."""
        # Clear existing clusters
        self.clusters = {}
        self.next_cluster_id = 0
        
        # Create a root cluster containing all points
        root_cluster = SemanticCluster(
            id=self._get_next_cluster_id(),
            particles=list(range(len(particles))),
            level=0,
            parent=None,
            children=[],
            center=torch.mean(torch.stack([p.vector for p in particles]), dim=0)
        )
        self.clusters[root_cluster.id] = root_cluster
        
        # Process HDBSCAN's cluster hierarchy
        if hasattr(clusterer, 'condensed_tree_'):
            self._process_condensed_tree(clusterer, particles, root_cluster.id)
    
    def _process_condensed_tree(self, clusterer, particles, parent_id):
        """Process HDBSCAN's condensed tree to build our hierarchy."""
        # This is a simplified version - in practice, you'd traverse the tree
        # structure provided by HDBSCAN's condensed_tree_
        pass  # Implementation would go here
    
    def _get_next_cluster_id(self) -> int:
        """Get the next available cluster ID."""
        cid = self.next_cluster_id
        self.next_cluster_id += 1
        return cid
    
    def get_cluster_at_resolution(self, particle_idx: int, resolution: float) -> SemanticCluster:
        """Get the most specific cluster containing the particle at the given resolution.
        
        Args:
            particle_idx: Index of the particle
            resolution: Resolution level (0.0 = coarsest, 1.0 = finest)
            
        Returns:
            The most specific SemanticCluster containing the particle at the given resolution
        """
        # This would be implemented to traverse the hierarchy based on resolution
        pass  # Implementation would go here
