import torch
import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple, Set, Any
from dataclasses import dataclass, field
import json
from semantic_hierarchy import SemanticHierarchy, SemanticCluster
import os

@dataclass
class SemanticParticle:
    """Represents a single semantic particle with a 12D vector."""
    
    # Axis indices for semantic dimensions
    LEXICAL = 0
    ETYMOLOGICAL = 1
    SYNTACTIC = 2
    PRAGMATIC = 3
    EMOTIONAL = 4
    TEMPORAL = 5
    SPATIAL = 6
    CAUSAL = 7
    SOCIAL = 8
    MODAL = 9
    THEMATIC = 10
    FUNCTIONAL = 11
    
    # Axis names for better readability
    AXIS_NAMES = {
        LEXICAL: "lexical",
        ETYMOLOGICAL: "etymological",
        SYNTACTIC: "syntactic",
        PRAGMATIC: "pragmatic",
        EMOTIONAL: "emotional",
        TEMPORAL: "temporal",
        SPATIAL: "spatial",
        CAUSAL: "causal",
        SOCIAL: "social",
        MODAL: "modal",
        THEMATIC: "thematic",
        FUNCTIONAL: "functional"
    }
    
    def __init__(self, concept: str, vector, metadata: dict = None):
        """Initialize a semantic particle.
        
        Args:
            concept: The name or identifier of the concept
            vector: The 12D semantic vector (list, numpy array, or torch.Tensor)
            metadata: Optional metadata dictionary
        """
        self.concept = concept
        self.metadata = metadata or {}
        
        # Convert vector to torch.Tensor if it isn't already
        if not isinstance(vector, torch.Tensor):
            try:
                if isinstance(vector, (list, tuple)):
                    self.vector = torch.tensor(vector, dtype=torch.float32)
                else:
                    # Try to convert to numpy array first
                    np_array = np.array(vector, dtype=np.float32)
                    self.vector = torch.from_numpy(np_array)
            except Exception as e:
                raise ValueError(f"Could not convert vector to torch.Tensor: {e}")
        else:
            self.vector = vector.clone().detach().float()
        
        # Ensure vector has 12 dimensions
        if len(self.vector) != 12:
            raise ValueError(f"Vector must have 12 dimensions, got {len(self.vector)}")
            
        # Ensure values are in [0, 1] range
        self.vector = torch.clamp(self.vector, 0, 1)
    
    def get_axis_value(self, axis: int) -> float:
        """Get the value of a specific semantic axis."""
        if not 0 <= axis < 12:
            raise ValueError("Axis must be between 0 and 11")
        return float(self.vector[axis])
    
    def to_dict(self) -> Dict:
        """Convert to a serializable dictionary."""
        return {
            'vector': self.vector.tolist(),
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SemanticParticle':
        """Create from a dictionary."""
        return cls(
            vector=torch.tensor(data['vector'], dtype=torch.float32),
            metadata=data.get('metadata', {})
        )

class SemanticField:
    """A field of semantic particles with adaptive sensitivity tuning."""
    
    def __init__(self, target_contrast: float = 0.5, min_cluster_size: int = 5):
        self.particles: List[SemanticParticle] = []
        self.axis_names = [
            "lexical", "etymological", "syntactic", "pragmatic",
            "emotional", "temporal", "spatial", "causal",
            "social", "modal", "thematic", "functional"
        ]
        self.sensitivities = torch.ones(12) / 12  # Initial uniform sensitivities
        self.target_contrast = target_contrast
        self.correlation_history = []
        self.axis_means = None
        self.axis_stds = None
        
        # Initialize hierarchical clustering
        self.hierarchy = SemanticHierarchy(min_cluster_size=min_cluster_size)
        self._hierarchy_dirty = True  # Flag to track if hierarchy needs rebuilding
        
    def add_particle(self, particle: 'SemanticParticle') -> None:
        """Add a semantic particle to the field."""
        self.particles.append(particle)
        self._update_statistics()
        self._hierarchy_dirty = True  # Mark hierarchy as needing update
        self.axis_means = None
        self.axis_stds = None
        
    def _update_statistics(self) -> None:
        """Update statistics about the semantic space and rebuild hierarchy if needed."""
        if not self.particles:
            return
            
        vectors = torch.stack([p.vector for p in self.particles])
        self.axis_means = torch.mean(vectors, dim=0)
        self.axis_stds = torch.std(vectors, dim=0)
        
        # Avoid division by zero for constant dimensions
        self.axis_stds = torch.where(
            self.axis_stds < 1e-6,
            torch.ones_like(self.axis_stds),
            self.axis_stds
        )
        
        # Update correlation matrix history (keep last 10)
        corr_matrix = self._compute_correlation_matrix()
        self.correlation_history.append(corr_matrix)
        if len(self.correlation_history) > 10:
            self.correlation_history.pop(0)
            
        # Rebuild hierarchy if needed
        if self._hierarchy_dirty and len(self.particles) >= 5:  # Minimum particles for clustering
            self.rebuild_hierarchy()
    
    def rebuild_hierarchy(self) -> None:
        """Rebuild the hierarchical clustering of particles."""
        if len(self.particles) < 5:  # HDBSCAN needs at least min_cluster_size + 1 points
            return
            
        self.hierarchy.build_hierarchy(self.particles)
        self._hierarchy_dirty = False
    
    def _compute_correlation_matrix(self) -> np.ndarray:
        """Compute the correlation matrix of the semantic space."""
        if len(self.particles) < 2:
            return np.eye(12)  # Identity matrix if not enough data
            
        vectors = torch.stack([p.vector for p in self.particles]).numpy()
        return np.corrcoef(vectors.T)
    
    def _measure_contrast(self, matrix: np.ndarray) -> float:
        """Calculate the contrast ratio of a matrix."""
        if matrix.size == 0:
            return 0.0
        return float(np.std(matrix) / (np.mean(np.abs(matrix)) + 1e-6))
    
    def _adaptive_smoothing(self, matrix: np.ndarray) -> np.ndarray:
        """Apply adaptive smoothing to reduce contrast while preserving structure."""
        if matrix.size == 0:
            return matrix
            
        current_contrast = self._measure_contrast(matrix)
        smoothed = matrix.copy()
        window_size = 3
        
        while current_contrast > self.target_contrast and window_size < min(matrix.shape):
            # Apply local averaging with increasing window size
            kernel = np.ones((window_size, window_size)) / (window_size ** 2)
            smoothed = cv2.filter2D(smoothed, -1, kernel)
            current_contrast = self._measure_contrast(smoothed)
            window_size += 2  # Increase window size for next iteration if needed
            
        return smoothed
    
    def tune_sensitivities(self) -> None:
        """Adjust dimensional sensitivities based on current data distribution."""
        if len(self.particles) < 2:
            return
            
        # Get current correlation matrix
        corr_matrix = self._compute_correlation_matrix()
        
        # Apply adaptive smoothing
        smoothed = self._adaptive_smoothing(corr_matrix)
        
        # Calculate dimensional variances (diagonal of covariance matrix)
        vectors = torch.stack([p.vector for p in self.particles])
        variances = torch.var(vectors, dim=0)
        
        # Calculate dimensional importances (inverse of variance, normalized)
        # Add small epsilon to avoid division by zero
        importances = 1 / (variances + 1e-6)
        
        # Adjust sensitivities based on correlation structure
        # Dimensions that are highly correlated with others should have reduced sensitivity
        correlation_scores = torch.tensor(
            np.sum(np.abs(smoothed), axis=1) - 1,  # Subtract self-correlation
            dtype=torch.float32
        )
        correlation_weights = 1 / (1 + correlation_scores)  # Less sensitive for highly correlated dims
        
        # Combine importances with correlation weights
        self.sensitivities = importances * correlation_weights
        self.sensitivities = self.sensitivities / (torch.sum(self.sensitivities) + 1e-6)  # Normalize
    
    def calculate_similarity(self, v1: torch.Tensor, v2: torch.Tensor, 
                           context: Optional[Dict[str, Any]] = None) -> float:
        """Calculate similarity between two vectors using current sensitivities and optional context.
        
        Args:
            v1: First vector
            v2: Second vector
            context: Optional context dictionary that may include:
                - resolution: float between 0.0 (coarse) and 1.0 (fine)
                - focus_dimensions: list of dimension indices to focus on
                
        Returns:
            Similarity score between 0.0 and 1.0
        """
        # Base similarity using current sensitivities
        diff = (v1 - v2).abs()
        
        # Apply context-aware adjustments if provided
        sensitivities = self.sensitivities.clone()
        
        if context:
            # Adjust for resolution
            resolution = context.get('resolution', 1.0)
            if resolution < 1.0:
                # At lower resolutions, reduce sensitivity to fine details
                sensitivities = sensitivities * (0.5 + 0.5 * resolution)
                
            # Focus on specific dimensions if requested
            focus_dims = context.get('focus_dimensions')
            if focus_dims is not None:
                mask = torch.zeros_like(sensitivities)
                for dim in focus_dims:
                    if 0 <= dim < len(mask):
                        mask[dim] = 1.0
                if mask.sum() > 0:  # Only apply if we have valid dimensions
                    sensitivities = sensitivities * mask
        
        # Calculate weighted difference and similarity
        weighted_diff = diff * sensitivities.to(v1.device)
        return float(torch.exp(-torch.sum(weighted_diff)))
    
    def get_cluster_context(self, particle_idx: int, resolution: float = 0.5) -> Dict[str, Any]:
        """Get context for a particle based on its cluster at the given resolution.
        
        Args:
            particle_idx: Index of the particle in self.particles
            resolution: Desired resolution level (0.0 = coarsest, 1.0 = finest)
            
        Returns:
            Context dictionary with cluster information
        """
        if self._hierarchy_dirty:
            self.rebuild_hierarchy()
            
        # Default context
        context = {
            'resolution': resolution,
            'focus_dimensions': None,
            'cluster_size': 1,
            'depth': 0
        }
        
        if not self.hierarchy.clusters:
            return context
            
        # Get the appropriate cluster for this resolution
        cluster = self.hierarchy.get_cluster_at_resolution(particle_idx, resolution)
        if cluster:
            # Calculate which dimensions vary the least in this cluster
            # (these are the most defining dimensions for this cluster)
            cluster_vectors = torch.stack([self.particles[i].vector for i in cluster.particles])
            variances = torch.var(cluster_vectors, dim=0)
            
            # Focus on the most stable dimensions (lowest variance)
            num_focus_dims = max(1, int(len(variances) * (1.0 - resolution)))
            focus_dims = torch.topk(variances, k=num_focus_dims, largest=False).indices.tolist()
            
            context.update({
                'focus_dimensions': focus_dims,
                'cluster_size': len(cluster.particles),
                'depth': cluster.level
            })
            
        return context
    
    def _get_axis_importance(self, tolerance: torch.Tensor) -> torch.Tensor:
        """Calculate axis importance based on tolerance and data statistics.
        
        This version implements a more sophisticated importance calculation that:
        1. Strongly emphasizes dimensions with low tolerance
        2. Considers both tolerance and data variability
        3. Uses non-linear transformations for better discrimination
        """
        if self.axis_means is None:
            self._update_statistics()
            if self.axis_means is None:  # Fallback if no statistics
                return torch.ones(12) / 12.0
        
        # Convert tolerance to importance with strong non-linearity
        # This makes small tolerances MUCH more important
        importance = 1.0 / (tolerance + 1e-6)
        importance = torch.pow(importance, 3)  # Cube to strongly emphasize differences
        
        # Normalize importance
        importance = importance / (torch.sum(importance) + 1e-6)
        
        # Adjust by variance - dimensions with higher variance get lower weight
        # But we don't want to completely ignore high-variance dimensions
        variance_weights = 1.0 / (self.axis_stds + 1e-6)
        variance_weights = torch.pow(variance_weights, 0.7)  # Reduce extreme weights
        variance_weights = variance_weights / (torch.sum(variance_weights) + 1e-6)
        
        # Combine with emphasis on tolerance-based importance
        combined = 0.8 * importance + 0.2 * variance_weights
        
        # Ensure no dimension gets zero weight and renormalize
        combined = torch.clamp(combined, min=1e-4)
        return combined / (torch.sum(combined) + 1e-6)
    
    def _calculate_similarity(self, v1: torch.Tensor, v2: torch.Tensor, 
                            axis_importance: torch.Tensor) -> float:
        """Calculate similarity between two vectors using a multi-scale approach.
        
        This version combines multiple similarity measures:
        1. Weighted cosine similarity for global structure
        2. Jaccard similarity for set-like matching of dimensions
        3. A non-linear transformation to emphasize important differences
        """
        # Calculate element-wise differences
        diff = torch.abs(v1 - v2)
        
        # Calculate weighted cosine similarity (global structure)
        dot_product = torch.sum(v1 * v2 * axis_importance)
        norm1 = torch.sqrt(torch.sum((v1 * axis_importance) ** 2))
        norm2 = torch.sqrt(torch.sum((v2 * axis_importance) ** 2))
        cosine_sim = dot_product / (norm1 * norm2 + 1e-6)
        
        # Calculate Jaccard similarity (set-like matching)
        threshold = 0.3  # Threshold for considering values 'similar'
        v1_set = (v1 > threshold).float()
        v2_set = (v2 > threshold).float()
        intersection = torch.sum(torch.min(v1_set, v2_set) * axis_importance)
        union = torch.sum(torch.max(v1_set, v2_set) * axis_importance)
        jaccard_sim = intersection / (union + 1e-6)
        
        # Calculate weighted Manhattan distance (local differences)
        weighted_diff = diff * axis_importance
        manhattan_dist = torch.sum(weighted_diff)
        
        # Calculate dimension matching score
        matching_dims = torch.sum((diff < 0.25).float() * axis_importance)
        
        # Combine all components with adaptive weights
        # When vectors are very similar, emphasize cosine similarity
        # When they differ more, put more weight on set-like matching
        similarity_balance = torch.sigmoid(5.0 * (manhattan_dist - 1.0))
        
        # Combine cosine and Jaccard similarities
        combined_sim = (1.0 - similarity_balance) * cosine_sim + similarity_balance * jaccard_sim
        
        # Apply non-linear scaling to emphasize differences
        # This makes the similarity score more discriminative
        similarity = torch.sigmoid(5.0 * (combined_sim - 0.7)) * 0.8 + 0.2 * matching_dims
        
        return float(similarity)
        
    def find_similar(self, query: torch.Tensor, tolerance: torch.Tensor, 
                    k: int = 5) -> List[Tuple[SemanticParticle, float]]:
        """Find particles most similar to the query vector using axis-wise tolerance.
        
        Args:
            query: The query vector (12D tensor)
            tolerance: Per-axis tolerance values (12D tensor), lower means more weight
            k: Number of results to return
            
        Returns:
            List of (particle, similarity_score) tuples, sorted by similarity
        """
        if not self.particles:
            return []
            
        # Ensure query is a tensor
        if not isinstance(query, torch.Tensor):
            query = torch.tensor(query, dtype=torch.float32)
            
        # Calculate axis importance based on tolerance and data statistics
        axis_importance = self._get_axis_importance(tolerance)
        
        # Calculate similarities to all particles
        similarities = []
        for particle in self.particles:
            similarity = self._calculate_similarity(
                query, particle.vector, axis_importance)
            similarities.append((particle, similarity))
        
        # Sort by similarity (descending) and return top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:min(k, len(similarities))]
    
    def find_by_axis_range(self, axis: int, min_val: float, max_val: float) -> List[SemanticParticle]:
        """Find particles where the specified axis value falls within the given range."""
        if not 0 <= axis < 12:
            raise ValueError("Axis must be between 0 and 11")
            
        return [p for p in self.particles 
                if min_val <= p.get_axis_value(axis) <= max_val]
        
    def save(self, filepath: str) -> None:
        """Save the semantic field to a file."""
        data = {
            'particles': [p.to_dict() for p in self.particles]
        }
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'SemanticField':
        """Load a semantic field from a file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        field = cls()
        for p_data in data['particles']:
            field.add_particle(SemanticParticle.from_dict(p_data))
        
        return field
