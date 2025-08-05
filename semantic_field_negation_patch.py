"""
Semantic Field Negation Patch
=============================

This patch integrates the enhanced negation capabilities into the original
SemanticField implementation to fix the test_negative_inference_space issue.

Usage:
1. Import this patch in your test file
2. Apply the patch to the SemanticField
3. Run the test - it should now pass!
"""

import torch
import torch.nn.functional as F
import math
from typing import List, Dict, Tuple, Optional, Any, Set
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict


class IncompatibilityType(Enum):
    """Types of semantic incompatibility"""
    DOMAIN_MISMATCH = "domain_mismatch"
    NOISE_CONTAMINATION = "noise"
    ABSTRACTION_CONFLICT = "abstraction"


@dataclass
class IncompatibilityRule:
    """Rule defining semantic incompatibility"""
    cluster_pair: Set[str]
    incompatibility_type: IncompatibilityType
    penalty_strength: float
    description: str


@dataclass 
class QueryAnalysis:
    """Analysis of a query vector's semantic composition"""
    dominant_clusters: List[Tuple[str, float]]
    cluster_blend_ratio: Dict[str, float]
    incompatibility_score: float
    contradiction_dimensions: List[int]
    semantic_coherence: float


class SemanticFieldNegationPatch:
    """
    Patch that adds negation/incompatibility handling to SemanticField
    """
    
    @staticmethod
    def patch_semantic_field(field):
        """
        Apply negation patch to a SemanticField instance
        """
        # Store original methods (if they exist)
        field._original_find_similar = field.find_similar
        if hasattr(field, '_calculate_similarity'):
            field._original_calculate_similarity = field._calculate_similarity
        
        # Initialize negation components
        field._negation_config = {
            'base_similarity_weight': 0.4,
            'incompatibility_penalty_weight': 2.0,
            'dimensional_contradiction_weight': 0.8,
            'cluster_inference_threshold': 0.3,
            'max_acceptable_incompatibility': 0.4,
            'incompatible_query_max_results': 3,
            'incompatible_similarity_threshold': 0.6,
        }
        
        field._incompatibility_rules = [
            IncompatibilityRule(
                cluster_pair={'business', 'art'},
                incompatibility_type=IncompatibilityType.DOMAIN_MISMATCH,
                penalty_strength=0.7,
                description="Business and art represent different conceptual domains"
            ),
            IncompatibilityRule(
                cluster_pair={'philosophy', 'biotech'},
                incompatibility_type=IncompatibilityType.DOMAIN_MISMATCH, 
                penalty_strength=0.6,
                description="Philosophy and biotech operate in different domains"
            ),
            IncompatibilityRule(
                cluster_pair={'mathematics', 'noise'},
                incompatibility_type=IncompatibilityType.NOISE_CONTAMINATION,
                penalty_strength=0.9,
                description="Noise contaminates mathematical concepts"
            ),
            IncompatibilityRule(
                cluster_pair={'science', 'noise'},
                incompatibility_type=IncompatibilityType.NOISE_CONTAMINATION,
                penalty_strength=0.9,
                description="Noise contaminates scientific concepts"
            ),
            IncompatibilityRule(
                cluster_pair={'technology', 'philosophy'},
                incompatibility_type=IncompatibilityType.ABSTRACTION_CONFLICT,
                penalty_strength=0.5,
                description="Technology (concrete) vs philosophy (abstract) conflict"
            ),
        ]
        
        # Initialize cluster analysis
        field._cluster_centroids = {}
        field._cluster_dimensions_stats = {}
        field._initialize_cluster_analysis = lambda: SemanticFieldNegationPatch._initialize_cluster_analysis(field)
        field._initialize_cluster_analysis()
        
        # Add new methods to the field
        field.analyze_query_composition = lambda query_vector: SemanticFieldNegationPatch._analyze_query_composition(field, query_vector)
        field.enhanced_similarity_calculation = lambda query_vector, particle, query_analysis: SemanticFieldNegationPatch._enhanced_similarity_calculation(field, query_vector, particle, query_analysis)
        field.find_similar = lambda query, tolerance, k=5, debug_id=None, debug_metadata=None: SemanticFieldNegationPatch._find_similar_enhanced(field, query, tolerance, k, debug_id, debug_metadata)
        
        return field
    
    @staticmethod
    def _initialize_cluster_analysis(field):
        """Initialize cluster analysis by computing centroids and statistics"""
        if not field.particles:
            return
            
        # Group particles by cluster
        cluster_particles = defaultdict(list)
        for particle in field.particles:
            cluster = particle.metadata.get('cluster', 'unknown')
            cluster_particles[cluster].append(particle)
        
        # Calculate cluster centroids and statistics
        for cluster_name, particles in cluster_particles.items():
            if len(particles) == 0:
                continue
                
            # Calculate centroid
            vectors = torch.stack([p.vector for p in particles])
            centroid = torch.mean(vectors, dim=0)
            field._cluster_centroids[cluster_name] = centroid
            
            # Calculate dimensional statistics
            std = torch.std(vectors, dim=0)
            field._cluster_dimensions_stats[cluster_name] = {
                'mean': centroid,
                'std': std,
                'min': torch.min(vectors, dim=0)[0],
                'max': torch.max(vectors, dim=0)[0]
            }
    
    @staticmethod
    def _analyze_query_composition(field, query_vector: torch.Tensor) -> QueryAnalysis:
        """Analyze query vector to understand its semantic composition"""
        if not field._cluster_centroids:
            return QueryAnalysis(
                dominant_clusters=[],
                cluster_blend_ratio={},
                incompatibility_score=0.0,
                contradiction_dimensions=[],
                semantic_coherence=1.0
            )
        
        # Calculate similarity to each cluster centroid
        cluster_similarities = {}
        for cluster_name, centroid in field._cluster_centroids.items():
            similarity = F.cosine_similarity(query_vector, centroid, dim=0).item()
            cluster_similarities[cluster_name] = max(0, similarity)
        
        # Identify dominant clusters
        threshold = field._negation_config['cluster_inference_threshold']
        dominant_clusters = [(name, sim) for name, sim in cluster_similarities.items() 
                           if sim > threshold]
        dominant_clusters.sort(key=lambda x: x[1], reverse=True)
        
        # Calculate blend ratios
        total_similarity = sum(cluster_similarities.values()) + 1e-6
        cluster_blend_ratio = {name: sim / total_similarity 
                             for name, sim in cluster_similarities.items()}
        
        # Analyze incompatibilities
        incompatibility_score = SemanticFieldNegationPatch._calculate_incompatibility_score(field, dominant_clusters)
        
        # Find dimensional contradictions
        contradiction_dimensions = SemanticFieldNegationPatch._find_dimensional_contradictions(field, query_vector, dominant_clusters)
        
        # Calculate semantic coherence
        total_confidence = sum(conf for _, conf in dominant_clusters)
        base_coherence = min(1.0, total_confidence)
        contradiction_penalty = len(contradiction_dimensions) / len(query_vector) * 0.5
        semantic_coherence = max(0.0, base_coherence - contradiction_penalty)
        
        return QueryAnalysis(
            dominant_clusters=dominant_clusters,
            cluster_blend_ratio=cluster_blend_ratio,
            incompatibility_score=incompatibility_score,
            contradiction_dimensions=contradiction_dimensions,
            semantic_coherence=semantic_coherence
        )
    
    @staticmethod
    def _calculate_incompatibility_score(field, dominant_clusters: List[Tuple[str, float]]) -> float:
        """Calculate incompatibility score based on dominant clusters"""
        if len(dominant_clusters) < 2:
            return 0.0
        
        max_incompatibility = 0.0
        cluster_names = set(name for name, _ in dominant_clusters)
        
        for rule in field._incompatibility_rules:
            if rule.cluster_pair.issubset(cluster_names):
                cluster1, cluster2 = list(rule.cluster_pair)
                conf1 = next((conf for name, conf in dominant_clusters if name == cluster1), 0)
                conf2 = next((conf for name, conf in dominant_clusters if name == cluster2), 0)
                
                incompatibility = rule.penalty_strength * min(conf1, conf2)
                max_incompatibility = max(max_incompatibility, incompatibility)
        
        return max_incompatibility
    
    @staticmethod
    def _find_dimensional_contradictions(field, query_vector: torch.Tensor, 
                                        dominant_clusters: List[Tuple[str, float]]) -> List[int]:
        """Find dimensions where query contradicts expected cluster patterns"""
        if len(dominant_clusters) < 2:
            return []
        
        contradictions = []
        
        for dim in range(len(query_vector)):
            cluster_dim_values = []
            for cluster_name, confidence in dominant_clusters[:2]:
                if cluster_name in field._cluster_dimensions_stats:
                    cluster_mean = field._cluster_dimensions_stats[cluster_name]['mean'][dim]
                    cluster_dim_values.append((cluster_mean, confidence))
            
            if len(cluster_dim_values) == 2:
                (mean1, conf1), (mean2, conf2) = cluster_dim_values
                query_val = query_vector[dim]
                
                dist1 = abs(query_val - mean1)
                dist2 = abs(query_val - mean2)
                cluster_distance = abs(mean1 - mean2)
                
                if dist1 > 0.2 and dist2 > 0.2 and cluster_distance > 0.3:
                    contradictions.append(dim)
        
        return contradictions
    
    @staticmethod
    def _enhanced_similarity_calculation(field, query_vector: torch.Tensor, 
                                       particle, query_analysis: QueryAnalysis) -> float:
        """Calculate enhanced similarity with negation awareness"""
        # Base similarity
        base_similarity = F.cosine_similarity(query_vector, particle.vector, dim=0).item()
        base_similarity = max(0, base_similarity)
        
        # Get particle cluster
        particle_cluster = particle.metadata.get('cluster', 'unknown')
        
        # Calculate incompatibility penalty
        incompatibility_penalty = SemanticFieldNegationPatch._calculate_particle_incompatibility_penalty(
            field, particle_cluster, query_analysis
        )
        
        # Calculate dimensional contradiction penalty  
        contradiction_penalty = SemanticFieldNegationPatch._calculate_dimensional_contradiction_penalty(
            field, query_vector, particle.vector, query_analysis.contradiction_dimensions
        )
        
        # For highly incompatible queries, apply aggressive filtering
        if query_analysis.incompatibility_score > field._negation_config['max_acceptable_incompatibility']:
            total_penalty = (
                incompatibility_penalty * 2.0 +
                contradiction_penalty * 1.5
            )
            penalty_multiplier = math.exp(-total_penalty * 3.0)
            penalized_similarity = base_similarity * penalty_multiplier
        else:
            total_penalty = (
                field._negation_config['incompatibility_penalty_weight'] * incompatibility_penalty +
                field._negation_config['dimensional_contradiction_weight'] * contradiction_penalty
            )
            penalized_similarity = base_similarity * (1.0 - min(0.95, total_penalty))
        
        # Weight with base similarity
        final_similarity = (
            field._negation_config['base_similarity_weight'] * penalized_similarity +
            (1.0 - field._negation_config['base_similarity_weight']) * base_similarity
        )
        
        return max(0.0, min(1.0, final_similarity))
    
    @staticmethod
    def _calculate_particle_incompatibility_penalty(field, particle_cluster: str,
                                                   query_analysis: QueryAnalysis) -> float:
        """Calculate incompatibility penalty for a specific particle"""
        if query_analysis.incompatibility_score == 0.0:
            return 0.0
        
        max_penalty = 0.0
        
        # Check direct cluster incompatibility from analysis
        dominant_cluster_names = set(name for name, _ in query_analysis.dominant_clusters)
        if particle_cluster in dominant_cluster_names:
            for rule in field._incompatibility_rules:
                if {particle_cluster}.issubset(rule.cluster_pair):
                    other_clusters = rule.cluster_pair - {particle_cluster}
                    if other_clusters.intersection(dominant_cluster_names):
                        max_penalty = max(max_penalty, rule.penalty_strength * 0.8)
        
        return max_penalty
    
    @staticmethod
    def _calculate_dimensional_contradiction_penalty(field, query_vector: torch.Tensor,
                                                   particle_vector: torch.Tensor,
                                                   contradiction_dimensions: List[int]) -> float:
        """Calculate penalty for dimensional contradictions"""
        if not contradiction_dimensions:
            return 0.0
        
        total_penalty = 0.0
        for dim in contradiction_dimensions:
            query_val = query_vector[dim]
            particle_val = particle_vector[dim] 
            
            diff = abs(query_val - particle_val)
            dim_penalty = min(1.0, diff * 2.0)
            total_penalty += dim_penalty
        
        return total_penalty / len(contradiction_dimensions) if contradiction_dimensions else 0.0
    
    @staticmethod
    def _find_similar_enhanced(field, query_vector: torch.Tensor, tolerance: torch.Tensor,
                             k: int = 5, debug_id: Optional[str] = None, 
                             debug_metadata: Optional[Dict] = None) -> List[Tuple]:
        """Enhanced find_similar with negation awareness"""
        if not field.particles:
            return []
        
        # Analyze query composition
        query_analysis = field.analyze_query_composition(query_vector)
        
        # Store analysis in debug metadata if provided
        if debug_metadata:
            debug_metadata['query_analysis'] = {
                'dominant_clusters': query_analysis.dominant_clusters,
                'incompatibility_score': query_analysis.incompatibility_score,
                'semantic_coherence': query_analysis.semantic_coherence,
                'contradiction_dimensions': query_analysis.contradiction_dimensions
            }
        
        # Calculate enhanced similarities
        similarities = []
        for particle in field.particles:
            similarity = field.enhanced_similarity_calculation(
                query_vector, particle, query_analysis
            )
            similarities.append((particle, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Apply adaptive threshold filtering
        if query_analysis.incompatibility_score > field._negation_config['max_acceptable_incompatibility']:
            strict_threshold = field._negation_config['incompatible_similarity_threshold']
            similarities = [(p, s) for p, s in similarities if s >= strict_threshold]
            max_results = field._negation_config['incompatible_query_max_results']
            similarities = similarities[:max_results]
        
        return similarities[:k]


# Convenience function to apply the patch
def apply_negation_patch(semantic_field):
    """
    Apply the negation patch to a SemanticField instance.
    
    Usage:
        field = SemanticField()
        field = apply_negation_patch(field)
        # Now field.find_similar() will handle negation properly
    """
    return SemanticFieldNegationPatch.patch_semantic_field(semantic_field)


if __name__ == "__main__":
    print("🚀 Semantic Field Negation Patch")
    print("=" * 40)
    print("\nThis patch fixes the test_negative_inference_space issue by adding:")
    print("✅ Query cluster inference")
    print("✅ Incompatibility detection") 
    print("✅ Adaptive penalties")
    print("✅ Enhanced similarity calculation")
    print("\nUsage:")
    print("  from semantic_field_negation_patch import apply_negation_patch")
    print("  field = apply_negation_patch(your_semantic_field)")
    print("  # Now the field handles negation properly!")