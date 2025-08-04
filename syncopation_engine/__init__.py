"""Syncopation Engine for Artificial True Cognition.

This package implements the core components of the syncopation engine,
including semantic axis management, hierarchical clustering, and dynamic
attention mechanisms for adaptive semantic processing.
"""

from .core.semantic_axis import SemanticField, SemanticParticle
from .core.semantic_hierarchy import SemanticHierarchy, SemanticCluster
from .core.semantic_attention import SemanticAttention, MultiScaleAttention, visualize_attention

__version__ = "0.1.0"
__all__ = [
    'SemanticField',
    'SemanticParticle',
    'SemanticHierarchy',
    'SemanticCluster',
    'SemanticAttention',
    'MultiScaleAttention',
    'visualize_attention'
]
