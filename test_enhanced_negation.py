"""
Test script for Enhanced Negation Semantic Field
===============================================

This script tests the enhanced negation capabilities by:
1. Running the problematic test_negative_inference_space 
2. Comparing original vs enhanced behavior
3. Providing detailed analysis of improvements
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import sys
import os

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'syncopation_engine', 'core'))
sys.path.append(os.path.dirname(__file__))

try:
    from semantic_axis import SemanticParticle, SemanticField
except ImportError:
    # Fallback implementations
    class SemanticParticle:
        def __init__(self, concept: str, vector: torch.Tensor, metadata: Dict = None):
            self.concept = concept
            self.vector = vector  
            self.metadata = metadata or {}

from enhanced_negation_semantic_field import EnhancedNegationSemanticField


def create_test_semantic_field():
    """Create the same test field as in the failing test"""
    
    # Create semantic particles based on the test setup
    field_data = {
        # Core Clusters (from the original test)
        'science': {'vector': [0.9, 0.8, 0.2, 0.3, 0.1, 0.4, 0.2, 0.7, 0.3, 0.8, 0.8, 0.6], 'size': 20},
        'art': {'vector': [0.5, 0.6, 0.9, 0.8, 0.7, 0.5, 0.8, 0.4, 0.8, 0.5, 0.6, 0.7], 'size': 20},
        'business': {'vector': [0.8, 0.7, 0.3, 0.7, 0.2, 0.8, 0.1, 0.5, 0.6, 0.4, 0.5, 0.8], 'size': 20},
        'technology': {'vector': [0.85, 0.75, 0.7, 0.6, 0.3, 0.6, 0.3, 0.8, 0.4, 0.7, 0.8, 0.7], 'size': 15},
        'philosophy': {'vector': [0.7, 0.8, 0.8, 0.9, 0.6, 0.5, 0.4, 0.7, 0.5, 0.9, 0.9, 0.5], 'size': 15},
        'mathematics': {'vector': [0.9, 0.9, 0.1, 0.2, 0.1, 0.3, 0.1, 0.9, 0.2, 0.9, 0.8, 0.7], 'size': 15},
        'biotech': {'vector': [0.8, 0.7, 0.3, 0.4, 0.3, 0.5, 0.3, 0.6, 0.4, 0.7, 0.7, 0.7], 'size': 8},
        # Noise cluster for testing
        'noise': {'vector': [0.1 if i%2==0 else 0.9 for i in range(12)], 'size': 5}
    }
    
    particles = []
    
    # Generate particles for each cluster
    for cluster_name, cluster_data in field_data.items():
        if cluster_data['size'] > 0:
            particles.extend(_add_cluster_particles(
                base_vector=cluster_data['vector'],
                n_particles=cluster_data['size'],
                cluster_id=cluster_name
            ))
    
    return particles


def _add_cluster_particles(base_vector: List[float], n_particles: int, 
                          cluster_id: str, noise_scale: float = 0.03) -> List[SemanticParticle]:
    """Helper to create a cluster of similar particles."""
    particles = []
    base = torch.tensor(base_vector, dtype=torch.float32)
    
    for i in range(n_particles):
        noise = torch.normal(0, noise_scale, size=base.shape)
        vector = torch.clamp(base + noise, 0, 1)
        particle = SemanticParticle(
            concept=f"{cluster_id}_{i}",
            vector=vector,
            metadata={
                'cluster': cluster_id,
                'original_vector': base.tolist()
            }
        )
        particles.append(particle)
    
    return particles


def test_negative_inference_original_vs_enhanced():
    """Test the negative inference space problem with both approaches"""
    
    print("🧪 Testing Negative Inference Space: Original vs Enhanced")
    print("=" * 70)
    
    # Create test data
    particles = create_test_semantic_field()
    
    # Create enhanced field
    enhanced_field = EnhancedNegationSemanticField()
    enhanced_field.particles = particles
    enhanced_field._initialize_cluster_analysis()
    
    # Define the problematic test cases
    test_cases = [
        ('business', 'art'),         # Business + Art (unrelated domains)
        ('mathematics', 'noise'),    # Math + Noise (noise should be ignored) 
        ('philosophy', 'biotech')    # Philosophy + Biotech (distant domains)
    ]
    
    # Test cluster vectors for reference
    cluster_vectors = {
        'business': [0.8, 0.7, 0.3, 0.7, 0.2, 0.8, 0.1, 0.5, 0.6, 0.4, 0.5, 0.8],
        'art': [0.5, 0.6, 0.9, 0.8, 0.7, 0.5, 0.8, 0.4, 0.8, 0.5, 0.6, 0.7],
        'mathematics': [0.9, 0.9, 0.1, 0.2, 0.1, 0.3, 0.1, 0.9, 0.2, 0.9, 0.8, 0.7],
        'noise': [0.1 if i%2==0 else 0.9 for i in range(12)],
        'philosophy': [0.7, 0.8, 0.8, 0.9, 0.6, 0.5, 0.4, 0.7, 0.5, 0.9, 0.9, 0.5],
        'biotech': [0.8, 0.7, 0.3, 0.4, 0.3, 0.5, 0.3, 0.6, 0.4, 0.7, 0.7, 0.7]
    }
    
    for cluster1, cluster2 in test_cases:
        print(f"\n🔍 Testing blend: {cluster1} + {cluster2}")
        print("-" * 50)
        
        # Create blend vector (average of two clusters)
        blend_vector = (torch.tensor(cluster_vectors[cluster1]) + 
                       torch.tensor(cluster_vectors[cluster2])) / 2
        
        # Test with enhanced field
        tolerance = torch.ones(12) * 0.15
        debug_metadata = {}
        
        enhanced_results = enhanced_field.find_similar_enhanced(
            blend_vector, tolerance, k=10, debug_metadata=debug_metadata
        )
        
        print(f"Enhanced Field Results: {len(enhanced_results)} matches")
        
        # Show query analysis
        if 'query_analysis' in debug_metadata:
            analysis = debug_metadata['query_analysis']
            print(f"  📊 Incompatibility Score: {analysis['incompatibility_score']:.3f}")
            print(f"  🧠 Semantic Coherence: {analysis['semantic_coherence']:.3f}")
            print(f"  🏷️  Dominant Clusters: {analysis['dominant_clusters'][:3]}")
            if analysis['contradiction_dimensions']:
                print(f"  ⚠️  Contradiction Dims: {analysis['contradiction_dimensions']}")
        
        # Show top results
        print("  🎯 Top Results:")
        for i, (particle, score) in enumerate(enhanced_results[:5]):
            cluster = particle.metadata.get('cluster', 'unknown')
            print(f"     {i+1}. {cluster} (score: {score:.3f})")
        
        # Generate detailed incompatibility report
        report = enhanced_field.get_incompatibility_report(blend_vector)
        
        # Show triggered rules
        if report['triggered_rules']:
            print("  🚫 Triggered Incompatibility Rules:")
            for rule in report['triggered_rules']:
                print(f"     - {rule['clusters']}: {rule['description']} (penalty: {rule['penalty_strength']:.2f})")
        
        # Check if it passes the test criteria
        passes_test = len(enhanced_results) < 5
        if enhanced_results and len(enhanced_results) > 0:
            top_similarity = enhanced_results[0][1]
            similarity_ok = top_similarity < 0.7
        else:
            similarity_ok = True
            
        overall_pass = passes_test and similarity_ok
        
        print(f"  ✅ Test Result: {'PASS' if overall_pass else 'FAIL'}")
        if not passes_test:
            print(f"     - Too many matches: {len(enhanced_results)} >= 5")
        if enhanced_results and not similarity_ok:
            print(f"     - Similarity too high: {enhanced_results[0][1]:.3f} >= 0.7")
        
        print()


def test_query_analysis_capabilities():
    """Test the query analysis capabilities in detail"""
    
    print("🔬 Testing Query Analysis Capabilities")
    print("=" * 50)
    
    # Create enhanced field
    particles = create_test_semantic_field()
    enhanced_field = EnhancedNegationSemanticField()
    enhanced_field.particles = particles
    enhanced_field._initialize_cluster_analysis()
    
    # Test different types of queries
    test_queries = [
        ("Pure Science", [0.9, 0.8, 0.2, 0.3, 0.1, 0.4, 0.2, 0.7, 0.3, 0.8, 0.8, 0.6]),
        ("Business+Art Blend", [(0.8+0.5)/2, (0.7+0.6)/2, (0.3+0.9)/2, (0.7+0.8)/2, (0.2+0.7)/2, (0.8+0.5)/2, (0.1+0.8)/2, (0.5+0.4)/2, (0.6+0.8)/2, (0.4+0.5)/2, (0.5+0.6)/2, (0.8+0.7)/2]),
        ("Math+Noise Contamination", [(0.9+0.1)/2, (0.9+0.9)/2, (0.1+0.1)/2, (0.2+0.9)/2, (0.1+0.1)/2, (0.3+0.9)/2, (0.1+0.1)/2, (0.9+0.9)/2, (0.2+0.1)/2, (0.9+0.9)/2, (0.8+0.1)/2, (0.7+0.9)/2])
    ]
    
    for query_name, query_vector in test_queries:
        print(f"\n🎯 Analyzing: {query_name}")
        query_tensor = torch.tensor(query_vector, dtype=torch.float32)
        
        analysis = enhanced_field.analyze_query_composition(query_tensor)
        report = enhanced_field.get_incompatibility_report(query_tensor)
        
        print(f"  📊 Incompatibility Score: {analysis.incompatibility_score:.3f}")
        print(f"  🧠 Semantic Coherence: {analysis.semantic_coherence:.3f}")
        print(f"  🏷️  Dominant Clusters: {analysis.dominant_clusters[:3]}")
        
        if analysis.contradiction_dimensions:
            print(f"  ⚠️  Contradictory Dimensions: {analysis.contradiction_dimensions}")
            
        if report['triggered_rules']:
            print("  🚫 Incompatibility Rules:")
            for rule in report['triggered_rules']:
                print(f"     - {rule['description']} (penalty: {rule['penalty_strength']:.2f})")
                
        if report['recommendations']:
            print("  💡 Recommendations:")
            for rec in report['recommendations']:
                print(f"     - {rec}")


def run_comprehensive_test():
    """Run comprehensive test suite"""
    
    print("🚀 Enhanced Negation Semantic Field - Comprehensive Test")
    print("=" * 70)
    
    print("\n📝 Problem Description:")
    print("The original semantic field fails test_negative_inference_space because")
    print("it returns too many matches (10) for invalid concept blends like 'business+art'")
    print("when it should return fewer than 5 matches with low similarity scores.")
    
    print("\n🔧 Solution Implemented:")
    print("1. ✅ Query Cluster Inference - Determines constituent clusters of blended queries")
    print("2. ✅ Incompatibility Rule Engine - Detects semantically incompatible combinations") 
    print("3. ✅ Multi-Strategy Penalties - Multiplicative penalties for stronger filtering")
    print("4. ✅ Dimensional Contradiction Analysis - Detects contradictory patterns")
    print("5. ✅ Adaptive Thresholding - Context-aware similarity filtering")
    
    # Run the main test
    test_negative_inference_original_vs_enhanced()
    
    # Run detailed analysis
    test_query_analysis_capabilities()
    
    print("\n🎉 Testing Complete!")
    print("\nThe enhanced field should now properly handle semantic negation by:")
    print("- Detecting incompatible concept combinations")  
    print("- Applying appropriate penalties to reduce false matches")
    print("- Providing detailed analysis for debugging and understanding")


if __name__ == "__main__":
    run_comprehensive_test()