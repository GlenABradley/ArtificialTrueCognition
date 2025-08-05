"""
Test Patch Integration
=====================

This script demonstrates how to integrate the negation patch with the original
test to fix the test_negative_inference_space issue.
"""

import torch
import numpy as np
import sys
import os
from typing import List, Dict, Tuple, Optional, Any

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'syncopation_engine', 'core'))
sys.path.append(os.path.dirname(__file__))

# Import the patch
from semantic_field_negation_patch import apply_negation_patch

# Import or define SemanticParticle and SemanticField
try:
    from semantic_axis import SemanticParticle, SemanticField
    print("✅ Using original SemanticField implementation")
except ImportError:
    print("⚠️  Using minimal SemanticField implementation")
    
    class SemanticParticle:
        def __init__(self, concept: str, vector: torch.Tensor, metadata: Dict = None):
            self.concept = concept
            self.vector = vector
            self.metadata = metadata or {}
    
    class SemanticField:
        def __init__(self):
            self.particles = []
        
        def add_particle(self, particle: SemanticParticle):
            self.particles.append(particle)
        
        def find_similar(self, query, tolerance, k=5, debug_id=None, debug_metadata=None):
            # Basic implementation for testing
            similarities = []
            for particle in self.particles:
                similarity = torch.cosine_similarity(query, particle.vector, dim=0).item()
                similarities.append((particle, max(0, similarity)))
            
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:k]


def create_test_semantic_field_original():
    """Create the exact same test field as in the original test"""
    field = SemanticField()
    
    # Test data from the original failing test
    clusters = {
        # Core Clusters
        'science': {'vector': [0.9, 0.8, 0.2, 0.3, 0.1, 0.4, 0.2, 0.7, 0.3, 0.8, 0.8, 0.6], 'size': 20},
        'art': {'vector': [0.5, 0.6, 0.9, 0.8, 0.7, 0.5, 0.8, 0.4, 0.8, 0.5, 0.6, 0.7], 'size': 20},
        'business': {'vector': [0.8, 0.7, 0.3, 0.7, 0.2, 0.8, 0.1, 0.5, 0.6, 0.4, 0.5, 0.8], 'size': 20},
        'technology': {'vector': [0.85, 0.75, 0.7, 0.6, 0.3, 0.6, 0.3, 0.8, 0.4, 0.7, 0.8, 0.7], 'size': 15},
        'philosophy': {'vector': [0.7, 0.8, 0.8, 0.9, 0.6, 0.5, 0.4, 0.7, 0.5, 0.9, 0.9, 0.5], 'size': 15},
        'mathematics': {'vector': [0.9, 0.9, 0.1, 0.2, 0.1, 0.3, 0.1, 0.9, 0.2, 0.9, 0.8, 0.7], 'size': 15},
        'biotech': {'vector': [0.8, 0.7, 0.3, 0.4, 0.3, 0.5, 0.3, 0.6, 0.4, 0.7, 0.7, 0.7], 'size': 8},
        'noise': {'vector': [0.1 if i%2==0 else 0.9 for i in range(12)], 'size': 5}
    }
    
    # Generate particles for each cluster
    for cluster_name, cluster_data in clusters.items():
        if cluster_data['size'] > 0:
            base = torch.tensor(cluster_data['vector'], dtype=torch.float32)
            for i in range(cluster_data['size']):
                noise = torch.normal(0, 0.03, size=base.shape)
                vector = torch.clamp(base + noise, 0, 1)
                particle = SemanticParticle(
                    concept=f"{cluster_name}_{i}",
                    vector=vector,
                    metadata={
                        'cluster': cluster_name,
                        'original_vector': base.tolist()
                    }
                )
                field.add_particle(particle)
    
    return field


def test_negative_inference_space_original(field):
    """Run the original failing test"""
    print("🔴 Testing ORIGINAL field behavior (should FAIL):")
    print("-" * 50)
    
    # Define the problematic test cases from the original test
    test_cases = [
        ('business', 'art'),
        ('mathematics', 'noise'), 
        ('philosophy', 'biotech')
    ]
    
    cluster_vectors = {
        'business': [0.8, 0.7, 0.3, 0.7, 0.2, 0.8, 0.1, 0.5, 0.6, 0.4, 0.5, 0.8],
        'art': [0.5, 0.6, 0.9, 0.8, 0.7, 0.5, 0.8, 0.4, 0.8, 0.5, 0.6, 0.7],
        'mathematics': [0.9, 0.9, 0.1, 0.2, 0.1, 0.3, 0.1, 0.9, 0.2, 0.9, 0.8, 0.7],
        'noise': [0.1 if i%2==0 else 0.9 for i in range(12)],
        'philosophy': [0.7, 0.8, 0.8, 0.9, 0.6, 0.5, 0.4, 0.7, 0.5, 0.9, 0.9, 0.5],
        'biotech': [0.8, 0.7, 0.3, 0.4, 0.3, 0.5, 0.3, 0.6, 0.4, 0.7, 0.7, 0.7]
    }
    
    total_passed = 0
    
    for cluster1, cluster2 in test_cases:
        print(f"\n🔍 Testing blend: {cluster1} + {cluster2}")
        
        # Create blend
        blend = (torch.tensor(cluster_vectors[cluster1]) + 
                torch.tensor(cluster_vectors[cluster2])) / 2
        
        # Find similar particles
        results = field.find_similar(blend, torch.ones(12) * 0.15, k=10)
        
        print(f"  Results: {len(results)} matches")
        if results:
            print(f"  Top similarity: {results[0][1]:.3f}")
        
        # Check test conditions
        count_pass = len(results) < 5
        similarity_pass = not results or results[0][1] < 0.7
        test_pass = count_pass and similarity_pass
        
        print(f"  Count test: {'PASS' if count_pass else 'FAIL'} ({len(results)} < 5)")
        print(f"  Similarity test: {'PASS' if similarity_pass else 'FAIL'} ({results[0][1]:.3f} < 0.7)" if results else "  Similarity test: PASS (no results)")
        print(f"  Overall: {'PASS' if test_pass else 'FAIL'}")
        
        if test_pass:
            total_passed += 1
    
    print(f"\n📊 Original Field Results: {total_passed}/{len(test_cases)} tests passed")
    return total_passed == len(test_cases)


def test_negative_inference_space_patched(field):
    """Run the test with the patched field"""
    print("\n🟢 Testing PATCHED field behavior (should PASS):")
    print("-" * 50)
    
    # Define the same test cases
    test_cases = [
        ('business', 'art'),
        ('mathematics', 'noise'), 
        ('philosophy', 'biotech')
    ]
    
    cluster_vectors = {
        'business': [0.8, 0.7, 0.3, 0.7, 0.2, 0.8, 0.1, 0.5, 0.6, 0.4, 0.5, 0.8],
        'art': [0.5, 0.6, 0.9, 0.8, 0.7, 0.5, 0.8, 0.4, 0.8, 0.5, 0.6, 0.7],
        'mathematics': [0.9, 0.9, 0.1, 0.2, 0.1, 0.3, 0.1, 0.9, 0.2, 0.9, 0.8, 0.7],
        'noise': [0.1 if i%2==0 else 0.9 for i in range(12)],
        'philosophy': [0.7, 0.8, 0.8, 0.9, 0.6, 0.5, 0.4, 0.7, 0.5, 0.9, 0.9, 0.5],
        'biotech': [0.8, 0.7, 0.3, 0.4, 0.3, 0.5, 0.3, 0.6, 0.4, 0.7, 0.7, 0.7]
    }
    
    total_passed = 0
    
    for cluster1, cluster2 in test_cases:
        print(f"\n🔍 Testing blend: {cluster1} + {cluster2}")
        
        # Create blend
        blend = (torch.tensor(cluster_vectors[cluster1]) + 
                torch.tensor(cluster_vectors[cluster2])) / 2
        
        # Find similar particles (now with negation awareness!)
        debug_metadata = {}
        results = field.find_similar(blend, torch.ones(12) * 0.15, k=10, debug_metadata=debug_metadata)
        
        print(f"  Results: {len(results)} matches")
        if results:
            print(f"  Top similarity: {results[0][1]:.3f}")
            
        # Show analysis if available
        if 'query_analysis' in debug_metadata:
            analysis = debug_metadata['query_analysis']
            print(f"  Incompatibility: {analysis['incompatibility_score']:.3f}")
            print(f"  Semantic coherence: {analysis['semantic_coherence']:.3f}")
        
        # Check test conditions
        count_pass = len(results) < 5
        similarity_pass = not results or results[0][1] < 0.7
        test_pass = count_pass and similarity_pass
        
        print(f"  Count test: {'PASS' if count_pass else 'FAIL'} ({len(results)} < 5)")
        print(f"  Similarity test: {'PASS' if similarity_pass else 'FAIL'} ({results[0][1]:.3f} < 0.7)" if results else "  Similarity test: PASS (no results)")
        print(f"  Overall: {'PASS' if test_pass else 'FAIL'}")
        
        if test_pass:
            total_passed += 1
    
    print(f"\n📊 Patched Field Results: {total_passed}/{len(test_cases)} tests passed")
    return total_passed == len(test_cases)


def main():
    """Main test function"""
    print("🚀 Semantic Field Negation Patch Integration Test")
    print("=" * 60)
    
    print("\n📝 Problem:")
    print("The original SemanticField returns too many matches (10) for invalid")
    print("concept blends like 'business+art' when it should return < 5 matches.")
    
    print("\n🔧 Solution:")
    print("Apply the negation patch that adds incompatibility detection and")
    print("adaptive penalties to filter out semantically invalid combinations.")
    
    # Create test field
    print("\n🏗️  Setting up test field...")
    original_field = create_test_semantic_field_original()
    print(f"Created field with {len(original_field.particles)} particles")
    
    # Test original behavior (should fail)
    original_passes = test_negative_inference_space_original(original_field)
    
    # Apply the patch
    print("\n🔧 Applying negation patch...")
    patched_field = apply_negation_patch(original_field)
    print("✅ Patch applied successfully!")
    
    # Test patched behavior (should pass)
    patched_passes = test_negative_inference_space_patched(patched_field)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 FINAL RESULTS:")
    print(f"   Original Field: {'PASS' if original_passes else 'FAIL'}")
    print(f"   Patched Field:  {'PASS' if patched_passes else 'FAIL'}")
    
    if not original_passes and patched_passes:
        print("\n🎉 SUCCESS! The negation patch successfully fixed the issue!")
        print("   ✅ Original field failed (as expected)")
        print("   ✅ Patched field passed all tests")
        print("\n💡 Integration Instructions:")
        print("   1. Import: from semantic_field_negation_patch import apply_negation_patch")
        print("   2. Apply:  field = apply_negation_patch(your_semantic_field)")
        print("   3. Use:    field.find_similar() now handles negation properly")
    else:
        print("\n❌ Something went wrong. Check the implementation.")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()