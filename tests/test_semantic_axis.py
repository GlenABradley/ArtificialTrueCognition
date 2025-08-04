import unittest
import torch
import numpy as np
import tempfile
import os
from semantic_axis import SemanticParticle, SemanticField
from typing import List, Dict, Set, Tuple

class TestSemanticParticle(unittest.TestCase):
    def test_creation(self):
        """Test particle creation and basic properties."""
        vector = torch.rand(12)
        particle = SemanticParticle(vector, {'test': 'data'})
        self.assertTrue(torch.allclose(particle.vector, vector))
        self.assertEqual(particle.metadata['test'], 'data')
        
    def test_axis_access(self):
        """Test accessing individual semantic axes."""
        vector = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.0, 0.1], 
                            dtype=torch.float32)
        particle = SemanticParticle(vector)
        self.assertAlmostEqual(particle.get_axis_value(0), 0.1)
        self.assertAlmostEqual(particle.get_axis_value(11), 0.1)
        with self.assertRaises(ValueError):
            particle.get_axis_value(12)  # Invalid axis

class TestSemanticField(unittest.TestCase):
    def setUp(self):
        """Set up test environment with sample semantic clusters."""
        self.field = SemanticField()
        self.clusters = {
            'science': {'vector': [0.9, 0.8, 0.2, 0.3, 0.1, 0.4, 0.2, 0.7, 0.3, 0.6, 0.8, 0.5], 'size': 10},
            'art': {'vector': [0.1, 0.2, 0.9, 0.8, 0.7, 0.1, 0.8, 0.2, 0.8, 0.7, 0.6, 0.7], 'size': 10},
            'business': {'vector': [0.8, 0.7, 0.6, 0.9, 0.2, 0.8, 0.1, 0.5, 0.6, 0.8, 0.9, 0.8], 'size': 10},
            'cognitive_science': {'vector': [0.7, 0.6, 0.7, 0.6, 0.5, 0.3, 0.4, 0.6, 0.4, 0.7, 0.8, 0.6], 'size': 0},
            'digital_art': {'vector': [0.5, 0.6, 0.8, 0.7, 0.6, 0.5, 0.7, 0.4, 0.7, 0.8, 0.7, 0.7], 'size': 0},
            'noise': {'vector': [0.1 if i%2==0 else 0.9 for i in range(12)], 'size': 5}
        }
        
        # Generate particles for each cluster
        for cluster_name, cluster_data in self.clusters.items():
            if cluster_data['size'] > 0:  # Skip zero-sized clusters
                self._add_cluster(
                    base_vector=cluster_data['vector'],
                    n_particles=cluster_data['size'],
                    cluster_id=cluster_name
                )
    
    def _add_cluster(self, base_vector: List[float], n_particles: int, cluster_id: str, noise_scale: float = 0.03):
        """Helper to add a cluster of similar particles."""
        base = torch.tensor(base_vector, dtype=torch.float32)
        for _ in range(n_particles):
            noise = torch.normal(0, noise_scale, size=base.shape)
            vector = torch.clamp(base + noise, 0, 1)
            self.field.add_particle(
                SemanticParticle(vector, {
                    'cluster': cluster_id,
                    'original_vector': base.tolist()
                })
            )

    # Test 1: Basic Cluster Separation
    def test_cluster_separation(self):
        """Verify that clusters remain distinct and identifiable."""
        for cluster_name, cluster_data in self.clusters.items():
            if cluster_data['size'] > 0:  # Only test existing clusters
                results = self.field.find_similar(
                    torch.tensor(cluster_data['vector']),
                    torch.ones(12) * 0.1,
                    k=5
                )
                # Verify top results match the expected cluster
                for particle, _ in results[:3]:  # Check top 3 matches
                    self.assertEqual(particle.metadata['cluster'], cluster_name)

    # Test 2: Blended Concept Inference
    def test_blended_concept_inference(self):
        """Test if the system can identify blended concepts."""
        # Cognitive science is a blend of science and art
        cognitive_science = torch.tensor(self.clusters['cognitive_science']['vector'])
        results = self.field.find_similar(cognitive_science, torch.ones(12)*0.15, k=10)
        
        # Should find both science and art particles
        found_clusters = {p.metadata['cluster'] for p, _ in results}
        self.assertIn('science', found_clusters)
        self.assertIn('art', found_clusters)

    # Test 3: Negative Inference Space
    def test_negative_inference_space(self):
        """Test that the system doesn't make invalid inferences."""
        # Create an invalid blend of features
        invalid_blend = torch.tensor([
            0.9, 0.8,  # Science-like
            0.8, 0.8,  # Art-like
            0.1, 0.1,  # Contradictory
            0.8, 0.1,  # Mixed
            0.8, 0.8,  # Art-like
            0.1, 0.1   # Contradictory
        ])
        
        results = self.field.find_similar(invalid_blend, torch.ones(12)*0.2, k=5)
        
        # Should not strongly match any single cluster
        cluster_counts = {}
        for particle, _ in results:
            cluster = particle.metadata['cluster']
            cluster_counts[cluster] = cluster_counts.get(cluster, 0) + 1
        
        # No single cluster should dominate
        max_count = max(cluster_counts.values()) if cluster_counts else 0
        self.assertLess(max_count, 3, "System is overfitting to invalid blends")

    # Test 4: Dimensional Sensitivity
    def test_dimensional_sensitivity(self):
        """Test how changes in each dimension affect similarity."""
        base_vector = torch.tensor(self.clusters['science']['vector'])
        results = []
        
        # Test sensitivity of each dimension
        for dim in range(12):
            modified = base_vector.clone()
            modified[dim] = 1.0 - modified[dim]  # Flip the dimension
            
            # Find similar particles
            similar = self.field.find_similar(modified, torch.ones(12)*0.1, k=5)
            
            # Record how many still match original cluster
            matches = sum(1 for p, _ in similar 
                         if p.metadata['cluster'] == 'science')
            results.append((dim, matches))
        
        # At least some dimensions should be discriminative
        critical_dims = [dim for dim, matches in results if matches < 3]
        self.assertGreaterEqual(len(critical_dims), 3, 
                              "Should have at least 3 critical dimensions")

    # Test 5: Progressive Concept Formation
    def test_progressive_concept_formation(self):
        """Test if the system can form new concepts through blending."""
        base_concepts = ['science', 'art', 'business']
        base_vectors = [torch.tensor(self.clusters[name]['vector']) 
                       for name in base_concepts]
        
        # Create progressive blends
        for i in range(1, 4):  # 3 levels of blending
            blend_weight = i * 0.25
            blended = sum(vec * (1-blend_weight)/(len(base_vectors)-1) 
                         for vec in base_vectors[1:])
            blended += base_vectors[0] * blend_weight
            
            # Find similar particles
            results = self.field.find_similar(blended, torch.ones(12)*0.15, k=8)
            
            # Should find multiple source clusters
            source_clusters = [p.metadata['cluster'] for p, _ in results]
            unique_sources = set(source_clusters)
            self.assertGreaterEqual(len(unique_sources), 2,
                                  f"Blend {i} should connect multiple concepts")

    # Test 6: Contextual Tolerance
    def test_contextual_tolerance(self):
        """Test dynamic tolerance adjustment."""
        # Create a query between science and art
        query = (torch.tensor(self.clusters['science']['vector']) + 
                torch.tensor(self.clusters['art']['vector'])) / 2
        
        # First with equal tolerances
        equal_tols = torch.ones(12) * 0.1
        equal_results = self.field.find_similar(query, equal_tols, k=5)
        
        # Then with adjusted tolerances emphasizing science dimensions
        sci_tols = torch.ones(12) * 0.2  # Default high tolerance
        sci_tols[0] = 0.05  # Tight tolerance on first science dimension
        sci_tols[1] = 0.05  # Tight tolerance on second science dimension
        sci_results = self.field.find_similar(query, sci_tols, k=5)
        
        # Should get more science results with adjusted tolerances
        sci_count = sum(1 for p, _ in sci_results 
                       if 'sci' in p.metadata['cluster'].lower())
        equal_sci_count = sum(1 for p, _ in equal_results 
                            if 'sci' in p.metadata['cluster'].lower())
        self.assertGreater(sci_count, equal_sci_count,
                         "Tolerance adjustment should affect concept retrieval")

    # Test 7: Temporal Concept Evolution
    def test_temporal_concept_evolution(self):
        """Test how concepts evolve over time."""
        # Simulate concept evolution
        concepts = {
            'early_tech': [0.9, 0.8, 0.1, 0.2, 0.1, 0.3, 0.1, 0.6, 0.2, 0.5, 0.7, 0.4],
            'early_art': [0.1, 0.2, 0.9, 0.8, 0.7, 0.1, 0.8, 0.1, 0.7, 0.6, 0.5, 0.6]
        }
        
        # Add initial concepts
        for name, vector in concepts.items():
            self.field.add_particle(SemanticParticle(
                torch.tensor(vector),
                {'cluster': 'timeline', 'generation': 0, 'source': name}
            ))
        
        # Verify we can trace lineage
        early_tech = torch.tensor(concepts['early_tech'])
        results = self.field.find_similar(early_tech, torch.ones(12)*0.1, k=5)
        
        # Should find our early tech concept
        found = any(p.metadata.get('source') == 'early_tech' for p, _ in results)
        self.assertTrue(found, "Should find early tech concept")

    # Test 8: Cross-Concept Analogy
    def test_cross_concept_analogy(self):
        """Test analogical reasoning across concepts."""
        # science : experiment :: art : ?
        science = torch.tensor(self.clusters['science']['vector'])
        experiment = torch.tensor([0.85, 0.75, 0.3, 0.7, 0.4, 0.5, 0.3, 0.6, 0.4, 0.6, 0.7, 0.6])
        art = torch.tensor(self.clusters['art']['vector'])
        
        # Calculate analogy: art + (experiment - science)
        analogy_vector = art + (experiment - science)
        analogy_vector = torch.clamp(analogy_vector, 0, 1)
        
        # Find similar concepts
        results = self.field.find_similar(analogy_vector, torch.ones(12)*0.15, k=5)
        
        # Should find art-related concepts
        art_related = any('art' in p.metadata['cluster'].lower() for p, _ in results)
        self.assertTrue(art_related, "Should find art-related concepts in analogy")

    # Test 9: Serialization
    def test_serialization(self):
        """Test saving and loading the semantic field."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            try:
                # Save and load
                self.field.save(tmp.name)
                loaded_field = SemanticField.load(tmp.name)
                
                # Check basic properties
                self.assertEqual(len(loaded_field.particles), len(self.field.particles))
                
                # Check a few particles
                for orig, loaded in zip(self.field.particles, loaded_field.particles):
                    self.assertTrue(torch.allclose(orig.vector, loaded.vector))
                    self.assertEqual(orig.metadata['cluster'], loaded.metadata['cluster'])
                    
            finally:
                # Clean up
                try:
                    os.unlink(tmp.name)
                except:
                    pass

if __name__ == '__main__':
    unittest.main()
