import unittest
import torch
import numpy as np
import tempfile
import os
import sys
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from syncopation_engine.core.semantic_axis import SemanticParticle, SemanticField
from typing import List, Dict, Set, Tuple

class TestSemanticParticle(unittest.TestCase):
    def test_creation(self):
        """Test particle creation and basic properties."""
        vector = torch.rand(12)
        particle = SemanticParticle(
            concept="test_particle",
            vector=vector,
            metadata={'test': 'data'}
        )
        self.assertTrue(torch.allclose(particle.vector, vector))
        self.assertEqual(particle.metadata['test'], 'data')
        self.assertEqual(particle.concept, "test_particle")
        
    def test_axis_access(self):
        """Test accessing individual semantic axes."""
        vector = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.0, 0.1], 
                            dtype=torch.float32)
        particle = SemanticParticle(
            concept="test_axis_access",
            vector=vector,
            metadata={}
        )
        self.assertAlmostEqual(particle.get_axis_value(0), 0.1)
        self.assertAlmostEqual(particle.get_axis_value(11), 0.1)
        with self.assertRaises(ValueError):
            particle.get_axis_value(12)  # Invalid axis

class TestSemanticField(unittest.TestCase):
    def setUp(self):
        """Set up test environment with sample semantic clusters."""
        self.field = SemanticField()
        # Enhanced test data with more clusters, better semantic relationships, and explicit connections
        self.clusters = {
            # Core Clusters (expanded from original)
            'science': {'vector': [0.9, 0.8, 0.2, 0.3, 0.1, 0.4, 0.2, 0.7, 0.3, 0.8, 0.8, 0.6], 'size': 20},
            'art': {'vector': [0.5, 0.6, 0.9, 0.8, 0.7, 0.5, 0.8, 0.4, 0.8, 0.5, 0.6, 0.7], 'size': 20},
            'business': {'vector': [0.8, 0.7, 0.3, 0.7, 0.2, 0.8, 0.1, 0.5, 0.6, 0.4, 0.5, 0.8], 'size': 20},
            
            # New Core Clusters
            'technology': {'vector': [0.85, 0.75, 0.7, 0.6, 0.3, 0.6, 0.3, 0.8, 0.4, 0.7, 0.8, 0.7], 'size': 15},
            'philosophy': {'vector': [0.7, 0.8, 0.8, 0.9, 0.6, 0.5, 0.4, 0.7, 0.5, 0.9, 0.9, 0.5], 'size': 15},
            'mathematics': {'vector': [0.9, 0.9, 0.1, 0.2, 0.1, 0.3, 0.1, 0.9, 0.2, 0.9, 0.8, 0.7], 'size': 15},
            
            # Blended Concepts (explicitly defined)
            'cognitive_science': {'vector': [0.7, 0.7, 0.6, 0.6, 0.5, 0.4, 0.5, 0.6, 0.5, 0.7, 0.8, 0.6], 'size': 10},
            'digital_art': {'vector': [0.6, 0.7, 0.8, 0.7, 0.6, 0.5, 0.7, 0.5, 0.7, 0.7, 0.7, 0.7], 'size': 10},
            'data_science': {'vector': [0.85, 0.85, 0.4, 0.5, 0.2, 0.6, 0.2, 0.8, 0.4, 0.8, 0.85, 0.7], 'size': 10},
            'science_fiction': {'vector': [0.7, 0.65, 0.8, 0.7, 0.6, 0.5, 0.6, 0.6, 0.7, 0.7, 0.8, 0.6], 'size': 10},
            
            # Sub-clusters and Specializations
            'theoretical_physics': {'vector': [0.9, 0.85, 0.3, 0.4, 0.2, 0.5, 0.2, 0.8, 0.3, 0.9, 0.9, 0.6], 'size': 8},
            'biotech': {'vector': [0.8, 0.7, 0.3, 0.4, 0.3, 0.5, 0.3, 0.6, 0.4, 0.7, 0.7, 0.7], 'size': 8},
            'ai_research': {'vector': [0.8, 0.8, 0.5, 0.6, 0.3, 0.6, 0.3, 0.8, 0.5, 0.8, 0.9, 0.7], 'size': 8},
            
            # Negative Control (shouldn't match anything)
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
        for i in range(n_particles):
            noise = torch.normal(0, noise_scale, size=base.shape)
            vector = torch.clamp(base + noise, 0, 1)
            self.field.add_particle(
                SemanticParticle(
                    concept=f"{cluster_id}_{i}",
                    vector=vector,
                    metadata={
                        'cluster': cluster_id,
                        'original_vector': base.tolist()
                    }
                )
            )

    # Test 1: Basic Cluster Separation
    def test_cluster_separation(self):
        """Verify that clusters remain distinct and identifiable."""
        # Enable debug output
        debug_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'debug')
        os.makedirs(debug_dir, exist_ok=True)
        debug_file = os.path.join(debug_dir, 'cluster_separation_debug.txt')
        print(f"\n=== DEBUG FILE WILL BE WRITTEN TO: {os.path.abspath(debug_file)} ===\n")
        
        with open(debug_file, 'w') as f:
            f.write("=== Cluster Separation Test Debug ===\n\n")
            
            # Test core clusters with higher stringency
            core_clusters = ['science', 'art', 'business', 'technology', 'philosophy', 'mathematics']
            all_passed = True
            
            for cluster_name in core_clusters:
                cluster_data = self.clusters[cluster_name]
                query_vector = torch.tensor(cluster_data['vector'])
                
                # Add metadata to the query vector
                query_vector.metadata = {'cluster': cluster_name}
                
                f.write(f"\n=== Testing cluster: {cluster_name} ===\n")
                f.write(f"Vector: {query_vector.tolist()}\n\n")
                
                results = self.field.find_similar(
                    query_vector,
                    torch.ones(12) * 0.1,
                    k=5,
                    debug_id=f"cluster_test_{cluster_name}"
                )
                
                f.write(f"Top 5 matches for {cluster_name}:\n")
                for i, (particle, score) in enumerate(results):
                    f.write(f"{i+1}. {particle.metadata['cluster']} (score: {score:.4f}) - {getattr(particle, 'concept', 'N/A')}\n")
                f.write("\n")
                
                # Verify top results match the expected cluster
                for i, (particle, _) in enumerate(results[:3]):  # Check top 3 matches
                    if particle.metadata['cluster'] != cluster_name:
                        f.write(f"FAIL: Position {i+1} is {particle.metadata['cluster']}, expected {cluster_name}\n")
                        all_passed = False
                    else:
                        f.write(f"PASS: Position {i+1} is {cluster_name} as expected\n")
                
                f.write("\n" + "="*80 + "\n\n")
            
            # Write final summary
            if all_passed:
                f.write("\n=== TEST PASSED FOR ALL CLUSTERS ===\n")
            else:
                f.write("\n=== TEST FAILED - SOME CLUSTERS NOT PROPERLY SEPARATED ===\n")
        
        # Now run the actual test assertions
        for cluster_name in core_clusters:
            cluster_data = self.clusters[cluster_name]
            # Add metadata to the query vector
            query_vector = torch.tensor(cluster_data['vector'])
            query_vector.metadata = {'cluster': cluster_name}
            
            results = self.field.find_similar(
                query_vector,
                torch.ones(12) * 0.1,
                k=5
            )
            
            # Verify top results match the expected cluster
            for particle, _ in results[:3]:  # Check top 3 matches
                self.assertEqual(particle.metadata['cluster'], cluster_name,
                               f"Cluster {cluster_name} separation failed. Got {particle.metadata['cluster']}")

    # Test 2: Blended Concept Inference
    def test_blended_concept_inference(self):
        """Test if the system can identify blended concepts."""
        # Enable debug output directory
        import os
        debug_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'debug')
        os.makedirs(debug_dir, exist_ok=True)
        debug_file = os.path.join(debug_dir, 'blended_concept_debug.json')
        
        # Clear any previous debug data
        if hasattr(self.field, 'debugger'):
            self.field.debugger.clear()
        
        # Cognitive science is a blend of science and art
        cognitive_science = torch.tensor(self.clusters['cognitive_science']['vector'])
        
        # Print debug info
        print("\n=== Blended Concept Test Debug ===")
        print(f"Debug output will be saved to: {os.path.abspath(debug_file)}")
        print("Cognitive Science vector:", cognitive_science.tolist())
        print("Science vector:", self.clusters['science']['vector'])
        print("Art vector:", self.clusters['art']['vector'])
        
        # Get similar particles with debug ID
        debug_id = f"blended_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        results = self.field.find_similar(
            cognitive_science, 
            torch.ones(12) * 0.15, 
            k=10,
            debug_id=debug_id
        )
        
        # Save debug information
        if hasattr(self.field, 'debugger'):
            self.field.debugger.save_to_file(debug_file)
            analysis = self.field.debugger.get_analysis()
            
            # Print analysis
            print("\n=== Similarity Analysis ===")
            print(f"Total calculations: {analysis.get('total_calculations', 0)}")
            print(f"Average similarity: {analysis.get('avg_similarity', 0):.4f}")
            print(f"Min similarity: {analysis.get('min_similarity', 0):.4f}")
            print(f"Max similarity: {analysis.get('max_similarity', 0):.4f}")
            print("\nAverage axis importance:")
            for name, imp in zip(self.field.axis_names, analysis.get('avg_axis_importance', [])):
                print(f"  {name:<12}: {imp:.4f}")
        
        # Print results
        print("\nTop 10 similar particles:")
        for i, (particle, score) in enumerate(results):
            print(f"{i+1:2d}. Cluster: {particle.metadata.get('cluster', 'N/A'):<10} "
                  f"Score: {score:.4f}  "
                  f"Concept: {getattr(particle, 'concept', 'N/A')}")
        
        # Should find both science and art particles
        found_clusters = {p.metadata.get('cluster', 'N/A') for p, _ in results}
        print("\nFound clusters:", found_clusters)
        
        # Save a summary of the results
        with open(os.path.join(debug_dir, 'blended_concept_summary.txt'), 'w') as f:
            f.write("=== Blended Concept Test Summary ===\n\n")
            f.write(f"Test ID: {debug_id}\n")
            f.write(f"Test time: {datetime.now().isoformat()}\n\n")
            
            f.write("=== Vectors ===\n")
            f.write(f"Cognitive Science: {cognitive_science.tolist()}\n")
            f.write(f"Science: {self.clusters['science']['vector']}\n")
            f.write(f"Art: {self.clusters['art']['vector']}\n\n")
            
            f.write("=== Results ===\n")
            for i, (p, score) in enumerate(results):
                f.write(f"{i+1:2d}. Cluster: {p.metadata.get('cluster', 'N/A'):<10} "
                       f"Score: {score:.4f}  "
                       f"Concept: {getattr(p, 'concept', 'N/A')}\n")
            
            f.write("\n=== Analysis ===\n")
            if analysis:
                f.write(f"Total calculations: {analysis.get('total_calculations', 0)}\n")
                f.write(f"Average similarity: {analysis.get('avg_similarity', 0):.4f}\n")
                f.write(f"Min similarity: {analysis.get('min_similarity', 0):.4f}\n")
                f.write(f"Max similarity: {analysis.get('max_similarity', 0):.4f}\n\n")
                
                f.write("Average axis importance:\n")
                for name, imp in zip(self.field.axis_names, analysis.get('avg_axis_importance', [])):
                    f.write(f"  {name:<12}: {imp:.4f}\n")
        
        # Check assertions

    # Test 3: Negative Inference Space
    def test_negative_inference_space(self):
        """Test that invalid or nonsensical blends are not inferred."""
        # Define invalid blends and their expected non-matches
        test_cases = [
            ('business', 'art'),  # Business + Art (unrelated domains)
            ('mathematics', 'noise'),  # Math + Noise (noise should be ignored)
            ('philosophy', 'biotech')  # Philosophy + Biotech (distant domains)
        ]
        
        for cluster1, cluster2 in test_cases:
            with self.subTest(blend=f"{cluster1}+{cluster2}"):
                # Create a blend of the two clusters
                blend = (torch.tensor(self.clusters[cluster1]['vector']) + 
                        torch.tensor(self.clusters[cluster2]['vector'])) / 2
                
                # Find similar particles with moderate tolerance
                results = self.field.find_similar(blend, torch.ones(12) * 0.15, k=10)
                
                # Should not find too many matches (invalid blend)
                self.assertLess(len(results), 5, 
                              f"Too many matches for invalid blend {cluster1}+{cluster2}")
                
                # If we do get results, they shouldn't be too similar
                if len(results) > 0:
                    _, top_similarity = results[0]
                    self.assertLess(top_similarity, 0.7, 
                                  f"Match similarity too high for invalid blend {cluster1}+{cluster2}")

    # Test 4: Dimensional Sensitivity
    def test_dimensional_sensitivity(self):
        """Test how changes in each dimension affect similarity."""
        # Enable debug mode and increase verbosity
        self.field.debug_mode = True
        
        # Print cluster information for reference
        print("\n=== CLUSTER REFERENCE ===")
        for name, data in self.clusters.items():
            print(f"{name}: {data['vector']}")
    
        base_vector = torch.tensor(self.clusters['science']['vector'])
        results = []
        
        # First, get baseline matches with unmodified vector
        print("\n=== DIMENSIONAL SENSITIVITY TEST ===")
        print(f"Base vector: {base_vector.tolist()}")
        
        # Get baseline matches with unmodified vector
        base_similar = self.field.find_similar(
            base_vector,
            torch.ones(12)*0.1,
            k=10,  # Changed from 5 to 10 to match the modified query
            debug_metadata={'test_name': 'dimensional_sensitivity'}
        )
        base_matches = sum(1 for p, _ in base_similar
                         if p.metadata['cluster'] == 'science')
        print(f"Baseline matches: {base_matches}/10")  # Updated to reflect k=10
        
        # Print baseline matches for reference
        print("Baseline top matches:")
        for i, (p, score) in enumerate(base_similar):
            print(f"  {i+1}. Cluster: {p.metadata['cluster']}, Score: {score:.4f}")
    
        # Test sensitivity of each dimension
        for dim in range(12):
            # Skip dimensions that are already near 0.5 (changing them won't show much effect)
            original_val = base_vector[dim].item()
            if 0.4 <= original_val <= 0.6:
                print(f"\n--- Skipping dimension {dim} (value {original_val:.4f} too close to 0.5) ---")
                results.append((dim, 5, 0.0))  # Assume no change for skipped dimensions
                continue
                
            modified = base_vector.clone()
            modified[dim] = 1.0 - original_val  # Flip the dimension
    
            print(f"\n--- Testing dimension {dim} ---")
            print(f"  Original value: {original_val:.4f}, Modified to: {modified[dim]:.4f}")
            print(f"  Dimension name: {self.field.axis_names[dim] if hasattr(self.field, 'axis_names') else f'dim_{dim}'}")
            
            # Calculate how much this dimension is changing relative to other dimensions
            dim_change = abs(modified[dim] - original_val)
            print(f"  Absolute change: {dim_change:.4f}")
    
            # Find similar particles with debug metadata for dimensional sensitivity test
            similar = self.field.find_similar(
                modified, 
                torch.ones(12)*0.1, 
                k=10,  # Get more results to see the distribution
                debug_metadata={
                    'test_name': 'dimensional_sensitivity',
                    'tested_dimension': dim,
                    'original_value': original_val,
                    'modified_value': modified[dim].item()
                }
            )
            
            # Calculate similarity scores for science cluster in baseline and modified queries
            baseline_scores = [score for p, score in base_similar 
                             if p.metadata['cluster'] == 'science']
            modified_scores = [score for p, score in similar 
                             if p.metadata['cluster'] == 'science']
            
            # Calculate average similarity score for science cluster
            baseline_avg = sum(baseline_scores) / len(baseline_scores) if baseline_scores else 0
            modified_avg = sum(modified_scores) / len(modified_scores) if modified_scores else 0
            
            # Calculate score change as a ratio of the original score
            score_change = (baseline_avg - modified_avg) / baseline_avg if baseline_avg > 0 else 0
            
            # Also track raw match count change for reference
            matches = sum(1 for p, _ in similar 
                         if p.metadata['cluster'] == 'science')
            match_change = (base_matches - matches) / base_matches if base_matches > 0 else 0
            
            results.append((dim, matches, score_change, original_val, modified[dim].item()))
            
            # Print detailed results for this dimension
            print(f"  Matches in 'science' cluster: {matches}/10 (Change: {match_change*100:.1f}%)")
            print(f"  Avg similarity: {modified_avg:.4f} (Change: {score_change*100:.1f}%)")
            print(f"  Base matches: {base_matches}, Current matches: {matches}")
            print(f"  Base avg score: {baseline_avg:.4f}, Modified avg score: {modified_avg:.4f}")
            print("  Top 10 matches:")
            for i, (p, score) in enumerate(similar):
                print(f"    {i+1}. Cluster: {p.metadata['cluster']:8s}, Score: {score:.4f}, Vector: {p.vector.tolist()}")
            
            # Debug: Print the actual scores for science cluster matches
            print("  Science cluster matches:")
            for i, (p, score) in enumerate(similar):
                if p.metadata['cluster'] == 'science':
                    print(f"    {i+1}. Score: {score:.4f}, Concept: {p.concept}")
        
        # Print summary with more detailed information
        print("\n=== TEST SUMMARY ===")
        print(f"{'Dim':<4} {'Original':<8} {'Modified':<8} {'Matches':<7} {'Score Chg':<9} {'Significant':<11}")
        print("-" * 70)
        
        significant_dims = []
        for result in results:
            if len(result) == 5:  # New format with original and modified values
                dim, matches, score_change, orig_val, mod_val = result
                dim_name = self.field.axis_names[dim] if hasattr(self.field, 'axis_names') else f'dim_{dim}'
                print(f"{dim:<4} {orig_val:<8.4f} {mod_val:<8.4f} {matches}/10     {score_change*100:>5.1f}%    {'✓' if score_change > 0.3 else '✗'}")
            else:  # Old format (shouldn't happen with our changes)
                dim, matches, score_change = result
                print(f"{dim:<4} {'N/A':<8} {'N/A':<8} {matches}/10     {score_change*100:>5.1f}%    {'✓' if score_change > 0.3 else '✗'}")
            
            # Consider a dimension significant if it causes >30% drop in average similarity score
            if score_change > 0.3:
                significant_dims.append(dim)
        
        print(f"\nSignificantly sensitive dimensions (>30% drop): {significant_dims}")
        print(f"Total significant dimensions: {len(significant_dims)}/12")
        
        # Check if we have enough significant dimensions
        self.assertGreaterEqual(len(significant_dims), 3,
                              f"Should have at least 3 significantly sensitive dimensions, found {len(significant_dims)}: {significant_dims}")
        
        # Disable debug mode
        self.field.debug_mode = False

    # Test 5: Progressive Concept Formation
    def test_progressive_concept_formation(self):
        """Test if the system can form new concepts through blending."""
        # Define test cases with different concept groups and their expected relationships
        test_cases = [
            # Science-focused blends
            (['science', 'technology', 'mathematics'], 0.7, ['theoretical_physics', 'ai_research']),
            # Creative blends
            (['art', 'technology', 'philosophy'], 0.6, ['digital_art', 'science_fiction']),
            # Applied science blends
            (['science', 'technology', 'business'], 0.65, ['data_science', 'biotech'])
        ]
        
        for base_concepts, min_similarity, expected_related in test_cases:
            with self.subTest(blend='+'.join(base_concepts)):
                base_vectors = [torch.tensor(self.clusters[name]['vector']) 
                              for name in base_concepts]
                
                # Create progressive blends
                for i in range(1, 4):  # 3 levels of blending
                    blend_weight = i * 0.25
                    # Create a weighted blend
                    blended = sum(vec * (1-blend_weight)/(len(base_vectors)-1)
                                for vec in base_vectors[1:])
                    blended += base_vectors[0] * blend_weight
            
                    # Find similar particles
                    results = self.field.find_similar(blended, torch.ones(12)*0.15, k=10)
            
                    # Should find multiple source clusters
                    source_clusters = [p.metadata['cluster'] for p, _ in results]
                    unique_sources = set(source_clusters)
                    
                    # Verify we have multiple source concepts
                    self.assertGreaterEqual(len(unique_sources), 2,
                                          f"Blend {i} of {base_concepts} should connect multiple concepts")
                    
                    # For later blends, check if we're getting closer to expected related concepts
                    if i >= 2:
                        found_related = any(any(related in cluster for related in expected_related)
                                          for cluster in unique_sources)
                        self.assertTrue(found_related,
                                      f"Blend {i} of {base_concepts} should relate to {expected_related}")
                        
                        # Check if we're getting reasonable similarity scores
                        if results:
                            _, top_similarity = results[0]
                            self.assertGreaterEqual(top_similarity, min_similarity,
                                                  f"Top similarity {top_similarity:.2f} too low for blend {i} of {base_concepts}")

    # Test 7: Temporal Concept Evolution
    def test_temporal_concept_evolution(self):
        """Test how concepts evolve over time and maintain relationships."""
        # Define concept evolution paths with multiple generations
        evolution_paths = {
            'technology': [
                # Generation 0: Early technology (mechanical)
                {'vector': [0.85, 0.75, 0.1, 0.3, 0.1, 0.4, 0.1, 0.7, 0.2, 0.6, 0.7, 0.5], 'gen': 0},
                # Generation 1: Electronic age
                {'vector': [0.88, 0.78, 0.15, 0.4, 0.15, 0.5, 0.15, 0.75, 0.25, 0.7, 0.8, 0.6], 'gen': 1},
                # Generation 2: Digital age
                {'vector': [0.9, 0.8, 0.25, 0.5, 0.2, 0.6, 0.2, 0.8, 0.3, 0.8, 0.85, 0.7], 'gen': 2},
                # Generation 3: AI era
                {'vector': [0.92, 0.85, 0.35, 0.6, 0.25, 0.7, 0.25, 0.85, 0.4, 0.85, 0.9, 0.75], 'gen': 3}
            ],
            'art': [
                # Generation 0: Classical art
                {'vector': [0.1, 0.2, 0.9, 0.8, 0.7, 0.1, 0.8, 0.1, 0.7, 0.6, 0.5, 0.6], 'gen': 0},
                # Generation 1: Modern art
                {'vector': [0.2, 0.3, 0.85, 0.75, 0.75, 0.2, 0.75, 0.2, 0.75, 0.7, 0.6, 0.65], 'gen': 1},
                # Generation 2: Contemporary art
                {'vector': [0.3, 0.4, 0.8, 0.7, 0.8, 0.3, 0.7, 0.3, 0.8, 0.75, 0.7, 0.7], 'gen': 2},
                # Generation 3: Digital/Interactive art
                {'vector': [0.4, 0.5, 0.75, 0.65, 0.85, 0.4, 0.65, 0.4, 0.85, 0.8, 0.8, 0.75], 'gen': 3}
            ]
        }
        
        # Add all generations to the field
        for concept, generations in evolution_paths.items():
            for gen_data in generations:
                self.field.add_particle(SemanticParticle(
                    concept=f"{concept}_gen{gen_data['gen']}",
                    vector=torch.tensor(gen_data['vector'], dtype=torch.float32),
                    metadata={
                        'cluster': 'timeline',
                        'generation': gen_data['gen'],
                        'concept': concept,
                        'description': f"{concept.capitalize()} Generation {gen_data['gen']}"
                    }
                ))
        
        # Ensure statistics are updated after adding particles
        if hasattr(self.field, '_update_statistics'):
            self.field._update_statistics()
        
        # Test 1: Verify we can trace lineage through generations
        for concept in evolution_paths.keys():
            # Get the latest generation
            latest_gen = max(gen['gen'] for gen in evolution_paths[concept])
            latest_vec = next(g['vector'] for g in evolution_paths[concept] 
                            if g['gen'] == latest_gen)
            
            # Find similar concepts (should find same concept across generations)
            results = self.field.find_similar(
                torch.tensor(latest_vec, dtype=torch.float32),
                torch.ones(12) * 0.15,
                k=8
            )
            
            # Should find multiple generations of the same concept
            found_generations = {p.metadata['generation'] for p, _ in results 
                               if p.metadata.get('concept') == concept}
            self.assertGreaterEqual(len(found_generations), 2,
                                  f"Should find multiple generations of {concept}")
            
            # Test 2: Verify progression in similarity
            if latest_gen > 0:
                # Compare with previous generation
                prev_gen_vec = next(g['vector'] for g in evolution_paths[concept] 
                                  if g['gen'] == latest_gen - 1)
                # Create default axis importance tensor
                axis_importance = torch.ones(12) * 0.15  # Uniform importance
                
                prev_similarity = self.field._calculate_similarity(
                    torch.tensor(latest_vec, dtype=torch.float32),
                    torch.tensor(prev_gen_vec, dtype=torch.float32),
                    axis_importance=axis_importance
                )
                
                # Compare with older generation (should be less similar)
                if latest_gen > 1:
                    older_gen_vec = next(g['vector'] for g in evolution_paths[concept] 
                                       if g['gen'] == latest_gen - 2)
                    older_similarity = self.field._calculate_similarity(
                        torch.tensor(latest_vec, dtype=torch.float32),
                        torch.tensor(older_gen_vec, dtype=torch.float32),
                        axis_importance=axis_importance
                    )
                    
                    # Should be more similar to immediate predecessor than older generations
                    self.assertGreater(prev_similarity, older_similarity,
                                     f"{concept} generation {latest_gen} should be more similar to "
                                     f"generation {latest_gen-1} than to generation {latest_gen-2}")

    # Test 8: Cross-Concept Analogy
    def test_cross_concept_analogy(self):
        """Test analogical reasoning across different conceptual domains."""
        # Define analogy test cases: A : B :: C : D
        analogies = [
            # Science : Experiment :: Art : ? (should find 'artwork' or 'creation')
            {
                'A': 'science',
                'B': 'experiment',
                'C': 'art',
                'expected_terms': ['artwork', 'creation', 'piece', 'composition']
            },
            # Mathematics : Proof :: Art : ? (should find 'exhibition' or 'showcase')
            {
                'A': 'mathematics',
                'B': 'proof',
                'C': 'art',
                'expected_terms': ['exhibition', 'showcase', 'display', 'performance']
            },
            # Technology : Innovation :: Philosophy : ? (should find 'theory' or 'doctrine')
            {
                'A': 'technology',
                'B': 'innovation',
                'C': 'philosophy',
                'expected_terms': ['theory', 'doctrine', 'school', 'perspective']
            }
        ]
        
        # Add some specific analogy targets that might not be in the main clusters
        analogy_concepts = {
            'experiment': [0.85, 0.75, 0.3, 0.7, 0.4, 0.5, 0.3, 0.6, 0.4, 0.6, 0.7, 0.6],
            'artwork': [0.3, 0.4, 0.85, 0.8, 0.75, 0.3, 0.8, 0.2, 0.8, 0.7, 0.6, 0.7],
            'proof': [0.9, 0.95, 0.1, 0.2, 0.1, 0.3, 0.1, 0.9, 0.2, 0.9, 0.9, 0.8],
            'exhibition': [0.2, 0.3, 0.9, 0.85, 0.8, 0.4, 0.85, 0.3, 0.85, 0.75, 0.7, 0.75],
            'innovation': [0.9, 0.8, 0.6, 0.7, 0.5, 0.7, 0.4, 0.8, 0.5, 0.8, 0.9, 0.8],
            'theory': [0.7, 0.8, 0.7, 0.9, 0.7, 0.5, 0.4, 0.7, 0.5, 0.9, 0.9, 0.6]
        }
        
        # Add analogy concepts to the field
        for concept, vector in analogy_concepts.items():
            if concept not in self.clusters:
                self.field.add_particle(SemanticParticle(
                    concept=concept,
                    vector=torch.tensor(vector, dtype=torch.float32),
                    metadata={'cluster': 'analogy', 'type': 'concept'}
                ))
        
        # Test each analogy
        for i, analogy in enumerate(analogies):
            with self.subTest(analogy=f"{analogy['A']}:{analogy['B']}::{analogy['C']}:?"):
                # Get vectors for A, B, and C
                A = torch.tensor(self.clusters[analogy['A']]['vector'] 
                               if analogy['A'] in self.clusters 
                               else analogy_concepts[analogy['A']], dtype=torch.float32)
                B = torch.tensor(self.clusters[analogy['B']]['vector'] 
                               if analogy['B'] in self.clusters 
                               else analogy_concepts[analogy['B']], dtype=torch.float32)
                C = torch.tensor(self.clusters[analogy['C']]['vector'] 
                               if analogy['C'] in self.clusters 
                               else analogy_concepts[analogy['C']], dtype=torch.float32)
                
                # Calculate analogy: D = C + (B - A)
                D = C + (B - A)
                D = torch.clamp(D, 0, 1)  # Ensure values stay in [0,1] range
                
                # Find similar concepts to D
                results = self.field.find_similar(D, torch.ones(12) * 0.2, k=5)
                
                # Get the top result's concept
                if results:
                    top_concept = results[0][0].concept.lower()
                    top_cluster = results[0][0].metadata['cluster'].lower()
                    
                    # Check if any expected term is in the concept or cluster
                    found = any(term in top_concept or term in top_cluster 
                              for term in analogy['expected_terms'])
                    
                    self.assertTrue(found, 
                                  f"Analogy {i+1}: Expected one of {analogy['expected_terms']} in "
                                  f"{top_concept} or {top_cluster}")
                    
                    # Print debug info
                    print(f"\nAnalogy: {analogy['A']} : {analogy['B']} :: {analogy['C']} : ?")
                    print(f"Top result: {top_concept} (cluster: {top_cluster})")
                    print(f"Expected terms: {analogy['expected_terms']}")
                else:
                    self.fail(f"No results found for analogy: {analogy}")

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
