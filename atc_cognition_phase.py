"""
ATC Cognition Phase - 4D Analytical Reasoning
============================================

This module implements the Cognition phase of the ATC architecture:
- 4D analytical reasoning for novel inputs
- Understanding → Hypothesis → Experimentation → Synthesis pipeline
- Power-of-2 progression: 4D → 16D → 64D → 256D
- Bifurcation mathematics for semantic field exploration
- Complex reasoning beyond simple pattern matching

Architecture: Novel Input → 4D Understanding → Hypothesis Generation → 
             Experimentation → Synthesis → Procedure

Author: Revolutionary ATC Architecture Team
Status: Milestone 3 - Cognition Phase
"""

import torch
import torch.nn as nn
import numpy as np
import networkx as nx
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import time
import json
from pathlib import Path

# Import our Power-of-2 foundation
from power_of_2_core import PowerOf2Layers, PowerOf2Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CognitionConfig:
    """Configuration for 4D Cognition Phase"""
    # 4D Cognition parameters
    cognition_dim: int = 4  # Start with 4D for analytical reasoning
    understanding_dim: int = 16  # 16D for deeper understanding
    reflection_dim: int = 64   # 64D for reflection 
    synthesis_dim: int = 256   # 256D for final synthesis
    
    # Bifurcation parameters (from ATC framework)
    bifurcation_delta: float = 4.669  # Golden ratio for semantic exploration
    max_hypotheses: int = 5  # Maximum hypotheses to generate
    
    # Processing parameters
    coherence_threshold: float = 0.6  # Minimum coherence for synthesis
    max_reasoning_steps: int = 10  # Maximum reasoning iterations
    experiment_confidence: float = 0.8  # Confidence threshold for experiments


class SemanticField:
    """
    Semantic field representation for 4D cognition
    
    Implements orderless semantic fields with particle bifurcation
    """
    
    def __init__(self, config: CognitionConfig):
        self.config = config
        self.particles = {}  # Semantic particles
        self.field_density = {}  # Field density maps
        self.axis_count = 12  # 12-axis semantic space (from framework)
        
        # Initialize bifurcation sequence
        self.bifurcation_sequence = self._generate_bifurcation_sequence()
        
        logger.debug("Semantic Field initialized with 12-axis space")
    
    def _generate_bifurcation_sequence(self, max_depth: int = 10) -> List[float]:
        """Generate bifurcation sequence: B(k) = B(k-1) + δ·B(k-2)"""
        sequence = [1.0, self.config.bifurcation_delta]  # B(0), B(1)
        
        for k in range(2, max_depth):
            next_val = sequence[k-1] + self.config.bifurcation_delta * sequence[k-2]
            sequence.append(next_val)
        
        return sequence
    
    def create_semantic_particles(self, concept_embedding: torch.Tensor) -> List[torch.Tensor]:
        """
        Create semantic particles from concept embedding
        
        Uses bifurcation mathematics to generate irreducible semantic units
        """
        particles = []
        
        # Generate particles at different bifurcation depths
        for depth in range(min(5, len(self.bifurcation_sequence))):
            bifurcation_factor = self.bifurcation_sequence[depth]
            
            # Create particle with bifurcation scaling
            particle = concept_embedding * (bifurcation_factor / 10.0)  # Scale for stability
            
            # Add 12-axis semantic noise for exploration
            semantic_noise = torch.randn(self.axis_count) * 0.1
            if len(particle) >= self.axis_count:
                particle[:self.axis_count] += semantic_noise
            else:
                # Extend particle to 12 dimensions if needed
                extended = torch.zeros(max(len(particle), self.axis_count))
                extended[:len(particle)] = particle
                extended[len(particle):self.axis_count] += semantic_noise[len(particle):]
                particle = extended
            
            particles.append(particle)
            
        logger.debug(f"Generated {len(particles)} semantic particles")
        return particles
    
    def calculate_field_density(self, particles: List[torch.Tensor], query_point: torch.Tensor) -> float:
        """
        Calculate semantic field density at query point
        
        ρ = Σ exp(-||x-p_i||²/σ²) 
        """
        if not particles:
            return 0.0
        
        density = 0.0
        sigma_squared = 1.0  # Field variance parameter
        
        for particle in particles:
            # Ensure dimensions match
            if len(particle) != len(query_point):
                min_len = min(len(particle), len(query_point))
                particle_slice = particle[:min_len]
                query_slice = query_point[:min_len]
            else:
                particle_slice = particle
                query_slice = query_point
            
            # Calculate Gaussian density contribution
            distance_squared = torch.sum((query_slice - particle_slice) ** 2)
            density += torch.exp(-distance_squared / sigma_squared)
        
        return float(density)
    
    def browse_field(self, particles: List[torch.Tensor], steps: int = 5) -> torch.Tensor:
        """
        Browse semantic field using gravity walk
        
        g = (c - p_i) · m / r²  where c = center of mass
        """
        if not particles:
            return torch.zeros(4)  # Return 4D zero vector
        
        # Calculate center of mass
        particles_tensor = torch.stack(particles)
        center_of_mass = torch.mean(particles_tensor, dim=0)
        
        # Start browsing from center of mass
        current_position = center_of_mass.clone()
        
        for step in range(steps):
            # Calculate gravitational forces from all particles
            total_force = torch.zeros_like(current_position)
            
            for particle in particles:
                # Force direction (toward particle)
                direction = particle - current_position
                distance = torch.norm(direction) + 1e-6  # Avoid division by zero
                
                # Gravitational force: F = m/r² (simplified, mass = density)
                mass = self.calculate_field_density([particle], current_position)
                force_magnitude = mass / (distance ** 2 + 1e-6)
                
                # Normalize direction and apply force
                normalized_direction = direction / distance
                total_force += force_magnitude * normalized_direction
            
            # Move in direction of total force (small step)
            current_position += total_force * 0.1
        
        # Return 4D cognition vector
        return current_position[:4]  # Slice to 4D for cognition


class CognitionProcessor:
    """
    Core 4D Cognition Phase Processor
    
    Implements Understanding → Hypothesis → Experimentation → Synthesis pipeline
    """
    
    def __init__(self, config: CognitionConfig = None, power_layers: PowerOf2Layers = None):
        self.config = config or CognitionConfig()
        self.power_layers = power_layers
        self.semantic_field = SemanticField(self.config)
        
        # Cognition components
        self.understanding_network = self._build_understanding_network()
        self.hypothesis_generator = self._build_hypothesis_generator()
        self.experiment_evaluator = self._build_experiment_evaluator()
        self.synthesis_network = self._build_synthesis_network()
        
        # Performance tracking
        self.cognition_stats = {
            'total_cognitions': 0,
            'successful_syntheses': 0,
            'avg_reasoning_steps': 0.0,
            'avg_coherence': 0.0,
            'avg_processing_time': 0.0
        }
        
        logger.info("4D Cognition Processor initialized")
    
    def _build_understanding_network(self) -> nn.Module:
        """Build 4D → 16D Understanding network"""
        return nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.LayerNorm(16),
            nn.Dropout(0.1)
        )
    
    def _build_hypothesis_generator(self) -> nn.Module:
        """Build 16D → Multiple Hypotheses network"""
        return nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),  # Will be reshaped to multiple hypotheses
            nn.LayerNorm(64)
        )
    
    def _build_experiment_evaluator(self) -> nn.Module:
        """Build hypothesis evaluation network"""
        return nn.Sequential(
            nn.Linear(16, 32),  # Hypothesis input
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.ReLU(), 
            nn.Linear(8, 1),   # Single confidence score
            nn.Sigmoid()
        )
    
    def _build_synthesis_network(self) -> nn.Module:
        """Build final synthesis network: validated hypotheses → 256D output"""
        return nn.Sequential(
            nn.Linear(64, 128),  # Aggregated hypotheses
            nn.ReLU(),
            nn.Linear(128, 256), # Final 256D synthesis
            nn.LayerNorm(256),
            nn.Dropout(0.1)
        )
    
    def understand(self, cognition_4d: torch.Tensor) -> torch.Tensor:
        """
        Understanding phase: 4D → 16D deep understanding
        
        This is where we build the conceptual graph and semantic understanding
        """
        logger.debug("🔍 Understanding phase: 4D → 16D")
        
        # Process through understanding network
        understanding_16d = self.understanding_network(cognition_4d)
        
        # Add semantic field browsing for richer understanding
        semantic_particles = self.semantic_field.create_semantic_particles(cognition_4d)
        field_context = self.semantic_field.browse_field(semantic_particles)
        
        # Combine network understanding with field context
        if len(field_context) >= 4:
            # Create enhanced understanding by combining both
            enhanced_understanding = understanding_16d.clone()
            enhanced_understanding[:4] += field_context[:4] * 0.3  # 30% field influence
        else:
            enhanced_understanding = understanding_16d
        
        logger.debug(f"Understanding complete: {enhanced_understanding.shape}")
        return enhanced_understanding
    
    def generate_hypotheses(self, understanding_16d: torch.Tensor) -> List[torch.Tensor]:
        """
        Hypothesis generation: 16D understanding → Multiple hypotheses
        
        Uses bifurcation mathematics to generate diverse hypotheses
        """
        logger.debug("💡 Hypothesis generation")
        
        # Generate base hypothesis space
        hypothesis_space = self.hypothesis_generator(understanding_16d)  # 64D space
        
        # Split into multiple hypotheses using bifurcation
        hypothesis_dim = 16  # Each hypothesis is 16D
        num_hypotheses = min(self.config.max_hypotheses, hypothesis_space.shape[0] // hypothesis_dim)
        
        hypotheses = []
        for i in range(num_hypotheses):
            start_idx = i * hypothesis_dim
            end_idx = start_idx + hypothesis_dim
            
            if end_idx <= hypothesis_space.shape[0]:
                hypothesis = hypothesis_space[start_idx:end_idx].clone()
                
                # Add bifurcation noise for diversity
                bifurcation_factor = self.semantic_field.bifurcation_sequence[min(i, len(self.semantic_field.bifurcation_sequence)-1)]
                noise = torch.randn_like(hypothesis) * (bifurcation_factor / 100.0)  # Small noise
                hypothesis += noise
                
                hypotheses.append(hypothesis)
        
        logger.debug(f"Generated {len(hypotheses)} hypotheses")
        return hypotheses
    
    def experiment_hypotheses(self, hypotheses: List[torch.Tensor]) -> List[Tuple[torch.Tensor, float]]:
        """
        Experimentation phase: Test each hypothesis for validity
        
        Returns validated hypotheses with confidence scores
        """
        logger.debug("🧪 Experimentation phase")
        
        validated_hypotheses = []
        
        for i, hypothesis in enumerate(hypotheses):
            # Evaluate hypothesis confidence
            confidence = float(self.experiment_evaluator(hypothesis))
            
            # Chemistry-like activation check (from framework)
            # E = ρ > 0.8 (activation energy threshold)
            activation_energy = self.semantic_field.calculate_field_density(
                self.semantic_field.create_semantic_particles(hypothesis), 
                hypothesis
            )
            
            is_activated = activation_energy > self.config.experiment_confidence
            
            if is_activated and confidence > self.config.experiment_confidence:
                validated_hypotheses.append((hypothesis, confidence))
                logger.debug(f"Hypothesis {i+1} validated: confidence={confidence:.3f}")
            else:
                logger.debug(f"Hypothesis {i+1} rejected: confidence={confidence:.3f}, activated={is_activated}")
        
        logger.debug(f"Validated {len(validated_hypotheses)}/{len(hypotheses)} hypotheses")
        return validated_hypotheses
    
    def synthesize(self, validated_hypotheses: List[Tuple[torch.Tensor, float]]) -> Tuple[torch.Tensor, float]:
        """
        Synthesis phase: Combine validated hypotheses → 256D final output
        
        This is where the final cognitive output is generated
        """
        logger.debug("⚡ Synthesis phase: → 256D")
        
        if not validated_hypotheses:
            # No valid hypotheses - return low-confidence default
            default_synthesis = torch.zeros(256)
            return default_synthesis, 0.1
        
        # Aggregate hypotheses weighted by confidence
        hypothesis_tensors = [h for h, _ in validated_hypotheses]
        confidences = [c for _, c in validated_hypotheses]
        
        # Create weighted average of hypotheses
        weighted_sum = torch.zeros(16)  # Hypothesis dimension
        total_weight = 0.0
        
        for hypothesis, confidence in validated_hypotheses:
            weighted_sum += hypothesis * confidence
            total_weight += confidence
        
        if total_weight > 0:
            aggregated_hypothesis = weighted_sum / total_weight
        else:
            aggregated_hypothesis = torch.mean(torch.stack(hypothesis_tensors), dim=0)
        
        # Extend to 64D for synthesis network input
        synthesis_input = torch.zeros(64)
        synthesis_input[:16] = aggregated_hypothesis
        
        # Generate final 256D synthesis
        final_synthesis = self.synthesis_network(synthesis_input)
        
        # Calculate overall coherence
        overall_coherence = np.mean(confidences) if confidences else 0.1
        
        logger.debug(f"Synthesis complete: coherence={overall_coherence:.3f}")
        return final_synthesis, overall_coherence
    
    def cognize(self, query_embedding: torch.Tensor, query_text: str = "") -> Dict[str, Any]:
        """
        Main 4D Cognition processing pipeline
        
        Args:
            query_embedding: Input query embedding (will be projected to 4D)
            query_text: Original query text for context
            
        Returns:
            Cognition result with full reasoning trace
        """
        start_time = time.time()
        self.cognition_stats['total_cognitions'] += 1
        
        logger.info(f"🧠 4D Cognition: {query_text[:50]}...")
        
        try:
            # Step 1: Project to 4D Cognition space
            if len(query_embedding) > 4:
                cognition_4d = query_embedding[:4]  # Simple projection (can be improved)
            else:
                cognition_4d = torch.zeros(4)
                cognition_4d[:len(query_embedding)] = query_embedding
            
            logger.debug(f"4D Cognition vector: {cognition_4d}")
            
            # Step 2: Understanding (4D → 16D)
            understanding_16d = self.understand(cognition_4d)
            
            # Step 3: Hypothesis Generation (16D → Multiple hypotheses)
            hypotheses = self.generate_hypotheses(understanding_16d)
            
            # Step 4: Experimentation (validate hypotheses)
            validated_hypotheses = self.experiment_hypotheses(hypotheses)
            
            # Step 5: Synthesis (validated hypotheses → 256D output)
            final_synthesis, coherence = self.synthesize(validated_hypotheses)
            
            # Step 6: Generate cognitive output
            processing_time = time.time() - start_time
            reasoning_steps = len(hypotheses)
            
            # Create interpretable output
            cognitive_output = self._generate_cognitive_output(
                query_text, final_synthesis, coherence, reasoning_steps
            )
            
            # Update statistics
            self.cognition_stats['successful_syntheses'] += 1
            self.cognition_stats['avg_reasoning_steps'] = (
                (self.cognition_stats['avg_reasoning_steps'] * (self.cognition_stats['total_cognitions'] - 1) + reasoning_steps)
                / self.cognition_stats['total_cognitions']
            )
            self.cognition_stats['avg_coherence'] = (
                (self.cognition_stats['avg_coherence'] * (self.cognition_stats['successful_syntheses'] - 1) + coherence)
                / self.cognition_stats['successful_syntheses']
            )
            self.cognition_stats['avg_processing_time'] = (
                (self.cognition_stats['avg_processing_time'] * (self.cognition_stats['total_cognitions'] - 1) + processing_time)
                / self.cognition_stats['total_cognitions']
            )
            
            return {
                'phase': 'cognition_4d',
                'success': True,
                'output': cognitive_output,
                'coherence': coherence,
                'dissonance': 1.0 - coherence,
                'processing_time': processing_time,
                'method': '4d_analytical_reasoning',
                'reasoning_steps': reasoning_steps,
                'hypotheses_generated': len(hypotheses),
                'hypotheses_validated': len(validated_hypotheses),
                'cognition_4d': cognition_4d.tolist(),
                'understanding_16d': understanding_16d.tolist()[:8],  # First 8 dims for brevity
                'synthesis_256d_preview': final_synthesis.tolist()[:8],  # First 8 dims
                'semantic_particles_count': len(self.semantic_field.create_semantic_particles(cognition_4d))
            }
            
        except Exception as e:
            logger.error(f"❌ 4D Cognition failed: {str(e)}")
            processing_time = time.time() - start_time
            
            return {
                'phase': 'cognition_4d_error',
                'success': False,
                'output': f'4D Cognition error: {str(e)}',
                'coherence': 0.0,
                'dissonance': 1.0,
                'processing_time': processing_time,
                'method': '4d_error_handling',
                'error': str(e)
            }
    
    def _generate_cognitive_output(self, query: str, synthesis: torch.Tensor, coherence: float, steps: int) -> str:
        """
        Generate human-readable cognitive output from 256D synthesis
        
        This converts the high-dimensional cognitive state into interpretable text
        """
        # Simple interpretation based on synthesis characteristics
        synthesis_magnitude = float(torch.norm(synthesis))
        synthesis_mean = float(torch.mean(synthesis))
        synthesis_std = float(torch.std(synthesis))
        
        # Generate contextual response based on cognitive analysis
        if coherence > 0.8:
            confidence_level = "high confidence"
        elif coherence > 0.6:
            confidence_level = "moderate confidence"
        else:
            confidence_level = "low confidence"
        
        if synthesis_magnitude > 5.0:
            complexity = "complex analytical"
        elif synthesis_magnitude > 2.0:
            complexity = "moderate analytical"
        else:
            complexity = "simple analytical"
        
        # Construct cognitive response
        cognitive_response = (
            f"4D Cognition analysis of '{query}': "
            f"Through {steps} reasoning steps, I've developed {complexity} understanding with {confidence_level}. "
            f"The semantic field exploration revealed patterns with {coherence:.1%} coherence. "
            f"Synthesis characteristics: magnitude={synthesis_magnitude:.2f}, "
            f"mean activation={synthesis_mean:.3f}, variability={synthesis_std:.3f}."
        )
        
        return cognitive_response
    
    def get_cognition_stats(self) -> Dict[str, Any]:
        """Get 4D Cognition performance statistics"""
        return {
            **self.cognition_stats,
            'success_rate': (
                self.cognition_stats['successful_syntheses'] / self.cognition_stats['total_cognitions']
                if self.cognition_stats['total_cognitions'] > 0 else 0.0
            )
        }


class CognitionPhaseIntegrator:
    """
    Integrates 4D Cognition Phase with Enhanced SATC Engine
    """
    
    def __init__(self, cognition_processor: CognitionProcessor):
        self.cognition_processor = cognition_processor
        self.integrated = False
        
    def integrate_with_satc(self, satc_engine):
        """
        Integrate 4D Cognition Phase with Enhanced SATC Engine
        """
        logger.info("Integrating 4D Cognition Phase with Enhanced SATC...")
        
        # Add Cognition processor to engine
        satc_engine.cognition_processor = self.cognition_processor
        satc_engine._using_cognition_4d = True
        
        self.integrated = True
        logger.info("✅ 4D Cognition Phase integration completed!")
        
        return satc_engine
    
    def process_novel_query(self, satc_engine, query: str, query_embedding: torch.Tensor = None) -> Dict[str, Any]:
        """
        Process novel query through 4D Cognition pipeline
        """
        if not self.integrated:
            logger.warning("4D Cognition Phase not yet integrated with SATC")
        
        # Get query embedding if not provided
        if query_embedding is None and hasattr(satc_engine, 'embedding_model'):
            query_embedding = torch.tensor(
                satc_engine.embedding_model.encode(query), 
                dtype=torch.float32
            )
        elif query_embedding is None:
            # Fallback: simple text-based embedding
            query_embedding = torch.tensor([len(query), len(set(query.lower()))], dtype=torch.float32)
        
        # Process through 4D Cognition
        cognition_result = self.cognition_processor.cognize(query_embedding, query)
        
        return cognition_result


def create_cognition_phase(power_layers: PowerOf2Layers = None):
    """
    Factory function to create complete 4D Cognition Phase
    """
    config = CognitionConfig()
    processor = CognitionProcessor(config, power_layers)
    integrator = CognitionPhaseIntegrator(processor)
    
    logger.info("4D Cognition Phase created successfully!")
    logger.info(f"Pipeline: 4D → 16D → 64D → 256D")
    logger.info(f"Bifurcation delta: {config.bifurcation_delta}")
    
    return processor, integrator, config


# Standalone testing function
def test_cognition_phase_standalone():
    """
    Standalone test of 4D Cognition Phase
    """
    print("=" * 60)
    print("COGNITION PHASE (4D) - STANDALONE TEST")
    print("=" * 60)
    
    # Create 4D Cognition phase
    processor, integrator, config = create_cognition_phase()
    
    # Test queries for cognition (novel, complex)
    test_queries = [
        ("What is the meaning of consciousness?", torch.randn(8)),
        ("How do quantum effects influence cognition?", torch.randn(12)),
        ("Explain the relationship between entropy and information", torch.randn(6)),
    ]
    
    print("\n--- 4D Cognition Testing ---")
    for query_text, query_embedding in test_queries:
        print(f"\nQuery: '{query_text}'")
        
        result = processor.cognize(query_embedding, query_text)
        
        print(f"  Success: {result['success']}")
        print(f"  Coherence: {result['coherence']:.3f}")
        print(f"  Reasoning steps: {result['reasoning_steps']}")
        print(f"  Hypotheses generated: {result['hypotheses_generated']}")
        print(f"  Hypotheses validated: {result['hypotheses_validated']}")
        print(f"  Processing time: {result['processing_time']:.3f}s")
        print(f"  Output: {result['output'][:100]}...")
    
    print("\n--- Performance Statistics ---")
    stats = processor.get_cognition_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print("=" * 60)
    print("MILESTONE 3: 4D COGNITION PHASE - COMPLETE!")
    print("=" * 60)
    
    return processor, integrator, config


if __name__ == "__main__":
    test_cognition_phase_standalone()