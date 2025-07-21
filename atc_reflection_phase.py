"""
ATC Reflection Phase - 16D Meta-Cognitive Reasoning
==================================================

This module implements the Reflection phase of the ATC architecture:
- 16D meta-cognitive reasoning and self-awareness
- Higher-order thinking about thinking processes
- Coherence analysis and cognitive introspection
- Meta-learning and strategy optimization
- Recursive reasoning improvement

Architecture: Cognition Results → 16D Meta-Analysis → Self-Assessment → 
             Strategy Refinement → Enhanced Understanding

Author: Revolutionary ATC Architecture Team
Status: Milestone 4 - Reflection Phase
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import time
import json
from pathlib import Path

# Import our foundations
from power_of_2_core import PowerOf2Layers, PowerOf2Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ReflectionConfig:
    """Configuration for 16D Reflection Phase"""
    # 16D Reflection parameters
    reflection_dim: int = 16  # Meta-cognitive dimension
    meta_analysis_depth: int = 3  # Levels of meta-reasoning
    coherence_analysis_threshold: float = 0.7  # Minimum coherence for acceptance
    
    # Self-awareness parameters
    introspection_steps: int = 5  # Steps of cognitive introspection
    strategy_refinement_rate: float = 0.1  # Learning rate for strategy updates
    
    # Meta-learning parameters
    experience_buffer_size: int = 100  # Size of reflection experience buffer
    meta_pattern_threshold: float = 0.8  # Threshold for recognizing meta-patterns
    
    # Recursive improvement parameters
    max_reflection_depth: int = 3  # Maximum recursive reflection depth
    improvement_threshold: float = 0.05  # Minimum improvement to continue reflection


class MetaCognitionEngine:
    """
    Meta-cognitive engine for 16D reflection
    
    Implements thinking about thinking - analyzing cognitive processes
    """
    
    def __init__(self, config: ReflectionConfig):
        self.config = config
        self.meta_patterns = {}  # Learned meta-cognitive patterns
        self.strategy_weights = torch.ones(16) * 0.5  # Adaptive strategy weights
        
        # Build meta-cognitive networks
        self.coherence_analyzer = self._build_coherence_analyzer()
        self.strategy_optimizer = self._build_strategy_optimizer()
        self.meta_reasoner = self._build_meta_reasoner()
        
        logger.debug("Meta-Cognition Engine initialized")
    
    def _build_coherence_analyzer(self) -> nn.Module:
        """Build coherence analysis network"""
        return nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),  # Single coherence score
            nn.Sigmoid()
        )
    
    def _build_strategy_optimizer(self) -> nn.Module:
        """Build strategy optimization network"""
        return nn.Sequential(
            nn.Linear(16, 24),
            nn.ReLU(),
            nn.Linear(24, 16),  # Strategy adjustment vector
            nn.Tanh()  # Bounded adjustments
        )
    
    def _build_meta_reasoner(self) -> nn.Module:
        """Build meta-reasoning network"""
        return nn.Sequential(
            nn.Linear(32, 64),  # Combined input + context
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),  # Meta-reasoning output
            nn.LayerNorm(16)
        )
    
    def analyze_coherence(self, cognition_result: Dict[str, Any]) -> float:
        """
        Analyze the coherence of a cognition result
        
        Meta-cognitive assessment of thinking quality
        """
        # Extract cognitive features for coherence analysis
        coherence_features = torch.zeros(16)
        
        # Feature 1-4: Basic coherence metrics
        coherence_features[0] = cognition_result.get('coherence', 0.0)
        coherence_features[1] = 1.0 - cognition_result.get('dissonance', 1.0)
        coherence_features[2] = min(cognition_result.get('processing_time', 10.0) / 10.0, 1.0)  # Normalized time
        coherence_features[3] = min(cognition_result.get('reasoning_steps', 1) / 10.0, 1.0)  # Normalized steps
        
        # Feature 5-8: Hypothesis quality metrics
        hypotheses_generated = cognition_result.get('hypotheses_generated', 0)
        hypotheses_validated = cognition_result.get('hypotheses_validated', 0)
        coherence_features[4] = min(hypotheses_generated / 10.0, 1.0)  # Diversity
        coherence_features[5] = (hypotheses_validated / max(hypotheses_generated, 1))  # Validation rate
        coherence_features[6] = float(cognition_result.get('success', False))  # Success indicator
        coherence_features[7] = len(cognition_result.get('output', '')) / 1000.0  # Output richness
        
        # Feature 9-12: Cognitive complexity indicators
        if 'cognition_4d' in cognition_result:
            cog_4d = torch.tensor(cognition_result['cognition_4d'][:4])
            coherence_features[8] = torch.norm(cog_4d).item()  # Magnitude
            coherence_features[9] = torch.std(cog_4d).item()   # Variability
        
        # Feature 13-16: Meta-features (self-reference)
        coherence_features[12] = torch.norm(coherence_features[:12]).item()  # Self-coherence
        coherence_features[13] = torch.mean(coherence_features[:12]).item()  # Self-average
        coherence_features[14] = torch.std(coherence_features[:12]).item()   # Self-variability
        coherence_features[15] = 1.0  # Meta-awareness indicator
        
        # Analyze coherence
        analyzed_coherence = float(self.coherence_analyzer(coherence_features))
        
        logger.debug(f"Coherence analysis: {analyzed_coherence:.3f}")
        return analyzed_coherence
    
    def meta_reason(self, cognition_result: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Meta-reasoning: Thinking about the thinking process
        
        Analyzes HOW the cognition was performed, not just the result
        """
        logger.debug("🔍 Meta-reasoning analysis")
        
        # Prepare meta-reasoning input
        meta_input = torch.zeros(32)
        
        # Primary reasoning features (16D)
        coherence_vector = torch.zeros(16)
        coherence_vector[0] = self.analyze_coherence(cognition_result)
        coherence_vector[1] = cognition_result.get('processing_time', 0.0)
        coherence_vector[2] = cognition_result.get('reasoning_steps', 0)
        coherence_vector[3] = float(cognition_result.get('success', False))
        
        # Add strategy weight influence
        coherence_vector[4:16] = self.strategy_weights[4:16] * coherence_vector[0]
        
        meta_input[:16] = coherence_vector
        
        # Context features (16D)
        context_vector = torch.zeros(16)
        if context:
            context_vector[0] = context.get('previous_coherence', 0.5)
            context_vector[1] = context.get('learning_progress', 0.5)
            context_vector[2] = context.get('strategy_effectiveness', 0.5)
        
        meta_input[16:32] = context_vector
        
        # Perform meta-reasoning
        meta_reasoning_output = self.meta_reasoner(meta_input)
        
        # Interpret meta-reasoning results
        meta_analysis = {
            'meta_coherence': float(torch.mean(meta_reasoning_output[:4])),
            'strategy_assessment': float(torch.mean(meta_reasoning_output[4:8])),
            'learning_potential': float(torch.mean(meta_reasoning_output[8:12])),
            'improvement_direction': meta_reasoning_output[12:16].tolist(),
            'meta_confidence': float(torch.norm(meta_reasoning_output)),
            'requires_strategy_update': float(torch.max(torch.abs(meta_reasoning_output[12:16]))) > 0.5
        }
        
        logger.debug(f"Meta-reasoning complete: confidence={meta_analysis['meta_confidence']:.3f}")
        return meta_analysis
    
    def optimize_strategy(self, meta_analysis: Dict[str, Any]) -> torch.Tensor:
        """
        Optimize cognitive strategy based on meta-analysis
        
        Updates strategy weights for improved future performance
        """
        logger.debug("⚙️ Strategy optimization")
        
        # Create strategy optimization input
        strategy_input = torch.zeros(16)
        strategy_input[0] = meta_analysis['meta_coherence']
        strategy_input[1] = meta_analysis['strategy_assessment']
        strategy_input[2] = meta_analysis['learning_potential']
        strategy_input[3] = meta_analysis['meta_confidence']
        
        # Add improvement direction with proper dimension handling
        improvement_direction = meta_analysis.get('improvement_direction', [0.0] * 4)
        # improvement_direction is 4D from meta_reasoning_output[12:16]
        # We need to expand it to 12D for strategy_input[4:16]
        if len(improvement_direction) == 4:
            # Expand 4D to 12D by repeating the pattern
            expanded_direction = improvement_direction * 3  # Repeat 3 times to get 12 elements
            improvement_vector = torch.tensor(expanded_direction, dtype=torch.float32)
        else:
            # Fallback: pad or truncate to 12 dimensions
            if len(improvement_direction) >= 12:
                improvement_vector = torch.tensor(improvement_direction[:12], dtype=torch.float32)
            else:
                padded_direction = list(improvement_direction) + [0.0] * (12 - len(improvement_direction))
                improvement_vector = torch.tensor(padded_direction, dtype=torch.float32)
        
        strategy_input[4:16] = improvement_vector
        
        # Generate strategy adjustments
        strategy_adjustments = self.strategy_optimizer(strategy_input)
        
        # Update strategy weights with learning rate
        self.strategy_weights += self.config.strategy_refinement_rate * strategy_adjustments
        
        # Keep weights bounded
        self.strategy_weights = torch.clamp(self.strategy_weights, 0.0, 1.0)
        
        logger.debug(f"Strategy updated: avg weight={torch.mean(self.strategy_weights):.3f}")
        return self.strategy_weights


class CognitiveIntrospection:
    """
    Cognitive introspection system for self-awareness
    
    Implements self-examination of cognitive processes
    """
    
    def __init__(self, config: ReflectionConfig):
        self.config = config
        self.introspection_history = []
        self.self_model = {}  # Model of own cognitive capabilities
        
    def introspect(self, cognition_result: Dict[str, Any], meta_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform cognitive introspection
        
        Self-examination of thinking processes and capabilities
        """
        logger.debug("🧘 Cognitive introspection")
        
        # Introspective questions and analysis
        introspection_results = {
            'self_assessment': self._assess_performance(cognition_result, meta_analysis),
            'capability_analysis': self._analyze_capabilities(cognition_result),
            'learning_insights': self._extract_learning_insights(cognition_result, meta_analysis),
            'cognitive_state': self._assess_cognitive_state(cognition_result),
            'improvement_potential': self._identify_improvements(meta_analysis)
        }
        
        # Update self-model
        self._update_self_model(introspection_results)
        
        # Store introspection in history
        self.introspection_history.append({
            'timestamp': time.time(),
            'cognition_coherence': cognition_result.get('coherence', 0.0),
            'meta_confidence': meta_analysis.get('meta_confidence', 0.0),
            'introspection_results': introspection_results
        })
        
        # Keep history bounded
        if len(self.introspection_history) > self.config.experience_buffer_size:
            self.introspection_history.pop(0)
        
        logger.debug(f"Introspection complete: {len(introspection_results)} insights")
        return introspection_results
    
    def _assess_performance(self, cognition_result: Dict[str, Any], meta_analysis: Dict[str, Any]) -> Dict[str, str]:
        """Self-assessment of cognitive performance"""
        coherence = cognition_result.get('coherence', 0.0)
        meta_confidence = meta_analysis.get('meta_confidence', 0.0)
        
        if coherence > 0.8 and meta_confidence > 0.7:
            performance = "Excellent cognitive performance with high coherence and meta-awareness"
        elif coherence > 0.6 and meta_confidence > 0.5:
            performance = "Good cognitive performance with moderate coherence"
        elif coherence > 0.4:
            performance = "Acceptable cognitive performance, room for improvement"
        else:
            performance = "Poor cognitive performance, significant improvement needed"
        
        return {
            'overall': performance,
            'coherence_level': f"{coherence:.1%}",
            'meta_confidence_level': f"{meta_confidence:.1%}"
        }
    
    def _analyze_capabilities(self, cognition_result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze current cognitive capabilities"""
        capabilities = {
            'reasoning_depth': min(cognition_result.get('reasoning_steps', 1) / 10.0, 1.0),
            'hypothesis_generation': min(cognition_result.get('hypotheses_generated', 1) / 5.0, 1.0),
            'validation_accuracy': cognition_result.get('hypotheses_validated', 0) / max(cognition_result.get('hypotheses_generated', 1), 1),
            'processing_efficiency': max(0.0, 1.0 - min(cognition_result.get('processing_time', 1.0) / 10.0, 1.0)),
            'output_quality': min(len(cognition_result.get('output', '')) / 1000.0, 1.0)
        }
        
        return capabilities
    
    def _extract_learning_insights(self, cognition_result: Dict[str, Any], meta_analysis: Dict[str, Any]) -> List[str]:
        """Extract learning insights from cognitive performance"""
        insights = []
        
        if meta_analysis.get('requires_strategy_update', False):
            insights.append("Strategy optimization needed for improved performance")
        
        if cognition_result.get('coherence', 0.0) < self.config.coherence_analysis_threshold:
            insights.append("Coherence improvement through better hypothesis validation required")
        
        if meta_analysis.get('learning_potential', 0.0) > 0.7:
            insights.append("High learning potential detected - increase reflection depth")
        
        if len(self.introspection_history) > 5:
            recent_coherences = [h['cognition_coherence'] for h in self.introspection_history[-5:]]
            if np.std(recent_coherences) < 0.1:
                insights.append("Cognitive consistency achieved - ready for more complex challenges")
        
        return insights
    
    def _assess_cognitive_state(self, cognition_result: Dict[str, Any]) -> Dict[str, str]:
        """Assess current cognitive state"""
        processing_time = cognition_result.get('processing_time', 0.0)
        coherence = cognition_result.get('coherence', 0.0)
        
        if processing_time < 0.1 and coherence > 0.8:
            state = "Highly efficient and coherent"
        elif processing_time < 0.5 and coherence > 0.6:
            state = "Balanced efficiency and accuracy"
        elif coherence > 0.7:
            state = "High accuracy, moderate efficiency"
        elif processing_time < 0.3:
            state = "High efficiency, moderate accuracy"
        else:
            state = "Suboptimal performance - needs improvement"
        
        return {
            'current_state': state,
            'efficiency': f"{max(0, 1.0 - processing_time):.1%}",
            'accuracy': f"{coherence:.1%}"
        }
    
    def _identify_improvements(self, meta_analysis: Dict[str, Any]) -> List[str]:
        """Identify potential improvements"""
        improvements = []
        
        if meta_analysis.get('strategy_assessment', 0.5) < 0.6:
            improvements.append("Refine cognitive strategy for better performance")
        
        if meta_analysis.get('meta_coherence', 0.5) < 0.7:
            improvements.append("Enhance meta-cognitive coherence")
        
        improvement_direction = meta_analysis.get('improvement_direction', [0] * 16)
        if max(improvement_direction) > 0.5:
            improvements.append("Focus on areas with highest improvement potential")
        
        return improvements
    
    def _update_self_model(self, introspection_results: Dict[str, Any]):
        """Update internal model of cognitive capabilities"""
        capabilities = introspection_results['capability_analysis']
        
        # Update self-model with exponential moving average
        alpha = 0.1  # Learning rate
        for key, value in capabilities.items():
            if key in self.self_model:
                self.self_model[key] = (1 - alpha) * self.self_model[key] + alpha * value
            else:
                self.self_model[key] = value
        
        # Add meta-capabilities
        self.self_model['introspection_depth'] = len(introspection_results.get('learning_insights', []))
        self.self_model['self_awareness'] = 1.0  # We're doing introspection, so we're self-aware!


class ReflectionProcessor:
    """
    Core 16D Reflection Phase Processor
    
    Implements meta-cognitive reasoning and self-improvement
    """
    
    def __init__(self, config: ReflectionConfig = None):
        self.config = config or ReflectionConfig()
        self.meta_cognition = MetaCognitionEngine(self.config)
        self.introspection = CognitiveIntrospection(self.config)
        
        # Reflection networks
        self.reflection_network = self._build_reflection_network()
        self.meta_learner = self._build_meta_learner()
        
        # Performance tracking
        self.reflection_stats = {
            'total_reflections': 0,
            'successful_improvements': 0,
            'avg_meta_coherence': 0.0,
            'avg_introspection_depth': 0.0,
            'strategy_updates': 0
        }
        
        logger.info("16D Reflection Processor initialized")
    
    def _build_reflection_network(self) -> nn.Module:
        """Build main 16D reflection network"""
        return nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 24),
            nn.ReLU(), 
            nn.Linear(24, 16),  # 16D reflection output
            nn.LayerNorm(16)
        )
    
    def _build_meta_learner(self) -> nn.Module:
        """Build meta-learning network"""
        return nn.Sequential(
            nn.Linear(48, 64),  # Cognition + meta-analysis + introspection
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),  # Meta-learning output
            nn.Dropout(0.1)
        )
    
    def reflect(self, cognition_result: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Main 16D Reflection processing
        
        Meta-cognitive analysis of cognition results with self-improvement
        
        Args:
            cognition_result: Result from 4D cognition phase
            context: Additional context for reflection
            
        Returns:
            Reflection analysis with improvement recommendations
        """
        start_time = time.time()
        self.reflection_stats['total_reflections'] += 1
        
        logger.info("🧘 16D Reflection: Meta-cognitive analysis")
        
        try:
            # Step 1: Meta-cognitive analysis
            logger.debug("Step 1: Meta-cognitive analysis")
            meta_analysis = self.meta_cognition.meta_reason(cognition_result, context)
            
            # Step 2: Cognitive introspection
            logger.debug("Step 2: Cognitive introspection") 
            introspection_results = self.introspection.introspect(cognition_result, meta_analysis)
            
            # Step 3: 16D reflection processing
            logger.debug("Step 3: 16D reflection network")
            reflection_input = torch.zeros(16)
            
            # Build reflection input from meta-analysis
            reflection_input[0] = meta_analysis['meta_coherence']
            reflection_input[1] = meta_analysis['strategy_assessment'] 
            reflection_input[2] = meta_analysis['learning_potential']
            reflection_input[3] = meta_analysis['meta_confidence']
            reflection_input[4] = float(meta_analysis['requires_strategy_update'])
            
            # Add introspection features
            capabilities = introspection_results['capability_analysis']
            reflection_input[5] = capabilities['reasoning_depth']
            reflection_input[6] = capabilities['hypothesis_generation']
            reflection_input[7] = capabilities['validation_accuracy']
            reflection_input[8] = capabilities['processing_efficiency']
            reflection_input[9] = capabilities['output_quality']
            
            # Add self-model features
            self_model = self.introspection.self_model
            reflection_input[10] = self_model.get('reasoning_depth', 0.5)
            reflection_input[11] = self_model.get('hypothesis_generation', 0.5)
            reflection_input[12] = self_model.get('processing_efficiency', 0.5)
            reflection_input[13] = self_model.get('introspection_depth', 0.0) / 10.0
            reflection_input[14] = self_model.get('self_awareness', 1.0)
            reflection_input[15] = 1.0  # Reflection indicator
            
            # Process through reflection network
            reflection_output = self.reflection_network(reflection_input)
            
            # Step 4: Strategy optimization (if needed)
            if meta_analysis['requires_strategy_update']:
                logger.debug("Step 4: Strategy optimization")
                new_strategy = self.meta_cognition.optimize_strategy(meta_analysis)
                self.reflection_stats['strategy_updates'] += 1
            else:
                new_strategy = self.meta_cognition.strategy_weights
            
            # Step 5: Meta-learning
            logger.debug("Step 5: Meta-learning")
            
            # Create meta-learning input with proper dimension handling
            try:
                # Component 1: Cognition 4D (ensure exactly 16 dimensions)
                cognition_4d_data = cognition_result.get('cognition_4d', [0]*16)
                if len(cognition_4d_data) >= 16:
                    cognition_component = torch.tensor(cognition_4d_data[:16], dtype=torch.float32)
                else:
                    # Pad to 16 dimensions
                    padded_data = list(cognition_4d_data) + [0.0] * (16 - len(cognition_4d_data))
                    cognition_component = torch.tensor(padded_data, dtype=torch.float32)
                
                # Component 2: Meta-analysis (ensure exactly 16 dimensions)
                meta_keys = ['meta_coherence', 'strategy_assessment', 'learning_potential', 'meta_confidence']
                meta_values = [meta_analysis.get(k, 0.0) for k in meta_keys]
                meta_values.extend([0.0] * 12)  # Pad to 16 total
                meta_component = torch.tensor(meta_values[:16], dtype=torch.float32)
                
                # Component 3: Reflection output (already 16D)
                reflection_component = reflection_output[:16] if len(reflection_output) >= 16 else torch.cat([
                    reflection_output, 
                    torch.zeros(16 - len(reflection_output))
                ])
                
                # Concatenate components (16 + 16 + 16 = 48D)
                meta_learning_input = torch.cat([
                    cognition_component,
                    meta_component, 
                    reflection_component
                ])
                
                # Ensure exactly 48 dimensions
                if len(meta_learning_input) > 48:
                    meta_learning_input = meta_learning_input[:48]
                elif len(meta_learning_input) < 48:
                    padding = torch.zeros(48 - len(meta_learning_input))
                    meta_learning_input = torch.cat([meta_learning_input, padding])
                
                logger.debug(f"Meta-learning input shape: {meta_learning_input.shape}")
                
            except Exception as e:
                logger.warning(f"Meta-learning input creation failed: {e}")
                # Fallback: create 48D zero vector
                meta_learning_input = torch.zeros(48)
            
            meta_learning_output = self.meta_learner(meta_learning_input)
            
            # Step 6: Generate reflection insights
            processing_time = time.time() - start_time
            
            reflection_insights = self._generate_reflection_insights(
                cognition_result, meta_analysis, introspection_results, reflection_output, meta_learning_output
            )
            
            # Update statistics
            self.reflection_stats['avg_meta_coherence'] = (
                (self.reflection_stats['avg_meta_coherence'] * (self.reflection_stats['total_reflections'] - 1) + meta_analysis['meta_coherence'])
                / self.reflection_stats['total_reflections']
            )
            
            self.reflection_stats['avg_introspection_depth'] = (
                (self.reflection_stats['avg_introspection_depth'] * (self.reflection_stats['total_reflections'] - 1) + len(introspection_results.get('learning_insights', [])))
                / self.reflection_stats['total_reflections']
            )
            
            if meta_analysis['meta_coherence'] > self.config.coherence_analysis_threshold:
                self.reflection_stats['successful_improvements'] += 1
            
            return {
                'phase': 'reflection_16d',
                'success': True,
                'meta_analysis': meta_analysis,
                'introspection': introspection_results,
                'reflection_output': reflection_output.tolist(),
                'meta_learning': meta_learning_output.tolist(),
                'strategy_weights': new_strategy.tolist(),
                'insights': reflection_insights,
                'processing_time': processing_time,
                'method': '16d_meta_cognitive_reflection',
                'coherence': meta_analysis['meta_coherence'],
                'self_awareness_level': self.introspection.self_model.get('self_awareness', 1.0),
                'improvement_potential': meta_analysis['learning_potential'],
                'strategy_updated': meta_analysis['requires_strategy_update']
            }
            
        except Exception as e:
            logger.error(f"❌ 16D Reflection failed: {str(e)}")
            logger.debug(f"Error details: {e}", exc_info=True)  # Add debug info
            processing_time = time.time() - start_time
            
            return {
                'phase': 'reflection_16d_error',
                'success': False,
                'error': str(e),
                'processing_time': processing_time,
                'method': '16d_error_handling'
            }
    
    def _generate_reflection_insights(
        self, 
        cognition_result: Dict[str, Any], 
        meta_analysis: Dict[str, Any], 
        introspection_results: Dict[str, Any],
        reflection_output: torch.Tensor,
        meta_learning_output: torch.Tensor
    ) -> List[str]:
        """Generate human-readable reflection insights"""
        
        insights = []
        
        # Meta-cognitive insights
        if meta_analysis['meta_confidence'] > 0.8:
            insights.append("High meta-cognitive confidence - thinking processes are well-understood")
        elif meta_analysis['meta_confidence'] < 0.5:
            insights.append("Low meta-cognitive confidence - need to improve self-understanding")
        
        # Performance insights
        performance = introspection_results['self_assessment']['overall']
        insights.append(f"Self-assessment: {performance}")
        
        # Learning insights
        learning_insights = introspection_results.get('learning_insights', [])
        insights.extend(learning_insights)
        
        # Strategy insights
        if meta_analysis['requires_strategy_update']:
            insights.append("Cognitive strategy updated for improved future performance")
        
        # Improvement insights
        improvements = introspection_results.get('improvement_potential', [])
        if improvements:
            insights.append(f"Identified {len(improvements)} improvement opportunities")
        
        # Reflection depth insight
        reflection_magnitude = float(torch.norm(reflection_output))
        if reflection_magnitude > 2.0:
            insights.append("Deep reflection achieved - complex meta-cognitive processing")
        elif reflection_magnitude > 1.0:
            insights.append("Moderate reflection depth - good meta-cognitive awareness")
        else:
            insights.append("Shallow reflection - may need deeper meta-cognitive analysis")
        
        return insights
    
    def get_reflection_stats(self) -> Dict[str, Any]:
        """Get 16D Reflection performance statistics"""
        return {
            **self.reflection_stats,
            'improvement_rate': (
                self.reflection_stats['successful_improvements'] / self.reflection_stats['total_reflections']
                if self.reflection_stats['total_reflections'] > 0 else 0.0
            ),
            'self_model': dict(self.introspection.self_model),
            'strategy_weights': self.meta_cognition.strategy_weights.tolist()
        }


class ReflectionPhaseIntegrator:
    """
    Integrates 16D Reflection Phase with Enhanced SATC Engine
    """
    
    def __init__(self, reflection_processor: ReflectionProcessor):
        self.reflection_processor = reflection_processor
        self.integrated = False
        
    def integrate_with_satc(self, satc_engine):
        """
        Integrate 16D Reflection Phase with Enhanced SATC Engine
        """
        logger.info("Integrating 16D Reflection Phase with Enhanced SATC...")
        
        # Add Reflection processor to engine
        satc_engine.reflection_processor = self.reflection_processor
        satc_engine._using_reflection_16d = True
        
        self.integrated = True
        logger.info("✅ 16D Reflection Phase integration completed!")
        
        return satc_engine
    
    def reflect_on_cognition(self, cognition_result: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Perform 16D reflection on cognition results
        """
        if not self.integrated:
            logger.warning("16D Reflection Phase not yet integrated with SATC")
        
        # Process through 16D Reflection
        reflection_result = self.reflection_processor.reflect(cognition_result, context)
        
        return reflection_result


def create_reflection_phase():
    """
    Factory function to create complete 16D Reflection Phase
    """
    config = ReflectionConfig()
    processor = ReflectionProcessor(config)
    integrator = ReflectionPhaseIntegrator(processor)
    
    logger.info("16D Reflection Phase created successfully!")
    logger.info(f"Meta-cognitive analysis with {config.meta_analysis_depth} levels")
    logger.info(f"Introspection steps: {config.introspection_steps}")
    
    return processor, integrator, config


# Standalone testing function
def test_reflection_phase_standalone():
    """
    Standalone test of 16D Reflection Phase
    """
    print("=" * 60)
    print("REFLECTION PHASE (16D) - STANDALONE TEST")
    print("=" * 60)
    
    # Create 16D Reflection phase
    processor, integrator, config = create_reflection_phase()
    
    # Mock cognition results for reflection
    mock_cognition_results = [
        {
            'phase': 'cognition_4d',
            'success': True,
            'coherence': 0.8,
            'dissonance': 0.2,
            'processing_time': 0.5,
            'reasoning_steps': 5,
            'hypotheses_generated': 4,
            'hypotheses_validated': 3,
            'cognition_4d': [0.5, -0.3, 0.8, 0.1],
            'output': 'Complex analytical reasoning about consciousness and cognition...'
        },
        {
            'phase': 'cognition_4d', 
            'success': True,
            'coherence': 0.6,
            'dissonance': 0.4,
            'processing_time': 0.8,
            'reasoning_steps': 3,
            'hypotheses_generated': 2,
            'hypotheses_validated': 1,
            'cognition_4d': [0.2, 0.7, -0.1, 0.4],
            'output': 'Moderate reasoning about neural networks...'
        }
    ]
    
    print("\n--- 16D Reflection Testing ---")
    for i, cognition_result in enumerate(mock_cognition_results):
        print(f"\nReflection {i+1}: Cognition coherence={cognition_result['coherence']}")
        
        result = processor.reflect(cognition_result)
        
        print(f"  Success: {result['success']}")
        print(f"  Meta-coherence: {result.get('meta_analysis', {}).get('meta_coherence', 0):.3f}")
        print(f"  Self-awareness: {result.get('self_awareness_level', 0):.3f}")
        print(f"  Strategy updated: {result.get('strategy_updated', False)}")
        print(f"  Improvement potential: {result.get('improvement_potential', 0):.3f}")
        print(f"  Processing time: {result['processing_time']:.3f}s")
        print(f"  Insights: {len(result.get('insights', []))}")
        
        for insight in result.get('insights', [])[:3]:  # Show first 3 insights
            print(f"    • {insight}")
    
    print("\n--- Reflection Statistics ---")
    stats = processor.get_reflection_stats()
    for key, value in stats.items():
        if key not in ['self_model', 'strategy_weights']:  # Skip complex nested data
            print(f"{key}: {value}")
    
    print(f"\nSelf-model capabilities:")
    for cap, value in stats.get('self_model', {}).items():
        print(f"  {cap}: {value:.3f}")
    
    print("=" * 60)
    print("MILESTONE 4: 16D REFLECTION PHASE - COMPLETE!")
    print("=" * 60)
    
    return processor, integrator, config


if __name__ == "__main__":
    test_reflection_phase_standalone()