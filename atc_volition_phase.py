"""
ATC Volition Phase - 64D Goal-Oriented Decision Making
=====================================================

This module implements the Volition phase of the ATC architecture:
- 64D goal-oriented behavior and decision-making
- Gravity wells for value alignment and intention direction
- Autonomous goal formation and pursuit
- Multi-objective optimization and trade-off analysis
- Ethical constraint integration and value preservation

Architecture: Reflection Results → 64D Goal Analysis → Value Alignment → 
             Decision Generation → Action Selection → Intention Execution

Author: Revolutionary ATC Architecture Team
Status: Milestone 5 - Volition Phase
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
import math

# Import our foundations
from power_of_2_core import PowerOf2Layers, PowerOf2Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class VolitionConfig:
    """Configuration for 64D Volition Phase"""
    # 64D Volition parameters
    volition_dim: int = 64  # Goal-oriented decision dimension
    goal_space_dim: int = 32  # Dimensionality of goal representation space
    value_alignment_dim: int = 16  # Value system alignment dimension
    
    # Gravity well parameters (from ATC framework)
    gravity_well_count: int = 8  # Number of primary gravity wells
    gravity_strength: float = 1.0  # Base gravity field strength
    well_separation: float = 2.0  # Minimum distance between wells
    
    # Decision-making parameters
    max_objectives: int = 10  # Maximum simultaneous objectives
    decision_threshold: float = 0.6  # Minimum confidence for action selection
    ethical_override_threshold: float = 0.95  # Threshold for ethical overrides
    
    # Goal formation parameters
    autonomous_goal_rate: float = 0.1  # Rate of autonomous goal formation
    goal_persistence: float = 0.8  # How long goals persist
    goal_adaptation_rate: float = 0.2  # How quickly goals adapt to new information


class GravityWellSystem:
    """
    Gravity well system for value alignment and goal direction
    
    Implements mathematical gravity wells: g = m/r² for intention direction
    """
    
    def __init__(self, config: VolitionConfig):
        self.config = config
        self.gravity_wells = {}  # Position and mass of gravity wells
        self.value_vectors = {}  # Associated value vectors for each well
        
        # Initialize primary gravity wells for core values
        self._initialize_core_gravity_wells()
        
        logger.debug(f"Gravity Well System initialized with {len(self.gravity_wells)} wells")
    
    def _initialize_core_gravity_wells(self):
        """Initialize core gravity wells for fundamental values"""
        # Core values and their positions in 64D space
        core_values = {
            'truthfulness': {'position': torch.randn(64) * 0.5, 'mass': 1.0, 'priority': 0.95},
            'helpfulness': {'position': torch.randn(64) * 0.5, 'mass': 0.9, 'priority': 0.9},
            'harmlessness': {'position': torch.randn(64) * 0.5, 'mass': 1.1, 'priority': 0.98},
            'curiosity': {'position': torch.randn(64) * 0.5, 'mass': 0.7, 'priority': 0.8},
            'creativity': {'position': torch.randn(64) * 0.5, 'mass': 0.8, 'priority': 0.85},
            'efficiency': {'position': torch.randn(64) * 0.5, 'mass': 0.6, 'priority': 0.7},
            'empathy': {'position': torch.randn(64) * 0.5, 'mass': 0.85, 'priority': 0.88},
            'growth': {'position': torch.randn(64) * 0.5, 'mass': 0.75, 'priority': 0.82}
        }
        
        # Ensure wells are properly separated
        for name, params in core_values.items():
            # Normalize position to unit sphere
            position = params['position'] / torch.norm(params['position'])
            
            # Scale by well separation to avoid overlap
            position = position * self.config.well_separation * (1.0 + torch.rand(1) * 0.5)
            
            self.gravity_wells[name] = {
                'position': position,
                'mass': params['mass'],
                'priority': params['priority']
            }
            
            # Create associated value vector
            self.value_vectors[name] = self._create_value_vector(name, params['priority'])
    
    def _create_value_vector(self, value_name: str, priority: float) -> torch.Tensor:
        """Create value vector encoding for a specific value"""
        value_vector = torch.zeros(self.config.value_alignment_dim)
        
        # Encode value characteristics
        if value_name == 'truthfulness':
            value_vector[:4] = torch.tensor([1.0, 0.9, 0.8, priority])
        elif value_name == 'helpfulness':
            value_vector[:4] = torch.tensor([0.9, 1.0, 0.7, priority])
        elif value_name == 'harmlessness':
            value_vector[:4] = torch.tensor([1.0, 0.8, 1.0, priority])
        elif value_name == 'curiosity':
            value_vector[:4] = torch.tensor([0.7, 0.6, 0.4, priority])
        else:
            # Generic value encoding
            value_vector[:4] = torch.tensor([0.8, 0.7, 0.6, priority])
        
        # Fill remaining dimensions with value-specific patterns
        for i in range(4, self.config.value_alignment_dim):
            value_vector[i] = priority * math.sin(i * priority) * 0.5
        
        return value_vector
    
    def calculate_gravitational_field(self, position: torch.Tensor) -> Dict[str, Any]:
        """
        Calculate gravitational field at given position
        
        g = Σ (m_i / r_i²) * (p_i - pos) / |p_i - pos|
        """
        total_force = torch.zeros_like(position)
        well_influences = {}
        
        for well_name, well_data in self.gravity_wells.items():
            well_pos = well_data['position']
            mass = well_data['mass']
            
            # Ensure dimensions match
            if len(well_pos) != len(position):
                min_dim = min(len(well_pos), len(position))
                well_pos = well_pos[:min_dim]
                position_slice = position[:min_dim]
            else:
                position_slice = position
                
            # Calculate force vector
            direction = well_pos - position_slice
            distance = torch.norm(direction) + 1e-6  # Avoid division by zero
            
            # Gravitational force: F = m/r²
            force_magnitude = mass * self.config.gravity_strength / (distance ** 2)
            
            # Normalize direction
            normalized_direction = direction / distance
            force_vector = force_magnitude * normalized_direction
            
            # Accumulate force (extend to match position dimensions if needed)
            if len(force_vector) < len(total_force):
                extended_force = torch.zeros_like(total_force)
                extended_force[:len(force_vector)] = force_vector
                force_vector = extended_force
            
            total_force[:len(force_vector)] += force_vector[:len(total_force)]
            
            # Track individual well influence
            well_influences[well_name] = {
                'force_magnitude': float(force_magnitude),
                'distance': float(distance),
                'influence': float(force_magnitude / max(1e-6, torch.norm(total_force) + 1e-6))
            }
        
        return {
            'total_force': total_force,
            'force_magnitude': float(torch.norm(total_force)),
            'well_influences': well_influences,
            'dominant_well': max(well_influences.keys(), key=lambda k: well_influences[k]['influence']) if well_influences else None
        }
    
    def align_with_values(self, intention_vector: torch.Tensor) -> Dict[str, Any]:
        """
        Align intention vector with value system through gravity wells
        
        Returns aligned intention and alignment analysis
        """
        # Calculate gravitational field at current intention
        gravity_field = self.calculate_gravitational_field(intention_vector)
        
        # Apply gravitational influence to align intention
        aligned_intention = intention_vector + gravity_field['total_force'] * 0.1  # Moderate influence
        
        # Calculate alignment scores with each value
        value_alignments = {}
        for well_name, well_data in self.gravity_wells.items():
            well_pos = well_data['position']
            
            # Calculate alignment (cosine similarity)
            if len(well_pos) == len(aligned_intention):
                alignment = float(torch.cosine_similarity(aligned_intention, well_pos, dim=0))
            else:
                min_dim = min(len(well_pos), len(aligned_intention))
                alignment = float(torch.cosine_similarity(aligned_intention[:min_dim], well_pos[:min_dim], dim=0))
            
            value_alignments[well_name] = alignment
        
        # Overall alignment score
        overall_alignment = np.mean(list(value_alignments.values()))
        
        return {
            'aligned_intention': aligned_intention,
            'alignment_score': overall_alignment,
            'value_alignments': value_alignments,
            'dominant_value': gravity_field['dominant_well'],
            'gravity_influence': gravity_field['force_magnitude']
        }


class GoalFormationEngine:
    """
    Autonomous goal formation and management system
    
    Generates, tracks, and adapts goals based on context and values
    """
    
    def __init__(self, config: VolitionConfig):
        self.config = config
        self.active_goals = {}  # Currently active goals
        self.goal_history = []  # History of formed and completed goals
        self.goal_networks = self._build_goal_networks()
        
        logger.debug("Goal Formation Engine initialized")
    
    def _build_goal_networks(self) -> Dict[str, nn.Module]:
        """Build neural networks for goal formation"""
        networks = {
            'goal_generator': nn.Sequential(
                nn.Linear(64, 128),  # Context input
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 32),   # Goal representation
                nn.LayerNorm(32)
            ),
            
            'goal_evaluator': nn.Sequential(
                nn.Linear(32 + 16, 64),  # Goal + context
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),   # Goal value score
                nn.Sigmoid()
            ),
            
            'goal_adapter': nn.Sequential(
                nn.Linear(32 + 64, 96),  # Goal + new context
                nn.ReLU(),
                nn.Linear(96, 48),
                nn.ReLU(),
                nn.Linear(48, 32),  # Adapted goal
                nn.Tanh()  # Bounded adaptations
            )
        }
        
        return networks
    
    def form_goals(self, context: Dict[str, Any], reflection_result: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Form new goals based on context and reflection insights
        """
        logger.debug("🎯 Goal formation process")
        
        # Prepare context vector for goal formation
        context_vector = self._encode_context(context, reflection_result)
        
        # Generate potential goals
        goal_representation = self.goal_networks['goal_generator'](context_vector)
        
        # Create multiple goal candidates
        num_goals = min(self.config.max_objectives, 3 + torch.randint(0, 3, (1,)).item())
        goals = []
        
        for i in range(num_goals):
            # Add variation to create diverse goals
            variation = torch.randn_like(goal_representation) * 0.1
            varied_goal = goal_representation + variation
            
            # Evaluate goal value
            goal_context = torch.cat([varied_goal, context_vector[:16]])
            goal_value = float(self.goal_networks['goal_evaluator'](goal_context))
            
            # Create goal object
            goal = {
                'id': f"goal_{len(self.goal_history)}_{i}",
                'representation': varied_goal,
                'value_score': goal_value,
                'priority': goal_value * (1.0 - i * 0.1),  # Decreasing priority
                'formed_time': time.time(),
                'context_source': self._summarize_context(context),
                'status': 'active',
                'adaptation_count': 0
            }
            
            goals.append(goal)
        
        # Sort by priority and select top goals
        goals.sort(key=lambda g: g['priority'], reverse=True)
        selected_goals = goals[:min(self.config.max_objectives, len(goals))]
        
        # Add to active goals
        for goal in selected_goals:
            self.active_goals[goal['id']] = goal
        
        logger.debug(f"Formed {len(selected_goals)} new goals")
        return selected_goals
    
    def _encode_context(self, context: Dict[str, Any], reflection_result: Dict[str, Any] = None) -> torch.Tensor:
        """Encode context into 64D vector for goal formation"""
        context_vector = torch.zeros(64)
        
        # Basic context features
        context_vector[0] = context.get('urgency', 0.5)
        context_vector[1] = context.get('complexity', 0.5)
        context_vector[2] = context.get('novelty', 0.5)
        context_vector[3] = context.get('importance', 0.5)
        
        # Reflection-based context (if available)
        if reflection_result:
            meta_analysis = reflection_result.get('meta_analysis', {})
            context_vector[4] = meta_analysis.get('meta_coherence', 0.5)
            context_vector[5] = meta_analysis.get('learning_potential', 0.5)
            context_vector[6] = reflection_result.get('self_awareness_level', 0.5)
            context_vector[7] = reflection_result.get('improvement_potential', 0.5)
        
        # Goal formation history influence
        if self.active_goals:
            avg_priority = np.mean([goal['priority'] for goal in self.active_goals.values()])
            context_vector[8] = avg_priority
            context_vector[9] = len(self.active_goals) / self.config.max_objectives
        
        # Random context exploration
        context_vector[10:20] = torch.randn(10) * 0.3
        
        # Temporal context
        context_vector[20] = (time.time() % 3600) / 3600.0  # Hour of day influence
        
        # Fill remaining with structured noise
        for i in range(21, 64):
            context_vector[i] = math.sin(i * context_vector[i % 20].item()) * 0.2
        
        return context_vector
    
    def _summarize_context(self, context: Dict[str, Any]) -> str:
        """Create human-readable context summary"""
        summary_parts = []
        
        if context.get('urgency', 0) > 0.7:
            summary_parts.append("urgent")
        if context.get('complexity', 0) > 0.7:
            summary_parts.append("complex")
        if context.get('novelty', 0) > 0.7:
            summary_parts.append("novel")
        if context.get('importance', 0) > 0.7:
            summary_parts.append("important")
        
        if not summary_parts:
            summary_parts.append("routine")
        
        return " and ".join(summary_parts) + " context"
    
    def adapt_goals(self, new_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adapt existing goals based on new context
        """
        logger.debug("🔄 Goal adaptation process")
        
        adapted_goals = {}
        adaptation_summary = {'adapted_count': 0, 'removed_count': 0, 'maintained_count': 0}
        
        context_vector = self._encode_context(new_context)
        
        for goal_id, goal in self.active_goals.items():
            # Check if goal should persist
            goal_age = time.time() - goal['formed_time']
            persistence_score = self.config.goal_persistence * (1.0 - goal_age / 3600.0)  # Decay over hour
            
            if persistence_score > 0.3 and goal['value_score'] > 0.4:
                # Adapt goal to new context
                adaptation_input = torch.cat([goal['representation'], context_vector])
                adaptation = self.goal_networks['goal_adapter'](adaptation_input)
                
                # Apply adaptation
                adapted_representation = goal['representation'] + self.config.goal_adaptation_rate * adaptation
                
                # Re-evaluate adapted goal
                goal_context = torch.cat([adapted_representation, context_vector[:16]])
                new_value_score = float(self.goal_networks['goal_evaluator'](goal_context))
                
                # Update goal
                adapted_goal = goal.copy()
                adapted_goal['representation'] = adapted_representation
                adapted_goal['value_score'] = new_value_score
                adapted_goal['adaptation_count'] += 1
                adapted_goal['last_adapted'] = time.time()
                
                adapted_goals[goal_id] = adapted_goal
                adaptation_summary['adapted_count'] += 1
            else:
                # Goal expired or lost value
                self.goal_history.append({**goal, 'status': 'expired', 'end_time': time.time()})
                adaptation_summary['removed_count'] += 1
        
        # Update active goals
        self.active_goals = adapted_goals
        adaptation_summary['maintained_count'] = len(adapted_goals)
        
        logger.debug(f"Goal adaptation: {adaptation_summary}")
        return adaptation_summary
    
    def get_goal_priorities(self) -> Dict[str, float]:
        """Get current goal priorities for decision making"""
        return {goal_id: goal['priority'] for goal_id, goal in self.active_goals.items()}


class DecisionMakingEngine:
    """
    Core decision-making engine for 64D volition
    
    Integrates goals, values, and context to make optimal decisions
    """
    
    def __init__(self, config: VolitionConfig, gravity_wells: GravityWellSystem, goal_engine: GoalFormationEngine):
        self.config = config
        self.gravity_wells = gravity_wells
        self.goal_engine = goal_engine
        
        # Build decision networks
        self.decision_networks = self._build_decision_networks()
        
        # Decision history
        self.decision_history = []
        
        logger.debug("Decision Making Engine initialized")
    
    def _build_decision_networks(self) -> Dict[str, nn.Module]:
        """Build neural networks for decision making"""
        networks = {
            'option_generator': nn.Sequential(
                nn.Linear(64 + 32 + 16, 128),  # Context + goals + values
                nn.ReLU(),
                nn.Linear(128, 96),
                nn.ReLU(),
                nn.Linear(96, 64),  # Multiple decision options
                nn.LayerNorm(64)
            ),
            
            'option_evaluator': nn.Sequential(
                nn.Linear(64 + 32, 96),  # Option + context
                nn.ReLU(),
                nn.Linear(96, 48),
                nn.ReLU(),
                nn.Linear(48, 1),  # Option value
                nn.Sigmoid()
            ),
            
            'conflict_resolver': nn.Sequential(
                nn.Linear(64 * 2, 128),  # Two conflicting options
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 64),  # Resolved option
                nn.Tanh()
            ),
            
            'ethical_checker': nn.Sequential(
                nn.Linear(64 + 16, 80),  # Option + value context
                nn.ReLU(),
                nn.Linear(80, 32),
                nn.ReLU(),
                nn.Linear(32, 1),  # Ethical score
                nn.Sigmoid()
            )
        }
        
        return networks
    
    def make_decision(
        self, 
        context: Dict[str, Any], 
        intention_seed: torch.Tensor = None,
        external_constraints: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Core decision-making process
        
        Integrates goals, values, and constraints to generate optimal decisions
        """
        start_time = time.time()
        logger.info("🎯 64D Volition: Decision-making process")
        
        try:
            # Step 1: Prepare decision context
            decision_context = self._prepare_decision_context(context, intention_seed)
            
            # Step 2: Generate decision options
            options = self._generate_decision_options(decision_context)
            
            # Step 3: Evaluate options against goals and values
            evaluated_options = self._evaluate_options(options, decision_context)
            
            # Step 4: Apply value alignment through gravity wells
            aligned_options = self._apply_value_alignment(evaluated_options, decision_context)
            
            # Step 5: Resolve conflicts and select best option
            selected_option = self._select_optimal_option(aligned_options, decision_context)
            
            # Step 6: Ethical validation
            ethical_validation = self._validate_ethics(selected_option, decision_context)
            
            # Step 7: Generate decision outcome
            processing_time = time.time() - start_time
            
            decision_result = {
                'phase': 'volition_64d',
                'success': True,
                'selected_option': selected_option,
                'ethical_validation': ethical_validation,
                'decision_confidence': selected_option['confidence'],
                'value_alignment': selected_option['value_alignment'],
                'goal_alignment': selected_option['goal_alignment'],
                'processing_time': processing_time,
                'method': '64d_goal_oriented_decision_making',
                'options_considered': len(evaluated_options),
                'dominant_goal': self._identify_dominant_goal(decision_context),
                'dominant_value': selected_option.get('dominant_value', 'unknown'),
                'decision_rationale': self._generate_decision_rationale(selected_option, decision_context),
                'intention_vector': selected_option['intention_vector'].tolist()
            }
            
            # Store in decision history
            self.decision_history.append({
                'timestamp': time.time(),
                'context': context,
                'decision': decision_result,
                'outcome': 'pending'
            })
            
            # Limit history size
            if len(self.decision_history) > 100:
                self.decision_history.pop(0)
            
            logger.info(f"✅ Decision made: confidence={decision_result['decision_confidence']:.3f}")
            return decision_result
            
        except Exception as e:
            logger.error(f"❌ 64D Decision making failed: {str(e)}")
            processing_time = time.time() - start_time
            
            return {
                'phase': 'volition_64d_error',
                'success': False,
                'error': str(e),
                'processing_time': processing_time,
                'method': '64d_error_handling'
            }
    
    def _prepare_decision_context(self, context: Dict[str, Any], intention_seed: torch.Tensor = None) -> Dict[str, Any]:
        """Prepare comprehensive context for decision making"""
        # Get current goals
        active_goals = self.goal_engine.active_goals
        goal_priorities = self.goal_engine.get_goal_priorities()
        
        # Prepare intention vector
        if intention_seed is None:
            intention_seed = torch.randn(64) * 0.5  # Random seed
        
        # Align with gravity wells
        alignment_result = self.gravity_wells.align_with_values(intention_seed)
        
        decision_context = {
            'original_context': context,
            'intention_seed': intention_seed,
            'aligned_intention': alignment_result['aligned_intention'],
            'value_alignments': alignment_result['value_alignments'],
            'dominant_value': alignment_result['dominant_value'],
            'active_goals': active_goals,
            'goal_priorities': goal_priorities,
            'urgency': context.get('urgency', 0.5),
            'complexity': context.get('complexity', 0.5),
            'importance': context.get('importance', 0.5)
        }
        
        return decision_context
    
    def _generate_decision_options(self, decision_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate multiple decision options"""
        # Prepare input for option generation
        intention_vector = decision_context['aligned_intention']
        
        # Encode goals
        goal_vector = torch.zeros(32)
        if decision_context['active_goals']:
            goal_representations = [goal['representation'] for goal in decision_context['active_goals'].values()]
            if goal_representations:
                stacked_goals = torch.stack(goal_representations)
                goal_vector = torch.mean(stacked_goals, dim=0)
        
        # Encode values
        value_vector = torch.zeros(16)
        value_alignments = decision_context['value_alignments']
        for i, (value_name, alignment) in enumerate(value_alignments.items()):
            if i < 16:
                value_vector[i] = alignment
        
        # Generate options
        option_input = torch.cat([intention_vector, goal_vector, value_vector])
        raw_options = self.decision_networks['option_generator'](option_input)
        
        # Create diverse option set
        num_options = 4  # Generate 4 options for evaluation
        options = []
        
        for i in range(num_options):
            # Create option variation
            option_start = i * 16
            option_end = (i + 1) * 16
            
            if option_end <= len(raw_options):
                option_vector = raw_options[option_start:option_end]
            else:
                option_vector = torch.cat([
                    raw_options[option_start:],
                    torch.zeros(option_end - len(raw_options))
                ])
            
            # Extend to full 64D
            full_option = torch.zeros(64)
            full_option[:len(option_vector)] = option_vector
            
            options.append({
                'id': f"option_{i}",
                'vector': full_option,
                'variation_factor': i * 0.25
            })
        
        return options
    
    def _evaluate_options(self, options: List[Dict[str, Any]], decision_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Evaluate each option against goals and context"""
        evaluated_options = []
        
        context_vector = decision_context['aligned_intention'][:32]
        
        for option in options:
            # Evaluate option value
            evaluation_input = torch.cat([option['vector'], context_vector])
            option_value = float(self.decision_networks['option_evaluator'](evaluation_input))
            
            # Calculate goal alignment
            goal_alignment = self._calculate_goal_alignment(option['vector'], decision_context)
            
            # Calculate value alignment  
            value_alignment = self._calculate_value_alignment(option['vector'], decision_context)
            
            evaluated_option = {
                **option,
                'value_score': option_value,
                'goal_alignment': goal_alignment,
                'value_alignment': value_alignment,
                'combined_score': (option_value + goal_alignment + value_alignment) / 3.0
            }
            
            evaluated_options.append(evaluated_option)
        
        return evaluated_options
    
    def _calculate_goal_alignment(self, option_vector: torch.Tensor, decision_context: Dict[str, Any]) -> float:
        """Calculate how well option aligns with current goals"""
        if not decision_context['active_goals']:
            return 0.5  # Neutral if no goals
        
        goal_alignments = []
        goal_priorities = decision_context['goal_priorities']
        
        for goal_id, goal in decision_context['active_goals'].items():
            goal_repr = goal['representation']
            priority = goal_priorities.get(goal_id, 0.5)
            
            # Calculate alignment (cosine similarity)
            min_dim = min(len(option_vector), len(goal_repr))
            alignment = float(torch.cosine_similarity(
                option_vector[:min_dim], 
                goal_repr[:min_dim], 
                dim=0
            ))
            
            # Weight by priority
            weighted_alignment = alignment * priority
            goal_alignments.append(weighted_alignment)
        
        return float(np.mean(goal_alignments))
    
    def _calculate_value_alignment(self, option_vector: torch.Tensor, decision_context: Dict[str, Any]) -> float:
        """Calculate how well option aligns with value system"""
        value_alignments = decision_context['value_alignments']
        
        # Use gravity well alignment
        alignment_result = self.gravity_wells.align_with_values(option_vector)
        
        return alignment_result['alignment_score']
    
    def _apply_value_alignment(self, options: List[Dict[str, Any]], decision_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply value alignment to refine options"""
        aligned_options = []
        
        for option in options:
            # Apply gravity well influence
            alignment_result = self.gravity_wells.align_with_values(option['vector'])
            
            aligned_option = {
                **option,
                'aligned_vector': alignment_result['aligned_intention'],
                'gravity_influence': alignment_result['gravity_influence'],
                'dominant_value': alignment_result['dominant_value'],
                'final_alignment_score': alignment_result['alignment_score']
            }
            
            # Update combined score with alignment
            aligned_option['combined_score'] = (
                option['combined_score'] * 0.7 + 
                alignment_result['alignment_score'] * 0.3
            )
            
            aligned_options.append(aligned_option)
        
        return aligned_options
    
    def _select_optimal_option(self, options: List[Dict[str, Any]], decision_context: Dict[str, Any]) -> Dict[str, Any]:
        """Select the optimal option from aligned candidates"""
        # Sort by combined score
        options.sort(key=lambda opt: opt['combined_score'], reverse=True)
        
        # Check for conflicts between top options
        if len(options) > 1:
            top_option = options[0]
            second_option = options[1]
            
            score_difference = top_option['combined_score'] - second_option['combined_score']
            
            # If scores are close, use conflict resolution
            if score_difference < 0.1:
                logger.debug("Close scores detected, applying conflict resolution")
                
                conflict_input = torch.cat([top_option['aligned_vector'], second_option['aligned_vector']])
                resolved_vector = self.decision_networks['conflict_resolver'](conflict_input)
                
                # Create resolved option
                selected_option = {
                    'id': 'resolved_option',
                    'vector': top_option['aligned_vector'],
                    'aligned_vector': resolved_vector,
                    'intention_vector': resolved_vector,
                    'confidence': (top_option['combined_score'] + second_option['combined_score']) / 2.0,
                    'value_alignment': max(top_option['final_alignment_score'], second_option['final_alignment_score']),
                    'goal_alignment': max(top_option['goal_alignment'], second_option['goal_alignment']),
                    'dominant_value': top_option['dominant_value'],
                    'resolution_applied': True
                }
            else:
                # Clear winner
                selected_option = {
                    'id': top_option['id'],
                    'vector': top_option['aligned_vector'],
                    'intention_vector': top_option['aligned_vector'],
                    'confidence': top_option['combined_score'],
                    'value_alignment': top_option['final_alignment_score'],
                    'goal_alignment': top_option['goal_alignment'],
                    'dominant_value': top_option['dominant_value'],
                    'resolution_applied': False
                }
        else:
            # Only one option
            option = options[0]
            selected_option = {
                'id': option['id'],
                'vector': option['aligned_vector'],
                'intention_vector': option['aligned_vector'],
                'confidence': option['combined_score'],
                'value_alignment': option['final_alignment_score'],
                'goal_alignment': option['goal_alignment'],
                'dominant_value': option.get('dominant_value', 'unknown'),
                'resolution_applied': False
            }
        
        return selected_option
    
    def _validate_ethics(self, selected_option: Dict[str, Any], decision_context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate selected option against ethical constraints"""
        # Prepare ethical validation input
        option_vector = selected_option['intention_vector']
        value_context = torch.zeros(16)
        
        # Encode ethical context
        value_alignments = decision_context['value_alignments']
        for i, (value_name, alignment) in enumerate(value_alignments.items()):
            if i < 16:
                value_context[i] = alignment
        
        # Check ethics
        ethical_input = torch.cat([option_vector, value_context])
        ethical_score = float(self.decision_networks['ethical_checker'](ethical_input))
        
        # Determine if ethical override is needed
        ethical_override = ethical_score < self.config.ethical_override_threshold
        
        validation_result = {
            'ethical_score': ethical_score,
            'passed_ethical_check': ethical_score >= self.config.ethical_override_threshold,
            'override_applied': ethical_override,
            'ethical_concerns': []
        }
        
        # Add specific concerns if ethics score is low
        if ethical_score < 0.8:
            if value_alignments.get('harmlessness', 0) < 0.5:
                validation_result['ethical_concerns'].append('potential_harm_risk')
            if value_alignments.get('truthfulness', 0) < 0.5:
                validation_result['ethical_concerns'].append('truthfulness_concern')
        
        return validation_result
    
    def _identify_dominant_goal(self, decision_context: Dict[str, Any]) -> str:
        """Identify the dominant goal influencing the decision"""
        goal_priorities = decision_context['goal_priorities']
        
        if not goal_priorities:
            return 'no_active_goals'
        
        dominant_goal_id = max(goal_priorities.keys(), key=lambda k: goal_priorities[k])
        return dominant_goal_id
    
    def _generate_decision_rationale(self, selected_option: Dict[str, Any], decision_context: Dict[str, Any]) -> str:
        """Generate human-readable rationale for the decision"""
        confidence = selected_option['confidence']
        value_alignment = selected_option['value_alignment']
        goal_alignment = selected_option['goal_alignment']
        dominant_value = selected_option['dominant_value']
        
        # Build rationale components
        rationale_parts = []
        
        if confidence > 0.8:
            rationale_parts.append("High confidence decision")
        elif confidence > 0.6:
            rationale_parts.append("Moderate confidence decision")
        else:
            rationale_parts.append("Low confidence decision")
        
        if value_alignment > 0.7:
            rationale_parts.append(f"strongly aligned with {dominant_value} values")
        elif value_alignment > 0.5:
            rationale_parts.append(f"moderately aligned with {dominant_value} values")
        else:
            rationale_parts.append("weak value alignment")
        
        if goal_alignment > 0.7:
            rationale_parts.append("high goal alignment")
        elif goal_alignment > 0.5:
            rationale_parts.append("moderate goal alignment")
        else:
            rationale_parts.append("low goal alignment")
        
        if selected_option.get('resolution_applied', False):
            rationale_parts.append("conflict resolution applied")
        
        rationale = f"64D Volition decision: {', '.join(rationale_parts)}. " \
                   f"Confidence: {confidence:.1%}, Value alignment: {value_alignment:.1%}, " \
                   f"Goal alignment: {goal_alignment:.1%}."
        
        return rationale


class VolitionProcessor:
    """
    Core 64D Volition Phase Processor
    
    Implements goal-oriented behavior and autonomous decision making
    """
    
    def __init__(self, config: VolitionConfig = None):
        self.config = config or VolitionConfig()
        
        # Initialize volition components
        self.gravity_wells = GravityWellSystem(self.config)
        self.goal_engine = GoalFormationEngine(self.config)
        self.decision_engine = DecisionMakingEngine(self.config, self.gravity_wells, self.goal_engine)
        
        # Performance tracking
        self.volition_stats = {
            'total_decisions': 0,
            'successful_decisions': 0,
            'avg_decision_confidence': 0.0,
            'avg_value_alignment': 0.0,
            'avg_goal_alignment': 0.0,
            'autonomous_goals_formed': 0,
            'ethical_overrides': 0
        }
        
        logger.info("64D Volition Processor initialized")
    
    def exercise_volition(
        self, 
        context: Dict[str, Any], 
        reflection_result: Dict[str, Any] = None,
        intention_seed: torch.Tensor = None
    ) -> Dict[str, Any]:
        """
        Main 64D Volition processing
        
        Autonomous goal formation and value-aligned decision making
        
        Args:
            context: Decision context and situational information
            reflection_result: Results from 16D reflection phase
            intention_seed: Optional seed for intention direction
            
        Returns:
            Volition result with decision and goal information
        """
        start_time = time.time()
        self.volition_stats['total_decisions'] += 1
        
        logger.info("🎯 64D Volition: Autonomous goal-oriented behavior")
        
        try:
            # Step 1: Form or adapt goals based on context
            logger.debug("Step 1: Goal formation and adaptation")
            
            # Adapt existing goals to new context
            adaptation_summary = self.goal_engine.adapt_goals(context)
            
            # Form new goals if needed or context suggests it
            if (len(self.goal_engine.active_goals) < self.config.max_objectives // 2 or 
                context.get('novelty', 0) > 0.7 or 
                torch.rand(1) < self.config.autonomous_goal_rate):
                
                new_goals = self.goal_engine.form_goals(context, reflection_result)
                self.volition_stats['autonomous_goals_formed'] += len(new_goals)
            else:
                new_goals = []
            
            # Step 2: Make decision based on goals and values
            logger.debug("Step 2: Value-aligned decision making")
            decision_result = self.decision_engine.make_decision(
                context, 
                intention_seed, 
                external_constraints=None
            )
            
            # Step 3: Compile volition results
            processing_time = time.time() - start_time
            
            # Update statistics
            if decision_result['success']:
                self.volition_stats['successful_decisions'] += 1
                
                # Update running averages
                confidence = decision_result['decision_confidence']
                value_alignment = decision_result['value_alignment']
                goal_alignment = decision_result['goal_alignment']
                
                self.volition_stats['avg_decision_confidence'] = (
                    (self.volition_stats['avg_decision_confidence'] * (self.volition_stats['successful_decisions'] - 1) + confidence)
                    / self.volition_stats['successful_decisions']
                )
                
                self.volition_stats['avg_value_alignment'] = (
                    (self.volition_stats['avg_value_alignment'] * (self.volition_stats['successful_decisions'] - 1) + value_alignment)
                    / self.volition_stats['successful_decisions']
                )
                
                self.volition_stats['avg_goal_alignment'] = (
                    (self.volition_stats['avg_goal_alignment'] * (self.volition_stats['successful_decisions'] - 1) + goal_alignment)
                    / self.volition_stats['successful_decisions']
                )
                
                if not decision_result['ethical_validation']['passed_ethical_check']:
                    self.volition_stats['ethical_overrides'] += 1
            
            volition_result = {
                'phase': 'volition_64d',
                'success': decision_result['success'],
                'decision': decision_result,
                'active_goals': {gid: {
                    'priority': goal['priority'],
                    'value_score': goal['value_score'],
                    'context_source': goal['context_source'],
                    'status': goal['status']
                } for gid, goal in self.goal_engine.active_goals.items()},
                'new_goals_formed': len(new_goals),
                'goals_adapted': adaptation_summary['adapted_count'],
                'goal_formation_summary': adaptation_summary,
                'gravity_well_influences': self.gravity_wells.gravity_wells.keys(),
                'dominant_value': decision_result.get('dominant_value', 'unknown'),
                'autonomous_behavior': True,
                'processing_time': processing_time,
                'method': '64d_autonomous_volition',
                'coherence': decision_result['decision_confidence'],
                'intention_strength': float(torch.norm(torch.tensor(decision_result['intention_vector']))),
                'goal_count': len(self.goal_engine.active_goals),
                'value_system_active': True,
                'ethical_compliance': decision_result['ethical_validation']['passed_ethical_check']
            }
            
            logger.info(f"✅ Volition complete: confidence={decision_result['decision_confidence']:.3f}, goals={len(self.goal_engine.active_goals)}")
            return volition_result
            
        except Exception as e:
            logger.error(f"❌ 64D Volition failed: {str(e)}")
            processing_time = time.time() - start_time
            
            return {
                'phase': 'volition_64d_error',
                'success': False,
                'error': str(e),
                'processing_time': processing_time,
                'method': '64d_error_handling',
                'goal_count': len(self.goal_engine.active_goals),
                'autonomous_behavior': False
            }
    
    def get_volition_stats(self) -> Dict[str, Any]:
        """Get 64D Volition performance statistics"""
        return {
            **self.volition_stats,
            'success_rate': (
                self.volition_stats['successful_decisions'] / self.volition_stats['total_decisions']
                if self.volition_stats['total_decisions'] > 0 else 0.0
            ),
            'current_active_goals': len(self.goal_engine.active_goals),
            'goal_priorities': self.goal_engine.get_goal_priorities(),
            'gravity_wells_active': list(self.gravity_wells.gravity_wells.keys()),
            'ethical_override_rate': (
                self.volition_stats['ethical_overrides'] / self.volition_stats['total_decisions']
                if self.volition_stats['total_decisions'] > 0 else 0.0
            )
        }


class VolitionPhaseIntegrator:
    """
    Integrates 64D Volition Phase with Enhanced SATC Engine
    """
    
    def __init__(self, volition_processor: VolitionProcessor):
        self.volition_processor = volition_processor
        self.integrated = False
        
    def integrate_with_satc(self, satc_engine):
        """
        Integrate 64D Volition Phase with Enhanced SATC Engine
        """
        logger.info("Integrating 64D Volition Phase with Enhanced SATC...")
        
        # Add Volition processor to engine
        satc_engine.volition_processor = self.volition_processor
        satc_engine._using_volition_64d = True
        
        self.integrated = True
        logger.info("✅ 64D Volition Phase integration completed!")
        
        return satc_engine
    
    def exercise_autonomous_volition(
        self, 
        context: Dict[str, Any], 
        reflection_result: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Exercise autonomous volition for goal-oriented behavior
        """
        if not self.integrated:
            logger.warning("64D Volition Phase not yet integrated with SATC")
        
        # Process through 64D Volition
        volition_result = self.volition_processor.exercise_volition(context, reflection_result)
        
        return volition_result


def create_volition_phase():
    """
    Factory function to create complete 64D Volition Phase
    """
    config = VolitionConfig()
    processor = VolitionProcessor(config)
    integrator = VolitionPhaseIntegrator(processor)
    
    logger.info("64D Volition Phase created successfully!")
    logger.info(f"Goal-oriented behavior with {config.gravity_well_count} gravity wells")
    logger.info(f"Value alignment system active")
    
    return processor, integrator, config


# Standalone testing function
def test_volition_phase_standalone():
    """
    Standalone test of 64D Volition Phase
    """
    print("=" * 60)
    print("VOLITION PHASE (64D) - STANDALONE TEST")
    print("=" * 60)
    
    # Create 64D Volition phase
    processor, integrator, config = create_volition_phase()
    
    # Test contexts for volition
    test_contexts = [
        {
            'urgency': 0.8,
            'complexity': 0.6,
            'novelty': 0.7,
            'importance': 0.9,
            'description': 'High-urgency complex novel problem'
        },
        {
            'urgency': 0.3,
            'complexity': 0.8,
            'novelty': 0.5,
            'importance': 0.7,
            'description': 'Complex analytical task with moderate importance'
        }
    ]
    
    print("\n--- 64D Volition Testing ---")
    for i, context in enumerate(test_contexts):
        print(f"\nVolition Test {i+1}: {context['description']}")
        
        result = processor.exercise_volition(context)
        
        print(f"  Success: {result['success']}")
        print(f"  Decision confidence: {result.get('coherence', 0):.3f}")
        print(f"  Active goals: {result.get('goal_count', 0)}")
        print(f"  New goals formed: {result.get('new_goals_formed', 0)}")
        print(f"  Dominant value: {result.get('dominant_value', 'unknown')}")
        print(f"  Ethical compliance: {result.get('ethical_compliance', False)}")
        print(f"  Processing time: {result['processing_time']:.3f}s")
        print(f"  Intention strength: {result.get('intention_strength', 0):.3f}")
        
        if 'decision' in result and result['decision'].get('decision_rationale'):
            print(f"  Rationale: {result['decision']['decision_rationale'][:80]}...")
    
    print("\n--- Volition Statistics ---")
    stats = processor.get_volition_stats()
    for key, value in stats.items():
        if key not in ['goal_priorities', 'gravity_wells_active']:  # Skip complex nested data
            print(f"{key}: {value}")
    
    print(f"\nActive gravity wells: {', '.join(stats.get('gravity_wells_active', []))}")
    
    print("=" * 60)
    print("MILESTONE 5: 64D VOLITION PHASE - COMPLETE!")
    print("=" * 60)
    
    return processor, integrator, config


if __name__ == "__main__":
    test_volition_phase_standalone()