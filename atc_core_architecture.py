"""
ATC Core Architecture - Recognition and Cognition Phases
=======================================================

This module implements the dual-phase architecture of the ATC model:
- Recognition Phase (System 1 - Fast)
- Cognition Phase (System 2 - Slow)

Integrates with the Brain Wiggle Engine for deep semantic processing.

Author: ATC Model Creator
Status: Core Implementation Framework
"""

import numpy as np
import torch
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import logging

from brain_wiggle_implementation import BrainWiggleEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProcessingPhase(Enum):
    """Enumeration of processing phases"""
    RECOGNITION = "recognition"
    COGNITION = "cognition"
    COMPLETE = "complete"

@dataclass
class ProcessingResult:
    """Result of ATC processing"""
    output: np.ndarray
    phase_used: ProcessingPhase
    processing_time: float
    confidence: float
    coherence_score: float
    success: bool
    metadata: Dict[str, Any] = field(default_factory=dict)

class Element(ABC):
    """Abstract base class for the six ATC elements"""
    
    def __init__(self, name: str):
        self.name = name
        self.usage_count = 0
        self.success_rate = 0.0
        
    @abstractmethod
    def process(self, input_data: Any, context: Dict[str, Any]) -> Any:
        """Process input according to element's function"""
        pass
    
    def update_metrics(self, success: bool):
        """Update element performance metrics"""
        self.usage_count += 1
        self.success_rate = (self.success_rate * (self.usage_count - 1) + (1 if success else 0)) / self.usage_count

class ObservationElement(Element):
    """Observation Element - Perceive and parse input"""
    
    def __init__(self):
        super().__init__("Observation")
        
    def process(self, input_data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features, facts, conditions, and goals from input"""
        logger.debug(f"Processing observation: {type(input_data)}")
        
        # Convert input to semantic vector if needed
        if isinstance(input_data, str):
            # In real implementation, this would use sentence embeddings
            semantic_vector = np.random.randn(12)  # Placeholder
        elif isinstance(input_data, np.ndarray):
            semantic_vector = input_data
        else:
            semantic_vector = np.array(input_data)
        
        # Extract features
        features = {
            'semantic_vector': semantic_vector,
            'input_type': type(input_data).__name__,
            'timestamp': time.time(),
            'feature_count': len(semantic_vector),
            'magnitude': np.linalg.norm(semantic_vector),
            'dominant_dimensions': np.argsort(np.abs(semantic_vector))[-3:]  # Top 3 dimensions
        }
        
        logger.debug(f"Extracted {len(features)} features")
        return features

class ExperienceElement(Element):
    """Experience Element - Recall similar past scenarios"""
    
    def __init__(self):
        super().__init__("Experience")
        self.memory_bank = []  # List of past experiences
        self.similarity_threshold = 0.7
        
    def process(self, input_data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """Find similar past experiences"""
        logger.debug(f"Searching experience for similar patterns")
        
        if 'semantic_vector' in context:
            query_vector = context['semantic_vector']
        else:
            query_vector = np.random.randn(12)  # Fallback
        
        # Search memory bank for similar experiences
        similar_experiences = []
        for experience in self.memory_bank:
            similarity = np.dot(query_vector, experience['vector'])
            if similarity > self.similarity_threshold:
                similar_experiences.append({
                    'experience': experience,
                    'similarity': similarity
                })
        
        # Sort by similarity
        similar_experiences.sort(key=lambda x: x['similarity'], reverse=True)
        
        result = {
            'similar_experiences': similar_experiences[:5],  # Top 5
            'total_found': len(similar_experiences),
            'memory_bank_size': len(self.memory_bank)
        }
        
        logger.debug(f"Found {len(similar_experiences)} similar experiences")
        return result
    
    def add_experience(self, vector: np.ndarray, outcome: str, success: bool):
        """Add new experience to memory bank"""
        experience = {
            'vector': vector,
            'outcome': outcome,
            'success': success,
            'timestamp': time.time()
        }
        self.memory_bank.append(experience)
        
        # Limit memory bank size
        if len(self.memory_bank) > 1000:
            self.memory_bank.pop(0)

class KnowledgeElement(Element):
    """Knowledge Element - Retrieve established facts and rules"""
    
    def __init__(self):
        super().__init__("Knowledge")
        self.knowledge_base = {}  # Dictionary of facts and rules
        
    def process(self, input_data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve relevant knowledge"""
        logger.debug(f"Querying knowledge base")
        
        # Query knowledge base (simplified)
        relevant_knowledge = []
        
        # In real implementation, this would use semantic search
        for key, value in self.knowledge_base.items():
            if isinstance(value, dict) and 'vector' in value:
                # Check semantic similarity
                if 'semantic_vector' in context:
                    similarity = np.dot(context['semantic_vector'], value['vector'])
                    if similarity > 0.5:
                        relevant_knowledge.append({
                            'key': key,
                            'value': value,
                            'similarity': similarity
                        })
        
        result = {
            'relevant_knowledge': relevant_knowledge,
            'knowledge_base_size': len(self.knowledge_base)
        }
        
        logger.debug(f"Found {len(relevant_knowledge)} relevant knowledge items")
        return result
    
    def add_knowledge(self, key: str, value: Any, vector: np.ndarray):
        """Add new knowledge to base"""
        self.knowledge_base[key] = {
            'value': value,
            'vector': vector,
            'timestamp': time.time()
        }

class UnderstandingElement(Element):
    """Understanding Element - Synthesize information into conceptual model"""
    
    def __init__(self):
        super().__init__("Understanding")
        
    def process(self, input_data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create conceptual model from available information"""
        logger.debug(f"Synthesizing understanding")
        
        # Gather all available information
        observations = context.get('observations', {})
        experiences = context.get('experiences', {})
        knowledge = context.get('knowledge', {})
        
        # Create conceptual model (simplified)
        conceptual_model = {
            'primary_concept': self._extract_primary_concept(observations),
            'related_concepts': self._find_related_concepts(experiences, knowledge),
            'causal_relationships': self._identify_causal_relationships(context),
            'confidence_level': self._calculate_confidence(context)
        }
        
        logger.debug(f"Created conceptual model with {len(conceptual_model)} components")
        return conceptual_model
    
    def _extract_primary_concept(self, observations: Dict) -> str:
        """Extract the primary concept from observations"""
        # Simplified concept extraction
        if 'semantic_vector' in observations:
            dominant_dim = np.argmax(np.abs(observations['semantic_vector']))
            return f"concept_{dominant_dim}"
        return "unknown_concept"
    
    def _find_related_concepts(self, experiences: Dict, knowledge: Dict) -> List[str]:
        """Find concepts related to primary concept"""
        related = []
        
        # From experiences
        for exp in experiences.get('similar_experiences', []):
            if 'experience' in exp:
                related.append(f"exp_{exp['experience'].get('outcome', 'unknown')}")
        
        # From knowledge
        for item in knowledge.get('relevant_knowledge', []):
            related.append(f"know_{item['key']}")
        
        return related[:10]  # Limit to top 10
    
    def _identify_causal_relationships(self, context: Dict) -> List[Tuple[str, str]]:
        """Identify causal relationships"""
        # Simplified causal analysis
        relationships = []
        
        # This would be much more sophisticated in real implementation
        experiences = context.get('experiences', {})
        for exp in experiences.get('similar_experiences', []):
            if 'experience' in exp:
                outcome = exp['experience'].get('outcome', 'unknown')
                relationships.append(('input', outcome))
        
        return relationships
    
    def _calculate_confidence(self, context: Dict) -> float:
        """Calculate confidence in understanding"""
        # Simplified confidence calculation
        exp_count = len(context.get('experiences', {}).get('similar_experiences', []))
        know_count = len(context.get('knowledge', {}).get('relevant_knowledge', []))
        
        confidence = min(1.0, (exp_count * 0.3 + know_count * 0.7) / 5.0)
        return confidence

class ExperimentationElement(Element):
    """Experimentation Element - Test hypotheses via simulations"""
    
    def __init__(self):
        super().__init__("Experimentation")
        
    def process(self, input_data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """Test hypotheses through simulation"""
        logger.debug(f"Running experimentation")
        
        understanding = context.get('understanding', {})
        hypotheses = self._generate_hypotheses(understanding)
        
        # Test each hypothesis
        experiment_results = []
        for hypothesis in hypotheses:
            result = self._test_hypothesis(hypothesis, context)
            experiment_results.append(result)
        
        # Rank results by success probability
        experiment_results.sort(key=lambda x: x['success_probability'], reverse=True)
        
        result = {
            'hypotheses_tested': len(hypotheses),
            'experiment_results': experiment_results,
            'best_hypothesis': experiment_results[0] if experiment_results else None
        }
        
        logger.debug(f"Tested {len(hypotheses)} hypotheses")
        return result
    
    def _generate_hypotheses(self, understanding: Dict) -> List[Dict]:
        """Generate hypotheses to test"""
        hypotheses = []
        
        primary_concept = understanding.get('primary_concept', 'unknown')
        related_concepts = understanding.get('related_concepts', [])
        
        # Generate simple hypotheses
        for i, concept in enumerate(related_concepts[:3]):  # Top 3 concepts
            hypothesis = {
                'id': f"hyp_{i}",
                'description': f"Apply {concept} to {primary_concept}",
                'concept': concept,
                'probability': 0.5 + i * 0.1  # Simple probability assignment
            }
            hypotheses.append(hypothesis)
        
        return hypotheses
    
    def _test_hypothesis(self, hypothesis: Dict, context: Dict) -> Dict:
        """Test a single hypothesis"""
        # Simplified hypothesis testing
        # In real implementation, this would run actual simulations
        
        base_probability = hypothesis.get('probability', 0.5)
        
        # Adjust probability based on context
        experiences = context.get('experiences', {})
        similar_count = len(experiences.get('similar_experiences', []))
        
        adjusted_probability = min(1.0, base_probability + similar_count * 0.1)
        
        # Simulate test result
        success = np.random.random() < adjusted_probability
        
        return {
            'hypothesis': hypothesis,
            'success': success,
            'success_probability': adjusted_probability,
            'evidence': f"Tested with {similar_count} similar experiences"
        }

class ProcedureElement(Element):
    """Procedure Element - Formulate and execute step-by-step plan"""
    
    def __init__(self):
        super().__init__("Procedure")
        
    def process(self, input_data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create and execute procedure"""
        logger.debug(f"Creating procedure")
        
        # Gather information from previous elements
        understanding = context.get('understanding', {})
        experiments = context.get('experiments', {})
        
        # Create procedure based on best experimental result
        best_hypothesis = experiments.get('best_hypothesis')
        if best_hypothesis:
            procedure = self._create_procedure_from_hypothesis(best_hypothesis)
        else:
            procedure = self._create_default_procedure(understanding)
        
        # Execute procedure
        execution_result = self._execute_procedure(procedure, context)
        
        result = {
            'procedure': procedure,
            'execution_result': execution_result,
            'success': execution_result.get('success', False)
        }
        
        logger.debug(f"Created and executed procedure with {len(procedure.get('steps', []))} steps")
        return result
    
    def _create_procedure_from_hypothesis(self, hypothesis: Dict) -> Dict:
        """Create procedure from successful hypothesis"""
        steps = [
            f"Apply {hypothesis['hypothesis']['concept']}",
            "Monitor results",
            "Adjust if necessary",
            "Verify outcome"
        ]
        
        return {
            'steps': steps,
            'based_on': hypothesis['hypothesis']['id'],
            'confidence': hypothesis['success_probability']
        }
    
    def _create_default_procedure(self, understanding: Dict) -> Dict:
        """Create default procedure when no hypothesis available"""
        primary_concept = understanding.get('primary_concept', 'unknown')
        
        steps = [
            f"Process {primary_concept}",
            "Apply standard approach",
            "Generate output",
            "Verify result"
        ]
        
        return {
            'steps': steps,
            'based_on': 'default_reasoning',
            'confidence': 0.5
        }
    
    def _execute_procedure(self, procedure: Dict, context: Dict) -> Dict:
        """Execute the procedure"""
        # Simplified execution
        steps = procedure.get('steps', [])
        
        execution_log = []
        for i, step in enumerate(steps):
            execution_log.append(f"Step {i+1}: {step} - Completed")
        
        return {
            'steps_executed': len(steps),
            'execution_log': execution_log,
            'success': True,
            'output': f"Procedure completed with {len(steps)} steps"
        }

class RecognitionPhase:
    """Recognition Phase - Fast processing for familiar patterns"""
    
    def __init__(self):
        self.observation = ObservationElement()
        self.experience = ExperienceElement()
        self.knowledge = KnowledgeElement()
        self.understanding = UnderstandingElement()
        self.procedure = ProcedureElement()
        
        self.similarity_threshold = 0.8
        
    def process(self, input_data: Any) -> Optional[ProcessingResult]:
        """Process input through recognition phase"""
        start_time = time.time()
        logger.info("Starting Recognition Phase")
        
        try:
            # Step 1: Observation
            observations = self.observation.process(input_data, {})
            
            # Step 2: Experience lookup
            experiences = self.experience.process(input_data, observations)
            
            # Step 3: Knowledge query
            knowledge = self.knowledge.process(input_data, observations)
            
            # Step 4: Quick understanding check
            context = {
                'observations': observations,
                'experiences': experiences,
                'knowledge': knowledge
            }
            understanding = self.understanding.process(input_data, context)
            
            # Decision: Can we handle this with existing patterns?
            confidence = understanding.get('confidence_level', 0.0)
            similar_experiences = experiences.get('similar_experiences', [])
            
            if confidence > self.similarity_threshold and similar_experiences:
                # Step 5: Execute quick procedure
                context['understanding'] = understanding
                procedure_result = self.procedure.process(input_data, context)
                
                processing_time = time.time() - start_time
                
                # Create result
                result = ProcessingResult(
                    output=np.array([1.0]),  # Placeholder output
                    phase_used=ProcessingPhase.RECOGNITION,
                    processing_time=processing_time,
                    confidence=confidence,
                    coherence_score=0.9,  # High coherence for recognized patterns
                    success=procedure_result.get('success', False),
                    metadata={
                        'similar_experiences': len(similar_experiences),
                        'procedure_steps': len(procedure_result.get('procedure', {}).get('steps', []))
                    }
                )
                
                logger.info(f"Recognition Phase completed in {processing_time:.4f}s")
                return result
            else:
                logger.info("Recognition Phase: Pattern not recognized, escalating to Cognition")
                return None
                
        except Exception as e:
            logger.error(f"Recognition Phase failed: {str(e)}")
            return None

class CognitionPhase:
    """Cognition Phase - Deliberate processing for novel inputs"""
    
    def __init__(self):
        self.observation = ObservationElement()
        self.experience = ExperienceElement()
        self.knowledge = KnowledgeElement()
        self.understanding = UnderstandingElement()
        self.experimentation = ExperimentationElement()
        self.procedure = ProcedureElement()
        
        # Initialize Brain Wiggle Engine
        self.brain_wiggle = BrainWiggleEngine()
        
    def process(self, input_data: Any) -> ProcessingResult:
        """Process input through cognition phase"""
        start_time = time.time()
        logger.info("Starting Cognition Phase")
        
        try:
            # Step 1: Detailed observation
            observations = self.observation.process(input_data, {})
            
            # Step 2: Experience analysis
            experiences = self.experience.process(input_data, observations)
            
            # Step 3: Knowledge retrieval
            knowledge = self.knowledge.process(input_data, observations)
            
            # Step 4: Deep understanding
            context = {
                'observations': observations,
                'experiences': experiences,
                'knowledge': knowledge
            }
            understanding = self.understanding.process(input_data, context)
            
            # Step 5: Experimentation
            context['understanding'] = understanding
            experiments = self.experimentation.process(input_data, context)
            
            # Step 6: Brain Wiggle for deep semantic processing
            semantic_vector = observations.get('semantic_vector', np.random.randn(12))
            wiggle_output, coherence_score, wiggle_success = self.brain_wiggle.wiggle(semantic_vector)
            
            # Step 7: Final procedure synthesis
            context['experiments'] = experiments
            context['brain_wiggle_output'] = wiggle_output
            procedure_result = self.procedure.process(input_data, context)
            
            processing_time = time.time() - start_time
            
            # Create result
            result = ProcessingResult(
                output=wiggle_output,
                phase_used=ProcessingPhase.COGNITION,
                processing_time=processing_time,
                confidence=understanding.get('confidence_level', 0.0),
                coherence_score=coherence_score,
                success=wiggle_success and procedure_result.get('success', False),
                metadata={
                    'hypotheses_tested': experiments.get('hypotheses_tested', 0),
                    'brain_wiggle_success': wiggle_success,
                    'procedure_steps': len(procedure_result.get('procedure', {}).get('steps', []))
                }
            )
            
            logger.info(f"Cognition Phase completed in {processing_time:.4f}s")
            return result
            
        except Exception as e:
            logger.error(f"Cognition Phase failed: {str(e)}")
            
            # Return error result
            return ProcessingResult(
                output=np.zeros(96),
                phase_used=ProcessingPhase.COGNITION,
                processing_time=time.time() - start_time,
                confidence=0.0,
                coherence_score=0.0,
                success=False,
                metadata={'error': str(e)}
            )

class ATCCoreEngine:
    """
    ATC Core Engine - Main orchestrator for the dual-phase architecture
    """
    
    def __init__(self):
        self.recognition_phase = RecognitionPhase()
        self.cognition_phase = CognitionPhase()
        
        # Performance tracking
        self.processing_history = []
        self.performance_metrics = {
            'recognition_success_rate': 0.0,
            'cognition_success_rate': 0.0,
            'average_processing_time': 0.0,
            'total_processed': 0
        }
        
        logger.info("ATC Core Engine initialized")
    
    def process(self, input_data: Any) -> ProcessingResult:
        """
        Main processing method - orchestrates Recognition and Cognition phases
        """
        logger.info("Starting ATC processing")
        
        # Reset Brain Wiggle iteration counter
        self.cognition_phase.brain_wiggle.reset_iteration_counter()
        
        # Try Recognition Phase first
        recognition_result = self.recognition_phase.process(input_data)
        
        if recognition_result is not None and recognition_result.success:
            # Recognition succeeded
            self._update_performance_metrics(recognition_result)
            self._add_to_history(recognition_result)
            return recognition_result
        else:
            # Recognition failed, proceed to Cognition
            logger.info("Recognition failed, proceeding to Cognition Phase")
            cognition_result = self.cognition_phase.process(input_data)
            
            self._update_performance_metrics(cognition_result)
            self._add_to_history(cognition_result)
            
            # Learn from cognition result
            self._learn_from_result(input_data, cognition_result)
            
            return cognition_result
    
    def _update_performance_metrics(self, result: ProcessingResult):
        """Update performance metrics"""
        self.performance_metrics['total_processed'] += 1
        
        if result.phase_used == ProcessingPhase.RECOGNITION:
            current_rate = self.performance_metrics['recognition_success_rate']
            total = self.performance_metrics['total_processed']
            self.performance_metrics['recognition_success_rate'] = (
                (current_rate * (total - 1) + (1 if result.success else 0)) / total
            )
        elif result.phase_used == ProcessingPhase.COGNITION:
            current_rate = self.performance_metrics['cognition_success_rate']
            total = self.performance_metrics['total_processed']
            self.performance_metrics['cognition_success_rate'] = (
                (current_rate * (total - 1) + (1 if result.success else 0)) / total
            )
        
        # Update average processing time
        current_avg = self.performance_metrics['average_processing_time']
        total = self.performance_metrics['total_processed']
        self.performance_metrics['average_processing_time'] = (
            (current_avg * (total - 1) + result.processing_time) / total
        )
    
    def _add_to_history(self, result: ProcessingResult):
        """Add result to processing history"""
        self.processing_history.append({
            'timestamp': time.time(),
            'phase': result.phase_used,
            'success': result.success,
            'processing_time': result.processing_time,
            'coherence_score': result.coherence_score,
            'confidence': result.confidence
        })
        
        # Limit history size
        if len(self.processing_history) > 1000:
            self.processing_history.pop(0)
    
    def _learn_from_result(self, input_data: Any, result: ProcessingResult):
        """Learn from processing result to improve future performance"""
        if result.success:
            # Add successful experience to memory
            if hasattr(self.recognition_phase.experience, 'add_experience'):
                self.recognition_phase.experience.add_experience(
                    vector=result.output[:12] if len(result.output) >= 12 else result.output,
                    outcome="success",
                    success=True
                )
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        return {
            'metrics': self.performance_metrics,
            'recent_history': self.processing_history[-10:],  # Last 10 results
            'total_history_size': len(self.processing_history)
        }

# Example usage and testing
if __name__ == "__main__":
    # Initialize ATC Core Engine
    atc_engine = ATCCoreEngine()
    
    # Test with various inputs
    test_inputs = [
        "Hello, can you help me understand quantum computing?",
        np.random.randn(12),
        {"type": "problem", "data": "solve equation x^2 + 5x + 6 = 0"},
        "What is the meaning of life?"
    ]
    
    for i, test_input in enumerate(test_inputs):
        print(f"\n--- Test {i+1} ---")
        print(f"Input: {test_input}")
        
        result = atc_engine.process(test_input)
        
        print(f"Phase used: {result.phase_used.value}")
        print(f"Success: {result.success}")
        print(f"Processing time: {result.processing_time:.4f}s")
        print(f"Confidence: {result.confidence:.4f}")
        print(f"Coherence score: {result.coherence_score:.4f}")
        print(f"Output shape: {result.output.shape}")
    
    # Get performance report
    print("\n--- Performance Report ---")
    report = atc_engine.get_performance_report()
    print(f"Total processed: {report['metrics']['total_processed']}")
    print(f"Recognition success rate: {report['metrics']['recognition_success_rate']:.4f}")
    print(f"Cognition success rate: {report['metrics']['cognition_success_rate']:.4f}")
    print(f"Average processing time: {report['metrics']['average_processing_time']:.4f}s")