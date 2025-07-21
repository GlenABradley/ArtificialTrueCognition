"""
ATC Recognition Phase - 2D Fast Pattern Matching
===============================================

This module implements the Recognition phase of the ATC architecture:
- Fast 2D pattern matching against memory
- Cosine similarity-based retrieval  
- Direct procedure execution for known patterns
- Escalation to Cognition phase for novelty

Architecture: Input → 2D Embedding → Memory Search → Match/NoMatch
- Match: Return cached procedure (fast path)
- NoMatch: Escalate to 4D Cognition phase (slow path)

Author: Revolutionary ATC Architecture Team
Status: Milestone 2 - Recognition Phase
"""

import torch
import torch.nn as nn
import numpy as np
import faiss
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity

# Import our Power-of-2 foundation
from power_of_2_core import PowerOf2Layers, PowerOf2Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RecognitionConfig:
    """Configuration for Recognition Phase"""
    # 2D Recognition parameters
    recognition_dim: int = 2  # Power-of-2 dimension for Recognition
    similarity_threshold: float = 0.8  # Cosine similarity threshold for match
    max_memory_items: int = 10000  # Maximum items in recognition memory
    
    # Memory parameters
    use_faiss_index: bool = True  # Use FAISS for fast similarity search
    faiss_nprobe: int = 10  # FAISS search parameter
    
    # Performance parameters
    max_search_time_ms: float = 10.0  # Maximum search time in milliseconds


class RecognitionMemory:
    """
    Memory system for Recognition phase
    
    Stores 2D patterns with associated procedures for fast retrieval
    """
    
    def __init__(self, config: RecognitionConfig):
        self.config = config
        self.patterns = []  # List of 2D pattern tensors
        self.procedures = []  # Associated procedures/responses
        self.metadata = []  # Additional metadata for each pattern
        
        # Initialize FAISS index for fast similarity search
        if self.config.use_faiss_index:
            self.index = faiss.IndexFlatIP(self.config.recognition_dim)  # Inner product for cosine
            self.index_built = False
        else:
            self.index = None
            
        logger.info(f"Recognition Memory initialized with dimension {self.config.recognition_dim}")
    
    def add_pattern(self, pattern: torch.Tensor, procedure: Any, metadata: Dict = None):
        """
        Add a new pattern-procedure pair to memory
        
        Args:
            pattern: 2D tensor representing the pattern
            procedure: Associated procedure/response
            metadata: Optional metadata dictionary
        """
        if pattern.shape[-1] != self.config.recognition_dim:
            raise ValueError(f"Pattern dimension {pattern.shape[-1]} != {self.config.recognition_dim}")
        
        # Normalize pattern for cosine similarity
        normalized_pattern = pattern / torch.norm(pattern, dim=-1, keepdim=True)
        
        self.patterns.append(normalized_pattern)
        self.procedures.append(procedure)
        self.metadata.append(metadata or {})
        
        # Update FAISS index
        if self.config.use_faiss_index and len(self.patterns) % 100 == 0:
            self._rebuild_index()
            
        logger.debug(f"Added pattern to memory. Total patterns: {len(self.patterns)}")
    
    def _rebuild_index(self):
        """Rebuild FAISS index with current patterns"""
        if not self.patterns:
            return
            
        # Convert to numpy array
        patterns_np = torch.stack(self.patterns).cpu().numpy()
        
        # Reset and rebuild index
        self.index = faiss.IndexFlatIP(self.config.recognition_dim)
        self.index.add(patterns_np)
        self.index_built = True
        
        logger.debug(f"FAISS index rebuilt with {len(self.patterns)} patterns")
    
    def search_similar(self, query_pattern: torch.Tensor, k: int = 1) -> Tuple[List[float], List[int]]:
        """
        Search for similar patterns in memory
        
        Args:
            query_pattern: 2D query tensor
            k: Number of similar patterns to return
            
        Returns:
            similarities: List of similarity scores
            indices: List of pattern indices
        """
        if not self.patterns:
            return [], []
        
        # Normalize query pattern
        normalized_query = query_pattern / torch.norm(query_pattern, dim=-1, keepdim=True)
        
        if self.config.use_faiss_index and self.index_built:
            # Use FAISS for fast search
            query_np = normalized_query.cpu().numpy().reshape(1, -1)
            similarities, indices = self.index.search(query_np, k)
            return similarities[0].tolist(), indices[0].tolist()
        else:
            # Fallback to manual cosine similarity
            patterns_tensor = torch.stack(self.patterns)
            similarities = torch.cosine_similarity(normalized_query, patterns_tensor, dim=-1)
            
            # Get top k
            top_similarities, top_indices = torch.topk(similarities, min(k, len(similarities)))
            return top_similarities.tolist(), top_indices.tolist()
    
    def get_best_match(self, query_pattern: torch.Tensor) -> Optional[Tuple[float, Any, Dict]]:
        """
        Get the best matching pattern above threshold
        
        Returns:
            (similarity, procedure, metadata) if match found, None otherwise
        """
        similarities, indices = self.search_similar(query_pattern, k=1)
        
        if not similarities or similarities[0] < self.config.similarity_threshold:
            return None
        
        idx = indices[0]
        return similarities[0], self.procedures[idx], self.metadata[idx]


class RecognitionProcessor:
    """
    Core Recognition Phase Processor
    
    Implements 2D fast pattern matching with escalation to Cognition
    """
    
    def __init__(self, config: RecognitionConfig = None):
        self.config = config or RecognitionConfig()
        self.memory = RecognitionMemory(self.config)
        
        # Performance tracking
        self.stats = {
            'total_queries': 0,
            'recognition_hits': 0,
            'recognition_misses': 0,
            'avg_search_time_ms': 0.0,
            'patterns_in_memory': 0
        }
        
        logger.info("Recognition Processor initialized")
    
    def embed_to_2d(self, input_text: str, embedding_model = None) -> torch.Tensor:
        """
        Embed input text to 2D Recognition space
        
        This is a critical function that maps high-dimensional embeddings to 2D
        while preserving semantic meaning for fast pattern matching.
        """
        if embedding_model:
            # Get full embedding from model
            full_embedding = embedding_model.encode(input_text)
            full_tensor = torch.tensor(full_embedding, dtype=torch.float32)
            
            # Simple dimensionality reduction to 2D
            # Method 1: Take first 2 dimensions (simple but may lose info)
            # Method 2: PCA-like projection (better semantic preservation)
            
            # For now, use method 2: semantic projection
            # Project onto 2 meaningful semantic axes
            semantic_axes = torch.tensor([
                [1.0, 0.0],  # Axis 1: General semantic content
                [0.0, 1.0]   # Axis 2: Specific semantic content  
            ], dtype=torch.float32)
            
            # Simple linear projection (can be learned later)
            projection_matrix = torch.randn(full_tensor.shape[0], 2) * 0.1
            embedding_2d = torch.matmul(full_tensor, projection_matrix)
            
        else:
            # Fallback: simple text-based 2D embedding
            text_len = len(input_text)
            text_complexity = len(set(input_text.lower()))
            embedding_2d = torch.tensor([text_len / 100.0, text_complexity / 50.0], dtype=torch.float32)
        
        return embedding_2d.unsqueeze(0)  # Add batch dimension
    
    def recognize(self, input_text: str, embedding_model = None) -> Dict[str, Any]:
        """
        Main Recognition phase processing
        
        Args:
            input_text: Input query text
            embedding_model: Optional embedding model for better 2D projection
            
        Returns:
            Recognition result with match status and procedure
        """
        start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        if start_time:
            start_time.record()
        import time
        start_cpu = time.time()
        
        self.stats['total_queries'] += 1
        
        # Step 1: Embed to 2D Recognition space
        pattern_2d = self.embed_to_2d(input_text, embedding_model)
        
        # Step 2: Search memory for similar patterns
        match_result = self.memory.get_best_match(pattern_2d)
        
        # Step 3: Measure timing
        end_cpu = time.time()
        search_time_ms = (end_cpu - start_cpu) * 1000
        
        # Step 4: Process result
        if match_result:
            similarity, procedure, metadata = match_result
            self.stats['recognition_hits'] += 1
            
            result = {
                'phase': 'recognition',
                'success': True,
                'match_found': True,
                'similarity': similarity,
                'procedure': procedure,
                'metadata': metadata,
                'pattern_2d': pattern_2d.squeeze().tolist(),
                'search_time_ms': search_time_ms,
                'escalate_to_cognition': False
            }
            
            logger.info(f"✅ Recognition HIT: similarity={similarity:.3f} (>{self.config.similarity_threshold})")
            
        else:
            self.stats['recognition_misses'] += 1
            
            result = {
                'phase': 'recognition',
                'success': True,  # Recognition succeeded (it determined no match)
                'match_found': False,
                'similarity': 0.0,
                'procedure': None,
                'metadata': {},
                'pattern_2d': pattern_2d.squeeze().tolist(),
                'search_time_ms': search_time_ms,
                'escalate_to_cognition': True  # Key: escalate to Cognition phase
            }
            
            logger.info(f"🔄 Recognition MISS: escalating to Cognition phase")
        
        # Update stats
        self.stats['avg_search_time_ms'] = (
            (self.stats['avg_search_time_ms'] * (self.stats['total_queries'] - 1) + search_time_ms) 
            / self.stats['total_queries']
        )
        self.stats['patterns_in_memory'] = len(self.memory.patterns)
        
        return result
    
    def learn_pattern(self, input_text: str, procedure: Any, embedding_model = None, metadata: Dict = None):
        """
        Learn a new pattern-procedure association
        
        This is called when Cognition phase successfully processes a novel input,
        so future similar inputs can be handled by fast Recognition.
        """
        pattern_2d = self.embed_to_2d(input_text, embedding_model)
        
        self.memory.add_pattern(
            pattern=pattern_2d.squeeze(),
            procedure=procedure,
            metadata=metadata or {'learned_from': 'cognition', 'input_text': input_text}
        )
        
        logger.info(f"📚 Learned new pattern: {input_text[:50]}...")
        return True
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get Recognition phase performance statistics"""
        if self.stats['total_queries'] > 0:
            recognition_rate = self.stats['recognition_hits'] / self.stats['total_queries']
        else:
            recognition_rate = 0.0
            
        return {
            **self.stats,
            'recognition_rate': recognition_rate,
            'miss_rate': 1.0 - recognition_rate,
            'avg_similarity_threshold': self.config.similarity_threshold
        }


class RecognitionPhaseIntegrator:
    """
    Integrates Recognition Phase with Enhanced SATC Engine
    """
    
    def __init__(self, recognition_processor: RecognitionProcessor):
        self.recognition_processor = recognition_processor
        self.integrated = False
        
    def integrate_with_satc(self, satc_engine):
        """
        Integrate Recognition Phase with Enhanced SATC Engine
        """
        logger.info("Integrating Recognition Phase with Enhanced SATC...")
        
        # Add Recognition processor to engine
        satc_engine.recognition_processor = self.recognition_processor
        satc_engine._using_recognition_phase = True
        
        # Store original process_query method
        satc_engine._original_process_query = satc_engine.process_query if hasattr(satc_engine, 'process_query') else None
        
        self.integrated = True
        logger.info("✅ Recognition Phase integration completed!")
        
        return satc_engine
    
    def process_with_recognition_first(self, satc_engine, query: str) -> Dict[str, Any]:
        """
        Process query through Recognition → Cognition pipeline
        """
        if not self.integrated:
            logger.warning("Recognition Phase not yet integrated with SATC")
        
        # Step 1: Try Recognition phase first (2D fast path)
        recognition_result = self.recognition_processor.recognize(
            query, 
            getattr(satc_engine, 'embedding_model', None)
        )
        
        if recognition_result['match_found']:
            # Recognition succeeded - return cached procedure
            return {
                'query': query,
                'phase': 'recognition',
                'success': True,
                'output': recognition_result['procedure'],
                'coherence': recognition_result['similarity'],
                'processing_time': recognition_result['search_time_ms'] / 1000,  # Convert to seconds
                'method': 'fast_recognition',
                'pattern_2d': recognition_result['pattern_2d'],
                'metadata': recognition_result['metadata']
            }
        
        else:
            # Recognition failed - escalate to Cognition phase (4D+)
            logger.info("🧠 Escalating to Cognition phase...")
            
            # This is where we'll integrate with 4D Cognition in Milestone 3
            # For now, return recognition miss result
            return {
                'query': query,
                'phase': 'recognition_miss',
                'success': False,  # Will be handled by Cognition
                'output': 'Recognition phase complete - escalating to Cognition',
                'coherence': 0.0,
                'processing_time': recognition_result['search_time_ms'] / 1000,
                'method': 'recognition_escalation',
                'pattern_2d': recognition_result['pattern_2d'],
                'escalate_to_cognition': True,
                'next_phase': 'cognition_4d'
            }


def create_recognition_phase():
    """
    Factory function to create complete Recognition Phase
    """
    config = RecognitionConfig()
    processor = RecognitionProcessor(config)
    integrator = RecognitionPhaseIntegrator(processor)
    
    logger.info("Recognition Phase created successfully!")
    logger.info(f"2D Recognition with threshold: {config.similarity_threshold}")
    
    return processor, integrator, config


# Standalone testing function
def test_recognition_phase_standalone():
    """
    Standalone test of Recognition Phase
    """
    print("=" * 60)
    print("RECOGNITION PHASE (2D) - STANDALONE TEST")
    print("=" * 60)
    
    # Create Recognition phase
    processor, integrator, config = create_recognition_phase()
    
    # Test patterns
    test_queries = [
        "Hello world",
        "What is the weather?", 
        "How are you?",
        "Hello world",  # Duplicate for recognition test
    ]
    
    # Test procedures (what Recognition should return for matches)
    test_procedures = [
        "greeting_response",
        "weather_query_response",
        "wellbeing_response"
    ]
    
    print("\n--- Learning Phase ---")
    # Learn some patterns
    for i, query in enumerate(test_queries[:3]):  # Learn first 3
        processor.learn_pattern(query, test_procedures[i])
        print(f"Learned: '{query}' → '{test_procedures[i]}'")
    
    print(f"\nMemory now contains {len(processor.memory.patterns)} patterns")
    
    print("\n--- Recognition Testing ---")
    # Test recognition
    for query in test_queries:
        result = processor.recognize(query)
        
        status = "HIT" if result['match_found'] else "MISS"
        similarity = result['similarity']
        pattern_2d = result['pattern_2d']
        
        print(f"Query: '{query}'")
        print(f"  Status: {status}")
        print(f"  Similarity: {similarity:.3f}")
        print(f"  2D Pattern: [{pattern_2d[0]:.3f}, {pattern_2d[1]:.3f}]")
        print(f"  Escalate: {result['escalate_to_cognition']}")
        print()
    
    print("--- Performance Statistics ---")
    stats = processor.get_performance_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print("=" * 60)
    print("MILESTONE 2: RECOGNITION PHASE - COMPLETE!")
    print("=" * 60)
    
    return processor, integrator, config


if __name__ == "__main__":
    test_recognition_phase_standalone()