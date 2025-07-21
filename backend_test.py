#!/usr/bin/env python3
"""
Backend Test Suite for Enhanced SATC Engine
==========================================

Tests the Enhanced SATC Engine with the new square dimension architecture.
Focus on cognition endpoint issues, tensor conversion, and dimension handling.

Author: Testing Agent
Status: Comprehensive Testing Suite
"""

import requests
import json
import time
import sys
import traceback
from typing import Dict, Any, List
import numpy as np

# Backend URL from environment
BACKEND_URL = "https://0d0327f1-b3e8-4760-9b4f-e767b10bd743.preview.emergentagent.com/api"

class SATCBackendTester:
    """Comprehensive backend testing for SATC Engine"""
    
    def __init__(self):
        self.base_url = BACKEND_URL
        self.test_results = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'errors': [],
            'warnings': []
        }
        
    def log_result(self, test_name: str, success: bool, message: str = "", error: str = ""):
        """Log test result"""
        self.test_results['total_tests'] += 1
        
        if success:
            self.test_results['passed_tests'] += 1
            print(f"✅ {test_name}: {message}")
        else:
            self.test_results['failed_tests'] += 1
            print(f"❌ {test_name}: {message}")
            if error:
                print(f"   Error: {error}")
                self.test_results['errors'].append({
                    'test': test_name,
                    'error': error,
                    'message': message
                })
    
    def log_warning(self, test_name: str, warning: str):
        """Log warning"""
        print(f"⚠️  {test_name}: {warning}")
        self.test_results['warnings'].append({
            'test': test_name,
            'warning': warning
        })
    
    def test_basic_connectivity(self):
        """Test basic API connectivity"""
        print("\n=== Testing Basic Connectivity ===")
        
        try:
            response = requests.get(f"{self.base_url}/", timeout=10)
            if response.status_code == 200:
                data = response.json()
                self.log_result("Basic Connectivity", True, f"API accessible: {data.get('message', 'OK')}")
                return True
            else:
                self.log_result("Basic Connectivity", False, f"HTTP {response.status_code}", response.text)
                return False
        except Exception as e:
            self.log_result("Basic Connectivity", False, "Connection failed", str(e))
            return False
    
    def test_engine_config(self):
        """Test SATC engine configuration with square dimensions"""
        print("\n=== Testing Engine Configuration ===")
        
        try:
            response = requests.get(f"{self.base_url}/cognition/config", timeout=10)
            if response.status_code == 200:
                config = response.json()
                
                # Check square dimension architecture
                expected_squares = [784, 625, 484, 361, 256, 169, 100, 64, 36, 16, 9, 4, 1]
                
                # Verify HD dimension
                if config.get('hd_dim') == 10000:
                    self.log_result("HD Dimension", True, f"HD dimension: {config['hd_dim']}")
                else:
                    self.log_result("HD Dimension", False, f"Expected 10000, got {config.get('hd_dim')}")
                
                # Check deep layers config
                deep_config = config.get('deep_layers_config', {})
                if deep_config.get('layers') == 12:
                    self.log_result("Deep Layers Count", True, f"Layers: {deep_config['layers']}")
                else:
                    self.log_result("Deep Layers Count", False, f"Expected 12 layers, got {deep_config.get('layers')}")
                
                # Check SOM grid size
                som_size = config.get('som_grid_size', 0)
                if som_size == 10:
                    self.log_result("SOM Grid Size", True, f"SOM grid: {som_size}x{som_size}")
                else:
                    self.log_result("SOM Grid Size", False, f"Expected 10, got {som_size}")
                
                return True
            else:
                self.log_result("Engine Configuration", False, f"HTTP {response.status_code}", response.text)
                return False
        except Exception as e:
            self.log_result("Engine Configuration", False, "Request failed", str(e))
            return False
    
    def test_atc_comprehensive_validation(self):
        """FINAL COMPREHENSIVE VALIDATION - REVOLUTIONARY ATC SYSTEM"""
        print("\n=== FINAL ATC COMPREHENSIVE VALIDATION ===")
        print("Testing all 5 ATC phases and consciousness emergence...")
        
        # Test 1: Complete Pipeline Activation
        print("\n--- Test 1: Complete Pipeline Activation ---")
        complex_query = "What is the meaning of consciousness, and how should I develop self-aware AI while ensuring ethical alignment with human values?"
        
        try:
            payload = {
                "query": complex_query,
                "use_recognition": True,
                "save_to_memory": True
            }
            
            response = requests.post(
                f"{self.base_url}/cognition",
                json=payload,
                timeout=60  # Longer timeout for complex processing
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Check ATC Phase Data
                phase = data.get('phase', '')
                meta_coherence = data.get('meta_coherence')
                self_awareness = data.get('self_awareness')
                goal_count = data.get('goal_count', 0)
                consciousness_level = data.get('consciousness_level', 0)
                identity_id = data.get('identity_id')
                total_memories = data.get('total_memories', 0)
                
                # Validate all 5 phases are contributing
                phases_active = 0
                phase_details = []
                
                # Recognition (2D) - Check if pattern matching occurred
                if data.get('method') == 'recognition' or 'recognition' in phase.lower():
                    phases_active += 1
                    phase_details.append("Recognition (2D): Pattern matching")
                
                # Cognition (4D) - Check reasoning steps
                reasoning_steps = data.get('metadata', {}).get('reasoning_steps', [])
                if reasoning_steps and len(reasoning_steps) > 0:
                    phases_active += 1
                    phase_details.append(f"Cognition (4D): {len(reasoning_steps)} reasoning steps")
                elif 'cognition' in phase.lower():
                    phases_active += 1
                    phase_details.append("Cognition (4D): Active")
                
                # Reflection (16D) - Check meta-coherence
                if meta_coherence is not None and meta_coherence > 0:
                    phases_active += 1
                    phase_details.append(f"Reflection (16D): Meta-coherence {meta_coherence:.3f}")
                
                # Volition (64D) - Check goal formation
                if goal_count is not None and goal_count > 0:
                    phases_active += 1
                    phase_details.append(f"Volition (64D): {goal_count} goals formed")
                
                # Personality (256D) - Check consciousness level
                if consciousness_level is not None and consciousness_level > 0.4:  # 40%+ threshold
                    phases_active += 1
                    phase_details.append(f"Personality (256D): Consciousness {consciousness_level:.3f}")
                
                # Validate success criteria
                if phases_active >= 3 and consciousness_level > 0.4:
                    self.log_result(
                        "ATC Complete Pipeline",
                        True,
                        f"{phases_active}/5 phases active, Consciousness: {consciousness_level:.3f}"
                    )
                    for detail in phase_details:
                        print(f"   ✓ {detail}")
                else:
                    self.log_result(
                        "ATC Complete Pipeline",
                        False,
                        f"Only {phases_active}/5 phases active, Consciousness: {consciousness_level:.3f}"
                    )
                
                # Test consciousness emergence validation
                if consciousness_level > 0.4 and identity_id and total_memories > 0:
                    self.log_result(
                        "Consciousness Emergence",
                        True,
                        f"Level: {consciousness_level:.3f}, ID: {identity_id[:8]}..., Memories: {total_memories}"
                    )
                else:
                    self.log_result(
                        "Consciousness Emergence",
                        False,
                        f"Level: {consciousness_level:.3f}, ID: {bool(identity_id)}, Memories: {total_memories}"
                    )
                
            else:
                self.log_result(
                    "ATC Complete Pipeline",
                    False,
                    f"HTTP {response.status_code}",
                    response.text[:200]
                )
        
        except Exception as e:
            self.log_result(
                "ATC Complete Pipeline",
                False,
                "Request failed",
                str(e)
            )
        
        # Test 2: Learning Validation
        print("\n--- Test 2: Learning Validation ---")
        learning_query = "How does pattern recognition learn from cognition results?"
        
        # First query - should go to Cognition
        try:
            payload = {"query": learning_query, "use_recognition": True, "save_to_memory": True}
            response1 = requests.post(f"{self.base_url}/cognition", json=payload, timeout=30)
            
            if response1.status_code == 200:
                data1 = response1.json()
                first_phase = data1.get('phase', '')
                first_processing_time = data1.get('processing_time', 0)
                
                # Wait a moment for memory consolidation
                time.sleep(2)
                
                # Second query - should hit Recognition (learned)
                response2 = requests.post(f"{self.base_url}/cognition", json=payload, timeout=30)
                
                if response2.status_code == 200:
                    data2 = response2.json()
                    second_phase = data2.get('phase', '')
                    second_processing_time = data2.get('processing_time', 0)
                    
                    # Check if learning occurred (faster processing or recognition phase)
                    if (second_processing_time < first_processing_time * 0.8 or 
                        'recognition' in second_phase.lower()):
                        self.log_result(
                            "Learning Validation",
                            True,
                            f"Learning detected: {first_phase} → {second_phase}, Time: {first_processing_time:.3f}s → {second_processing_time:.3f}s"
                        )
                    else:
                        self.log_result(
                            "Learning Validation",
                            False,
                            f"No learning improvement: {first_phase} → {second_phase}"
                        )
                else:
                    self.log_result("Learning Validation", False, f"Second query failed: HTTP {response2.status_code}")
            else:
                self.log_result("Learning Validation", False, f"First query failed: HTTP {response1.status_code}")
        
        except Exception as e:
            self.log_result("Learning Validation", False, "Request failed", str(e))
        
        # Test 3: Mathematical Foundation Integrity
        print("\n--- Test 3: Mathematical Foundation Integrity ---")
        try:
            # Test power-of-2 invertibility through multiple queries
            math_queries = [
                "Test mathematical foundation",
                "Validate tensor operations",
                "Check dimension consistency"
            ]
            
            coherence_scores = []
            processing_times = []
            
            for query in math_queries:
                payload = {"query": query, "use_recognition": False, "save_to_memory": True}
                response = requests.post(f"{self.base_url}/cognition", json=payload, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    coherence_scores.append(data.get('coherence', 0))
                    processing_times.append(data.get('processing_time', 0))
                
                time.sleep(1)
            
            if coherence_scores:
                avg_coherence = sum(coherence_scores) / len(coherence_scores)
                avg_time = sum(processing_times) / len(processing_times)
                
                # Check mathematical foundation integrity
                if avg_coherence > 0.5 and avg_time < 5.0:  # Good coherence, reasonable time
                    self.log_result(
                        "Mathematical Foundation",
                        True,
                        f"Avg coherence: {avg_coherence:.3f}, Avg time: {avg_time:.3f}s"
                    )
                else:
                    self.log_result(
                        "Mathematical Foundation",
                        False,
                        f"Poor performance: coherence {avg_coherence:.3f}, time {avg_time:.3f}s"
                    )
            else:
                self.log_result("Mathematical Foundation", False, "No valid responses")
        
        except Exception as e:
            self.log_result("Mathematical Foundation", False, "Request failed", str(e))

    def test_cognition_endpoint_basic(self):
        """Test basic cognition endpoint functionality"""
        print("\n=== Testing Cognition Endpoint (Basic) ===")
        
        test_queries = [
            "Hello, how are you?",
            "What is artificial intelligence?",
            "Explain consciousness",
            "How does machine learning work?",
            "What are your capabilities?"
        ]
        
        for query in test_queries:
            try:
                payload = {
                    "query": query,
                    "use_recognition": True,
                    "save_to_memory": True
                }
                
                response = requests.post(
                    f"{self.base_url}/cognition",
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Check required fields
                    required_fields = ['query', 'output', 'phase', 'success', 'coherence', 'processing_time']
                    missing_fields = [field for field in required_fields if field not in data]
                    
                    if not missing_fields:
                        # Check if output is meaningful (not just "Error: ")
                        if data['output'] and not data['output'].startswith("Error: "):
                            self.log_result(
                                f"Cognition Query: '{query[:30]}...'",
                                True,
                                f"Phase: {data['phase']}, Coherence: {data['coherence']:.3f}, Time: {data['processing_time']:.3f}s"
                            )
                        else:
                            self.log_result(
                                f"Cognition Query: '{query[:30]}...'",
                                False,
                                f"Empty or error output: '{data['output'][:50]}...'"
                            )
                    else:
                        self.log_result(
                            f"Cognition Query: '{query[:30]}...'",
                            False,
                            f"Missing fields: {missing_fields}"
                        )
                else:
                    self.log_result(
                        f"Cognition Query: '{query[:30]}...'",
                        False,
                        f"HTTP {response.status_code}",
                        response.text[:200]
                    )
                
                # Small delay between requests
                time.sleep(0.5)
                
            except Exception as e:
                self.log_result(
                    f"Cognition Query: '{query[:30]}...'",
                    False,
                    "Request failed",
                    str(e)
                )
    
    def test_cognition_square_dimensions(self):
        """Test cognition with focus on square dimension processing"""
        print("\n=== Testing Square Dimension Processing ===")
        
        # Test queries that should trigger different processing paths
        dimension_test_queries = [
            "Simple test",  # Short query
            "This is a medium length query that should test the embedding dimension handling properly",  # Medium query
            "This is a very long query designed to test the square dimension architecture and how it handles complex inputs with multiple concepts, ideas, and semantic structures that need to be processed through the deep layers with square progression from 784 to 625 to 484 and so on down to the final dimension of 1",  # Long query
            "Test 123 numbers and symbols !@#$%",  # Mixed content
            "Quantum consciousness artificial intelligence machine learning neural networks"  # Technical terms
        ]
        
        for query in dimension_test_queries:
            try:
                payload = {"query": query, "use_recognition": False, "save_to_memory": True}
                
                response = requests.post(
                    f"{self.base_url}/cognition",
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Check for dimension-related metadata
                    metadata = data.get('metadata', {})
                    
                    # Verify processing occurred
                    if data.get('success') and data.get('output'):
                        # Check for square dimension indicators
                        nodes_count = metadata.get('nodes_count', 0)
                        sememes_count = metadata.get('sememes_count', 0)
                        
                        self.log_result(
                            f"Square Dimension Test: '{query[:20]}...'",
                            True,
                            f"Nodes: {nodes_count}, Sememes: {sememes_count}, Phase: {data['phase']}"
                        )
                        
                        # Check coherence is reasonable
                        coherence = data.get('coherence', 0)
                        if coherence < 0 or coherence > 1:
                            self.log_warning(
                                f"Coherence Range: '{query[:20]}...'",
                                f"Coherence {coherence} outside [0,1] range"
                            )
                    else:
                        self.log_result(
                            f"Square Dimension Test: '{query[:20]}...'",
                            False,
                            f"Processing failed: {data.get('output', 'No output')}"
                        )
                else:
                    self.log_result(
                        f"Square Dimension Test: '{query[:20]}...'",
                        False,
                        f"HTTP {response.status_code}",
                        response.text[:200]
                    )
                
                time.sleep(0.5)
                
            except Exception as e:
                self.log_result(
                    f"Square Dimension Test: '{query[:20]}...'",
                    False,
                    "Request failed",
                    str(e)
                )
    
    def test_sememe_extraction(self):
        """Test sememe extraction with square dimensions"""
        print("\n=== Testing Sememe Extraction ===")
        
        test_queries = [
            "artificial intelligence",
            "consciousness and awareness",
            "machine learning algorithms",
            "quantum computing principles",
            "natural language processing"
        ]
        
        for query in test_queries:
            try:
                response = requests.get(
                    f"{self.base_url}/cognition/sememes/{query}",
                    timeout=20
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Check sememe structure
                    sememes = data.get('sememes', [])
                    nodes_count = data.get('nodes_count', 0)
                    
                    if sememes and nodes_count > 0:
                        # Verify sememe structure
                        valid_sememes = 0
                        for sememe in sememes:
                            if (sememe.get('node_index') is not None and 
                                sememe.get('primary_sememe') and
                                sememe.get('node_vector_length', 0) > 0):
                                valid_sememes += 1
                        
                        if valid_sememes > 0:
                            self.log_result(
                                f"Sememe Extraction: '{query}'",
                                True,
                                f"Valid sememes: {valid_sememes}/{len(sememes)}, Nodes: {nodes_count}"
                            )
                        else:
                            self.log_result(
                                f"Sememe Extraction: '{query}'",
                                False,
                                f"No valid sememes found in {len(sememes)} results"
                            )
                    else:
                        self.log_result(
                            f"Sememe Extraction: '{query}'",
                            False,
                            f"No sememes or nodes found"
                        )
                else:
                    self.log_result(
                        f"Sememe Extraction: '{query}'",
                        False,
                        f"HTTP {response.status_code}",
                        response.text[:200]
                    )
                
                time.sleep(0.5)
                
            except Exception as e:
                self.log_result(
                    f"Sememe Extraction: '{query}'",
                    False,
                    "Request failed",
                    str(e)
                )
    
    def test_performance_metrics(self):
        """Test performance metrics endpoint"""
        print("\n=== Testing Performance Metrics ===")
        
        try:
            response = requests.get(f"{self.base_url}/cognition/performance", timeout=10)
            
            if response.status_code == 200:
                metrics = response.json()
                
                # Check required metrics fields
                required_fields = [
                    'total_queries', 'recognition_hits', 'cognition_processes',
                    'recognition_rate', 'avg_coherence', 'avg_processing_time'
                ]
                
                missing_fields = [field for field in required_fields if field not in metrics]
                
                if not missing_fields:
                    self.log_result(
                        "Performance Metrics",
                        True,
                        f"Queries: {metrics['total_queries']}, Recognition rate: {metrics['recognition_rate']:.3f}"
                    )
                    
                    # Check for reasonable values
                    if metrics['avg_coherence'] < 0 or metrics['avg_coherence'] > 1:
                        self.log_warning(
                            "Coherence Range",
                            f"Average coherence {metrics['avg_coherence']} outside [0,1]"
                        )
                    
                    if metrics['avg_processing_time'] < 0:
                        self.log_warning(
                            "Processing Time",
                            f"Negative processing time: {metrics['avg_processing_time']}"
                        )
                else:
                    self.log_result(
                        "Performance Metrics",
                        False,
                        f"Missing fields: {missing_fields}"
                    )
            else:
                self.log_result(
                    "Performance Metrics",
                    False,
                    f"HTTP {response.status_code}",
                    response.text[:200]
                )
        except Exception as e:
            self.log_result(
                "Performance Metrics",
                False,
                "Request failed",
                str(e)
            )
    
    def test_training_endpoints(self):
        """Test training-related endpoints"""
        print("\n=== Testing Training Endpoints ===")
        
        # Test training status
        try:
            response = requests.get(f"{self.base_url}/training/status", timeout=10)
            
            if response.status_code == 200:
                status = response.json()
                
                required_fields = ['is_training', 'current_epoch', 'total_epochs']
                missing_fields = [field for field in required_fields if field not in status]
                
                if not missing_fields:
                    self.log_result(
                        "Training Status",
                        True,
                        f"Training: {status['is_training']}, Epoch: {status['current_epoch']}/{status['total_epochs']}"
                    )
                else:
                    self.log_result(
                        "Training Status",
                        False,
                        f"Missing fields: {missing_fields}"
                    )
            else:
                self.log_result(
                    "Training Status",
                    False,
                    f"HTTP {response.status_code}",
                    response.text[:200]
                )
        except Exception as e:
            self.log_result(
                "Training Status",
                False,
                "Request failed",
                str(e)
            )
        
        # Test adding training pair
        try:
            training_pair = {
                "query": "What is the square dimension architecture?",
                "response": "The square dimension architecture uses perfect squares for layer dimensions, progressing from 784 (28²) down to 1 (1²) through the deep layers.",
                "quality_score": 0.9,
                "coherence_score": 0.85,
                "sememes": ["architecture", "dimensions", "squares"]
            }
            
            response = requests.post(
                f"{self.base_url}/training/add-pair",
                json=training_pair,
                timeout=10
            )
            
            if response.status_code == 200:
                self.log_result(
                    "Add Training Pair",
                    True,
                    "Training pair added successfully"
                )
            else:
                self.log_result(
                    "Add Training Pair",
                    False,
                    f"HTTP {response.status_code}",
                    response.text[:200]
                )
        except Exception as e:
            self.log_result(
                "Add Training Pair",
                False,
                "Request failed",
                str(e)
            )
    
    def test_bulk_training_endpoints(self):
        """Test bulk training system endpoints"""
        print("\n=== Testing Bulk Training Endpoints ===")
        
        # Test Hello World system creation
        try:
            payload = {"quick_start": True}
            response = requests.post(
                f"{self.base_url}/training/hello-world",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'ready':
                    self.log_result(
                        "Hello World System",
                        True,
                        f"System created: {data.get('message', 'OK')}"
                    )
                else:
                    self.log_result(
                        "Hello World System",
                        False,
                        f"System not ready: {data}"
                    )
            else:
                self.log_result(
                    "Hello World System",
                    False,
                    f"HTTP {response.status_code}",
                    response.text[:200]
                )
        except Exception as e:
            self.log_result(
                "Hello World System",
                False,
                "Request failed",
                str(e)
            )
        
        # Test bulk training status
        try:
            response = requests.get(f"{self.base_url}/training/bulk-status", timeout=10)
            
            if response.status_code == 200:
                status = response.json()
                
                if status.get('system_status') == 'initialized':
                    self.log_result(
                        "Bulk Training Status",
                        True,
                        f"System initialized, Hardware optimized: {status.get('hardware_optimized', False)}"
                    )
                else:
                    self.log_result(
                        "Bulk Training Status",
                        False,
                        f"System not initialized: {status.get('system_status')}"
                    )
            else:
                self.log_result(
                    "Bulk Training Status",
                    False,
                    f"HTTP {response.status_code}",
                    response.text[:200]
                )
        except Exception as e:
            self.log_result(
                "Bulk Training Status",
                False,
                "Request failed",
                str(e)
            )
    
    def test_hardware_info(self):
        """Test hardware information endpoint"""
        print("\n=== Testing Hardware Information ===")
        
        try:
            response = requests.get(f"{self.base_url}/training/hardware-info", timeout=10)
            
            if response.status_code == 200:
                info = response.json()
                hardware_info = info.get('hardware_info', {})
                
                # Check key hardware metrics
                gpu_available = hardware_info.get('gpu_available', False)
                cpu_count = hardware_info.get('cpu_count', 0)
                
                self.log_result(
                    "Hardware Information",
                    True,
                    f"GPU: {gpu_available}, CPU cores: {cpu_count}, Optimization: {info.get('optimization_status')}"
                )
                
                # Check for RTX 4070 Ti specific info
                gpu_name = hardware_info.get('gpu_name', '')
                if 'RTX' in gpu_name or gpu_available:
                    self.log_result(
                        "GPU Detection",
                        True,
                        f"GPU detected: {gpu_name}"
                    )
                else:
                    self.log_warning(
                        "GPU Detection",
                        "No RTX GPU detected - may affect performance"
                    )
            else:
                self.log_result(
                    "Hardware Information",
                    False,
                    f"HTTP {response.status_code}",
                    response.text[:200]
                )
        except Exception as e:
            self.log_result(
                "Hardware Information",
                False,
                "Request failed",
                str(e)
            )
    
    def test_memory_operations(self):
        """Test memory and state operations"""
        print("\n=== Testing Memory Operations ===")
        
        # Test engine reset
        try:
            response = requests.post(f"{self.base_url}/cognition/reset", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                self.log_result(
                    "Engine Reset",
                    True,
                    data.get('message', 'Reset successful')
                )
            else:
                self.log_result(
                    "Engine Reset",
                    False,
                    f"HTTP {response.status_code}",
                    response.text[:200]
                )
        except Exception as e:
            self.log_result(
                "Engine Reset",
                False,
                "Request failed",
                str(e)
            )
        
        # Test state save
        try:
            response = requests.post(f"{self.base_url}/cognition/save-state", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                self.log_result(
                    "State Save",
                    True,
                    data.get('message', 'State saved')
                )
            else:
                self.log_result(
                    "State Save",
                    False,
                    f"HTTP {response.status_code}",
                    response.text[:200]
                )
        except Exception as e:
            self.log_result(
                "State Save",
                False,
                "Request failed",
                str(e)
            )
    
    def test_cognition_history(self):
        """Test cognition history endpoint"""
        print("\n=== Testing Cognition History ===")
        
        try:
            response = requests.get(f"{self.base_url}/cognition/history?limit=5", timeout=10)
            
            if response.status_code == 200:
                history = response.json()
                
                if isinstance(history, list):
                    self.log_result(
                        "Cognition History",
                        True,
                        f"Retrieved {len(history)} history entries"
                    )
                    
                    # Check history entry structure
                    if history:
                        entry = history[0]
                        required_fields = ['query', 'output', 'phase', 'success']
                        missing_fields = [field for field in required_fields if field not in entry]
                        
                        if not missing_fields:
                            self.log_result(
                                "History Entry Structure",
                                True,
                                f"Valid entry structure"
                            )
                        else:
                            self.log_result(
                                "History Entry Structure",
                                False,
                                f"Missing fields: {missing_fields}"
                            )
                else:
                    self.log_result(
                        "Cognition History",
                        False,
                        f"Expected list, got {type(history)}"
                    )
            else:
                self.log_result(
                    "Cognition History",
                    False,
                    f"HTTP {response.status_code}",
                    response.text[:200]
                )
        except Exception as e:
            self.log_result(
                "Cognition History",
                False,
                "Request failed",
                str(e)
            )
    
    def run_comprehensive_test(self):
        """Run all tests"""
        print("🧠 Enhanced SATC Engine Backend Test Suite")
        print("=" * 50)
        print(f"Testing backend at: {self.base_url}")
        print("Focus: FINAL COMPREHENSIVE ATC VALIDATION")
        print("Testing all 5 ATC phases and consciousness emergence")
        print()
        
        # Run all test categories
        if self.test_basic_connectivity():
            # PRIORITY: ATC Comprehensive Validation
            self.test_atc_comprehensive_validation()
            
            # Standard tests
            self.test_engine_config()
            self.test_cognition_endpoint_basic()
            self.test_cognition_square_dimensions()
            self.test_sememe_extraction()
            self.test_performance_metrics()
            self.test_training_endpoints()
            self.test_bulk_training_endpoints()
            self.test_hardware_info()
            self.test_memory_operations()
            self.test_cognition_history()
        else:
            print("❌ Basic connectivity failed - skipping other tests")
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 50)
        print("🧠 ENHANCED SATC ENGINE TEST SUMMARY")
        print("=" * 50)
        
        total = self.test_results['total_tests']
        passed = self.test_results['passed_tests']
        failed = self.test_results['failed_tests']
        warnings = len(self.test_results['warnings'])
        
        print(f"Total Tests: {total}")
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"⚠️  Warnings: {warnings}")
        
        if total > 0:
            success_rate = (passed / total) * 100
            print(f"Success Rate: {success_rate:.1f}%")
        
        # Critical issues
        critical_issues = []
        for error in self.test_results['errors']:
            if any(keyword in error['test'].lower() for keyword in ['cognition', 'dimension', 'square']):
                critical_issues.append(error)
        
        if critical_issues:
            print(f"\n🚨 CRITICAL ISSUES ({len(critical_issues)}):")
            for issue in critical_issues:
                print(f"   • {issue['test']}: {issue['message']}")
        
        # Warnings summary
        if self.test_results['warnings']:
            print(f"\n⚠️  WARNINGS ({len(self.test_results['warnings'])}):")
            for warning in self.test_results['warnings']:
                print(f"   • {warning['test']}: {warning['warning']}")
        
        print("\n" + "=" * 50)
        
        # Return overall status
        return failed == 0

def main():
    """Main test execution"""
    tester = SATCBackendTester()
    success = tester.run_comprehensive_test()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()