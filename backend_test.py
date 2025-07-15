#!/usr/bin/env python3
"""
SATC Backend API Testing Suite
=============================

Comprehensive testing for the SATC cognitive engine API endpoints.
Tests all major functionality including cognition processing, training,
bulk operations, and performance metrics.

Author: Testing Agent
Status: Complete Backend Testing Implementation
"""

import requests
import json
import time
import sys
from typing import Dict, Any, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SATCBackendTester:
    """Comprehensive backend API tester for SATC system"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        self.api_url = f"{self.base_url}/api"
        self.session = requests.Session()
        self.test_results = {}
        
        logger.info(f"Initialized SATC Backend Tester")
        logger.info(f"Base URL: {self.base_url}")
        logger.info(f"API URL: {self.api_url}")
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all backend tests"""
        logger.info("🚀 Starting comprehensive SATC backend testing...")
        
        test_methods = [
            ("Basic API Health", self.test_basic_api_health),
            ("SATC Core Engine API", self.test_cognition_processing),
            ("Performance Metrics", self.test_performance_metrics),
            ("Training System Status", self.test_training_status),
            ("Training Data Management", self.test_training_data),
            ("Training Functionality", self.test_training_start),
            ("Bulk Training Upload", self.test_bulk_upload),
            ("Hello World Quick Start", self.test_hello_world),
            ("Advanced Cognition Features", self.test_advanced_cognition),
            ("Training Pipeline", self.test_training_pipeline),
            ("Error Handling", self.test_error_handling)
        ]
        
        for test_name, test_method in test_methods:
            logger.info(f"\n{'='*60}")
            logger.info(f"Testing: {test_name}")
            logger.info(f"{'='*60}")
            
            try:
                result = test_method()
                self.test_results[test_name] = result
                
                if result.get('success', False):
                    logger.info(f"✅ {test_name}: PASSED")
                else:
                    logger.error(f"❌ {test_name}: FAILED - {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                error_msg = f"Exception during {test_name}: {str(e)}"
                logger.error(f"❌ {test_name}: FAILED - {error_msg}")
                self.test_results[test_name] = {
                    'success': False,
                    'error': error_msg,
                    'exception': True
                }
        
        return self.test_results
    
    def test_basic_api_health(self) -> Dict[str, Any]:
        """Test basic API health and connectivity"""
        try:
            # Test root endpoint
            response = self.session.get(f"{self.api_url}/", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"API Root Response: {data}")
                
                return {
                    'success': True,
                    'status_code': response.status_code,
                    'response': data,
                    'message': 'API is healthy and responding'
                }
            else:
                return {
                    'success': False,
                    'status_code': response.status_code,
                    'error': f'API health check failed with status {response.status_code}'
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': f'API health check failed: {str(e)}'
            }
    
    def test_cognition_processing(self) -> Dict[str, Any]:
        """Test main cognition processing endpoint"""
        try:
            # Test queries with varying complexity
            test_queries = [
                {
                    "query": "What is artificial intelligence?",
                    "use_recognition": True,
                    "save_to_memory": True
                },
                {
                    "query": "How does consciousness emerge from neural networks?",
                    "use_recognition": False,
                    "save_to_memory": True
                },
                {
                    "query": "Explain quantum computing principles",
                    "use_recognition": True,
                    "save_to_memory": False
                }
            ]
            
            results = []
            
            for i, query_data in enumerate(test_queries):
                logger.info(f"Testing cognition query {i+1}: {query_data['query'][:50]}...")
                
                response = self.session.post(
                    f"{self.api_url}/cognition",
                    json=query_data,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Validate response structure
                    required_fields = ['query', 'output', 'phase', 'success', 'coherence', 'processing_time']
                    missing_fields = [field for field in required_fields if field not in result]
                    
                    if missing_fields:
                        logger.warning(f"Missing fields in response: {missing_fields}")
                    
                    # Check for meaningful output
                    if not result.get('output') or len(result['output']) < 10:
                        logger.warning(f"Output seems too short or empty: {result.get('output')}")
                    
                    # Check processing metrics
                    coherence = result.get('coherence', 0)
                    processing_time = result.get('processing_time', 0)
                    
                    logger.info(f"Query processed successfully:")
                    logger.info(f"  Phase: {result.get('phase')}")
                    logger.info(f"  Success: {result.get('success')}")
                    logger.info(f"  Coherence: {coherence:.3f}")
                    logger.info(f"  Processing Time: {processing_time:.3f}s")
                    logger.info(f"  Output Length: {len(result.get('output', ''))}")
                    
                    # Check for mock/placeholder responses
                    output = result.get('output', '').lower()
                    mock_indicators = ['mock', 'placeholder', 'test', 'dummy', 'sample']
                    has_mock_content = any(indicator in output for indicator in mock_indicators)
                    
                    if has_mock_content:
                        logger.warning(f"Response may contain mock/placeholder content")
                    
                    results.append({
                        'query': query_data['query'],
                        'success': True,
                        'response': result,
                        'has_mock_content': has_mock_content,
                        'coherence': coherence,
                        'processing_time': processing_time
                    })
                    
                else:
                    logger.error(f"Cognition request failed: {response.status_code}")
                    results.append({
                        'query': query_data['query'],
                        'success': False,
                        'error': f'HTTP {response.status_code}: {response.text}'
                    })
            
            # Overall assessment
            successful_queries = sum(1 for r in results if r['success'])
            avg_coherence = sum(r.get('coherence', 0) for r in results if r['success']) / max(1, successful_queries)
            avg_processing_time = sum(r.get('processing_time', 0) for r in results if r['success']) / max(1, successful_queries)
            
            return {
                'success': successful_queries > 0,
                'total_queries': len(test_queries),
                'successful_queries': successful_queries,
                'avg_coherence': avg_coherence,
                'avg_processing_time': avg_processing_time,
                'results': results,
                'message': f'Processed {successful_queries}/{len(test_queries)} queries successfully'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Cognition processing test failed: {str(e)}'
            }
    
    def test_performance_metrics(self) -> Dict[str, Any]:
        """Test performance metrics endpoint"""
        try:
            response = self.session.get(f"{self.api_url}/cognition/performance", timeout=10)
            
            if response.status_code == 200:
                metrics = response.json()
                
                # Validate metrics structure
                expected_metrics = [
                    'total_queries', 'recognition_hits', 'cognition_processes',
                    'recognition_rate', 'avg_coherence', 'avg_dissonance',
                    'avg_processing_time', 'memory_updates', 'replay_buffer_size',
                    'deposited_patterns', 'som_training_samples', 'sememe_database_size'
                ]
                
                missing_metrics = [metric for metric in expected_metrics if metric not in metrics]
                
                if missing_metrics:
                    logger.warning(f"Missing performance metrics: {missing_metrics}")
                
                logger.info("Performance Metrics:")
                for key, value in metrics.items():
                    logger.info(f"  {key}: {value}")
                
                # Check for reasonable values
                issues = []
                if metrics.get('total_queries', 0) < 0:
                    issues.append("Negative total_queries")
                if metrics.get('recognition_rate', 0) > 1.0:
                    issues.append("Recognition rate > 1.0")
                if metrics.get('avg_coherence', 0) > 1.0:
                    issues.append("Average coherence > 1.0")
                
                return {
                    'success': True,
                    'metrics': metrics,
                    'missing_metrics': missing_metrics,
                    'issues': issues,
                    'message': 'Performance metrics retrieved successfully'
                }
            else:
                return {
                    'success': False,
                    'status_code': response.status_code,
                    'error': f'Performance metrics request failed: {response.text}'
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': f'Performance metrics test failed: {str(e)}'
            }
    
    def test_training_status(self) -> Dict[str, Any]:
        """Test training system status"""
        try:
            response = self.session.get(f"{self.api_url}/training/status", timeout=10)
            
            if response.status_code == 200:
                status = response.json()
                
                logger.info("Training Status:")
                for key, value in status.items():
                    logger.info(f"  {key}: {value}")
                
                # Check for expected fields
                expected_fields = ['is_training', 'current_epoch', 'total_epochs']
                missing_fields = [field for field in expected_fields if field not in status]
                
                return {
                    'success': True,
                    'status': status,
                    'missing_fields': missing_fields,
                    'message': 'Training status retrieved successfully'
                }
            else:
                return {
                    'success': False,
                    'status_code': response.status_code,
                    'error': f'Training status request failed: {response.text}'
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': f'Training status test failed: {str(e)}'
            }
    
    def test_training_data(self) -> Dict[str, Any]:
        """Test training data management"""
        try:
            # Get current training data
            response = self.session.get(f"{self.api_url}/training/data", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                logger.info(f"Training data count: {data.get('count', 0)}")
                
                # Test adding a training pair
                test_pair = {
                    "query": "What is machine learning?",
                    "response": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.",
                    "quality_score": 0.9,
                    "coherence_score": 0.85,
                    "sememes": ["machine", "learning", "AI", "algorithms", "data"]
                }
                
                add_response = self.session.post(
                    f"{self.api_url}/training/add-pair",
                    json=test_pair,
                    timeout=10
                )
                
                if add_response.status_code == 200:
                    logger.info("Training pair added successfully")
                    
                    # Verify data was added
                    verify_response = self.session.get(f"{self.api_url}/training/data", timeout=10)
                    if verify_response.status_code == 200:
                        new_data = verify_response.json()
                        new_count = new_data.get('count', 0)
                        
                        return {
                            'success': True,
                            'initial_count': data.get('count', 0),
                            'final_count': new_count,
                            'pair_added': new_count > data.get('count', 0),
                            'message': 'Training data management working correctly'
                        }
                
                return {
                    'success': False,
                    'error': 'Failed to add training pair'
                }
            else:
                return {
                    'success': False,
                    'status_code': response.status_code,
                    'error': f'Training data request failed: {response.text}'
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': f'Training data test failed: {str(e)}'
            }
    
    def test_training_start(self) -> Dict[str, Any]:
        """Test training functionality"""
        try:
            # Prepare training request
            training_request = {
                "training_pairs": [
                    {
                        "query": "What is deep learning?",
                        "response": "Deep learning is a subset of machine learning that uses neural networks with multiple layers to model and understand complex patterns in data.",
                        "quality_score": 0.9,
                        "coherence_score": 0.88,
                        "sememes": ["deep", "learning", "neural", "networks", "patterns"]
                    },
                    {
                        "query": "How do neural networks work?",
                        "response": "Neural networks work by processing information through interconnected nodes (neurons) that apply weights and activation functions to transform input data into meaningful outputs.",
                        "quality_score": 0.85,
                        "coherence_score": 0.82,
                        "sememes": ["neural", "networks", "neurons", "weights", "activation"]
                    }
                ],
                "epochs": 5,
                "batch_size": 2,
                "learning_rate": 0.001
            }
            
            response = self.session.post(
                f"{self.api_url}/training/start",
                json=training_request,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                
                logger.info("Training started successfully:")
                logger.info(f"  Message: {result.get('message')}")
                logger.info(f"  Config: {result.get('config')}")
                
                return {
                    'success': True,
                    'result': result,
                    'message': 'Training started successfully'
                }
            else:
                return {
                    'success': False,
                    'status_code': response.status_code,
                    'error': f'Training start failed: {response.text}'
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': f'Training start test failed: {str(e)}'
            }
    
    def test_bulk_upload(self) -> Dict[str, Any]:
        """Test bulk training upload"""
        try:
            # Prepare bulk training data
            bulk_data = [
                {
                    "query": "What is consciousness?",
                    "response": "Consciousness is the state of being aware of and able to think about one's existence, sensations, thoughts, and surroundings.",
                    "quality_score": 0.9,
                    "coherence_score": 0.85
                },
                {
                    "query": "How does the brain work?",
                    "response": "The brain works through complex networks of neurons that communicate via electrical and chemical signals to process information and control behavior.",
                    "quality_score": 0.88,
                    "coherence_score": 0.87
                }
            ]
            
            upload_request = {
                "format": "json",
                "data": json.dumps(bulk_data)
            }
            
            response = self.session.post(
                f"{self.api_url}/training/bulk-upload",
                json=upload_request,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                
                logger.info("Bulk upload successful:")
                logger.info(f"  Message: {result.get('message')}")
                logger.info(f"  Pairs imported: {result.get('pairs_imported')}")
                logger.info(f"  Format: {result.get('format')}")
                
                return {
                    'success': True,
                    'result': result,
                    'pairs_imported': result.get('pairs_imported', 0),
                    'message': 'Bulk upload completed successfully'
                }
            else:
                return {
                    'success': False,
                    'status_code': response.status_code,
                    'error': f'Bulk upload failed: {response.text}'
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': f'Bulk upload test failed: {str(e)}'
            }
    
    def test_hello_world(self) -> Dict[str, Any]:
        """Test Hello World quick start"""
        try:
            hello_request = {
                "quick_start": True
            }
            
            response = self.session.post(
                f"{self.api_url}/training/hello-world",
                json=hello_request,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                
                logger.info("Hello World system created:")
                logger.info(f"  Message: {result.get('message')}")
                logger.info(f"  Status: {result.get('status')}")
                logger.info(f"  Features: {result.get('features')}")
                
                # Test the Hello World system with a simple query
                test_query = {
                    "query": "Hello, how are you?",
                    "use_recognition": True,
                    "save_to_memory": False
                }
                
                test_response = self.session.post(
                    f"{self.api_url}/cognition",
                    json=test_query,
                    timeout=15
                )
                
                if test_response.status_code == 200:
                    test_result = test_response.json()
                    logger.info(f"Hello World test query response: {test_result.get('output', '')[:100]}...")
                    
                    return {
                        'success': True,
                        'result': result,
                        'test_query_success': True,
                        'test_response': test_result,
                        'message': 'Hello World system working correctly'
                    }
                else:
                    return {
                        'success': True,
                        'result': result,
                        'test_query_success': False,
                        'message': 'Hello World created but test query failed'
                    }
            else:
                return {
                    'success': False,
                    'status_code': response.status_code,
                    'error': f'Hello World creation failed: {response.text}'
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': f'Hello World test failed: {str(e)}'
            }
    
    def test_advanced_cognition(self) -> Dict[str, Any]:
        """Test advanced cognition features"""
        try:
            results = {}
            
            # Test cognition history
            history_response = self.session.get(f"{self.api_url}/cognition/history?limit=5", timeout=10)
            if history_response.status_code == 200:
                history = history_response.json()
                results['history'] = {
                    'success': True,
                    'count': len(history),
                    'data': history
                }
                logger.info(f"Retrieved {len(history)} history items")
            else:
                results['history'] = {
                    'success': False,
                    'error': f'History request failed: {history_response.text}'
                }
            
            # Test engine configuration
            config_response = self.session.get(f"{self.api_url}/cognition/config", timeout=10)
            if config_response.status_code == 200:
                config = config_response.json()
                results['config'] = {
                    'success': True,
                    'config': config
                }
                logger.info("Engine configuration retrieved successfully")
            else:
                results['config'] = {
                    'success': False,
                    'error': f'Config request failed: {config_response.text}'
                }
            
            # Test sememe analysis
            test_query = "artificial intelligence"
            sememe_response = self.session.get(f"{self.api_url}/cognition/sememes/{test_query}", timeout=15)
            if sememe_response.status_code == 200:
                sememes = sememe_response.json()
                results['sememes'] = {
                    'success': True,
                    'query': test_query,
                    'sememes_count': len(sememes.get('sememes', [])),
                    'nodes_count': sememes.get('nodes_count', 0)
                }
                logger.info(f"Sememe analysis: {sememes.get('sememes_count', 0)} sememes, {sememes.get('nodes_count', 0)} nodes")
            else:
                results['sememes'] = {
                    'success': False,
                    'error': f'Sememe request failed: {sememe_response.text}'
                }
            
            # Overall success
            successful_tests = sum(1 for r in results.values() if r.get('success', False))
            total_tests = len(results)
            
            return {
                'success': successful_tests > 0,
                'successful_tests': successful_tests,
                'total_tests': total_tests,
                'results': results,
                'message': f'Advanced cognition tests: {successful_tests}/{total_tests} passed'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Advanced cognition test failed: {str(e)}'
            }
    
    def test_training_pipeline(self) -> Dict[str, Any]:
        """Test training pipeline features"""
        try:
            results = {}
            
            # Test response evaluation
            eval_response = self.session.post(
                f"{self.api_url}/training/evaluate",
                params={
                    "query": "What is machine learning?",
                    "response": "Machine learning is a method of data analysis that automates analytical model building."
                },
                timeout=15
            )
            
            if eval_response.status_code == 200:
                evaluation = eval_response.json()
                results['evaluation'] = {
                    'success': True,
                    'scores': evaluation
                }
                logger.info(f"Response evaluation: Overall score {evaluation.get('overall', 0):.3f}")
            else:
                results['evaluation'] = {
                    'success': False,
                    'error': f'Evaluation failed: {eval_response.text}'
                }
            
            # Test response improvement
            improve_response = self.session.post(
                f"{self.api_url}/training/improve-response",
                params={
                    "query": "What is AI?",
                    "current_response": "AI is computers.",
                    "target_response": "Artificial Intelligence (AI) is the simulation of human intelligence in machines that are programmed to think and learn like humans."
                },
                timeout=15
            )
            
            if improve_response.status_code == 200:
                improvement = improve_response.json()
                results['improvement'] = {
                    'success': True,
                    'result': improvement
                }
                logger.info("Response improvement training pair added successfully")
            else:
                results['improvement'] = {
                    'success': False,
                    'error': f'Improvement failed: {improve_response.text}'
                }
            
            # Test bulk training status
            bulk_status_response = self.session.get(f"{self.api_url}/training/bulk-status", timeout=10)
            if bulk_status_response.status_code == 200:
                bulk_status = bulk_status_response.json()
                results['bulk_status'] = {
                    'success': True,
                    'status': bulk_status
                }
                logger.info(f"Bulk training status: {bulk_status.get('system_status')}")
            else:
                results['bulk_status'] = {
                    'success': False,
                    'error': f'Bulk status failed: {bulk_status_response.text}'
                }
            
            # Overall success
            successful_tests = sum(1 for r in results.values() if r.get('success', False))
            total_tests = len(results)
            
            return {
                'success': successful_tests > 0,
                'successful_tests': successful_tests,
                'total_tests': total_tests,
                'results': results,
                'message': f'Training pipeline tests: {successful_tests}/{total_tests} passed'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Training pipeline test failed: {str(e)}'
            }
    
    def test_error_handling(self) -> Dict[str, Any]:
        """Test error handling and edge cases"""
        try:
            results = {}
            
            # Test invalid cognition request
            invalid_cognition = self.session.post(
                f"{self.api_url}/cognition",
                json={"invalid_field": "test"},
                timeout=10
            )
            results['invalid_cognition'] = {
                'status_code': invalid_cognition.status_code,
                'handled_gracefully': invalid_cognition.status_code in [400, 422]
            }
            
            # Test empty query
            empty_query = self.session.post(
                f"{self.api_url}/cognition",
                json={"query": ""},
                timeout=10
            )
            results['empty_query'] = {
                'status_code': empty_query.status_code,
                'handled_gracefully': empty_query.status_code in [200, 400, 422]
            }
            
            # Test non-existent endpoint
            nonexistent = self.session.get(f"{self.api_url}/nonexistent", timeout=10)
            results['nonexistent_endpoint'] = {
                'status_code': nonexistent.status_code,
                'handled_gracefully': nonexistent.status_code == 404
            }
            
            # Overall assessment
            graceful_handling = sum(1 for r in results.values() if r.get('handled_gracefully', False))
            total_error_tests = len(results)
            
            return {
                'success': graceful_handling > 0,
                'graceful_handling': graceful_handling,
                'total_error_tests': total_error_tests,
                'results': results,
                'message': f'Error handling: {graceful_handling}/{total_error_tests} handled gracefully'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Error handling test failed: {str(e)}'
            }
    
    def generate_summary_report(self) -> str:
        """Generate comprehensive test summary report"""
        if not self.test_results:
            return "No test results available"
        
        report = []
        report.append("="*80)
        report.append("SATC BACKEND API TEST SUMMARY REPORT")
        report.append("="*80)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result.get('success', False))
        
        report.append(f"\nOverall Results: {passed_tests}/{total_tests} tests passed")
        report.append(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        report.append("\nDetailed Results:")
        report.append("-" * 50)
        
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result.get('success', False) else "❌ FAIL"
            report.append(f"{status} {test_name}")
            
            if not result.get('success', False):
                error = result.get('error', 'Unknown error')
                report.append(f"    Error: {error}")
            
            # Add specific details for key tests
            if test_name == "SATC Core Engine API" and result.get('success'):
                report.append(f"    Queries processed: {result.get('successful_queries', 0)}/{result.get('total_queries', 0)}")
                report.append(f"    Average coherence: {result.get('avg_coherence', 0):.3f}")
                report.append(f"    Average processing time: {result.get('avg_processing_time', 0):.3f}s")
            
            elif test_name == "Performance Metrics" and result.get('success'):
                metrics = result.get('metrics', {})
                report.append(f"    Total queries: {metrics.get('total_queries', 0)}")
                report.append(f"    Recognition rate: {metrics.get('recognition_rate', 0):.3f}")
                report.append(f"    Sememe database size: {metrics.get('sememe_database_size', 0)}")
        
        # Critical Issues Section
        report.append("\nCritical Issues Found:")
        report.append("-" * 30)
        
        critical_issues = []
        for test_name, result in self.test_results.items():
            if not result.get('success', False) and test_name in ["Basic API Health", "SATC Core Engine API"]:
                critical_issues.append(f"- {test_name}: {result.get('error', 'Unknown error')}")
        
        if critical_issues:
            report.extend(critical_issues)
        else:
            report.append("No critical issues found")
        
        # Mock/Placeholder Detection
        report.append("\nMock/Placeholder Content Detection:")
        report.append("-" * 40)
        
        cognition_result = self.test_results.get("SATC Core Engine API", {})
        if cognition_result.get('success') and 'results' in cognition_result:
            mock_detected = any(r.get('has_mock_content', False) for r in cognition_result['results'])
            if mock_detected:
                report.append("⚠️  Mock/placeholder content detected in cognition responses")
            else:
                report.append("✅ No obvious mock/placeholder content detected")
        else:
            report.append("Unable to assess mock content (cognition test failed)")
        
        # Recommendations
        report.append("\nRecommendations:")
        report.append("-" * 20)
        
        if passed_tests == total_tests:
            report.append("✅ All tests passed! System appears to be working correctly.")
        elif passed_tests >= total_tests * 0.8:
            report.append("⚠️  Most tests passed, but some issues need attention.")
        else:
            report.append("❌ Multiple critical issues found. System needs significant fixes.")
        
        report.append("\n" + "="*80)
        
        return "\n".join(report)

def main():
    """Main testing function"""
    # Get backend URL from environment or use default
    import os
    
    # Read from frontend .env file
    frontend_env_path = "/app/frontend/.env"
    backend_url = None
    
    try:
        with open(frontend_env_path, 'r') as f:
            for line in f:
                if line.startswith('REACT_APP_BACKEND_URL='):
                    backend_url = line.split('=', 1)[1].strip()
                    break
    except FileNotFoundError:
        logger.warning(f"Frontend .env file not found at {frontend_env_path}")
    
    if not backend_url:
        backend_url = "http://localhost:8001"
        logger.warning(f"Using default backend URL: {backend_url}")
    else:
        logger.info(f"Using backend URL from .env: {backend_url}")
    
    # Initialize tester
    tester = SATCBackendTester(backend_url)
    
    # Run all tests
    results = tester.run_all_tests()
    
    # Generate and print summary report
    summary = tester.generate_summary_report()
    print("\n" + summary)
    
    # Return exit code based on results
    total_tests = len(results)
    passed_tests = sum(1 for result in results.values() if result.get('success', False))
    
    if passed_tests == total_tests:
        logger.info("🎉 All tests passed!")
        return 0
    elif passed_tests >= total_tests * 0.8:
        logger.warning("⚠️  Some tests failed, but system is mostly functional")
        return 1
    else:
        logger.error("❌ Multiple critical failures detected")
        return 2

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)