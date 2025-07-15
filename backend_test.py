#!/usr/bin/env python3
"""
Comprehensive Backend Testing for SATC System
Tests all API endpoints for the Enhanced SATC Engine
"""

import requests
import json
import time
import sys
from typing import Dict, Any, List
from datetime import datetime

# Backend URL from frontend/.env
BACKEND_URL = "https://e6089aeb-2208-4f9e-8c74-cb69cb6d7583.preview.emergentagent.com/api"

class SATCBackendTester:
    def __init__(self):
        self.base_url = BACKEND_URL
        self.session = requests.Session()
        self.test_results = []
        self.failed_tests = []
        
    def log_test(self, test_name: str, success: bool, details: str = "", response_data: Any = None):
        """Log test results"""
        result = {
            "test": test_name,
            "success": success,
            "details": details,
            "timestamp": datetime.now().isoformat(),
            "response_data": response_data
        }
        self.test_results.append(result)
        
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if details:
            print(f"    Details: {details}")
        if not success:
            self.failed_tests.append(test_name)
        print()

    def test_basic_health_check(self):
        """Test basic API health endpoints"""
        print("=== BASIC HEALTH CHECK TESTS ===")
        
        # Test GET /api/
        try:
            response = self.session.get(f"{self.base_url}/")
            if response.status_code == 200:
                data = response.json()
                if "Enhanced SATC API" in data.get("message", ""):
                    self.log_test("Basic API Health Check", True, f"Status: {response.status_code}, Message: {data['message']}")
                else:
                    self.log_test("Basic API Health Check", False, f"Unexpected message: {data}")
            else:
                self.log_test("Basic API Health Check", False, f"Status: {response.status_code}, Response: {response.text}")
        except Exception as e:
            self.log_test("Basic API Health Check", False, f"Exception: {str(e)}")

        # Test GET /api/cognition/config
        try:
            response = self.session.get(f"{self.base_url}/cognition/config")
            if response.status_code == 200:
                data = response.json()
                required_keys = ["hd_dim", "som_grid_size", "deep_layers_config", "clustering_config", "performance_targets"]
                if all(key in data for key in required_keys):
                    self.log_test("Engine Configuration", True, f"Config loaded with keys: {list(data.keys())}")
                else:
                    self.log_test("Engine Configuration", False, f"Missing config keys. Got: {list(data.keys())}")
            else:
                self.log_test("Engine Configuration", False, f"Status: {response.status_code}, Response: {response.text}")
        except Exception as e:
            self.log_test("Engine Configuration", False, f"Exception: {str(e)}")

        # Test GET /api/training/hardware-info
        try:
            response = self.session.get(f"{self.base_url}/training/hardware-info")
            if response.status_code == 200:
                data = response.json()
                if "hardware_info" in data and "optimization_status" in data:
                    self.log_test("Hardware Info", True, f"Hardware status: {data['optimization_status']}")
                else:
                    self.log_test("Hardware Info", False, f"Missing hardware info keys: {list(data.keys())}")
            else:
                self.log_test("Hardware Info", False, f"Status: {response.status_code}, Response: {response.text}")
        except Exception as e:
            self.log_test("Hardware Info", False, f"Exception: {str(e)}")

    def test_core_cognition(self):
        """Test core cognition endpoints"""
        print("=== CORE COGNITION TESTS ===")
        
        test_queries = [
            "Hello, how are you?",
            "What is artificial intelligence?",
            "How does machine learning work?",
            "What is consciousness?"
        ]
        
        for query in test_queries:
            try:
                payload = {
                    "query": query,
                    "use_recognition": True,
                    "save_to_memory": True
                }
                response = self.session.post(f"{self.base_url}/cognition", json=payload)
                
                if response.status_code == 200:
                    data = response.json()
                    required_fields = ["output", "phase", "success", "coherence", "processing_time"]
                    
                    if all(field in data for field in required_fields):
                        coherence = data.get("coherence", 0)
                        processing_time = data.get("processing_time", 0)
                        self.log_test(f"Cognition Query: '{query[:30]}...'", True, 
                                    f"Phase: {data['phase']}, Coherence: {coherence:.3f}, Time: {processing_time:.3f}s")
                    else:
                        self.log_test(f"Cognition Query: '{query[:30]}...'", False, 
                                    f"Missing fields. Got: {list(data.keys())}")
                else:
                    self.log_test(f"Cognition Query: '{query[:30]}...'", False, 
                                f"Status: {response.status_code}, Response: {response.text}")
            except Exception as e:
                self.log_test(f"Cognition Query: '{query[:30]}...'", False, f"Exception: {str(e)}")

        # Test performance metrics
        try:
            response = self.session.get(f"{self.base_url}/cognition/performance")
            if response.status_code == 200:
                data = response.json()
                required_metrics = ["total_queries", "recognition_hits", "cognition_processes", "avg_coherence"]
                if all(metric in data for metric in required_metrics):
                    self.log_test("Performance Metrics", True, 
                                f"Total queries: {data['total_queries']}, Avg coherence: {data['avg_coherence']:.3f}")
                else:
                    self.log_test("Performance Metrics", False, f"Missing metrics: {list(data.keys())}")
            else:
                self.log_test("Performance Metrics", False, f"Status: {response.status_code}")
        except Exception as e:
            self.log_test("Performance Metrics", False, f"Exception: {str(e)}")

        # Test query history
        try:
            response = self.session.get(f"{self.base_url}/cognition/history?limit=5")
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list):
                    self.log_test("Query History", True, f"Retrieved {len(data)} history entries")
                else:
                    self.log_test("Query History", False, f"Expected list, got: {type(data)}")
            else:
                self.log_test("Query History", False, f"Status: {response.status_code}")
        except Exception as e:
            self.log_test("Query History", False, f"Exception: {str(e)}")

        # Test sememes endpoint
        try:
            test_query = "artificial intelligence"
            response = self.session.get(f"{self.base_url}/cognition/sememes/{test_query}")
            if response.status_code == 200:
                data = response.json()
                if "sememes" in data and "nodes_count" in data:
                    self.log_test("Sememes Extraction", True, 
                                f"Query: {data['query']}, Sememes: {len(data['sememes'])}, Nodes: {data['nodes_count']}")
                else:
                    self.log_test("Sememes Extraction", False, f"Missing sememe data: {list(data.keys())}")
            else:
                self.log_test("Sememes Extraction", False, f"Status: {response.status_code}")
        except Exception as e:
            self.log_test("Sememes Extraction", False, f"Exception: {str(e)}")

    def test_training_system(self):
        """Test training system endpoints"""
        print("=== TRAINING SYSTEM TESTS ===")
        
        # Test add training pair
        try:
            training_pair = {
                "query": "What is machine learning?",
                "response": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.",
                "quality_score": 0.9,
                "coherence_score": 0.85,
                "sememes": ["machine", "learning", "artificial", "intelligence"]
            }
            response = self.session.post(f"{self.base_url}/training/add-pair", json=training_pair)
            
            if response.status_code == 200:
                data = response.json()
                if "message" in data and "successfully" in data["message"]:
                    self.log_test("Add Training Pair", True, data["message"])
                else:
                    self.log_test("Add Training Pair", False, f"Unexpected response: {data}")
            else:
                self.log_test("Add Training Pair", False, f"Status: {response.status_code}, Response: {response.text}")
        except Exception as e:
            self.log_test("Add Training Pair", False, f"Exception: {str(e)}")

        # Test get training data
        try:
            response = self.session.get(f"{self.base_url}/training/data")
            if response.status_code == 200:
                data = response.json()
                if "training_pairs" in data and "count" in data:
                    self.log_test("Get Training Data", True, f"Retrieved {data['count']} training pairs")
                else:
                    self.log_test("Get Training Data", False, f"Missing data fields: {list(data.keys())}")
            else:
                self.log_test("Get Training Data", False, f"Status: {response.status_code}")
        except Exception as e:
            self.log_test("Get Training Data", False, f"Exception: {str(e)}")

        # Test training status
        try:
            response = self.session.get(f"{self.base_url}/training/status")
            if response.status_code == 200:
                data = response.json()
                required_fields = ["is_training", "current_epoch", "total_epochs"]
                if all(field in data for field in required_fields):
                    self.log_test("Training Status", True, 
                                f"Training: {data['is_training']}, Epoch: {data['current_epoch']}/{data['total_epochs']}")
                else:
                    self.log_test("Training Status", False, f"Missing status fields: {list(data.keys())}")
            else:
                self.log_test("Training Status", False, f"Status: {response.status_code}")
        except Exception as e:
            self.log_test("Training Status", False, f"Exception: {str(e)}")

        # Test start training
        try:
            training_request = {
                "training_pairs": [
                    {
                        "query": "What is deep learning?",
                        "response": "Deep learning is a subset of machine learning that uses neural networks with multiple layers to model and understand complex patterns in data.",
                        "quality_score": 0.9,
                        "coherence_score": 0.88,
                        "sememes": ["deep", "learning", "neural", "networks"]
                    }
                ],
                "epochs": 5,
                "batch_size": 8,
                "learning_rate": 0.001
            }
            response = self.session.post(f"{self.base_url}/training/start", json=training_request)
            
            if response.status_code == 200:
                data = response.json()
                if "message" in data and "config" in data:
                    self.log_test("Start Training", True, f"{data['message']}, Config: {data['config']}")
                else:
                    self.log_test("Start Training", False, f"Unexpected response: {data}")
            else:
                self.log_test("Start Training", False, f"Status: {response.status_code}, Response: {response.text}")
        except Exception as e:
            self.log_test("Start Training", False, f"Exception: {str(e)}")

        # Test response evaluation
        try:
            params = {
                "query": "What is AI?",
                "response": "Artificial Intelligence is the simulation of human intelligence in machines."
            }
            response = self.session.post(f"{self.base_url}/training/evaluate", params=params)
            
            if response.status_code == 200:
                data = response.json()
                required_scores = ["coherence", "relevance", "informativeness", "fluency", "overall"]
                if all(score in data for score in required_scores):
                    self.log_test("Response Evaluation", True, 
                                f"Overall score: {data['overall']:.3f}, Coherence: {data['coherence']:.3f}")
                else:
                    self.log_test("Response Evaluation", False, f"Missing evaluation scores: {list(data.keys())}")
            else:
                self.log_test("Response Evaluation", False, f"Status: {response.status_code}")
        except Exception as e:
            self.log_test("Response Evaluation", False, f"Exception: {str(e)}")

        # Test improve response
        try:
            params = {
                "query": "What is consciousness?",
                "current_response": "Consciousness is awareness.",
                "target_response": "Consciousness is the state of being aware of and able to think about one's existence, sensations, thoughts, and surroundings."
            }
            response = self.session.post(f"{self.base_url}/training/improve-response", params=params)
            
            if response.status_code == 200:
                data = response.json()
                if "message" in data and "evaluation" in data:
                    self.log_test("Improve Response", True, f"{data['message']}")
                else:
                    self.log_test("Improve Response", False, f"Unexpected response: {data}")
            else:
                self.log_test("Improve Response", False, f"Status: {response.status_code}")
        except Exception as e:
            self.log_test("Improve Response", False, f"Exception: {str(e)}")

    def test_bulk_training_system(self):
        """Test bulk training system endpoints"""
        print("=== BULK TRAINING SYSTEM TESTS ===")
        
        # Test Hello World system creation
        try:
            hello_world_request = {"quick_start": True}
            response = self.session.post(f"{self.base_url}/training/hello-world", json=hello_world_request)
            
            if response.status_code == 200:
                data = response.json()
                if "message" in data and "status" in data and data["status"] == "ready":
                    self.log_test("Hello World System", True, f"{data['message']}, Features: {len(data.get('features', []))}")
                else:
                    self.log_test("Hello World System", False, f"Unexpected response: {data}")
            else:
                self.log_test("Hello World System", False, f"Status: {response.status_code}, Response: {response.text}")
        except Exception as e:
            self.log_test("Hello World System", False, f"Exception: {str(e)}")

        # Test bulk training status
        try:
            response = self.session.get(f"{self.base_url}/training/bulk-status")
            if response.status_code == 200:
                data = response.json()
                required_fields = ["system_status", "hardware_optimized", "hardware_specs", "ready_for_deployment"]
                if all(field in data for field in required_fields):
                    self.log_test("Bulk Training Status", True, 
                                f"Status: {data['system_status']}, Hardware optimized: {data['hardware_optimized']}")
                else:
                    self.log_test("Bulk Training Status", False, f"Missing status fields: {list(data.keys())}")
            else:
                self.log_test("Bulk Training Status", False, f"Status: {response.status_code}")
        except Exception as e:
            self.log_test("Bulk Training Status", False, f"Exception: {str(e)}")

        # Test create sample dataset
        try:
            response = self.session.post(f"{self.base_url}/training/create-sample-dataset")
            if response.status_code == 200:
                data = response.json()
                if "message" in data and "pairs_count" in data:
                    self.log_test("Create Sample Dataset", True, 
                                f"{data['message']}, Pairs: {data['pairs_count']}")
                else:
                    self.log_test("Create Sample Dataset", False, f"Unexpected response: {data}")
            else:
                self.log_test("Create Sample Dataset", False, f"Status: {response.status_code}")
        except Exception as e:
            self.log_test("Create Sample Dataset", False, f"Exception: {str(e)}")

        # Test bulk upload (with sample JSON data)
        try:
            sample_data = json.dumps([
                {
                    "query": "What is neural plasticity?",
                    "response": "Neural plasticity is the brain's ability to reorganize and form new neural connections throughout life.",
                    "quality_score": 0.9,
                    "coherence_score": 0.85
                },
                {
                    "query": "How does memory work?",
                    "response": "Memory involves encoding, storing, and retrieving information through complex neural networks in the brain.",
                    "quality_score": 0.88,
                    "coherence_score": 0.82
                }
            ])
            
            bulk_upload = {
                "format": "json",
                "data": sample_data
            }
            response = self.session.post(f"{self.base_url}/training/bulk-upload", json=bulk_upload)
            
            if response.status_code == 200:
                data = response.json()
                if "message" in data and "pairs_imported" in data:
                    self.log_test("Bulk Upload", True, 
                                f"{data['message']}, Imported: {data['pairs_imported']} pairs")
                else:
                    self.log_test("Bulk Upload", False, f"Unexpected response: {data}")
            else:
                self.log_test("Bulk Upload", False, f"Status: {response.status_code}, Response: {response.text}")
        except Exception as e:
            self.log_test("Bulk Upload", False, f"Exception: {str(e)}")

        # Test automated training start
        try:
            automated_request = {
                "hours_per_day": 16,
                "rest_hours": 8,
                "max_epochs": 100,
                "save_every_n_epochs": 10
            }
            response = self.session.post(f"{self.base_url}/training/automated-start", json=automated_request)
            
            if response.status_code == 200:
                data = response.json()
                if "message" in data and "config" in data:
                    self.log_test("Automated Training", True, f"{data['message']}, Status: {data.get('status', 'unknown')}")
                else:
                    self.log_test("Automated Training", False, f"Unexpected response: {data}")
            else:
                self.log_test("Automated Training", False, f"Status: {response.status_code}")
        except Exception as e:
            self.log_test("Automated Training", False, f"Exception: {str(e)}")

    def test_advanced_features(self):
        """Test advanced features and edge cases"""
        print("=== ADVANCED FEATURES TESTS ===")
        
        # Test engine reset
        try:
            response = self.session.post(f"{self.base_url}/cognition/reset")
            if response.status_code == 200:
                data = response.json()
                if "message" in data and "reset" in data["message"]:
                    self.log_test("Engine Reset", True, data["message"])
                else:
                    self.log_test("Engine Reset", False, f"Unexpected response: {data}")
            else:
                self.log_test("Engine Reset", False, f"Status: {response.status_code}")
        except Exception as e:
            self.log_test("Engine Reset", False, f"Exception: {str(e)}")

        # Test error handling with invalid cognition query
        try:
            invalid_payload = {"query": ""}  # Empty query
            response = self.session.post(f"{self.base_url}/cognition", json=invalid_payload)
            
            # This should either work (handle empty query) or return proper error
            if response.status_code in [200, 400, 422]:
                self.log_test("Error Handling - Empty Query", True, f"Proper error handling, Status: {response.status_code}")
            else:
                self.log_test("Error Handling - Empty Query", False, f"Unexpected status: {response.status_code}")
        except Exception as e:
            self.log_test("Error Handling - Empty Query", False, f"Exception: {str(e)}")

        # Test invalid training data
        try:
            invalid_training = {
                "training_pairs": [],  # Empty training pairs
                "epochs": 0,
                "batch_size": 0
            }
            response = self.session.post(f"{self.base_url}/training/start", json=invalid_training)
            
            # Should handle gracefully
            if response.status_code in [200, 400, 422]:
                self.log_test("Error Handling - Invalid Training", True, f"Proper error handling, Status: {response.status_code}")
            else:
                self.log_test("Error Handling - Invalid Training", False, f"Unexpected status: {response.status_code}")
        except Exception as e:
            self.log_test("Error Handling - Invalid Training", False, f"Exception: {str(e)}")

    def run_all_tests(self):
        """Run all test suites"""
        print("🚀 Starting SATC Backend Comprehensive Testing")
        print(f"Backend URL: {self.base_url}")
        print("=" * 60)
        
        start_time = time.time()
        
        # Run test suites
        self.test_basic_health_check()
        self.test_core_cognition()
        self.test_training_system()
        self.test_bulk_training_system()
        self.test_advanced_features()
        
        end_time = time.time()
        
        # Print summary
        print("=" * 60)
        print("🏁 TESTING COMPLETE")
        print(f"Total tests run: {len(self.test_results)}")
        print(f"Passed: {len([t for t in self.test_results if t['success']])}")
        print(f"Failed: {len(self.failed_tests)}")
        print(f"Total time: {end_time - start_time:.2f} seconds")
        
        if self.failed_tests:
            print("\n❌ FAILED TESTS:")
            for test in self.failed_tests:
                print(f"  - {test}")
        else:
            print("\n✅ ALL TESTS PASSED!")
        
        return len(self.failed_tests) == 0

def main():
    """Main testing function"""
    tester = SATCBackendTester()
    success = tester.run_all_tests()
    
    # Save detailed results
    with open('/app/backend_test_results.json', 'w') as f:
        json.dump(tester.test_results, f, indent=2)
    
    print(f"\nDetailed results saved to: /app/backend_test_results.json")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())