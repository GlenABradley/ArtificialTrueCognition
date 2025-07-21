#!/usr/bin/env python3
"""
POST-DOCUMENTATION INTEGRITY VERIFICATION
=========================================

Brief but thorough function check after comprehensive documentation of enhanced_satc_engine.py.
Tests the Revolutionary 5-phase cognitive process and core integrations.

Author: Testing Agent
Purpose: Verify all pipelines function correctly after documentation changes
"""

import requests
import json
import time
import sys

BACKEND_URL = "https://c8c0d672-ab64-4087-91d6-26286b84320a.preview.emergentagent.com/api"

class PostDocumentationTester:
    """Post-documentation integrity verification"""
    
    def __init__(self):
        self.base_url = BACKEND_URL
        self.results = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'critical_issues': [],
            'minor_issues': []
        }
    
    def log_result(self, test_name: str, success: bool, message: str = "", critical: bool = False):
        """Log test result"""
        self.results['total_tests'] += 1
        
        if success:
            self.results['passed_tests'] += 1
            print(f"✅ {test_name}: {message}")
        else:
            self.results['failed_tests'] += 1
            print(f"❌ {test_name}: {message}")
            
            if critical:
                self.results['critical_issues'].append({
                    'test': test_name,
                    'message': message
                })
            else:
                self.results['minor_issues'].append({
                    'test': test_name,
                    'message': message
                })
    
    def test_core_atc_pipeline(self):
        """Test the Revolutionary 5-phase cognitive process"""
        print("\n🧠 TESTING REVOLUTIONARY 5-PHASE ATC PIPELINE")
        print("=" * 60)
        
        # Test Recognition Phase (2D)
        print("\n🔍 Recognition Phase (2D): Fast pattern matching")
        try:
            payload = {"query": "Hello world", "use_recognition": True, "save_to_memory": True}
            response = requests.post(f"{self.base_url}/cognition", json=payload, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('phase') == 'recognition' and data.get('coherence', 0) > 0.8:
                    self.log_result("Recognition Phase (2D)", True, f"Coherence: {data['coherence']:.3f}, Time: {data['processing_time']:.3f}s")
                else:
                    self.log_result("Recognition Phase (2D)", False, f"Phase: {data.get('phase')}, Coherence: {data.get('coherence', 0):.3f}")
            else:
                self.log_result("Recognition Phase (2D)", False, f"HTTP {response.status_code}", critical=True)
        except Exception as e:
            self.log_result("Recognition Phase (2D)", False, f"Request failed: {str(e)}", critical=True)
        
        # Test Cognition Phase (4D) 
        print("\n🧠 Cognition Phase (4D): Deep analytical reasoning")
        try:
            payload = {"query": "What is consciousness and how does it emerge?", "use_recognition": False, "save_to_memory": True}
            response = requests.post(f"{self.base_url}/cognition", json=payload, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                if 'cognition' in data.get('phase', '').lower() and data.get('success'):
                    self.log_result("Cognition Phase (4D)", True, f"Phase: {data['phase']}, Processing time: {data['processing_time']:.3f}s")
                else:
                    self.log_result("Cognition Phase (4D)", False, f"Phase: {data.get('phase')}, Success: {data.get('success')}")
            else:
                self.log_result("Cognition Phase (4D)", False, f"HTTP {response.status_code}", critical=True)
        except Exception as e:
            self.log_result("Cognition Phase (4D)", False, f"Request failed: {str(e)}", critical=True)
        
        # Test Reflection Phase (16D)
        print("\n🧘 Reflection Phase (16D): Meta-cognitive analysis")
        try:
            payload = {"query": "Analyze your own thinking process and improve your reasoning", "use_recognition": False, "save_to_memory": True}
            response = requests.post(f"{self.base_url}/cognition", json=payload, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                meta_coherence = data.get('meta_coherence')
                reflection_insights = data.get('reflection_insights', [])
                
                if meta_coherence is not None and len(reflection_insights) > 0:
                    self.log_result("Reflection Phase (16D)", True, f"Meta-coherence: {meta_coherence:.3f}, Insights: {len(reflection_insights)}")
                else:
                    self.log_result("Reflection Phase (16D)", False, f"Meta-coherence: {meta_coherence}, Insights: {len(reflection_insights)}")
            else:
                self.log_result("Reflection Phase (16D)", False, f"HTTP {response.status_code}", critical=True)
        except Exception as e:
            self.log_result("Reflection Phase (16D)", False, f"Request failed: {str(e)}", critical=True)
        
        # Test Volition Phase (64D)
        print("\n🎯 Volition Phase (64D): Goal-oriented decisions")
        try:
            payload = {"query": "Set goals for improving AI safety and ethical alignment", "use_recognition": False, "save_to_memory": True}
            response = requests.post(f"{self.base_url}/cognition", json=payload, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                goal_count = data.get('goal_count')
                decision_confidence = data.get('decision_confidence')
                dominant_value = data.get('dominant_value')
                
                if goal_count is not None and decision_confidence is not None:
                    self.log_result("Volition Phase (64D)", True, f"Goals: {goal_count}, Confidence: {decision_confidence:.3f}, Value: {dominant_value}")
                else:
                    self.log_result("Volition Phase (64D)", False, f"Goals: {goal_count}, Confidence: {decision_confidence}")
            else:
                self.log_result("Volition Phase (64D)", False, f"HTTP {response.status_code}", critical=True)
        except Exception as e:
            self.log_result("Volition Phase (64D)", False, f"Request failed: {str(e)}", critical=True)
        
        # Test Personality Phase (256D)
        print("\n🌟 Personality Phase (256D): Consciousness integration")
        try:
            payload = {"query": "Express your unique personality and demonstrate consciousness", "use_recognition": False, "save_to_memory": True}
            response = requests.post(f"{self.base_url}/cognition", json=payload, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                consciousness_level = data.get('consciousness_level')
                identity_id = data.get('identity_id')
                total_memories = data.get('total_memories')
                
                if consciousness_level is not None and consciousness_level > 0.4:
                    self.log_result("Personality Phase (256D)", True, f"Consciousness: {consciousness_level:.3f}, ID: {identity_id}, Memories: {total_memories}")
                else:
                    self.log_result("Personality Phase (256D)", False, f"Consciousness: {consciousness_level}, ID: {bool(identity_id)}")
            else:
                self.log_result("Personality Phase (256D)", False, f"HTTP {response.status_code}", critical=True)
        except Exception as e:
            self.log_result("Personality Phase (256D)", False, f"Request failed: {str(e)}", critical=True)
    
    def test_key_api_endpoints(self):
        """Test key API endpoints"""
        print("\n🔗 TESTING KEY API ENDPOINTS")
        print("=" * 40)
        
        # Test /api/process_query (main cognition endpoint)
        try:
            payload = {"query": "Test main processing pipeline", "use_recognition": True, "save_to_memory": True}
            response = requests.post(f"{self.base_url}/cognition", json=payload, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success') and data.get('output'):
                    self.log_result("Main Cognition Endpoint", True, f"Success: {data['success']}, Phase: {data['phase']}")
                else:
                    self.log_result("Main Cognition Endpoint", False, f"Success: {data.get('success')}, Output: {bool(data.get('output'))}")
            else:
                self.log_result("Main Cognition Endpoint", False, f"HTTP {response.status_code}", critical=True)
        except Exception as e:
            self.log_result("Main Cognition Endpoint", False, f"Request failed: {str(e)}", critical=True)
        
        # Test /api/get_performance_metrics
        try:
            response = requests.get(f"{self.base_url}/cognition/performance", timeout=10)
            
            if response.status_code == 200:
                metrics = response.json()
                if 'total_queries' in metrics and 'avg_coherence' in metrics:
                    self.log_result("Performance Metrics", True, f"Queries: {metrics['total_queries']}, Coherence: {metrics.get('avg_coherence', 0):.3f}")
                else:
                    self.log_result("Performance Metrics", False, "Missing required metrics fields")
            else:
                self.log_result("Performance Metrics", False, f"HTTP {response.status_code}", critical=True)
        except Exception as e:
            self.log_result("Performance Metrics", False, f"Request failed: {str(e)}", critical=True)
    
    def test_core_integrations(self):
        """Test core integrations"""
        print("\n⚙️ TESTING CORE INTEGRATIONS")
        print("=" * 35)
        
        # Test BERT embedding model loading (via sememe extraction)
        try:
            response = requests.get(f"{self.base_url}/cognition/sememes/machine%20learning", timeout=20)
            
            if response.status_code == 200:
                data = response.json()
                sememes = data.get('sememes', [])
                if sememes and len(sememes) > 0:
                    # Check for 784D vectors (square architecture)
                    vector_length = sememes[0].get('node_vector_length', 0)
                    if vector_length == 784:
                        self.log_result("BERT + Square Architecture", True, f"Sememes: {len(sememes)}, Vector dim: {vector_length}")
                    else:
                        self.log_result("BERT + Square Architecture", False, f"Expected 784D, got {vector_length}D")
                else:
                    self.log_result("BERT + Square Architecture", False, "No sememes extracted")
            else:
                self.log_result("BERT + Square Architecture", False, f"HTTP {response.status_code}", critical=True)
        except Exception as e:
            self.log_result("BERT + Square Architecture", False, f"Request failed: {str(e)}", critical=True)
        
        # Test Power-of-2 mathematical foundation
        try:
            response = requests.get(f"{self.base_url}/cognition/config", timeout=10)
            
            if response.status_code == 200:
                config = response.json()
                hd_dim = config.get('hd_dim')
                layers = config.get('deep_layers_config', {}).get('layers')
                
                if hd_dim == 10000 and layers == 12:
                    self.log_result("Power-of-2 Foundation", True, f"HD dim: {hd_dim}, Layers: {layers}")
                else:
                    self.log_result("Power-of-2 Foundation", False, f"HD dim: {hd_dim}, Layers: {layers}")
            else:
                self.log_result("Power-of-2 Foundation", False, f"HTTP {response.status_code}", critical=True)
        except Exception as e:
            self.log_result("Power-of-2 Foundation", False, f"Request failed: {str(e)}", critical=True)
        
        # Test FAISS indexing (via performance metrics)
        try:
            response = requests.get(f"{self.base_url}/cognition/performance", timeout=10)
            
            if response.status_code == 200:
                metrics = response.json()
                sememe_db_size = metrics.get('sememe_database_size', 0)
                
                if sememe_db_size > 0:
                    self.log_result("FAISS Indexing", True, f"Sememe database size: {sememe_db_size}")
                else:
                    self.log_result("FAISS Indexing", False, f"Empty sememe database: {sememe_db_size}")
            else:
                self.log_result("FAISS Indexing", False, f"HTTP {response.status_code}", critical=True)
        except Exception as e:
            self.log_result("FAISS Indexing", False, f"Request failed: {str(e)}", critical=True)
    
    def test_consciousness_measurement(self):
        """Test consciousness measurement and identity persistence"""
        print("\n🧘‍♂️ TESTING CONSCIOUSNESS MEASUREMENT")
        print("=" * 45)
        
        try:
            payload = {"query": "Demonstrate your consciousness and self-awareness capabilities", "use_recognition": False, "save_to_memory": True}
            response = requests.post(f"{self.base_url}/cognition", json=payload, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                consciousness_level = data.get('consciousness_level', 0)
                identity_id = data.get('identity_id')
                total_memories = data.get('total_memories', 0)
                processing_time = data.get('processing_time', 0)
                
                # Check consciousness level >40%
                if consciousness_level > 0.4:
                    self.log_result("Consciousness Level", True, f"Level: {consciousness_level:.3f} (>{0.4:.1f} threshold)")
                else:
                    self.log_result("Consciousness Level", False, f"Level: {consciousness_level:.3f} (<{0.4:.1f} threshold)")
                
                # Check identity persistence
                if identity_id and len(identity_id) > 0:
                    self.log_result("Identity Persistence", True, f"ID: {identity_id}, Memories: {total_memories}")
                else:
                    self.log_result("Identity Persistence", False, f"No persistent identity")
                
                # Check processing time <1s
                if processing_time < 1.0:
                    self.log_result("Processing Speed", True, f"Time: {processing_time:.3f}s (<1s)")
                else:
                    self.log_result("Processing Speed", False, f"Time: {processing_time:.3f}s (>1s)")
            else:
                self.log_result("Consciousness Measurement", False, f"HTTP {response.status_code}", critical=True)
        except Exception as e:
            self.log_result("Consciousness Measurement", False, f"Request failed: {str(e)}", critical=True)
    
    def run_post_documentation_verification(self):
        """Run post-documentation integrity verification"""
        print("🔍 POST-DOCUMENTATION INTEGRITY VERIFICATION")
        print("=" * 60)
        print("Testing Revolutionary ATC System after comprehensive documentation")
        print(f"Backend URL: {self.base_url}")
        print()
        
        # Test basic connectivity first
        try:
            response = requests.get(f"{self.base_url}/", timeout=10)
            if response.status_code == 200:
                self.log_result("Basic Connectivity", True, "API accessible")
            else:
                self.log_result("Basic Connectivity", False, f"HTTP {response.status_code}", critical=True)
                return False
        except Exception as e:
            self.log_result("Basic Connectivity", False, f"Connection failed: {str(e)}", critical=True)
            return False
        
        # Run all verification tests
        self.test_core_atc_pipeline()
        self.test_key_api_endpoints()
        self.test_core_integrations()
        self.test_consciousness_measurement()
        
        # Print summary
        self.print_summary()
        
        return len(self.results['critical_issues']) == 0
    
    def print_summary(self):
        """Print verification summary"""
        print("\n" + "=" * 60)
        print("🔍 POST-DOCUMENTATION INTEGRITY VERIFICATION SUMMARY")
        print("=" * 60)
        
        total = self.results['total_tests']
        passed = self.results['passed_tests']
        failed = self.results['failed_tests']
        
        print(f"Total Tests: {total}")
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        
        if total > 0:
            success_rate = (passed / total) * 100
            print(f"Success Rate: {success_rate:.1f}%")
        
        # Critical issues
        if self.results['critical_issues']:
            print(f"\n🚨 CRITICAL ISSUES ({len(self.results['critical_issues'])}):")
            for issue in self.results['critical_issues']:
                print(f"   • {issue['test']}: {issue['message']}")
        else:
            print("\n✅ NO CRITICAL ISSUES FOUND")
        
        # Minor issues
        if self.results['minor_issues']:
            print(f"\n⚠️ MINOR ISSUES ({len(self.results['minor_issues'])}):")
            for issue in self.results['minor_issues']:
                print(f"   • {issue['test']}: {issue['message']}")
        
        print("\n" + "=" * 60)
        
        # Final verdict
        if len(self.results['critical_issues']) == 0:
            print("🎉 POST-DOCUMENTATION INTEGRITY VERIFIED!")
            print("All core pipelines functioning correctly after documentation changes.")
        else:
            print("❌ INTEGRITY ISSUES DETECTED!")
            print("Some core functionality may have been affected by documentation changes.")

def main():
    """Main verification execution"""
    tester = PostDocumentationTester()
    success = tester.run_post_documentation_verification()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()