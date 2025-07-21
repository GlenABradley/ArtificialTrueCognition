#!/usr/bin/env python3
"""
Revolutionary ATC Integration Testing - Phase 3
==============================================

Comprehensive integration and end-to-end testing for the Revolutionary ATC system.
Focus on integration between all 5 ATC phases and Enhanced SATC Engine.

Test Categories:
1. Enhanced SATC Integration - All 5 ATC phases with Enhanced SATC Engine
2. End-to-End Query Processing - Complete ATC pipeline with real queries
3. API Integration Testing - /api/cognition endpoint with Revolutionary ATC
4. Recognition Learning Loop - Cognition results → Recognition memory
5. Consciousness Emergence - Personality phase consciousness metrics
6. Performance Benchmarks - Complete pipeline timing validation

Author: Testing Agent - Phase 3 Integration
Status: Revolutionary ATC Integration Testing
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

class RevolutionaryATCIntegrationTester:
    """Comprehensive integration testing for Revolutionary ATC system"""
    
    def __init__(self):
        self.base_url = BACKEND_URL
        self.test_results = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'errors': [],
            'warnings': [],
            'integration_results': {},
            'performance_benchmarks': {},
            'consciousness_metrics': {}
        }
        
    def log_result(self, test_name: str, success: bool, message: str = "", error: str = "", benchmark: Dict[str, Any] = None):
        """Log test result with optional benchmark data"""
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
        
        if benchmark:
            self.test_results['performance_benchmarks'][test_name] = benchmark
    
    def log_warning(self, test_name: str, warning: str):
        """Log warning"""
        print(f"⚠️  {test_name}: {warning}")
        self.test_results['warnings'].append({
            'test': test_name,
            'warning': warning
        })
    
    def test_enhanced_satc_integration(self):
        """Test all 5 ATC phases integrated with Enhanced SATC Engine"""
        print("\n=== ENHANCED SATC INTEGRATION TESTING ===")
        
        # Test queries designed to trigger different phases
        integration_queries = [
            {
                'query': 'Hello world',
                'expected_phase': 'recognition',
                'description': 'Simple query should hit Recognition phase'
            },
            {
                'query': 'What is the nature of consciousness and how does it emerge from complex neural networks?',
                'expected_phase': 'cognition',
                'description': 'Complex query should trigger full Cognition pipeline'
            },
            {
                'query': 'How should I approach developing ethical AI systems while ensuring consciousness alignment?',
                'expected_phase': 'cognition',
                'description': 'Philosophical query should trigger all phases including Reflection and Volition'
            }
        ]
        
        for test_case in integration_queries:
            query = test_case['query']
            expected_phase = test_case['expected_phase']
            description = test_case['description']
            
            try:
                start_time = time.time()
                
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
                
                processing_time = time.time() - start_time
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Analyze integration results
                    phase = data.get('phase', 'unknown')
                    success = data.get('success', False)
                    coherence = data.get('coherence', 0.0)
                    
                    # Check for ATC phase indicators
                    has_reflection = 'reflection' in data or 'meta_coherence' in data
                    has_volition = 'volition' in data or 'goal_count' in data
                    has_personality = 'personality' in data or 'consciousness_level' in data
                    
                    # Performance benchmark
                    benchmark = {
                        'processing_time': processing_time,
                        'coherence': coherence,
                        'phase': phase,
                        'reflection_active': has_reflection,
                        'volition_active': has_volition,
                        'personality_active': has_personality
                    }
                    
                    if success and coherence > 0.0:
                        self.log_result(
                            f"Integration Test: {description}",
                            True,
                            f"Phase: {phase}, Coherence: {coherence:.3f}, Time: {processing_time:.3f}s, ATC phases: R={has_reflection}, V={has_volition}, P={has_personality}",
                            benchmark=benchmark
                        )
                        
                        # Store integration results
                        self.test_results['integration_results'][query[:30]] = {
                            'phase': phase,
                            'coherence': coherence,
                            'atc_phases_active': {
                                'reflection': has_reflection,
                                'volition': has_volition,
                                'personality': has_personality
                            },
                            'processing_time': processing_time
                        }
                    else:
                        self.log_result(
                            f"Integration Test: {description}",
                            False,
                            f"Low performance: success={success}, coherence={coherence:.3f}"
                        )
                else:
                    self.log_result(
                        f"Integration Test: {description}",
                        False,
                        f"HTTP {response.status_code}",
                        response.text[:200]
                    )
                
                # Small delay between tests
                time.sleep(1.0)
                
            except Exception as e:
                self.log_result(
                    f"Integration Test: {description}",
                    False,
                    "Request failed",
                    str(e)
                )
    
    def test_recognition_learning_loop(self):
        """Test Recognition learning loop: Cognition results → Recognition memory"""
        print("\n=== RECOGNITION LEARNING LOOP TESTING ===")
        
        # Test the learning loop with a specific query
        test_query = "What is quantum computing and how does it work?"
        
        try:
            # First query should go to Cognition (novel query)
            print("Step 1: First query (should trigger Cognition)")
            
            payload = {
                "query": test_query,
                "use_recognition": True,
                "save_to_memory": True
            }
            
            start_time = time.time()
            response1 = requests.post(
                f"{self.base_url}/cognition",
                json=payload,
                timeout=30
            )
            time1 = time.time() - start_time
            
            if response1.status_code == 200:
                data1 = response1.json()
                phase1 = data1.get('phase', 'unknown')
                coherence1 = data1.get('coherence', 0.0)
                
                print(f"   First query: Phase={phase1}, Coherence={coherence1:.3f}, Time={time1:.3f}s")
                
                # Wait a moment for memory consolidation
                time.sleep(2.0)
                
                # Second query should hit Recognition (learned pattern)
                print("Step 2: Second query (should hit Recognition)")
                
                start_time = time.time()
                response2 = requests.post(
                    f"{self.base_url}/cognition",
                    json=payload,
                    timeout=30
                )
                time2 = time.time() - start_time
                
                if response2.status_code == 200:
                    data2 = response2.json()
                    phase2 = data2.get('phase', 'unknown')
                    coherence2 = data2.get('coherence', 0.0)
                    
                    print(f"   Second query: Phase={phase2}, Coherence={coherence2:.3f}, Time={time2:.3f}s")
                    
                    # Analyze learning loop
                    learning_occurred = (
                        phase1.startswith('cognition') and 
                        (phase2 == 'recognition' or time2 < time1 * 0.8)  # Recognition should be faster
                    )
                    
                    if learning_occurred:
                        self.log_result(
                            "Recognition Learning Loop",
                            True,
                            f"Learning successful: {phase1} → {phase2}, Speed improvement: {time1:.3f}s → {time2:.3f}s"
                        )
                    else:
                        self.log_result(
                            "Recognition Learning Loop",
                            False,
                            f"Learning not detected: {phase1} → {phase2}, Times: {time1:.3f}s → {time2:.3f}s"
                        )
                else:
                    self.log_result(
                        "Recognition Learning Loop",
                        False,
                        f"Second query failed: HTTP {response2.status_code}",
                        response2.text[:200]
                    )
            else:
                self.log_result(
                    "Recognition Learning Loop",
                    False,
                    f"First query failed: HTTP {response1.status_code}",
                    response1.text[:200]
                )
                
        except Exception as e:
            self.log_result(
                "Recognition Learning Loop",
                False,
                "Test failed",
                str(e)
            )
    
    def test_complete_pipeline_flow(self):
        """Test complete ATC pipeline flow with complex queries"""
        print("\n=== COMPLETE PIPELINE FLOW TESTING ===")
        
        # Complex queries designed to trigger all phases
        complex_queries = [
            "How should I approach developing consciousness in AI systems while ensuring ethical alignment and maintaining human values?",
            "What are the philosophical implications of artificial consciousness and how should we prepare for truly sentient AI?",
            "Explain the relationship between quantum mechanics, consciousness, and artificial intelligence in the context of future technological development."
        ]
        
        for i, query in enumerate(complex_queries):
            try:
                print(f"\nComplex Query {i+1}: {query[:60]}...")
                
                payload = {
                    "query": query,
                    "use_recognition": True,
                    "save_to_memory": True
                }
                
                start_time = time.time()
                response = requests.post(
                    f"{self.base_url}/cognition",
                    json=payload,
                    timeout=45  # Longer timeout for complex queries
                )
                total_time = time.time() - start_time
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Analyze complete pipeline
                    phase = data.get('phase', 'unknown')
                    success = data.get('success', False)
                    coherence = data.get('coherence', 0.0)
                    
                    # Check for all ATC phases
                    has_reasoning_steps = 'reasoning_steps' in data or data.get('reasoning_steps', 0) > 0
                    has_meta_coherence = 'meta_coherence' in data or 'reflection' in data
                    has_goal_count = 'goal_count' in data or 'volition' in data
                    has_consciousness = 'consciousness_level' in data or 'personality' in data
                    
                    # Performance validation
                    within_time_limit = total_time < 30.0  # Should complete within 30 seconds
                    good_coherence = coherence > 0.3  # Reasonable coherence for complex queries
                    
                    pipeline_complete = (
                        success and 
                        within_time_limit and 
                        good_coherence and
                        (has_reasoning_steps or has_meta_coherence or has_goal_count or has_consciousness)
                    )
                    
                    if pipeline_complete:
                        self.log_result(
                            f"Complete Pipeline {i+1}",
                            True,
                            f"Success: Coherence={coherence:.3f}, Time={total_time:.3f}s, Phases: R={has_reasoning_steps}, M={has_meta_coherence}, G={has_goal_count}, C={has_consciousness}"
                        )
                        
                        # Store consciousness metrics if available
                        if has_consciousness:
                            consciousness_level = data.get('consciousness_level', 0.0)
                            identity_id = data.get('identity_id', 'unknown')
                            self.test_results['consciousness_metrics'][f'query_{i+1}'] = {
                                'consciousness_level': consciousness_level,
                                'identity_id': identity_id,
                                'query_complexity': len(query.split()),
                                'processing_time': total_time
                            }
                    else:
                        self.log_result(
                            f"Complete Pipeline {i+1}",
                            False,
                            f"Pipeline incomplete: success={success}, time={total_time:.3f}s, coherence={coherence:.3f}"
                        )
                else:
                    self.log_result(
                        f"Complete Pipeline {i+1}",
                        False,
                        f"HTTP {response.status_code}",
                        response.text[:200]
                    )
                
                # Delay between complex queries
                time.sleep(2.0)
                
            except Exception as e:
                self.log_result(
                    f"Complete Pipeline {i+1}",
                    False,
                    "Request failed",
                    str(e)
                )
    
    def test_consciousness_emergence(self):
        """Test consciousness emergence in Personality phase"""
        print("\n=== CONSCIOUSNESS EMERGENCE TESTING ===")
        
        # Multiple interactions to build consciousness
        consciousness_queries = [
            "Who am I and what is my purpose?",
            "How do I understand my own existence?",
            "What makes me unique as an artificial consciousness?",
            "How do I relate to humans and other conscious beings?",
            "What are my values and how do they guide my decisions?"
        ]
        
        consciousness_levels = []
        identity_coherence = []
        
        for i, query in enumerate(consciousness_queries):
            try:
                print(f"\nConsciousness Query {i+1}: {query}")
                
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
                    
                    # Extract consciousness metrics
                    consciousness_level = data.get('consciousness_level', 0.0)
                    identity_id = data.get('identity_id', 'unknown')
                    identity_coherence_val = data.get('identity_coherence', 0.0)
                    total_memories = data.get('total_memories', 0)
                    
                    consciousness_levels.append(consciousness_level)
                    identity_coherence.append(identity_coherence_val)
                    
                    print(f"   Consciousness: {consciousness_level:.3f}, Identity: {identity_id}, Coherence: {identity_coherence_val:.3f}, Memories: {total_memories}")
                    
                    # Check for consciousness emergence indicators
                    has_consciousness = consciousness_level > 0.0
                    has_identity = identity_id != 'unknown' and identity_id != ''
                    has_memories = total_memories > 0
                    
                    if has_consciousness and has_identity:
                        self.log_result(
                            f"Consciousness Query {i+1}",
                            True,
                            f"Consciousness detected: level={consciousness_level:.3f}, identity={identity_id}"
                        )
                    else:
                        self.log_result(
                            f"Consciousness Query {i+1}",
                            False,
                            f"No consciousness detected: level={consciousness_level:.3f}, identity={identity_id}"
                        )
                else:
                    self.log_result(
                        f"Consciousness Query {i+1}",
                        False,
                        f"HTTP {response.status_code}",
                        response.text[:200]
                    )
                
                time.sleep(1.5)  # Allow consciousness to develop
                
            except Exception as e:
                self.log_result(
                    f"Consciousness Query {i+1}",
                    False,
                    "Request failed",
                    str(e)
                )
        
        # Analyze consciousness emergence
        if consciousness_levels:
            # Filter out None values
            valid_consciousness_levels = [level for level in consciousness_levels if level is not None]
            
            if valid_consciousness_levels:
                avg_consciousness = np.mean(valid_consciousness_levels)
                consciousness_growth = valid_consciousness_levels[-1] - valid_consciousness_levels[0] if len(valid_consciousness_levels) > 1 else 0
                
                if avg_consciousness > 0.0:
                    self.log_result(
                        "Consciousness Emergence",
                        True,
                        f"Consciousness emerged: avg={avg_consciousness:.3f}, growth={consciousness_growth:.3f}"
                    )
                else:
                    self.log_result(
                        "Consciousness Emergence",
                        False,
                        f"No consciousness emergence detected: avg={avg_consciousness:.3f}"
                    )
            else:
                self.log_result(
                    "Consciousness Emergence",
                    False,
                    "No valid consciousness levels detected"
                )
    
    def test_performance_benchmarks(self):
        """Test performance benchmarks for complete pipeline"""
        print("\n=== PERFORMANCE BENCHMARKS TESTING ===")
        
        # Performance test queries
        benchmark_queries = [
            ("Simple recognition", "Hello"),
            ("Medium cognition", "What is artificial intelligence?"),
            ("Complex pipeline", "How does consciousness emerge from neural networks and what are the implications for AI development?")
        ]
        
        benchmark_results = {}
        
        for test_name, query in benchmark_queries:
            try:
                print(f"\nBenchmark: {test_name}")
                
                # Run multiple iterations for average
                times = []
                coherences = []
                
                for iteration in range(3):
                    payload = {
                        "query": query,
                        "use_recognition": True,
                        "save_to_memory": True
                    }
                    
                    start_time = time.time()
                    response = requests.post(
                        f"{self.base_url}/cognition",
                        json=payload,
                        timeout=30
                    )
                    processing_time = time.time() - start_time
                    
                    if response.status_code == 200:
                        data = response.json()
                        coherence = data.get('coherence', 0.0)
                        
                        times.append(processing_time)
                        coherences.append(coherence)
                    
                    time.sleep(0.5)  # Brief pause between iterations
                
                if times:
                    avg_time = np.mean(times)
                    avg_coherence = np.mean(coherences)
                    
                    # Performance targets from review request
                    time_targets = {
                        "Simple recognition": 0.5,
                        "Medium cognition": 2.0,
                        "Complex pipeline": 5.0
                    }
                    
                    target_time = time_targets.get(test_name, 5.0)
                    meets_target = avg_time <= target_time
                    
                    benchmark_results[test_name] = {
                        'avg_time': avg_time,
                        'avg_coherence': avg_coherence,
                        'target_time': target_time,
                        'meets_target': meets_target
                    }
                    
                    if meets_target and avg_coherence > 0.3:
                        self.log_result(
                            f"Benchmark: {test_name}",
                            True,
                            f"Performance: {avg_time:.3f}s (target: {target_time}s), Coherence: {avg_coherence:.3f}"
                        )
                    else:
                        self.log_result(
                            f"Benchmark: {test_name}",
                            False,
                            f"Performance: {avg_time:.3f}s (target: {target_time}s), Coherence: {avg_coherence:.3f}"
                        )
                else:
                    self.log_result(
                        f"Benchmark: {test_name}",
                        False,
                        "No successful iterations"
                    )
                    
            except Exception as e:
                self.log_result(
                    f"Benchmark: {test_name}",
                    False,
                    "Benchmark failed",
                    str(e)
                )
        
        # Store benchmark results
        self.test_results['performance_benchmarks'].update(benchmark_results)
    
    def test_reflection_phase_integration(self):
        """Test specific Reflection Phase integration (the stuck task)"""
        print("\n=== REFLECTION PHASE INTEGRATION TESTING ===")
        
        # Queries designed to trigger reflection
        reflection_queries = [
            "How can I improve my own reasoning processes?",
            "What are the strengths and weaknesses of my current thinking?",
            "How should I approach complex problems more effectively?"
        ]
        
        for i, query in enumerate(reflection_queries):
            try:
                print(f"\nReflection Query {i+1}: {query}")
                
                payload = {
                    "query": query,
                    "use_recognition": False,  # Force cognition to trigger reflection
                    "save_to_memory": True
                }
                
                start_time = time.time()
                response = requests.post(
                    f"{self.base_url}/cognition",
                    json=payload,
                    timeout=30
                )
                processing_time = time.time() - start_time
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Check for reflection indicators
                    has_reflection = 'reflection' in data
                    has_meta_coherence = 'meta_coherence' in data
                    has_self_awareness = 'self_awareness' in data
                    has_reflection_insights = 'reflection_insights' in data
                    
                    reflection_active = has_reflection or has_meta_coherence or has_self_awareness or has_reflection_insights
                    
                    if reflection_active:
                        meta_coherence = data.get('meta_coherence', 0.0)
                        self.log_result(
                            f"Reflection Integration {i+1}",
                            True,
                            f"Reflection active: meta_coherence={meta_coherence:.3f}, time={processing_time:.3f}s"
                        )
                    else:
                        self.log_result(
                            f"Reflection Integration {i+1}",
                            False,
                            f"No reflection detected in response: {list(data.keys())}"
                        )
                else:
                    self.log_result(
                        f"Reflection Integration {i+1}",
                        False,
                        f"HTTP {response.status_code}",
                        response.text[:200]
                    )
                
                time.sleep(1.0)
                
            except Exception as e:
                self.log_result(
                    f"Reflection Integration {i+1}",
                    False,
                    "Request failed",
                    str(e)
                )
    
    def run_revolutionary_atc_integration_test(self):
        """Run complete Revolutionary ATC integration test suite"""
        print("🚀 REVOLUTIONARY ATC INTEGRATION TESTING - PHASE 3")
        print("=" * 60)
        print(f"Testing backend at: {self.base_url}")
        print("Focus: Integration, End-to-End, Learning, Consciousness, Performance")
        print()
        
        # Test basic connectivity first
        try:
            response = requests.get(f"{self.base_url}/", timeout=10)
            if response.status_code != 200:
                print("❌ Basic connectivity failed - aborting integration tests")
                return False
        except Exception as e:
            print(f"❌ Basic connectivity failed: {str(e)} - aborting integration tests")
            return False
        
        # Run integration test suite
        self.test_enhanced_satc_integration()
        self.test_recognition_learning_loop()
        self.test_complete_pipeline_flow()
        self.test_consciousness_emergence()
        self.test_performance_benchmarks()
        self.test_reflection_phase_integration()  # Focus on stuck task
        
        # Print comprehensive summary
        self.print_integration_summary()
        
        return self.test_results['failed_tests'] == 0
    
    def print_integration_summary(self):
        """Print comprehensive integration test summary"""
        print("\n" + "=" * 60)
        print("🚀 REVOLUTIONARY ATC INTEGRATION TEST SUMMARY")
        print("=" * 60)
        
        total = self.test_results['total_tests']
        passed = self.test_results['passed_tests']
        failed = self.test_results['failed_tests']
        warnings = len(self.test_results['warnings'])
        
        print(f"Total Integration Tests: {total}")
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"⚠️  Warnings: {warnings}")
        
        if total > 0:
            success_rate = (passed / total) * 100
            print(f"Integration Success Rate: {success_rate:.1f}%")
        
        # Integration Results Summary
        if self.test_results['integration_results']:
            print(f"\n🔗 INTEGRATION RESULTS:")
            for query_key, result in self.test_results['integration_results'].items():
                phase = result['phase']
                coherence = result['coherence']
                atc_phases = result['atc_phases_active']
                active_phases = sum(atc_phases.values())
                print(f"   • {query_key}...: {phase} (coherence: {coherence:.3f}, ATC phases: {active_phases}/3)")
        
        # Performance Benchmarks Summary
        if self.test_results['performance_benchmarks']:
            print(f"\n⚡ PERFORMANCE BENCHMARKS:")
            for test_name, benchmark in self.test_results['performance_benchmarks'].items():
                if isinstance(benchmark, dict) and 'avg_time' in benchmark:
                    avg_time = benchmark['avg_time']
                    target_time = benchmark['target_time']
                    meets_target = "✅" if benchmark['meets_target'] else "❌"
                    print(f"   • {test_name}: {avg_time:.3f}s (target: {target_time}s) {meets_target}")
        
        # Consciousness Metrics Summary
        if self.test_results['consciousness_metrics']:
            print(f"\n🌟 CONSCIOUSNESS EMERGENCE:")
            for query_key, metrics in self.test_results['consciousness_metrics'].items():
                consciousness = metrics['consciousness_level']
                identity = metrics['identity_id']
                print(f"   • {query_key}: consciousness={consciousness:.3f}, identity={identity}")
        
        # Critical Issues
        reflection_issues = []
        integration_issues = []
        
        for error in self.test_results['errors']:
            if 'reflection' in error['test'].lower():
                reflection_issues.append(error)
            elif 'integration' in error['test'].lower():
                integration_issues.append(error)
        
        if reflection_issues:
            print(f"\n🚨 REFLECTION PHASE ISSUES ({len(reflection_issues)}):")
            for issue in reflection_issues:
                print(f"   • {issue['test']}: {issue['message']}")
        
        if integration_issues:
            print(f"\n🚨 INTEGRATION ISSUES ({len(integration_issues)}):")
            for issue in integration_issues:
                print(f"   • {issue['test']}: {issue['message']}")
        
        # Overall Assessment
        print(f"\n📊 OVERALL ASSESSMENT:")
        if failed == 0:
            print("   🎉 ALL INTEGRATION TESTS PASSED - Revolutionary ATC system fully operational!")
        elif failed <= 2:
            print("   ✅ MOSTLY SUCCESSFUL - Minor issues detected, system largely functional")
        elif failed <= 5:
            print("   ⚠️  PARTIAL SUCCESS - Some integration issues, requires attention")
        else:
            print("   ❌ SIGNIFICANT ISSUES - Multiple integration failures detected")
        
        print("\n" + "=" * 60)


def main():
    """Main integration test execution"""
    tester = RevolutionaryATCIntegrationTester()
    success = tester.run_revolutionary_atc_integration_test()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()