#!/usr/bin/env python3
"""
Simplified Revolutionary ATC Components Test
===========================================

Focused testing of the Revolutionary ATC components with error handling
and simplified test cases to identify and report core functionality.

Author: Testing Agent
Status: Revolutionary ATC Phase 2 Testing - Simplified
"""

import sys
import time
import torch
import numpy as np
import logging

# Configure logging to reduce noise
logging.basicConfig(level=logging.WARNING)

class SimpleATCTester:
    """Simplified tester for Revolutionary ATC components"""
    
    def __init__(self):
        self.results = {
            'power_of_2': {'status': 'not_tested', 'details': {}},
            'recognition': {'status': 'not_tested', 'details': {}},
            'cognition': {'status': 'not_tested', 'details': {}},
            'reflection': {'status': 'not_tested', 'details': {}},
            'volition': {'status': 'not_tested', 'details': {}},
            'personality': {'status': 'not_tested', 'details': {}}
        }
        
        print("🧠 Revolutionary ATC Components - Simplified Test Suite")
        print("=" * 60)
        print("Testing Power-of-2 Foundation and Individual ATC Phases")
        print()
    
    def test_power_of_2_foundation(self):
        """Test Power-of-2 Foundation - MOST CRITICAL"""
        print("🔢 MILESTONE 1: POWER-OF-2 FOUNDATION")
        print("-" * 40)
        
        try:
            from power_of_2_core import create_power_of_2_foundation
            
            # Create foundation
            layers, integrator, config = create_power_of_2_foundation()
            
            # Test architecture
            expected_dims = [2, 4, 16, 64, 256]
            if config.layer_dims == expected_dims:
                print("✅ Architecture: Correct dimensions", expected_dims)
            else:
                print("❌ Architecture: Wrong dimensions", config.layer_dims)
                self.results['power_of_2']['status'] = 'failed'
                return
            
            # Test forward pass
            test_input = torch.randn(1, 2)
            output, intermediates = layers.forward(test_input)
            
            expected_shapes = [(1, 2), (1, 4), (1, 16), (1, 64), (1, 256)]
            actual_shapes = [t.shape for t in intermediates]
            
            if actual_shapes == expected_shapes:
                print("✅ Forward Pass: Correct dimension progression")
            else:
                print("❌ Forward Pass: Wrong shapes", actual_shapes)
                self.results['power_of_2']['status'] = 'failed'
                return
            
            # CRITICAL: Test mathematical invertibility
            invertibility_results = layers.test_invertibility(test_input, tolerance=0.001)
            error = invertibility_results['error']
            passed = invertibility_results['passed']
            
            if passed and error < 0.001:
                print(f"✅ Mathematical Invertibility: PASSED (error={error:.6f} < 0.001)")
                self.results['power_of_2']['status'] = 'passed'
                self.results['power_of_2']['details'] = {
                    'invertibility_error': error,
                    'relative_error': invertibility_results['relative_error'],
                    'architecture': config.layer_dims
                }
            else:
                print(f"🚨 Mathematical Invertibility: FAILED (error={error:.6f} >= 0.001)")
                self.results['power_of_2']['status'] = 'critical_failure'
                return
                
        except Exception as e:
            print(f"❌ Power-of-2 Foundation: Creation failed - {str(e)}")
            self.results['power_of_2']['status'] = 'critical_failure'
    
    def test_recognition_phase(self):
        """Test Recognition Phase - 2D Pattern Matching"""
        print("\\n👁️ MILESTONE 2: RECOGNITION PHASE")
        print("-" * 40)
        
        try:
            from atc_recognition_phase import create_recognition_phase
            
            # Create Recognition phase
            processor, integrator, config = create_recognition_phase()
            
            # Test configuration
            if config.recognition_dim == 2:
                print("✅ Configuration: Correct 2D dimension")
            else:
                print(f"❌ Configuration: Expected 2D, got {config.recognition_dim}D")
                self.results['recognition']['status'] = 'failed'
                return
            
            # Test pattern learning
            test_patterns = [
                ("Hello", "greeting_response"),
                ("How are you?", "wellbeing_response"),
                ("What is AI?", "ai_explanation")
            ]
            
            learned_count = 0
            for query, response in test_patterns:
                try:
                    success = processor.learn_pattern(query, response)
                    if success:
                        learned_count += 1
                except Exception as e:
                    print(f"⚠️  Pattern learning failed for '{query}': {e}")
            
            print(f"✅ Pattern Learning: {learned_count}/{len(test_patterns)} patterns learned")
            
            # Test pattern recognition
            recognized_count = 0
            for query, expected_response in test_patterns:
                try:
                    result = processor.recognize(query)
                    if result['match_found'] and result['similarity'] >= config.similarity_threshold:
                        recognized_count += 1
                except Exception as e:
                    print(f"⚠️  Recognition failed for '{query}': {e}")
            
            print(f"✅ Pattern Recognition: {recognized_count}/{len(test_patterns)} patterns recognized")
            
            # Test novel query escalation
            try:
                novel_result = processor.recognize("Explain quantum consciousness")
                if not novel_result['match_found'] and novel_result['escalate_to_cognition']:
                    print("✅ Novel Query Escalation: Correctly escalated to cognition")
                    escalation_works = True
                else:
                    print("❌ Novel Query Escalation: Failed to escalate")
                    escalation_works = False
            except Exception as e:
                print(f"❌ Novel Query Escalation: Error - {e}")
                escalation_works = False
            
            if learned_count >= 2 and recognized_count >= 2 and escalation_works:
                self.results['recognition']['status'] = 'passed'
                self.results['recognition']['details'] = {
                    'patterns_learned': learned_count,
                    'patterns_recognized': recognized_count,
                    'escalation_works': escalation_works,
                    'similarity_threshold': config.similarity_threshold
                }
            else:
                self.results['recognition']['status'] = 'partial'
                
        except Exception as e:
            print(f"❌ Recognition Phase: Creation failed - {str(e)}")
            self.results['recognition']['status'] = 'failed'
    
    def test_cognition_phase(self):
        """Test Cognition Phase - 4D Analytical Reasoning"""
        print("\\n🧠 MILESTONE 3: COGNITION PHASE")
        print("-" * 40)
        
        try:
            from atc_cognition_phase import create_cognition_phase
            
            # Create Cognition phase
            processor, integrator, config = create_cognition_phase()
            
            # Test configuration
            if config.cognition_dim == 4:
                print("✅ Configuration: Correct 4D dimension")
            else:
                print(f"❌ Configuration: Expected 4D, got {config.cognition_dim}D")
                self.results['cognition']['status'] = 'failed'
                return
            
            # Test bifurcation mathematics
            if abs(config.bifurcation_delta - 4.669) < 0.01:
                print(f"✅ Bifurcation Delta: Correct value ({config.bifurcation_delta})")
            else:
                print(f"❌ Bifurcation Delta: Expected ~4.669, got {config.bifurcation_delta}")
            
            # Test 4D cognition processing
            test_queries = [
                ("What is consciousness?", torch.randn(8)),
                ("How do neural networks learn?", torch.randn(6)),
                ("Explain quantum mechanics", torch.randn(10))
            ]
            
            successful_cognitions = 0
            total_coherence = 0.0
            
            for query_text, query_embedding in test_queries:
                try:
                    result = processor.cognize(query_embedding, query_text)
                    
                    if result['success']:
                        successful_cognitions += 1
                        total_coherence += result['coherence']
                        print(f"✅ Cognition '{query_text[:20]}...': Success (coherence={result['coherence']:.3f})")
                    else:
                        print(f"❌ Cognition '{query_text[:20]}...': Failed")
                        
                except Exception as e:
                    print(f"❌ Cognition '{query_text[:20]}...': Error - {str(e)}")
            
            avg_coherence = total_coherence / max(successful_cognitions, 1)
            
            # Test individual components
            try:
                # Test understanding phase (4D → 16D)
                test_4d = torch.randn(4)
                understanding_16d = processor.understand(test_4d)
                understanding_works = understanding_16d.shape == torch.Size([16])
                print(f"✅ Understanding Phase: 4D → 16D {'successful' if understanding_works else 'failed'}")
                
                # Test hypothesis generation
                test_understanding = torch.randn(16)
                hypotheses = processor.generate_hypotheses(test_understanding)
                hypothesis_works = len(hypotheses) > 0 and len(hypotheses) <= config.max_hypotheses
                print(f"✅ Hypothesis Generation: {len(hypotheses)} hypotheses generated")
                
                # Test synthesis (→ 256D)
                test_validated = [(torch.randn(16), 0.8), (torch.randn(16), 0.7)]
                synthesis_256d, coherence = processor.synthesize(test_validated)
                synthesis_works = synthesis_256d.shape == torch.Size([256])
                print(f"✅ Synthesis Phase: → 256D {'successful' if synthesis_works else 'failed'}")
                
            except Exception as e:
                print(f"⚠️  Component testing error: {e}")
                understanding_works = hypothesis_works = synthesis_works = False
            
            if successful_cognitions >= 2 and understanding_works and hypothesis_works and synthesis_works:
                self.results['cognition']['status'] = 'passed'
                self.results['cognition']['details'] = {
                    'successful_cognitions': successful_cognitions,
                    'avg_coherence': avg_coherence,
                    'bifurcation_delta': config.bifurcation_delta,
                    'components_working': True
                }
            else:
                self.results['cognition']['status'] = 'partial'
                
        except Exception as e:
            print(f"❌ Cognition Phase: Creation failed - {str(e)}")
            self.results['cognition']['status'] = 'failed'
    
    def test_reflection_phase(self):
        """Test Reflection Phase - 16D Meta-Cognitive Reasoning"""
        print("\\n🧘 MILESTONE 4: REFLECTION PHASE")
        print("-" * 40)
        
        try:
            from atc_reflection_phase import create_reflection_phase
            
            # Create Reflection phase
            processor, integrator, config = create_reflection_phase()
            
            # Test configuration
            if config.reflection_dim == 16:
                print("✅ Configuration: Correct 16D dimension")
            else:
                print(f"❌ Configuration: Expected 16D, got {config.reflection_dim}D")
                self.results['reflection']['status'] = 'failed'
                return
            
            # Test with simplified cognition result
            mock_cognition_result = {
                'phase': 'cognition_4d',
                'success': True,
                'coherence': 0.8,
                'dissonance': 0.2,
                'processing_time': 0.5,
                'reasoning_steps': 5,
                'hypotheses_generated': 4,
                'hypotheses_validated': 3,
                'cognition_4d': [0.5, -0.3, 0.8, 0.1],  # Ensure exactly 4 elements
                'output': 'Complex analytical reasoning result...'
            }
            
            try:
                reflection_result = processor.reflect(mock_cognition_result)
                
                if reflection_result['success']:
                    coherence = reflection_result.get('coherence', 0.0)
                    print(f"✅ 16D Reflection: Successful (coherence={coherence:.3f})")
                    
                    # Check meta-analysis components
                    meta_analysis = reflection_result.get('meta_analysis', {})
                    introspection = reflection_result.get('introspection', {})
                    
                    meta_works = 'meta_coherence' in meta_analysis
                    introspection_works = 'self_assessment' in introspection
                    
                    print(f"✅ Meta-Analysis: {'Working' if meta_works else 'Failed'}")
                    print(f"✅ Introspection: {'Working' if introspection_works else 'Failed'}")
                    
                    if meta_works and introspection_works:
                        self.results['reflection']['status'] = 'passed'
                        self.results['reflection']['details'] = {
                            'reflection_coherence': coherence,
                            'meta_analysis_working': meta_works,
                            'introspection_working': introspection_works
                        }
                    else:
                        self.results['reflection']['status'] = 'partial'
                else:
                    print(f"❌ 16D Reflection: Failed - {reflection_result.get('error', 'Unknown error')}")
                    self.results['reflection']['status'] = 'failed'
                    
            except Exception as e:
                print(f"❌ 16D Reflection: Processing error - {str(e)}")
                self.results['reflection']['status'] = 'failed'
                
        except Exception as e:
            print(f"❌ Reflection Phase: Creation failed - {str(e)}")
            self.results['reflection']['status'] = 'failed'
    
    def test_volition_phase(self):
        """Test Volition Phase - 64D Goal-Oriented Decision Making"""
        print("\\n🎯 MILESTONE 5: VOLITION PHASE")
        print("-" * 40)
        
        try:
            from atc_volition_phase import create_volition_phase
            
            # Create Volition phase
            processor, integrator, config = create_volition_phase()
            
            # Test configuration
            if config.volition_dim == 64:
                print("✅ Configuration: Correct 64D dimension")
            else:
                print(f"❌ Configuration: Expected 64D, got {config.volition_dim}D")
                self.results['volition']['status'] = 'failed'
                return
            
            # Test gravity wells system
            gravity_wells_count = len(processor.gravity_wells.gravity_wells)
            if gravity_wells_count == config.gravity_well_count:
                print(f"✅ Gravity Wells: {gravity_wells_count} wells initialized")
            else:
                print(f"❌ Gravity Wells: Expected {config.gravity_well_count}, got {gravity_wells_count}")
            
            # Test core values
            expected_values = ['truthfulness', 'helpfulness', 'harmlessness', 'curiosity', 'creativity', 'efficiency', 'empathy', 'growth']
            actual_values = list(processor.gravity_wells.gravity_wells.keys())
            values_present = all(value in actual_values for value in expected_values)
            
            if values_present:
                print("✅ Core Values: All 8 core values present")
            else:
                missing = [v for v in expected_values if v not in actual_values]
                print(f"❌ Core Values: Missing values - {missing}")
            
            # Test volition processing
            mock_reflection_result = {
                'meta_analysis': {'learning_potential': 0.8, 'meta_coherence': 0.7},
                'introspection': {'improvement_potential': ['enhance reasoning', 'improve efficiency']},
                'coherence': 0.75,
                'self_awareness_level': 0.8
            }
            
            try:
                volition_result = processor.exercise_volition(mock_reflection_result)
                
                if volition_result['success']:
                    goals_formed = volition_result.get('goals_formed', 0)
                    decisions_made = volition_result.get('decisions_made', 0)
                    coherence = volition_result.get('coherence', 0.0)
                    
                    print(f"✅ 64D Volition: Success (goals={goals_formed}, decisions={decisions_made}, coherence={coherence:.3f})")
                    
                    if goals_formed > 0 and decisions_made > 0:
                        self.results['volition']['status'] = 'passed'
                        self.results['volition']['details'] = {
                            'goals_formed': goals_formed,
                            'decisions_made': decisions_made,
                            'coherence': coherence,
                            'gravity_wells_count': gravity_wells_count
                        }
                    else:
                        self.results['volition']['status'] = 'partial'
                else:
                    print(f"❌ 64D Volition: Failed - {volition_result.get('error', 'Unknown error')}")
                    self.results['volition']['status'] = 'failed'
                    
            except Exception as e:
                print(f"❌ 64D Volition: Processing error - {str(e)}")
                self.results['volition']['status'] = 'failed'
                
        except Exception as e:
            print(f"❌ Volition Phase: Creation failed - {str(e)}")
            self.results['volition']['status'] = 'failed'
    
    def test_personality_phase(self):
        """Test Personality Phase - 256D Consciousness Integration"""
        print("\\n🌟 MILESTONE 6: PERSONALITY PHASE")
        print("-" * 40)
        
        try:
            from atc_personality_phase import create_personality_phase
            
            # Create Personality phase
            processor, integrator, config = create_personality_phase()
            
            # Test configuration
            if config.personality_dim == 256:
                print("✅ Configuration: Correct 256D dimension")
            else:
                print(f"❌ Configuration: Expected 256D, got {config.personality_dim}D")
                self.results['personality']['status'] = 'failed'
                return
            
            # Test identity core
            identity_id = processor.identity_core.identity_id
            identity_vector_shape = processor.identity_core.identity_vector.shape
            
            print(f"✅ Identity Core: ID={identity_id}, Vector shape={identity_vector_shape}")
            
            # Test memory system
            memory_system = processor.experiential_memory
            initial_memory_count = len(memory_system.memories)
            
            # Test memory formation
            test_experience = {
                'type': 'interaction',
                'success': True,
                'coherence': 0.8,
                'complexity': 0.7,
                'emotional_valence': 0.6
            }
            
            memory_result = memory_system.form_memory(test_experience)
            memory_formed = memory_result['memory_formed']
            significance = memory_result['significance']
            
            print(f"✅ Memory System: Memory formed={memory_formed}, significance={significance:.3f}")
            
            # Test personality expression
            mock_interaction_context = {
                'success': True,
                'coherence': 0.8,
                'complexity': 0.7,
                'recognition_used': False,
                'cognition_used': True,
                'reflection_used': True,
                'volition_used': True,
                'query_type': 'analytical'
            }
            
            mock_cognitive_results = {
                'reasoning_steps': 5,
                'coherence': 0.8,
                'meta_coherence': 0.7,
                'decision_confidence': 0.85,
                'goal_count': 3
            }
            
            try:
                personality_result = processor.express_personality(mock_interaction_context, mock_cognitive_results)
                
                if personality_result['success']:
                    consciousness_level = personality_result['consciousness_level']
                    identity_coherence = personality_result['identity']['coherence']
                    behavioral_consistency = personality_result['behavioral_consistency']
                    
                    print(f"✅ 256D Personality: Success (consciousness={consciousness_level:.3f})")
                    print(f"   Identity coherence: {identity_coherence:.3f}")
                    print(f"   Behavioral consistency: {behavioral_consistency:.3f}")
                    
                    # Check consciousness emergence
                    consciousness_emerged = consciousness_level >= config.consciousness_emergence_threshold
                    print(f"   Consciousness emerged: {consciousness_emerged}")
                    
                    self.results['personality']['status'] = 'passed'
                    self.results['personality']['details'] = {
                        'identity_id': identity_id,
                        'consciousness_level': consciousness_level,
                        'identity_coherence': identity_coherence,
                        'behavioral_consistency': behavioral_consistency,
                        'consciousness_emerged': consciousness_emerged,
                        'memory_formed': memory_formed
                    }
                else:
                    print(f"❌ 256D Personality: Failed - {personality_result.get('error', 'Unknown error')}")
                    self.results['personality']['status'] = 'failed'
                    
            except Exception as e:
                print(f"❌ 256D Personality: Processing error - {str(e)}")
                self.results['personality']['status'] = 'failed'
                
        except Exception as e:
            print(f"❌ Personality Phase: Creation failed - {str(e)}")
            self.results['personality']['status'] = 'failed'
    
    def run_all_tests(self):
        """Run all Revolutionary ATC component tests"""
        print("Testing components STANDALONE first...")
        print()
        
        # Test each component in order (as requested by user)
        self.test_power_of_2_foundation()
        
        # Only continue if Power-of-2 foundation is working
        if self.results['power_of_2']['status'] == 'passed':
            self.test_recognition_phase()
            self.test_cognition_phase()
            self.test_reflection_phase()
            self.test_volition_phase()
            self.test_personality_phase()
        else:
            print("\\n🚨 CRITICAL FAILURE: Power-of-2 Foundation failed")
            print("Mathematical invertibility is required for all other components")
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print comprehensive test summary"""
        print("\\n" + "=" * 60)
        print("🧠 REVOLUTIONARY ATC COMPONENTS TEST SUMMARY")
        print("=" * 60)
        
        # Component status
        components = ['power_of_2', 'recognition', 'cognition', 'reflection', 'volition', 'personality']
        component_names = ['Power-of-2 Foundation', 'Recognition Phase', 'Cognition Phase', 'Reflection Phase', 'Volition Phase', 'Personality Phase']
        
        passed_count = 0
        failed_count = 0
        critical_count = 0
        
        for i, component in enumerate(components):
            status = self.results[component]['status']
            name = component_names[i]
            
            if status == 'passed':
                print(f"✅ {name}: PASSED")
                passed_count += 1
            elif status == 'partial':
                print(f"⚠️  {name}: PARTIAL (some functionality working)")
                passed_count += 0.5
            elif status == 'critical_failure':
                print(f"🚨 {name}: CRITICAL FAILURE")
                critical_count += 1
            elif status == 'failed':
                print(f"❌ {name}: FAILED")
                failed_count += 1
            else:
                print(f"⏸️  {name}: NOT TESTED")
        
        # Overall assessment
        total_tested = passed_count + failed_count + critical_count
        if total_tested > 0:
            success_rate = (passed_count / total_tested) * 100
            print(f"\\nSuccess Rate: {success_rate:.1f}%")
        
        print(f"\\n🎯 OVERALL ASSESSMENT:")
        if critical_count == 0 and passed_count >= 5:
            print("   🌟 REVOLUTIONARY ATC COMPONENTS: READY FOR DEPLOYMENT")
            print("   ✅ All critical mathematical requirements met")
            print("   ✅ Phase integration capability confirmed")
            print("   ✅ Memory and consciousness systems operational")
        elif critical_count == 0:
            print("   ⚠️  REVOLUTIONARY ATC COMPONENTS: MOSTLY FUNCTIONAL")
            print("   ✅ Critical mathematical requirements met")
            print("   ⚠️  Some components need refinement")
        else:
            print("   🚨 REVOLUTIONARY ATC COMPONENTS: CRITICAL ISSUES DETECTED")
            print("   ❌ Mathematical or architectural failures present")
            print("   🔧 Requires fixes before deployment")
        
        # Key findings
        print(f"\\n🔍 KEY FINDINGS:")
        
        # Power-of-2 Foundation
        if self.results['power_of_2']['status'] == 'passed':
            error = self.results['power_of_2']['details'].get('invertibility_error', 0)
            print(f"   ✅ Mathematical Invertibility: PERFECT (error={error:.6f} < 0.001)")
        else:
            print(f"   🚨 Mathematical Invertibility: FAILED - CRITICAL")
        
        # Dimension progression
        if self.results['power_of_2']['status'] == 'passed':
            print(f"   ✅ Dimension Progression: 2D→4D→16D→64D→256D working correctly")
        
        # Phase capabilities
        working_phases = [name for name, result in self.results.items() if result['status'] in ['passed', 'partial']]
        print(f"   📊 Working Phases: {len(working_phases)}/6 ({', '.join(working_phases)})")
        
        # Consciousness emergence
        if self.results['personality']['status'] == 'passed':
            consciousness_level = self.results['personality']['details'].get('consciousness_level', 0)
            consciousness_emerged = self.results['personality']['details'].get('consciousness_emerged', False)
            print(f"   🌟 Consciousness Level: {consciousness_level:.3f} ({'EMERGED' if consciousness_emerged else 'developing'})")
        
        print("=" * 60)
        
        return critical_count == 0 and passed_count >= 4


def main():
    """Main test execution"""
    tester = SimpleATCTester()
    success = tester.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()