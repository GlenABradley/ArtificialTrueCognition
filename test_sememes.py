#!/usr/bin/env python3

import sys
import os
sys.path.append('/app')

from enhanced_satc_engine import EnhancedSATCEngine, SATCConfig

try:
    print("Creating SATC config...")
    config = SATCConfig()
    
    print("Creating SATC engine...")
    engine = EnhancedSATCEngine(config)
    
    print("Testing embed_query...")
    intent_vector = engine.embed_query("test query")
    print(f"Intent vector shape: {intent_vector.shape}")
    
    print("Testing deep_layers...")
    structure = engine.deep_layers(intent_vector)
    print(f"Structure shape: {structure.shape}")
    
    print("Testing som_project...")
    heat_map = engine.som_clustering.project(structure.detach().cpu().numpy())
    print(f"Heat map shape: {heat_map.shape}")
    
    print("Testing dynamic_cluster...")
    nodes = engine.dynamic_cluster(heat_map)
    print(f"Nodes shape: {nodes.shape}")
    
    print("Testing hd_encoder...")
    hd_nodes = engine.hd_encoder.encode(nodes)
    print(f"HD nodes shape: {hd_nodes.shape}")
    
    print("Testing sememe_population...")
    sememes = engine.sememe_population(hd_nodes)
    print(f"Sememes type: {type(sememes)}, length: {len(sememes)}")
    
    print("✅ All tests passed!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()