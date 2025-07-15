from fastapi import FastAPI, APIRouter, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import uuid
from datetime import datetime
import sys
import torch

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

# Import our Enhanced SATC Engine
from enhanced_satc_engine import EnhancedSATCEngine, SATCConfig
from satc_training_pipeline import SATCTrainer, TrainingConfig, ResponseQualityEvaluator

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI(title="Enhanced SATC API", description="Artificial True Cognition System")

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Initialize Enhanced SATC Engine
try:
    satc_config = SATCConfig()
    satc_engine = EnhancedSATCEngine(satc_config)
    
    # Initialize training components
    training_config = TrainingConfig()
    trainer = SATCTrainer(training_config)
    evaluator = ResponseQualityEvaluator()
    
    logger.info("Enhanced SATC Engine initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize SATC Engine: {str(e)}")
    satc_engine = None
    trainer = None
    evaluator = None

# Define Models
class StatusCheck(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    client_name: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class StatusCheckCreate(BaseModel):
    client_name: str

class CognitionQuery(BaseModel):
    query: str
    use_recognition: bool = True
    save_to_memory: bool = True

class CognitionResponse(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    query: str
    output: str
    phase: str
    success: bool
    coherence: float
    dissonance: float = None
    processing_time: float
    method: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class PerformanceMetrics(BaseModel):
    total_queries: int
    recognition_hits: int
    cognition_processes: int
    recognition_rate: float
    avg_coherence: float
    avg_dissonance: float
    avg_processing_time: float
    memory_updates: int
    replay_buffer_size: int
    deposited_patterns: int
    som_training_samples: int
    sememe_database_size: int

class TrainingPair(BaseModel):
    query: str
    response: str
    quality_score: float = 0.8
    coherence_score: float = 0.8
    sememes: List[str] = []

class TrainingRequest(BaseModel):
    training_pairs: List[TrainingPair]
    epochs: int = 10
    batch_size: int = 16
    learning_rate: float = 1e-4

class TrainingStatus(BaseModel):
    is_training: bool
    current_epoch: int = 0
    total_epochs: int = 0
    current_loss: float = 0.0
    current_coherence: float = 0.0
    estimated_time_remaining: float = 0.0

class ResponseEvaluation(BaseModel):
    query: str
    response: str
    coherence: float
    relevance: float
    informativeness: float
    fluency: float
    overall: float

# Original routes
@api_router.get("/")
async def root():
    return {"message": "Enhanced SATC API - Artificial True Cognition System"}

@api_router.post("/status", response_model=StatusCheck)
async def create_status_check(input: StatusCheckCreate):
    status_dict = input.dict()
    status_obj = StatusCheck(**status_dict)
    _ = await db.status_checks.insert_one(status_obj.dict())
    return status_obj

@api_router.get("/status", response_model=List[StatusCheck])
async def get_status_checks():
    status_checks = await db.status_checks.find().to_list(1000)
    return [StatusCheck(**status_check) for status_check in status_checks]

# Enhanced SATC Routes
@api_router.post("/cognition", response_model=CognitionResponse)
async def process_cognition(query: CognitionQuery):
    """Process a query through the SATC cognition engine"""
    if satc_engine is None:
        raise HTTPException(status_code=500, detail="SATC Engine not initialized")
    
    try:
        # Process the query
        result = satc_engine.process_query(query.query)
        
        # Create response
        response = CognitionResponse(
            query=query.query,
            output=result['output'],
            phase=result['phase'],
            success=result['success'],
            coherence=result.get('coherence', 0.0),
            dissonance=result.get('dissonance', 0.0),
            processing_time=result['processing_time'],
            method=result.get('method', 'unknown'),
            metadata={
                'nodes_count': len(result.get('nodes', [])),
                'sememes_count': len(result.get('sememes', [])),
                'variants_count': result.get('variants_count', 0)
            }
        )
        
        # Save to MongoDB
        await db.cognition_responses.insert_one(response.dict())
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cognition processing failed: {str(e)}")

@api_router.get("/cognition/history")
async def get_cognition_history(limit: int = 10):
    """Get recent cognition processing history"""
    try:
        history = await db.cognition_responses.find().sort("timestamp", -1).limit(limit).to_list(length=limit)
        return history
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve history: {str(e)}")

@api_router.get("/cognition/performance", response_model=PerformanceMetrics)
async def get_performance_metrics():
    """Get current performance metrics from SATC engine"""
    if satc_engine is None:
        raise HTTPException(status_code=500, detail="SATC Engine not initialized")
    
    try:
        metrics = satc_engine.get_performance_report()
        return PerformanceMetrics(**metrics)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve metrics: {str(e)}")

@api_router.post("/cognition/reset")
async def reset_engine():
    """Reset the SATC engine to initial state"""
    try:
        global satc_engine
        satc_engine = EnhancedSATCEngine(satc_config)
        return {"message": "SATC engine reset successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reset engine: {str(e)}")

@api_router.get("/cognition/config")
async def get_engine_config():
    """Get current SATC engine configuration"""
    try:
        return {
            "hd_dim": satc_config.hd_dim,
            "som_grid_size": satc_config.som_grid_size,
            "deep_layers_config": satc_config.deep_layers_config,
            "clustering_config": satc_config.clustering_config,
            "performance_targets": satc_config.performance_targets
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve config: {str(e)}")

@api_router.post("/cognition/save-state")
async def save_engine_state():
    """Save current engine state"""
    if satc_engine is None:
        raise HTTPException(status_code=500, detail="SATC Engine not initialized")
    
    try:
        state_path = "/tmp/satc_engine_state.pt"
        satc_engine.save_state(state_path)
        return {"message": f"Engine state saved to {state_path}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save state: {str(e)}")

@api_router.post("/cognition/load-state")
async def load_engine_state():
    """Load previously saved engine state"""
    if satc_engine is None:
        raise HTTPException(status_code=500, detail="SATC Engine not initialized")
    
    try:
        state_path = "/tmp/satc_engine_state.pt"
        satc_engine.load_state(state_path)
        return {"message": f"Engine state loaded from {state_path}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load state: {str(e)}")

@api_router.get("/cognition/sememes/{query}")
async def get_sememes_for_query(query: str):
    """Get sememes that would be used for a query (without full processing)"""
    if satc_engine is None:
        raise HTTPException(status_code=500, detail="SATC Engine not initialized")
    
    try:
        # Get intent vector
        intent_vector = satc_engine.embed_query(query)
        
        # Get structure
        structure = satc_engine.deep_layers(intent_vector)
        
        # Get nodes
        heat_map = satc_engine.som_clustering.project(structure.detach().cpu().numpy())
        nodes = satc_engine.dynamic_cluster(heat_map)
        
        # Get HD nodes
        hd_nodes = satc_engine.hd_encoder.encode(nodes)
        
        # Get sememes
        sememes = satc_engine.sememe_population(hd_nodes)
        
        return {
            "query": query,
            "sememes": sememes,
            "nodes_count": len(nodes),
            "structure_mean": float(torch.mean(structure))
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get sememes: {str(e)}")

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
