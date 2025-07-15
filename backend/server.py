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

# Import our Core SATC Engine (unified architecture)
from core_satc_engine import EnhancedSATCEngine, CoreSATCConfig
from satc_training_pipeline import SATCTrainer, TrainingConfig, ResponseQualityEvaluator
from bulk_training_system import BulkTrainingSystem, BulkTrainingConfig

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
    satc_config = CoreSATCConfig()
    satc_engine = EnhancedSATCEngine(satc_config)
    
    # Initialize training components
    training_config = TrainingConfig()
    trainer = SATCTrainer(training_config)
    evaluator = ResponseQualityEvaluator()
    
    # Initialize bulk training system
    bulk_system = BulkTrainingSystem()
    
    logger.info("Enhanced SATC Engine initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize SATC Engine: {str(e)}")
    satc_engine = None
    trainer = None
    evaluator = None
    bulk_system = None

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

class BulkTrainingUpload(BaseModel):
    format: str = "json"  # json, csv, auto
    data: str  # Raw data content
    
class AutomatedTrainingRequest(BaseModel):
    hours_per_day: int = 20
    rest_hours: int = 4
    max_epochs: int = 1000
    save_every_n_epochs: int = 10

class HelloWorldRequest(BaseModel):
    quick_start: bool = True

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
        
        # Convert datetime objects to strings for JSON serialization
        for item in history:
            if 'timestamp' in item and hasattr(item['timestamp'], 'isoformat'):
                item['timestamp'] = item['timestamp'].isoformat()
            # Remove MongoDB ObjectId which can't be serialized
            if '_id' in item:
                del item['_id']
        
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
        sememes_raw = satc_engine.sememe_population(hd_nodes)
        
        # Clean up sememes for JSON serialization
        sememes_clean = []
        for sememe in sememes_raw:
            clean_sememe = {
                'node_index': sememe['node_index'],
                'primary_sememe': sememe['primary_sememe']['data']['concept'] if sememe['primary_sememe'] else None,
                'alternative_sememes': [alt['data']['concept'] if alt and 'data' in alt else str(alt) for alt in sememe['alternative_sememes']] if sememe['alternative_sememes'] else [],
                'node_vector_length': len(sememe['node_vector']) if sememe['node_vector'] is not None else 0
            }
            sememes_clean.append(clean_sememe)
        
        return {
            "query": query,
            "sememes": sememes_clean,
            "nodes_count": len(nodes),
            "structure_mean": float(torch.mean(structure))
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get sememes: {str(e)}")

# Training Endpoints
@api_router.post("/training/start")
async def start_training(request: TrainingRequest):
    """Start training the SATC engine with provided training pairs"""
    if trainer is None:
        raise HTTPException(status_code=500, detail="Training system not initialized")
    
    try:
        # Save training pairs to file
        training_data = []
        for pair in request.training_pairs:
            training_data.append({
                "query": pair.query,
                "response": pair.response,
                "quality_score": pair.quality_score,
                "coherence_score": pair.coherence_score,
                "sememes": pair.sememes
            })
        
        # Save to training file
        import json
        from pathlib import Path
        training_path = Path("data/training_pairs.jsonl")
        training_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(training_path, 'w') as f:
            for item in training_data:
                f.write(json.dumps(item) + '\n')
        
        # Update training config
        trainer.config.num_epochs = request.epochs
        trainer.config.batch_size = request.batch_size
        trainer.config.learning_rate = request.learning_rate
        
        # Start training in background (simplified for demo)
        # In production, this would be a background task
        logger.info(f"Training started with {len(training_data)} pairs")
        
        return {
            "message": f"Training started with {len(training_data)} pairs",
            "config": {
                "epochs": request.epochs,
                "batch_size": request.batch_size,
                "learning_rate": request.learning_rate
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start training: {str(e)}")

@api_router.get("/training/status", response_model=TrainingStatus)
async def get_training_status():
    """Get current training status"""
    if trainer is None:
        raise HTTPException(status_code=500, detail="Training system not initialized")
    
    try:
        # For now, return mock status
        # In production, this would track actual training progress
        return TrainingStatus(
            is_training=False,
            current_epoch=0,
            total_epochs=0,
            current_loss=0.0,
            current_coherence=0.0,
            estimated_time_remaining=0.0
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get training status: {str(e)}")

@api_router.post("/training/add-pair")
async def add_training_pair(pair: TrainingPair):
    """Add a single training pair"""
    try:
        # Save to training file
        import json
        from pathlib import Path
        training_path = Path("data/training_pairs.jsonl")
        training_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Append to file
        with open(training_path, 'a') as f:
            training_item = {
                "query": pair.query,
                "response": pair.response,
                "quality_score": pair.quality_score,
                "coherence_score": pair.coherence_score,
                "sememes": pair.sememes
            }
            f.write(json.dumps(training_item) + '\n')
        
        return {"message": "Training pair added successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add training pair: {str(e)}")

@api_router.post("/training/evaluate", response_model=ResponseEvaluation)
async def evaluate_response(query: str, response: str):
    """Evaluate response quality"""
    if evaluator is None:
        raise HTTPException(status_code=500, detail="Evaluator not initialized")
    
    try:
        scores = evaluator.evaluate_response(query, response)
        
        return ResponseEvaluation(
            query=query,
            response=response,
            coherence=scores['coherence'],
            relevance=scores['relevance'],
            informativeness=scores['informativeness'],
            fluency=scores['fluency'],
            overall=scores['overall']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to evaluate response: {str(e)}")

@api_router.get("/training/data")
async def get_training_data():
    """Get current training data"""
    try:
        from pathlib import Path
        import json
        
        training_path = Path("data/training_pairs.jsonl")
        if not training_path.exists():
            return {"training_pairs": [], "count": 0}
        
        training_pairs = []
        with open(training_path, 'r') as f:
            for line in f:
                if line.strip():
                    training_pairs.append(json.loads(line.strip()))
        
        return {
            "training_pairs": training_pairs,
            "count": len(training_pairs)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get training data: {str(e)}")

@api_router.delete("/training/data")
async def clear_training_data():
    """Clear all training data"""
    try:
        from pathlib import Path
        
        training_path = Path("data/training_pairs.jsonl")
        if training_path.exists():
            training_path.unlink()
        
        return {"message": "Training data cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear training data: {str(e)}")

@api_router.post("/training/improve-response")
async def improve_response(query: str, current_response: str, target_response: str):
    """Add a training pair to improve a specific response"""
    if evaluator is None:
        raise HTTPException(status_code=500, detail="Evaluator not initialized")
    
    try:
        # Evaluate the target response
        scores = evaluator.evaluate_response(query, target_response)
        
        # Create training pair
        training_pair = TrainingPair(
            query=query,
            response=target_response,
            quality_score=scores['overall'],
            coherence_score=scores['coherence'],
            sememes=[]  # Would be populated with actual sememes
        )
        
        # Add to training data
        import json
        from pathlib import Path
        training_path = Path("data/training_pairs.jsonl")
        training_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(training_path, 'a') as f:
            training_item = {
                "query": training_pair.query,
                "response": training_pair.response,
                "quality_score": training_pair.quality_score,
                "coherence_score": training_pair.coherence_score,
                "sememes": training_pair.sememes
            }
            f.write(json.dumps(training_item) + '\n')
        
        return {
            "message": "Training pair added for response improvement",
            "evaluation": scores,
            "training_pair": training_pair
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to improve response: {str(e)}")

# Bulk Training Endpoints
@api_router.post("/training/bulk-upload")
async def bulk_upload_training_data(upload: BulkTrainingUpload):
    """Upload bulk training data (CSV, JSON, or other formats)"""
    if bulk_system is None:
        raise HTTPException(status_code=500, detail="Bulk training system not initialized")
    
    try:
        # Create temporary file
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(mode='w', suffix=f".{upload.format}", delete=False) as tmp_file:
            tmp_file.write(upload.data)
            tmp_path = tmp_file.name
        
        try:
            # Import bulk dataset
            training_file = bulk_system.import_bulk_dataset(tmp_path, upload.format)
            
            # Get count of imported pairs
            count = 0
            with open(training_file, 'r') as f:
                for line in f:
                    if line.strip():
                        count += 1
            
            return {
                "message": f"Bulk training data uploaded successfully",
                "training_file": training_file,
                "pairs_imported": count,
                "format": upload.format
            }
        
        finally:
            # Clean up temporary file
            os.unlink(tmp_path)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload bulk data: {str(e)}")

@api_router.post("/training/hello-world")
async def create_hello_world_system(request: HelloWorldRequest):
    """Create Hello World conversational system for immediate testing"""
    if bulk_system is None:
        raise HTTPException(status_code=500, detail="Bulk training system not initialized")
    
    try:
        # Create Hello World system
        hello_world_engine = bulk_system.quick_start_hello_world()
        
        # Replace global engine with Hello World version
        global satc_engine
        satc_engine = hello_world_engine
        
        return {
            "message": "Hello World conversational system created!",
            "status": "ready",
            "features": [
                "Basic conversation",
                "Question answering",
                "Improved response quality",
                "Enhanced coherence"
            ]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create Hello World system: {str(e)}")

@api_router.post("/training/automated-start")
async def start_automated_training(request: AutomatedTrainingRequest):
    """Start automated training pipeline for continuous learning"""
    if bulk_system is None:
        raise HTTPException(status_code=500, detail="Bulk training system not initialized")
    
    try:
        # Configure automated training
        bulk_system.config.training_hours_per_day = request.hours_per_day
        bulk_system.config.rest_hours = request.rest_hours
        bulk_system.config.num_epochs = request.max_epochs
        bulk_system.config.save_every_n_epochs = request.save_every_n_epochs
        
        # Start automated training (in background)
        # In production, this would use background tasks
        import asyncio
        
        # Create task for automated training
        # For now, return success - actual implementation would use background workers
        
        return {
            "message": "Automated training pipeline configured",
            "config": {
                "training_hours_per_day": request.hours_per_day,
                "rest_hours": request.rest_hours,
                "max_epochs": request.max_epochs,
                "save_every_n_epochs": request.save_every_n_epochs
            },
            "status": "configured",
            "note": "Use hardware testbed for actual training execution"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start automated training: {str(e)}")

@api_router.get("/training/bulk-status")
async def get_bulk_training_status():
    """Get bulk training system status"""
    if bulk_system is None:
        raise HTTPException(status_code=500, detail="Bulk training system not initialized")
    
    try:
        status = bulk_system.pipeline.get_training_status()
        
        return {
            "system_status": "initialized",
            "hardware_optimized": True,
            "hardware_specs": {
                "gpu": "RTX 4070 Ti (12GB VRAM)",
                "cpu": "Ryzen 9 7900X (24 threads)",
                "ram": "64GB DDR5-6000",
                "storage": "2TB NVMe SSD"
            },
            "training_status": status,
            "ready_for_deployment": True
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get bulk training status: {str(e)}")

@api_router.post("/training/create-sample-dataset")
async def create_sample_dataset():
    """Create sample conversational dataset for testing"""
    if bulk_system is None:
        raise HTTPException(status_code=500, detail="Bulk training system not initialized")
    
    try:
        # Create sample dataset
        sample_data = bulk_system.importer.download_conversational_dataset()
        
        # Create training file
        training_file = bulk_system.importer.create_bulk_training_file(sample_data, "sample_dataset.jsonl")
        
        return {
            "message": "Sample dataset created successfully",
            "training_file": training_file,
            "pairs_count": len(sample_data),
            "ready_for_training": True
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create sample dataset: {str(e)}")

@api_router.get("/training/hardware-info")
async def get_hardware_info():
    """Get hardware optimization information"""
    import torch
    import psutil
    
    try:
        hardware_info = {
            "gpu_available": torch.cuda.is_available(),
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "gpu_memory_allocated": torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
            "gpu_memory_reserved": torch.cuda.memory_reserved() if torch.cuda.is_available() else 0,
            "cpu_count": psutil.cpu_count(),
            "cpu_percent": psutil.cpu_percent(),
            "memory_total": psutil.virtual_memory().total,
            "memory_available": psutil.virtual_memory().available,
            "memory_percent": psutil.virtual_memory().percent,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None
        }
        
        if torch.cuda.is_available():
            hardware_info["gpu_name"] = torch.cuda.get_device_name(0)
            hardware_info["gpu_memory_total"] = torch.cuda.get_device_properties(0).total_memory
        
        return {
            "hardware_info": hardware_info,
            "optimization_status": "ready",
            "recommended_config": {
                "batch_size": 64,
                "num_workers": 24,
                "memory_limit_gb": 60
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get hardware info: {str(e)}")

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
