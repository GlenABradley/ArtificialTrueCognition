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
    logger.info("Enhanced SATC Engine initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize SATC Engine: {str(e)}")
    satc_engine = None

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

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
