from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from sentence_transformers import CrossEncoder
import torch
import asyncio
import time
import os

app = FastAPI()

# Security headers
@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)
    response.headers.update({
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block"
    })
    return response

# Model configuration
MODEL_NAME = os.getenv("MODEL_NAME", "jinaai/jina-reranker-v2-base-multilingual")
MAX_MODEL_LOAD_TIME = 300  # 5 minutes
model = CrossEncoder(
    MODEL_NAME,
    device='cuda' if torch.cuda.is_available() else 'cpu', trust_remote_code=True  # Add this back
)


class PredictRequest(BaseModel):
    query: str
    documents: List[str]
    batch_size: Optional[int] = 32

class RankRequest(BaseModel):
    query: str
    documents: List[str]
    top_k: Optional[int] = None
    return_documents: Optional[bool] = True
    batch_size: Optional[int] = 32

@app.on_event("startup")
async def initialize_services():
    await verify_cuda()
    await load_model()

async def verify_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available in Docker container")
    cuda_version = torch.version.cuda
    if cuda_version != "11.8":
        raise RuntimeError(f"Wrong CUDA version. Expected 11.8, got {cuda_version}")

async def load_model():
    try:
        if not hasattr(model, 'model') or model.model is None:
            model.load_model()
    except Exception as e:
        raise RuntimeError(f"Model loading failed: {str(e)}")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "torch_version": torch.__version__,
        "model_loaded": hasattr(model, 'model') and model.model is not None,
        "cache_dir": os.environ.get("TRANSFORMERS_CACHE", "")
    }

@app.post("/predict")
async def predict(request: PredictRequest):
    try:
        pairs = [[request.query, doc] for doc in request.documents]
        scores = model.predict(pairs, batch_size=request.batch_size).tolist()
        return {"scores": scores}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rank")
async def rank(request: RankRequest):
    try:
        pairs = [[request.query, doc] for doc in request.documents]
        scores = model.predict(pairs, batch_size=request.batch_size)
        
        # Combine scores with documents and sort
        scored_docs = sorted(
            [(float(score), doc) for score, doc in zip(scores, request.documents)],
            key=lambda x: x[0],
            reverse=True
        )
        
        # Apply top_k with default value
        top_k = request.top_k if request.top_k is not None else 3
        scored_docs = scored_docs[:top_k]
        
        # Format response
        return {
            "rankings": [{
                "score": score,
                "document": doc if request.return_documents else None
            } for score, doc in scored_docs]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
