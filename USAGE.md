# jina-reranker-v2-base-multilingual
Docker Container for jina-reranker-v2-base-multilingual with API Support

### Step 1: Create a Dockerfile

First, create a `Dockerfile` that sets up the environment and installs all required dependencies:

```dockerfile
# Use an official CUDA-enabled PyTorch image
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04 as builder

RUN apt-get update && \
    apt-get install -y python3.10 python3.10-distutils python3-pip && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install  --no-cache-dir --user -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118

FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

RUN apt-get update && \
    apt-get install -y python3.10 python3.10-distutils && \
    useradd -m appuser && \
    mkdir -p /cache/huggingface && \
    chown -R appuser:appuser /cache && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /root/.local /home/appuser/.local
COPY . /app

ENV PATH="/home/appuser/.local/bin:${PATH}" \
    PYTHONPATH="/app" \
    CUDA_VISIBLE_DEVICES="0" \
    TRANSFORMERS_CACHE="/cache/huggingface"

USER appuser

EXPOSE 8501
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8501"]


```

### Step 2: Create the FastAPI Application

Create a file named `main.py` that defines the FastAPI application:

```python
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
        # Create pairs of query and documents
        pairs = [[request.query, doc] for doc in request.documents]
        
        # Get scores using predict method
        scores = model.predict(pairs, batch_size=request.batch_size)
        
        # Create rankings with scores and documents
        rankings = []
        scored_docs = list(zip(scores, request.documents))
        sorted_docs = sorted(scored_docs, key=lambda x: x[0], reverse=True)
        
        # Apply top_k if specified
        if request.top_k:
            sorted_docs = sorted_docs[:request.top_k]
        
        # Format the response
        rankings = [
            {
                "score": float(score),
                "document": doc if request.return_documents else None
            }
            for score, doc in sorted_docs
        ]
        
        return {"rankings": rankings}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### Step 3: Create a Docker Compose File

Create a `docker-compose.yml` file to define and run the container:

```yaml
version: '3.8'

services:
  reranker-api:
    build:
      context: ./fastapi
      dockerfile: Dockerfile
    ports:
      - "8501:8501"
    environment:
      - MODEL_NAME=jinaai/jina-reranker-v2-base-multilingual
      - TRANSFORMERS_CACHE=/cache/huggingface
      - HF_TRUST_REMOTE_CODE=true
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    volumes:
      - model-cache:/cache
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped

volumes:
  model-cache:
```

### Step 4: Build and Run the Container

Run the following commands to build and start the container:

```bash
docker-compose build
docker-compose up
```

### Step 5: Test the API

Once the container is running, you can test the API using the following `curl` command:

```bash
curl -X POST http://localhost:8000/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "model": "jina-reranker-v2-base-multilingual",
    "query": "Organic skincare products for sensitive skin",
    "top_n": 3,
    "documents": [
        "Organic skincare for sensitive skin with aloe vera and chamomile.",
        "New makeup trends focus on bold colors and innovative techniques",
        "Bio-Hautpflege für empfindliche Haut mit Aloe Vera und Kamille",
        "Neue Make-up-Trends setzen auf kräftige Farben und innovative Techniken",
        "Cuidado de la piel orgánico para piel sensible con aloe vera y manzanilla",
        "Las nuevas tendencias de maquillaje se centran en colores vivos y técnicas innovadoras",
        "针对敏感肌专门设计的天然有机护肤产品",
        "新的化妆趋势注重鲜艳的颜色和创新的技巧",
        "敏感肌のために特別に設計された天然有機スキンケア製品",
        "新しいメイクのトレンドは鮮やかな色と革新的な技術に焦点を当てています"
    ]
}'
```

### Step 6: Verify the Output

The API should return the top 3 most relevant documents based on the query. The response should look something like this:

```json
[
    {"text": "Organic skincare for sensitive skin with aloe vera and chamomile.", "score": 0.8311430811882019},
    {"text": "Cuidado de la piel orgánico para piel sensible con aloe vera y manzanilla", "score": 0.7620701193809509},
    {"text": "Bio-Hautpflege für empfindliche Haut mit Aloe Vera und Kamille", "score": 0.6334102749824524}
]
```

### Notes:

1. **Hardware Requirements**: Make sure your system has an NVIDIA GPU with CUDA support. The Docker container uses CUDA-enabled PyTorch to leverage GPU acceleration.

2. **Dependencies**: The Dockerfile includes all necessary dependencies, including `transformers`, `einops`, `fastapi`, `uvicorn`, and `flash-attn` for flash attention support.

3. **Model Loading**: The model is loaded in evaluation mode and moved to the GPU if available. If no GPU is detected, it will fall back to CPU mode, but performance will be significantly reduced.

4. **API Endpoint**: The API is exposed on port 8501 and can be accessed at `http://localhost:8501/rerank`.

5. **Testing**: You can test the API using the provided `curl` command or any HTTP client like Postman.

This setup provides a scalable and efficient way to run the Jina Re-Ranker V2 model locally as an API endpoint, leveraging your GPU for faster computations.
