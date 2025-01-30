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
from transformers import AutoModelForSequenceClassification
import torch

app = FastAPI()

# Load the model
model = AutoModelForSequenceClassification.from_pretrained(
    'jinaai/jina-reranker-v2-base-multilingual',
    torch_dtype="auto",
    trust_remote_code=True,
)

# Move the model to the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

@app.post("/rerank")
async def rerank(
    data: dict
):
    try:
        query = data.get("query", "")
        documents = data.get("documents", [])
        top_n = data.get("top_n", 3)
        max_length = data.get("max_length", 1024)
        
        if not query or not documents:
            raise HTTPException(status_code=400, detail="Query and documents are required")
        
        # Construct sentence pairs
        sentence_pairs = [[query, doc] for doc in documents]
        
        # Compute scores
        scores = model.compute_score(sentence_pairs, max_length=max_length)
        
        # Combine documents and scores, then sort
        document_scores = list(zip(documents, scores))
        document_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_n documents
        top_documents = document_scores[:top_n]
        
        return [
            {"text": doc, "score": score} 
            for doc, score in top_documents
        ]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Step 3: Create a Docker Compose File

Create a `docker-compose.yml` file to define and run the container:

```yaml
version: '3.8'

services:
  reranker:
    build: .
    ports:
      - "8000:8000"
    environment:
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        limits:
          devices:
            - driver: nvidia
              device: /dev/nvidia0
              count: 1
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

4. **API Endpoint**: The API is exposed on port 8000 and can be accessed at `http://localhost:8000/rerank`.

5. **Testing**: You can test the API using the provided `curl` command or any HTTP client like Postman.

This setup provides a scalable and efficient way to run the Jina Re-Ranker V2 model locally as an API endpoint, leveraging your GPU for faster computations.
