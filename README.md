# Jina Re-Ranker API

A FastAPI based local hosted Docker application that provides endpoints for document re-ranking using the Jina Re-Ranker v2 model.

## Requirements
- Docker
- Docker Compose
- NVIDIA GPU with CUDA support
- NVIDIA Container Toolkit

## Setup
1. Clone the repository
2. Create .env file with required environment variables
3. Run `docker compose up --build`

## API Endpoints

### Predict Endpoint
POST /predict
- Returns raw relevance scores for query-document pairs

### Rank Endpoint
POST /rank
- Returns ranked documents with scores and metadata

## Testing
Run tests using:
