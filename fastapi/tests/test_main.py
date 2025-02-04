from fastapi.testclient import TestClient
from main import app
import pytest

client = TestClient(app)

@pytest.mark.parametrize("endpoint", ["/predict", "/rank"])
def test_base_endpoints(endpoint):
    response = client.get(endpoint)
    assert response.status_code == 405  # Method Not Allowed

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict_endpoint():
    response = client.post(
        "/predict",
        json={
            "query": "test query",
            "documents": ["valid doc 1", "valid doc 2"],
            "batch_size": 16
        }
    )
    assert response.status_code == 200
    assert len(response.json()["scores"]) == 2

def test_rank_endpoint():
    response = client.post(
        "/rank",
        json={
            "query": "test query",
            "documents": ["doc A", "doc B"],
            "top_k": 1,
            "batch_size": 8
        }
    )
    assert response.status_code == 200
    assert len(response.json()["rankings"]) == 1

def test_invalid_input():
    response = client.post(
        "/predict",
        json={
            "query": "test",
            "documents": [{"invalid": "format"}]
        }
    )
    assert response.status_code == 422

def test_large_batch():
    large_docs = ["doc"] * 500
    response = client.post(
        "/rank",
        json={
            "query": "test",
            "documents": large_docs,
            "batch_size": 64
        }
    )
    assert response.status_code == 200
    assert len(response.json()["rankings"]) == 500
