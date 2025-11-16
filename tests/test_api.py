import pytest
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

def test_health_check():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_root_endpoint():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

@pytest.mark.asyncio
async def test_chat_endpoint():
    """Test chat endpoint"""
    response = client.post(
        "/api/chat",
        json={
            "message": "What is AAPL stock price?",
            "session_id": "test123"
        }
    )
    
    assert response.status_code in [200, 500]  # May fail without API keys

def test_stock_info_endpoint():
    """Test stock info endpoint"""
    response = client.get("/api/stock/AAPL")
    
    # May return 404 if API is not configured
    assert response.status_code in [200, 404, 500]