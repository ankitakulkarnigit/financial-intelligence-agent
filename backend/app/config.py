# ==========================================
# app/config.py - Configuration Management
# ==========================================

from pydantic_settings import BaseSettings
from typing import Optional
import os
from pathlib import Path

class Settings(BaseSettings):
    """Application settings"""
    
    # Google Gemini
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    GOOGLE_PROJECT_ID: Optional[str] = os.getenv("GOOGLE_PROJECT_ID")
    
    # Financial Data APIs
    ALPHA_VANTAGE_API_KEY: str = os.getenv("ALPHA_VANTAGE_API_KEY", "demo")
    NEWS_API_KEY: Optional[str] = os.getenv("NEWS_API_KEY")
    
    # Application
    BACKEND_URL: str = os.getenv("BACKEND_URL", "http://localhost:8000")
    FRONTEND_URL: str = os.getenv("FRONTEND_URL", "http://localhost:8501")
    
    # Database
    CHROMA_DB_PATH: str = os.getenv("CHROMA_DB_PATH", "./data/vector_db")
    
    # AWS
    AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")
    AWS_ACCOUNT_ID: Optional[str] = os.getenv("AWS_ACCOUNT_ID")
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Initialize settings
settings = Settings()

# Create necessary directories
Path(settings.CHROMA_DB_PATH).mkdir(parents=True, exist_ok=True)


# ==========================================
# backend/requirements.txt
# ==========================================

"""
fastapi==0.109.0
uvicorn[standard]==0.27.0
pydantic==2.5.0
pydantic-settings==2.1.0

# Google AI
google-generativeai==0.3.2
google-cloud-aiplatform==1.38.0

# LangChain
langchain==0.1.0
langchain-google-genai==0.0.6
langchain-community==0.0.13

# Vector Database
chromadb==0.4.22

# Financial Data
yfinance==0.2.35
alpha-vantage==2.3.1
requests==2.31.0

# Data Processing
pandas==2.1.4
numpy==1.26.3
scikit-learn==1.4.0

# Utilities
python-dotenv==1.0.0
aiohttp==3.9.1
asyncio==3.4.3

# Monitoring
prometheus-client==0.19.0
"""


# ==========================================
# frontend/requirements.txt
# ==========================================

"""
streamlit==1.30.0
requests==2.31.0
plotly==5.18.0
pandas==2.1.4
numpy==1.26.3
python-dotenv==1.0.0
"""


# ==========================================
# deployment/.env.example
# ==========================================

"""
# Google Gemini API
GOOGLE_API_KEY=your_gemini_api_key_here
GOOGLE_PROJECT_ID=your_gcp_project_id

# Financial Data APIs
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
NEWS_API_KEY=your_news_api_key

# Application URLs
BACKEND_URL=http://localhost:8000
FRONTEND_URL=http://localhost:8501

# ChromaDB
CHROMA_DB_PATH=./data/vector_db

# AWS Configuration
AWS_REGION=us-east-1
AWS_ACCOUNT_ID=123456789012
ECR_REGISTRY=123456789012.dkr.ecr.us-east-1.amazonaws.com

# Logging
LOG_LEVEL=INFO
"""


# ==========================================
# backend/Dockerfile
# ==========================================

"""
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/

# Create data directory
RUN mkdir -p /app/data/vector_db

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
"""


# ==========================================
# frontend/Dockerfile
# ==========================================

"""
FROM python:3.10-slim

WORKDIR /app

# Install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY app.py .
COPY components/ ./components/

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
"""


# ==========================================
# deployment/docker-compose.yml
# ==========================================

"""
version: '3.8'

services:
  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    container_name: financial-agent-backend
    ports:
      - "8000:8000"
    environment:
      - GOOGLE_API_KEY=${GOOGLE_API_KEY}
      - ALPHA_VANTAGE_API_KEY=${ALPHA_VANTAGE_API_KEY}
      - NEWS_API_KEY=${NEWS_API_KEY}
      - CHROMA_DB_PATH=/app/data/vector_db
    volumes:
      - ./data/vector_db:/app/data/vector_db
      - ./backend/app:/app/app
    networks:
      - financial-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    container_name: financial-agent-frontend
    ports:
      - "8501:8501"
    environment:
      - BACKEND_URL=http://backend:8000
    depends_on:
      - backend
    networks:
      - financial-network
    restart: unless-stopped

networks:
  financial-network:
    driver: bridge

volumes:
  vector_db_data:
"""


# ==========================================
# tests/test_agents.py - Sample Tests
# ==========================================

"""
import pytest
from app.agents.adk_agents import ResearchAgent, AnalysisAgent
from app.agents.orchestrator import AgentOrchestrator

@pytest.mark.asyncio
async def test_research_agent():
    agent = ResearchAgent()
    result = await agent.research("AAPL", "What is Apple's current status?")
    
    assert result is not None
    assert "analysis" in result
    assert result["confidence"] >= 0

@pytest.mark.asyncio
async def test_orchestrator():
    orchestrator = AgentOrchestrator()
    result = await orchestrator.process_query("Analyze Tesla stock", session_id="test")
    
    assert result is not None
    assert "response" in result
    assert "reasoning_steps" in result
    assert len(result["reasoning_steps"]) > 0

def test_agent_initialization():
    orchestrator = AgentOrchestrator()
    
    assert orchestrator.research_agent is not None
    assert orchestrator.analysis_agent is not None
    assert orchestrator.sentiment_agent is not None
    assert orchestrator.prediction_agent is not None
"""


# ==========================================
# .gitignore
# ==========================================

"""
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
.venv

# Environment
.env
.env.local

# Data
data/
*.db
*.sqlite
chroma_db/

# IDEs
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
*.log
logs/

# Build
dist/
build/
*.egg-info/

# AWS
.aws/
credentials

# Docker
*.dockerfile.swp
"""
