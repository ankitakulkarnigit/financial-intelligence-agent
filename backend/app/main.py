"""
FastAPI Backend for Financial Market Intelligence Agent
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uvicorn
from datetime import datetime
import logging

from app.config import settings
from app.agents.orchestrator import AgentOrchestrator
from app.data.ingestion import DataIngestion
from app.rag.retrieval import RAGPipeline
from app.analysis.technical import TechnicalAnalyzer
from app.analysis.sentiment import SentimentAnalyzer
from app.analysis.predictions import PricePredictor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Financial Market Intelligence API",
    description="AI-powered financial analysis with RAG and multi-agent system",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
orchestrator = AgentOrchestrator()
data_ingestion = DataIngestion()
rag_pipeline = RAGPipeline()
technical_analyzer = TechnicalAnalyzer()
sentiment_analyzer = SentimentAnalyzer()
price_predictor = PricePredictor()

# Pydantic models
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    include_sources: bool = True

class ChatResponse(BaseModel):
    response: str
    agent_reasoning: List[Dict[str, Any]]
    sources: Optional[List[Dict[str, str]]]
    timestamp: str

class AnalysisRequest(BaseModel):
    symbol: str
    analysis_type: str = "full"  # full, technical, sentiment, prediction
    timeframe: str = "1mo"  # 1d, 5d, 1mo, 3mo, 1y

class AnalysisResponse(BaseModel):
    symbol: str
    analysis: Dict[str, Any]
    recommendations: List[str]
    risk_level: str
    timestamp: str

class PredictionRequest(BaseModel):
    symbol: str
    days: int = 7
    confidence_interval: float = 0.95

class StockData(BaseModel):
    symbol: str
    current_price: float
    change: float
    volume: int
    market_cap: Optional[float]

# Routes
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Financial Market Intelligence API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "agents": "operational",
            "rag": "operational",
            "data_ingestion": "operational"
        }
    }

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat with the financial intelligence agent
    """
    try:
        logger.info(f"Chat request: {request.message}")
        
        # Process query through agent orchestrator
        result = await orchestrator.process_query(
            query=request.message,
            session_id=request.session_id
        )
        
        # Get relevant context from RAG
        if request.include_sources:
            sources = await rag_pipeline.retrieve_context(request.message)
        else:
            sources = None
        
        return ChatResponse(
            response=result["response"],
            agent_reasoning=result["reasoning_steps"],
            sources=sources,
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_stock(request: AnalysisRequest):
    """
    Perform comprehensive stock analysis
    """
    try:
        logger.info(f"Analysis request for {request.symbol}")
        
        # Fetch stock data
        stock_data = await data_ingestion.get_stock_data(
            request.symbol, 
            request.timeframe
        )
        
        analysis_results = {}
        
        # Technical analysis
        if request.analysis_type in ["full", "technical"]:
            analysis_results["technical"] = technical_analyzer.analyze(stock_data)
        
        # Sentiment analysis
        if request.analysis_type in ["full", "sentiment"]:
            news_data = await data_ingestion.get_news(request.symbol)
            analysis_results["sentiment"] = sentiment_analyzer.analyze(news_data)
        
        # Predictions
        if request.analysis_type in ["full", "prediction"]:
            analysis_results["prediction"] = price_predictor.predict(
                stock_data, 
                days=7
            )
        
        # Get AI-powered recommendations
        recommendations = await orchestrator.generate_recommendations(
            symbol=request.symbol,
            analysis_data=analysis_results
        )
        
        # Calculate risk level
        risk_level = _calculate_risk_level(analysis_results)
        
        return AnalysisResponse(
            symbol=request.symbol,
            analysis=analysis_results,
            recommendations=recommendations,
            risk_level=risk_level,
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stock/{symbol}")
async def get_stock_info(symbol: str):
    """
    Get current stock information
    """
    try:
        data = await data_ingestion.get_stock_data(symbol, "1d")
        
        return {
            "symbol": symbol,
            "price": data["current_price"],
            "change": data["change_percent"],
            "volume": data["volume"],
            "market_cap": data.get("market_cap"),
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Stock info error: {str(e)}")
        raise HTTPException(status_code=404, detail=f"Stock {symbol} not found")

@app.post("/api/predict/{symbol}")
async def predict_price(symbol: str, days: int = 7):
    """
    Predict future stock prices
    """
    try:
        logger.info(f"Prediction request for {symbol}, {days} days")
        
        # Get historical data
        stock_data = await data_ingestion.get_stock_data(symbol, "6mo")
        
        # Generate predictions
        predictions = price_predictor.predict(stock_data, days=days)
        
        return {
            "symbol": symbol,
            "current_price": stock_data["current_price"],
            "predictions": predictions["forecast"],
            "confidence": predictions["confidence"],
            "trend": predictions["trend"],
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ingest/documents")
async def ingest_documents(background_tasks: BackgroundTasks):
    """
    Ingest financial documents into RAG pipeline
    """
    try:
        # Run ingestion in background
        background_tasks.add_task(rag_pipeline.ingest_documents)
        
        return {
            "status": "started",
            "message": "Document ingestion started in background"
        }
    
    except Exception as e:
        logger.error(f"Ingestion error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/compare")
async def compare_stocks(symbols: str):
    """
    Compare multiple stocks
    symbols: comma-separated stock symbols (e.g., "AAPL,MSFT,GOOGL")
    """
    try:
        symbol_list = [s.strip().upper() for s in symbols.split(",")]
        
        if len(symbol_list) > 5:
            raise HTTPException(
                status_code=400, 
                detail="Maximum 5 stocks can be compared"
            )
        
        # Get data for all symbols
        comparison_data = []
        for symbol in symbol_list:
            data = await data_ingestion.get_stock_data(symbol, "1mo")
            technical = technical_analyzer.analyze(data)
            
            comparison_data.append({
                "symbol": symbol,
                "price": data["current_price"],
                "change_1mo": data.get("change_1mo", 0),
                "rsi": technical["indicators"]["rsi"],
                "volatility": technical["volatility"]
            })
        
        # Generate comparison insights
        insights = await orchestrator.compare_stocks(comparison_data)
        
        return {
            "stocks": comparison_data,
            "insights": insights,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Comparison error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Helper functions
def _calculate_risk_level(analysis: Dict[str, Any]) -> str:
    """Calculate overall risk level from analysis"""
    risk_score = 0
    
    if "technical" in analysis:
        volatility = analysis["technical"].get("volatility", 0)
        risk_score += min(volatility * 10, 40)
    
    if "sentiment" in analysis:
        sentiment_score = analysis["sentiment"].get("score", 0)
        if sentiment_score < -0.3:
            risk_score += 30
        elif sentiment_score < 0:
            risk_score += 15
    
    if risk_score > 60:
        return "HIGH"
    elif risk_score > 30:
        return "MEDIUM"
    else:
        return "LOW"

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )