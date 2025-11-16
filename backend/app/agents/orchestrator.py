"""
Agent Orchestrator - Coordinates multiple agents for complex queries
"""
from typing import List, Dict, Any, Optional
import asyncio
import logging
from datetime import datetime

from app.agents.adk_agents import (
    ResearchAgent,
    AnalysisAgent,
    SentimentAgent,
    PredictionAgent,
    ReportAgent
)
from app.data.ingestion import DataIngestion
from app.analysis.technical import TechnicalAnalyzer
from app.analysis.predictions import PricePredictor

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """
    Orchestrates multiple agents to answer complex financial queries
    """
    
    def __init__(self):
        # Initialize all agents
        self.research_agent = ResearchAgent()
        self.analysis_agent = AnalysisAgent()
        self.sentiment_agent = SentimentAgent()
        self.prediction_agent = PredictionAgent()
        self.report_agent = ReportAgent()
        
        # Supporting components
        self.data_ingestion = DataIngestion()
        self.technical_analyzer = TechnicalAnalyzer()
        self.price_predictor = PricePredictor()
        
        # Session management
        self.sessions = {}
    
    async def process_query(self, query: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a user query by coordinating agents
        """
        logger.info(f"Processing query: {query}")
        
        # Parse query to extract stock symbols and intent
        query_info = self._parse_query(query)
        symbol = query_info.get("symbol")
        intent = query_info.get("intent")
        
        if not symbol:
            return {
                "response": "Please specify a stock symbol (e.g., AAPL, TSLA, MSFT).",
                "reasoning_steps": []
            }
        
        # Execute agent workflow based on intent
        if intent == "analysis":
            result = await self._run_full_analysis(symbol, query)
        elif intent == "prediction":
            result = await self._run_prediction(symbol, query)
        elif intent == "comparison":
            symbols = query_info.get("symbols", [symbol])
            result = await self._run_comparison(symbols, query)
        elif intent == "sentiment":
            result = await self._run_sentiment_analysis(symbol, query)
        else:
            result = await self._run_general_query(symbol, query)
        
        # Store in session if provided
        if session_id:
            self._update_session(session_id, query, result)
        
        return result
    
    async def _run_full_analysis(self, symbol: str, query: str) -> Dict[str, Any]:
        """Run complete multi-agent analysis"""
        reasoning_steps = []
        
        try:
            # Step 1: Research Agent gathers data
            logger.info(f"Step 1: Research Agent gathering data for {symbol}")
            research_result = await self.research_agent.research(symbol, query)
            reasoning_steps.append({
                "agent": "Research Agent",
                "action": "Data collection",
                "result": research_result.get("key_findings", [])
            })
            
            # Step 2: Get technical analysis
            logger.info(f"Step 2: Running technical analysis for {symbol}")
            stock_data = await self.data_ingestion.get_stock_data(symbol, "3mo")
            technical_data = self.technical_analyzer.analyze(stock_data)
            reasoning_steps.append({
                "agent": "Technical Analyzer",
                "action": "Technical indicators calculated",
                "result": {
                    "rsi": technical_data["indicators"]["rsi"],
                    "trend": technical_data["trend"]
                }
            })
            
            # Step 3: Analysis Agent evaluates
            logger.info(f"Step 3: Analysis Agent evaluating data")
            analysis_result = await self.analysis_agent.analyze(
                symbol, 
                research_result, 
                technical_data
            )
            reasoning_steps.append({
                "agent": "Analysis Agent",
                "action": "Comprehensive analysis",
                "result": analysis_result.get("key_findings", [])
            })
            
            # Step 4: Sentiment Agent analyzes news
            logger.info(f"Step 4: Sentiment Agent analyzing sentiment")
            news_data = await self.data_ingestion.get_news(symbol)
            sentiment_result = await self.sentiment_agent.analyze_sentiment(
                symbol, 
                news_data
            )
            reasoning_steps.append({
                "agent": "Sentiment Agent",
                "action": "Sentiment analysis",
                "result": {
                    "score": sentiment_result.get("analysis", ""),
                    "confidence": sentiment_result.get("confidence", 0)
                }
            })
            
            # Step 5: Prediction Agent forecasts
            logger.info(f"Step 5: Prediction Agent generating forecast")
            prediction_data = self.price_predictor.predict(stock_data, days=7)
            prediction_result = await self.prediction_agent.predict(
                symbol,
                analysis_result,
                prediction_data
            )
            reasoning_steps.append({
                "agent": "Prediction Agent",
                "action": "Price forecast",
                "result": prediction_result.get("key_findings", [])
            })
            
            # Step 6: Report Agent synthesizes
            logger.info(f"Step 6: Report Agent creating final report")
            all_results = {
                "research": research_result,
                "analysis": analysis_result,
                "sentiment": sentiment_result,
                "prediction": prediction_result
            }
            
            final_report = await self.report_agent.generate_report(symbol, all_results)
            reasoning_steps.append({
                "agent": "Report Agent",
                "action": "Report synthesis",
                "result": "Comprehensive report generated"
            })
            
            return {
                "response": final_report,
                "reasoning_steps": reasoning_steps,
                "raw_data": all_results
            }
        
        except Exception as e:
            logger.error(f"Error in full analysis: {str(e)}")
            return {
                "response": f"Analysis failed: {str(e)}",
                "reasoning_steps": reasoning_steps
            }
    
    async def _run_prediction(self, symbol: str, query: str) -> Dict[str, Any]:
        """Run prediction-focused analysis"""
        reasoning_steps = []
        
        try:
            # Get historical data
            stock_data = await self.data_ingestion.get_stock_data(symbol, "6mo")
            reasoning_steps.append({
                "agent": "Data Ingestion",
                "action": "Fetched 6 months historical data",
                "result": f"Retrieved data for {symbol}"
            })
            
            # Generate predictions
            prediction_data = self.price_predictor.predict(stock_data, days=7)
            reasoning_steps.append({
                "agent": "Price Predictor",
                "action": "ML prediction",
                "result": prediction_data.get("trend", "")
            })
            
            # Get technical context
            technical_data = self.technical_analyzer.analyze(stock_data)
            
            # Prediction agent interprets
            prediction_result = await self.prediction_agent.predict(
                symbol,
                {"technical": technical_data},
                prediction_data
            )
            reasoning_steps.append({
                "agent": "Prediction Agent",
                "action": "Forecast interpretation",
                "result": prediction_result.get("key_findings", [])
            })
            
            response = f"{prediction_result.get('analysis', '')}\n\n"
            response += f"Forecast: {prediction_data.get('trend', 'neutral').upper()}\n"
            response += f"Confidence: {prediction_data.get('confidence', 0)*100:.1f}%"
            
            return {
                "response": response,
                "reasoning_steps": reasoning_steps,
                "forecast": prediction_data
            }
        
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return {
                "response": f"Prediction failed: {str(e)}",
                "reasoning_steps": reasoning_steps
            }
    
    async def _run_sentiment_analysis(self, symbol: str, query: str) -> Dict[str, Any]:
        """Run sentiment-focused analysis"""
        reasoning_steps = []
        
        try:
            # Get news data
            news_data = await self.data_ingestion.get_news(symbol, limit=20)
            reasoning_steps.append({
                "agent": "Data Ingestion",
                "action": "Fetched news articles",
                "result": f"Retrieved {len(news_data)} articles"
            })
            
            # Sentiment analysis
            sentiment_result = await self.sentiment_agent.analyze_sentiment(
                symbol,
                news_data
            )
            reasoning_steps.append({
                "agent": "Sentiment Agent",
                "action": "Sentiment scoring",
                "result": sentiment_result.get("key_findings", [])
            })
            
            return {
                "response": sentiment_result.get("analysis", ""),
                "reasoning_steps": reasoning_steps,
                "sentiment_score": sentiment_result.get("confidence", 0)
            }
        
        except Exception as e:
            logger.error(f"Sentiment analysis error: {str(e)}")
            return {
                "response": f"Sentiment analysis failed: {str(e)}",
                "reasoning_steps": reasoning_steps
            }
    
    async def _run_comparison(self, symbols: List[str], query: str) -> Dict[str, Any]:
        """Compare multiple stocks"""
        reasoning_steps = []
        
        try:
            # Gather data for all symbols
            comparison_data = []
            
            for symbol in symbols[:5]:  # Limit to 5 stocks
                stock_data = await self.data_ingestion.get_stock_data(symbol, "1mo")
                technical = self.technical_analyzer.analyze(stock_data)
                
                comparison_data.append({
                    "symbol": symbol,
                    "price": stock_data["current_price"],
                    "change": stock_data.get("change_percent", 0),
                    "technical": technical
                })
            
            reasoning_steps.append({
                "agent": "Data Ingestion",
                "action": "Gathered comparison data",
                "result": f"Analyzed {len(comparison_data)} stocks"
            })
            
            # Analysis agent compares
            context = f"Comparison data:\n{comparison_data}"
            result = await self.analysis_agent.think(context, query)
            
            reasoning_steps.append({
                "agent": "Analysis Agent",
                "action": "Comparative analysis",
                "result": result.get("key_findings", [])
            })
            
            return {
                "response": result.get("analysis", ""),
                "reasoning_steps": reasoning_steps,
                "comparison_data": comparison_data
            }
        
        except Exception as e:
            logger.error(f"Comparison error: {str(e)}")
            return {
                "response": f"Comparison failed: {str(e)}",
                "reasoning_steps": reasoning_steps
            }
    
    async def _run_general_query(self, symbol: str, query: str) -> Dict[str, Any]:
        """Handle general queries"""
        reasoning_steps = []
        
        try:
            # Basic research
            research_result = await self.research_agent.research(symbol, query)
            reasoning_steps.append({
                "agent": "Research Agent",
                "action": "Information gathering",
                "result": research_result.get("key_findings", [])
            })
            
            return {
                "response": research_result.get("analysis", ""),
                "reasoning_steps": reasoning_steps
            }
        
        except Exception as e:
            return {
                "response": f"Query processing failed: {str(e)}",
                "reasoning_steps": reasoning_steps
            }
    
    def _parse_query(self, query: str) -> Dict[str, Any]:
        """Parse query to extract intent and entities"""
        query_lower = query.lower()
        
        # Extract stock symbols (simple pattern matching)
        import re
        symbols = re.findall(r'\b[A-Z]{1,5}\b', query.upper())
        
        # Determine intent
        intent = "general"
        if any(word in query_lower for word in ["predict", "forecast", "future", "next"]):
            intent = "prediction"
        elif any(word in query_lower for word in ["analyze", "analysis", "evaluate"]):
            intent = "analysis"
        elif any(word in query_lower for word in ["compare", "vs", "versus"]):
            intent = "comparison"
        elif any(word in query_lower for word in ["sentiment", "news", "feeling"]):
            intent = "sentiment"
        
        return {
            "symbol": symbols[0] if symbols else None,
            "symbols": symbols,
            "intent": intent
        }
    
    def _update_session(self, session_id: str, query: str, result: Dict[str, Any]):
        """Update session history"""
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        
        self.sessions[session_id].append({
            "query": query,
            "result": result,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep only last 10 interactions
        if len(self.sessions[session_id]) > 10:
            self.sessions[session_id] = self.sessions[session_id][-10:]
    
    async def generate_recommendations(self, symbol: str, 
                                      analysis_data: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations"""
        context = f"Analysis for {symbol}:\n{analysis_data}"
        query = "Based on this analysis, provide 3-5 actionable investment recommendations."
        
        result = await self.report_agent.think(context, query)
        
        recommendations = result.get("key_findings", [])
        if not recommendations:
            recommendations = ["Hold position", "Monitor key indicators", "Reassess in 30 days"]
        
        return recommendations
    
    async def compare_stocks(self, comparison_data: List[Dict[str, Any]]) -> str:
        """Generate comparison insights"""
        context = f"Stock comparison:\n{comparison_data}"
        query = "Compare these stocks and identify the best opportunity."
        
        result = await self.analysis_agent.think(context, query)
        
        return result.get("analysis", "Unable to generate comparison insights")