
"""
Google Agentic Development Kit (ADK) - Multi-Agent System
Implements specialized agents for financial market intelligence
"""
import google.generativeai as genai
from typing import List, Dict, Any, Optional
import json
from datetime import datetime
import asyncio

from app.config import settings
from app.data.ingestion import DataIngestion
from app.rag.retrieval import RAGPipeline

# Configure Google Gemini
genai.configure(api_key=settings.GOOGLE_API_KEY)

class BaseAgent:
    """Base class for all agents"""
    
    def __init__(self, name: str, role: str, instructions: str):
        self.name = name
        self.role = role
        self.instructions = instructions
        self.model = genai.GenerativeModel(
            model_name='gemini-2.0-flash-exp',
            generation_config={
                'temperature': 0.7,
                'top_p': 0.95,
                'top_k': 40,
                'max_output_tokens': 2048,
            }
        )
        self.chat_history = []
    
    async def think(self, context: str, query: str) -> Dict[str, Any]:
        """Agent reasoning process"""
        prompt = f"""
{self.instructions}

Context:
{context}

Query: {query}

Provide your analysis in JSON format with:
- analysis: your detailed analysis
- key_findings: list of key findings
- confidence: confidence score 0-1
- next_steps: suggested next steps if any
"""
        
        try:
            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt
            )
            
            result = self._parse_response(response.text)
            
            self.chat_history.append({
                "query": query,
                "response": result,
                "timestamp": datetime.now().isoformat()
            })
            
            return result
        
        except Exception as e:
            return {
                "analysis": f"Error in agent reasoning: {str(e)}",
                "key_findings": [],
                "confidence": 0.0,
                "next_steps": []
            }
    
    def _parse_response(self, text: str) -> Dict[str, Any]:
        """Parse agent response"""
        try:
            start_idx = text.find('{')
            end_idx = text.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_text = text[start_idx:end_idx]
                return json.loads(json_text)
            else:
                return {
                    "analysis": text,
                    "key_findings": [],
                    "confidence": 0.7,
                    "next_steps": []
                }
        except json.JSONDecodeError:
            return {
                "analysis": text,
                "key_findings": [],
                "confidence": 0.7,
                "next_steps": []
            }


class ResearchAgent(BaseAgent):
    """Agent responsible for data research and collection"""
    
    def __init__(self):
        super().__init__(
            name="Research Agent",
            role="Data Researcher",
            instructions="""
You are a financial research agent. Your role is to:
1. Identify relevant data sources for the query
2. Extract key information from available data
3. Summarize findings in a structured format
4. Flag any data gaps or inconsistencies

Focus on factual, quantitative data. Be precise with numbers and dates.
"""
        )
        self.data_ingestion = DataIngestion()
    
    async def research(self, symbol: str, query: str) -> Dict[str, Any]:
        """Conduct research on a stock"""
        stock_data = await self.data_ingestion.get_stock_data(symbol, "3mo")
        news_data = await self.data_ingestion.get_news(symbol, limit=10)
        
        context = f"""
Stock: {symbol}
Current Price: ${stock_data['current_price']}
Volume: {stock_data['volume']:,}
Market Cap: ${stock_data.get('market_cap', 'N/A')}

Recent News Headlines:
{self._format_news(news_data)}
"""
        
        return await self.think(context, query)
    
    def _format_news(self, news_data: List[Dict]) -> str:
        """Format news data for context"""
        if not news_data:
            return "No recent news available"
        
        lines = []
        for item in news_data[:5]:
            lines.append(f"- {item.get('title', 'N/A')} ({item.get('date', 'N/A')})")
        return "\n".join(lines)


class AnalysisAgent(BaseAgent):
    """Agent for technical and fundamental analysis"""
    
    def __init__(self):
        super().__init__(
            name="Analysis Agent",
            role="Financial Analyst",
            instructions="""
You are a financial analysis expert. Your role is to:
1. Perform technical analysis (trends, patterns, indicators)
2. Evaluate fundamental metrics
3. Identify risks and opportunities
4. Provide data-driven insights

Use quantitative methods and be specific about metrics.
"""
        )
    
    async def analyze(self, symbol: str, research_data: Dict[str, Any], 
                     technical_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive analysis"""
        
        context = f"""
Research Findings:
{json.dumps(research_data, indent=2)}

Technical Indicators:
{json.dumps(technical_data, indent=2)}
"""
        
        query = f"Analyze {symbol} based on the research and technical data."
        
        return await self.think(context, query)


class SentimentAgent(BaseAgent):
    """Agent for sentiment analysis"""
    
    def __init__(self):
        super().__init__(
            name="Sentiment Agent",
            role="Sentiment Analyzer",
            instructions="""
You are a sentiment analysis expert. Your role is to:
1. Analyze news sentiment and tone
2. Identify market sentiment trends
3. Detect potential sentiment shifts
4. Provide sentiment scores from -1 to +1
"""
        )
    
    async def analyze_sentiment(self, symbol: str, news_data: List[Dict]) -> Dict[str, Any]:
        """Analyze sentiment from news"""
        
        context = f"""
Stock: {symbol}
News Articles:
{json.dumps(news_data, indent=2)}
"""
        
        query = "Analyze the overall sentiment. Provide a sentiment score."
        
        return await self.think(context, query)


class PredictionAgent(BaseAgent):
    """Agent for generating predictions"""
    
    def __init__(self):
        super().__init__(
            name="Prediction Agent",
            role="Market Forecaster",
            instructions="""
You are a market prediction specialist. Your role is to:
1. Generate price forecasts based on patterns
2. Identify trend directions
3. Assess prediction confidence
4. Provide probabilistic outlooks
"""
        )
    
    async def predict(self, symbol: str, analysis_results: Dict[str, Any],
                     prediction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate predictions"""
        
        context = f"""
Stock: {symbol}
Analysis: {json.dumps(analysis_results, indent=2)}
Model Output: {json.dumps(prediction_data, indent=2)}
"""
        
        query = "Provide a forecast for the next 7 days."
        
        return await self.think(context, query)


class ReportAgent(BaseAgent):
    """Agent for synthesizing reports"""
    
    def __init__(self):
        super().__init__(
            name="Report Agent",
            role="Report Synthesizer",
            instructions="""
You are a report creation expert. Your role is to:
1. Synthesize insights from multiple agents
2. Create clear recommendations
3. Highlight risks and opportunities
4. Present information clearly
"""
        )
    
    async def generate_report(self, symbol: str, 
                            all_agent_results: Dict[str, Dict[str, Any]]) -> str:
        """Generate final report"""
        
        context = f"""
Stock: {symbol}
Research: {json.dumps(all_agent_results.get('research', {}), indent=2)}
Analysis: {json.dumps(all_agent_results.get('analysis', {}), indent=2)}
Sentiment: {json.dumps(all_agent_results.get('sentiment', {}), indent=2)}
Predictions: {json.dumps(all_agent_results.get('prediction', {}), indent=2)}
"""
        
        query = "Create a comprehensive investment report."
        
        result = await self.think(context, query)
        return result.get('analysis', 'Report generation failed')
