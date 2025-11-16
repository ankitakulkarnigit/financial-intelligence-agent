import pytest
from backend.app.agents.adk_agents import ResearchAgent, AnalysisAgent
from backend.app.agents.orchestrator import AgentOrchestrator

@pytest.mark.asyncio
async def test_research_agent():
    """Test research agent functionality"""
    agent = ResearchAgent()
    result = await agent.research("AAPL", "What is Apple's current status?")
    
    assert result is not None
    assert "analysis" in result
    assert result["confidence"] >= 0

@pytest.mark.asyncio
async def test_orchestrator():
    """Test agent orchestrator"""
    orchestrator = AgentOrchestrator()
    result = await orchestrator.process_query("Analyze Tesla stock", session_id="test")
    
    assert result is not None
    assert "response" in result
    assert "reasoning_steps" in result
    assert len(result["reasoning_steps"]) > 0

def test_agent_initialization():
    """Test agent initialization"""
    orchestrator = AgentOrchestrator()
    
    assert orchestrator.research_agent is not None
    assert orchestrator.analysis_agent is not None
    assert orchestrator.sentiment_agent is not None
    assert orchestrator.prediction_agent is not None