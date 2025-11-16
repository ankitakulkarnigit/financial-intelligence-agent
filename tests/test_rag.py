import pytest
from backend.app.rag.retrieval import RAGPipeline

@pytest.mark.asyncio
async def test_rag_initialization():
    """Test RAG pipeline initialization"""
    rag = RAGPipeline()
    
    assert rag.collection is not None
    assert rag.chroma_client is not None

@pytest.mark.asyncio
async def test_document_addition():
    """Test adding documents to RAG"""
    rag = RAGPipeline()
    
    documents = [
        {
            "content": "Test document about stock market",
            "metadata": {"source": "test", "type": "test"}
        }
    ]
    
    await rag.add_documents(documents)
    stats = rag.get_stats()
    
    assert stats["status"] == "operational"

@pytest.mark.asyncio
async def test_context_retrieval():
    """Test context retrieval"""
    rag = RAGPipeline()
    
    results = await rag.retrieve_context("stock market analysis", top_k=3)
    
    assert isinstance(results, list)