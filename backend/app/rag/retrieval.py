"""
RAG (Retrieval-Augmented Generation) Pipeline
Using ChromaDB for vector storage and Google Gemini for embeddings
"""
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import google.generativeai as genai
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path
import asyncio

from app.config import settings

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    RAG pipeline for financial document retrieval and context generation
    """
    
    def __init__(self):
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=settings.CHROMA_DB_PATH,
            settings=Settings(
                anonymized_telemetry=False
            )
        )
        
        # Initialize embedding function using Gemini
        genai.configure(api_key=settings.GOOGLE_API_KEY)
        
        # Get or create collection
        self.collection = self._get_or_create_collection()
        
        logger.info("RAG Pipeline initialized")
    
    def _get_or_create_collection(self):
        """Get or create ChromaDB collection"""
        try:
            return self.chroma_client.get_collection(
                name="financial_documents"
            )
        except:
            # Create collection with metadata
            return self.chroma_client.create_collection(
                name="financial_documents",
                metadata={"description": "Financial documents and market data"}
            )
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding using Gemini"""
        try:
            result = genai.embed_content(
                model="models/embedding-001",
                content=text,
                task_type="retrieval_document"
            )
            return result['embedding']
        except Exception as e:
            logger.error(f"Embedding generation error: {str(e)}")
            # Return zero vector as fallback
            return [0.0] * 768
    
    async def retrieve_context(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context from vector database
        """
        try:
            # Generate query embedding
            query_embedding = await asyncio.to_thread(
                self._generate_embedding,
                query
            )
            
            # Query ChromaDB
            results = await asyncio.to_thread(
                self.collection.query,
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            contexts = []
            if results and results['documents'] and len(results['documents'][0]) > 0:
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                    distance = results['distances'][0][i] if results['distances'] else 0
                    
                    contexts.append({
                        "content": doc,
                        "source": metadata.get("source", "Unknown"),
                        "type": metadata.get("type", "document"),
                        "relevance_score": 1 - distance,  # Convert distance to similarity
                        "metadata": metadata
                    })
            
            logger.info(f"Retrieved {len(contexts)} relevant contexts for query")
            return contexts
        
        except Exception as e:
            logger.error(f"Context retrieval error: {str(e)}")
            return []
    
    async def add_documents(self, documents: List[Dict[str, Any]]):
        """
        Add documents to the vector database
        
        documents: List of dicts with 'content', 'metadata', and optional 'id'
        """
        try:
            doc_ids = []
            doc_contents = []
            doc_embeddings = []
            doc_metadatas = []
            
            for i, doc in enumerate(documents):
                doc_id = doc.get('id', f"doc_{i}_{hash(doc['content'])}")
                content = doc['content']
                metadata = doc.get('metadata', {})
                
                # Generate embedding
                embedding = await asyncio.to_thread(
                    self._generate_embedding,
                    content
                )
                
                doc_ids.append(doc_id)
                doc_contents.append(content)
                doc_embeddings.append(embedding)
                doc_metadatas.append(metadata)
            
            # Add to collection
            await asyncio.to_thread(
                self.collection.add,
                ids=doc_ids,
                documents=doc_contents,
                embeddings=doc_embeddings,
                metadatas=doc_metadatas
            )
            
            logger.info(f"Added {len(documents)} documents to vector database")
        
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
    
    async def ingest_documents(self):
        """
        Ingest financial documents and reports
        This can be run periodically to update the knowledge base
        """
        logger.info("Starting document ingestion...")
        
        # Sample financial documents (in production, fetch from APIs/databases)
        sample_docs = [
            {
                "content": """
                Tesla Inc. (TSLA) is an American electric vehicle and clean energy company.
                Key metrics: Market Cap $800B+, Strong revenue growth, High valuation multiples.
                Main products: Model S, Model 3, Model X, Model Y, Solar panels, Energy storage.
                CEO: Elon Musk. Founded: 2003. Headquarters: Austin, Texas.
                """,
                "metadata": {
                    "source": "Company Overview",
                    "symbol": "TSLA",
                    "type": "company_profile",
                    "date": "2024"
                }
            },
            {
                "content": """
                Apple Inc. (AAPL) is a multinational technology company.
                Key metrics: Market Cap $3T+, Strong profitability, Consistent dividend growth.
                Main products: iPhone, iPad, Mac, Apple Watch, Services (App Store, iCloud).
                CEO: Tim Cook. Founded: 1976. Headquarters: Cupertino, California.
                Known for innovation, brand loyalty, and ecosystem lock-in.
                """,
                "metadata": {
                    "source": "Company Overview",
                    "symbol": "AAPL",
                    "type": "company_profile",
                    "date": "2024"
                }
            },
            {
                "content": """
                Microsoft Corporation (MSFT) is a leading technology company.
                Key metrics: Market Cap $3T+, Cloud growth driving revenue, Strong margins.
                Main products: Windows, Office 365, Azure, Xbox, LinkedIn.
                CEO: Satya Nadella. Founded: 1975. Headquarters: Redmond, Washington.
                Major growth areas: Cloud computing, AI integration, Enterprise software.
                """,
                "metadata": {
                    "source": "Company Overview",
                    "symbol": "MSFT",
                    "type": "company_profile",
                    "date": "2024"
                }
            },
            {
                "content": """
                Investment Strategy: Value Investing
                Value investing involves buying stocks trading below intrinsic value.
                Key principles: Margin of safety, Long-term perspective, Fundamental analysis.
                Metrics: P/E ratio, P/B ratio, Dividend yield, Free cash flow.
                Famous practitioners: Warren Buffett, Benjamin Graham.
                """,
                "metadata": {
                    "source": "Investment Strategies",
                    "type": "educational",
                    "date": "2024"
                }
            },
            {
                "content": """
                Technical Analysis: RSI (Relative Strength Index)
                RSI measures momentum and overbought/oversold conditions.
                Scale: 0-100. Above 70: Overbought. Below 30: Oversold.
                Used to identify potential reversal points and trend strength.
                Best combined with other indicators for confirmation.
                """,
                "metadata": {
                    "source": "Technical Indicators",
                    "type": "educational",
                    "date": "2024"
                }
            }
        ]
        
        await self.add_documents(sample_docs)
        logger.info("Document ingestion complete")
    
    async def search_similar(self, text: str, top_k: int = 3) -> List[str]:
        """Search for similar documents"""
        contexts = await self.retrieve_context(text, top_k)
        return [ctx['content'] for ctx in contexts]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        try:
            count = self.collection.count()
            return {
                "total_documents": count,
                "collection_name": self.collection.name,
                "status": "operational"
            }
        except Exception as e:
            return {
                "total_documents": 0,
                "error": str(e),
                "status": "error"
            }
    
    async def update_document(self, doc_id: str, new_content: str, 
                            new_metadata: Optional[Dict] = None):
        """Update an existing document"""
        try:
            embedding = await asyncio.to_thread(
                self._generate_embedding,
                new_content
            )
            
            await asyncio.to_thread(
                self.collection.update,
                ids=[doc_id],
                documents=[new_content],
                embeddings=[embedding],
                metadatas=[new_metadata] if new_metadata else None
            )
            
            logger.info(f"Updated document {doc_id}")
        except Exception as e:
            logger.error(f"Error updating document: {str(e)}")
    
    async def delete_document(self, doc_id: str):
        """Delete a document from the collection"""
        try:
            await asyncio.to_thread(
                self.collection.delete,
                ids=[doc_id]
            )
            logger.info(f"Deleted document {doc_id}")
        except Exception as e:
            logger.error(f"Error deleting document: {str(e)}")
    
    def clear_collection(self):
        """Clear all documents from collection (use with caution)"""
        try:
            self.chroma_client.delete_collection("financial_documents")
            self.collection = self._get_or_create_collection()
            logger.warning("Collection cleared")
        except Exception as e:
            logger.error(f"Error clearing collection: {str(e)}")