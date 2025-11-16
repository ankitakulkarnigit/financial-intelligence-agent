"""
Analysis Modules: Technical Analysis, Sentiment Analysis, Price Predictions
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """Sentiment analysis for news and social media"""
    
    def analyze(self, news_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze sentiment from news articles
        """
        try:
            if not news_data:
                return {
                    "score": 0.0,
                    "sentiment": "neutral",
                    "confidence": 0.0,
                    "positive_count": 0,
                    "negative_count": 0,
                    "neutral_count": 0
                }
            
            # Simple keyword-based sentiment (in production, use transformers)
            sentiments = []
            
            for article in news_data:
                text = f"{article.get('title', '')} {article.get('summary', '')}".lower()
                score = self._calculate_sentiment_score(text)
                sentiments.append(score)
            
            avg_sentiment = np.mean(sentiments)
            
            # Count sentiment categories
            positive_count = sum(1 for s in sentiments if s > 0.1)
            negative_count = sum(1 for s in sentiments if s < -0.1)
            neutral_count = len(sentiments) - positive_count - negative_count
            
            # Determine overall sentiment
            if avg_sentiment > 0.2:
                sentiment_label = "positive"
            elif avg_sentiment < -0.2:
                sentiment_label = "negative"
            else:
                sentiment_label = "neutral"
            
            return {
                "score": float(avg_sentiment),
                "sentiment": sentiment_label,
                "confidence": float(np.std(sentiments)),
                "positive_count": positive_count,
                "negative_count": negative_count,
                "neutral_count": neutral_count,
                "article_count": len(news_data)
            }
        
        except Exception as e:
            logger.error(f"Sentiment analysis error: {str(e)}")
            return {"error": str(e)}
    
    def _calculate_sentiment_score(self, text: str) -> float:
        """Simple keyword-based sentiment scoring"""
        positive_words = [
            'growth', 'profit', 'gain', 'increase', 'strong', 'beat',
            'surge', 'rally', 'bullish', 'positive', 'success', 'win',
            'high', 'record', 'breakthrough', 'innovative', 'excellent'
        ]
        
        negative_words = [
            'loss', 'decline', 'fall', 'weak', 'miss', 'drop',
            'crash', 'bearish', 'negative', 'fail', 'risk', 'concern',
            'warning', 'cut', 'reduce', 'poor', 'lawsuit', 'fraud'
        ]
        
        score = 0.0
        
        for word in positive_words:
            score += text.count(word) * 0.1
        
        for word in negative_words:
            score -= text.count(word) * 0.1
        
        # Normalize to -1 to 1
        return max(min(score, 1.0), -1.0)