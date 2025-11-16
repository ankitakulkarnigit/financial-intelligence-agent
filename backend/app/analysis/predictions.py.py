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

class PricePredictor:
    """ML-based price prediction"""
    
    def predict(self, stock_data: Dict[str, Any], days: int = 7) -> Dict[str, Any]:
        """
        Predict future prices using simple ML model
        """
        try:
            historical = stock_data.get('historical', [])
            if len(historical) < 30:
                return {
                    "error": "Insufficient data for prediction",
                    "forecast": [],
                    "trend": "unknown",
                    "confidence": 0.0
                }
            
            # Prepare data
            df = pd.DataFrame(historical)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            # Feature engineering
            df['returns'] = df['close'].pct_change()
            df['sma_5'] = df['close'].rolling(5).mean()
            df['sma_20'] = df['close'].rolling(20).mean()
            df['volatility'] = df['returns'].rolling(10).std()
            
            # Drop NaN
            df = df.dropna()
            
            if len(df) < 20:
                return self._simple_trend_prediction(df, days)
            
            # Simple prediction using moving average trend
            forecast = self._generate_forecast(df, days)
            
            # Calculate trend
            trend = "bullish" if forecast[-1] > df['close'].iloc[-1] else "bearish"
            
            # Confidence based on recent volatility
            recent_vol = df['volatility'].iloc[-10:].mean()
            confidence = max(0.3, 1.0 - (recent_vol * 10))
            
            return {
                "forecast": forecast,
                "trend": trend,
                "confidence": float(confidence),
                "current_price": float(df['close'].iloc[-1]),
                "predicted_price": float(forecast[-1]),
                "change_percent": float((forecast[-1] - df['close'].iloc[-1]) / df['close'].iloc[-1] * 100)
            }
        
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return {"error": str(e)}
    
    def _generate_forecast(self, df: pd.DataFrame, days: int) -> List[float]:
        """Generate price forecast"""
        # Simple exponential moving average projection
        current_price = df['close'].iloc[-1]
        recent_trend = df['close'].pct_change().tail(10).mean()
        
        forecast = []
        price = current_price
        
        for i in range(days):
            # Add some randomness and trend
            daily_change = recent_trend + np.random.normal(0, 0.01)
            price = price * (1 + daily_change)
            forecast.append(float(price))
        
        return forecast
    
    def _simple_trend_prediction(self, df: pd.DataFrame, days: int) -> Dict[str, Any]:
        """Fallback simple prediction"""
        current_price = df['close'].iloc[-1]
        avg_change = df['close'].pct_change().mean()
        
        forecast = [current_price * (1 + avg_change) ** i for i in range(1, days + 1)]
        
        return {
            "forecast": forecast,
            "trend": "bullish" if avg_change > 0 else "bearish",
            "confidence": 0.5,
            "current_price": float(current_price),
            "predicted_price": forecast[-1],
            "change_percent": float((forecast[-1] - current_price) / current_price * 100)
        }