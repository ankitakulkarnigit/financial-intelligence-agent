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


class TechnicalAnalyzer:
    """Technical analysis indicators and metrics"""
    
    def analyze(self, stock_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform technical analysis on stock data
        """
        try:
            historical = stock_data.get('historical', [])
            if not historical:
                return {"error": "No historical data available"}
            
            # Convert to pandas DataFrame
            df = pd.DataFrame(historical)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            # Calculate indicators
            indicators = self._calculate_indicators(df)
            
            # Determine trend
            trend = self._determine_trend(df, indicators)
            
            # Calculate volatility
            volatility = self._calculate_volatility(df)
            
            # Support and resistance
            support, resistance = self._find_support_resistance(df)
            
            return {
                "indicators": indicators,
                "trend": trend,
                "volatility": volatility,
                "support_level": support,
                "resistance_level": resistance,
                "signals": self._generate_signals(indicators, trend)
            }
        
        except Exception as e:
            logger.error(f"Technical analysis error: {str(e)}")
            return {"error": str(e)}
    
    def _calculate_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate technical indicators"""
        # RSI (Relative Strength Index)
        rsi = self._calculate_rsi(df['close'])
        
        # Moving Averages
        sma_20 = df['close'].rolling(window=20).mean().iloc[-1] if len(df) >= 20 else df['close'].mean()
        sma_50 = df['close'].rolling(window=50).mean().iloc[-1] if len(df) >= 50 else df['close'].mean()
        
        # MACD
        macd, signal = self._calculate_macd(df['close'])
        
        # Bollinger Bands
        bb_upper, bb_lower = self._calculate_bollinger_bands(df['close'])
        
        current_price = df['close'].iloc[-1]
        
        return {
            "rsi": float(rsi),
            "sma_20": float(sma_20),
            "sma_50": float(sma_50),
            "macd": float(macd),
            "macd_signal": float(signal),
            "bollinger_upper": float(bb_upper),
            "bollinger_lower": float(bb_lower),
            "current_price": float(current_price)
        }
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI"""
        if len(prices) < period + 1:
            return 50.0  # Neutral
        
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0
    
    def _calculate_macd(self, prices: pd.Series) -> tuple:
        """Calculate MACD"""
        if len(prices) < 26:
            return 0.0, 0.0
        
        ema_12 = prices.ewm(span=12, adjust=False).mean()
        ema_26 = prices.ewm(span=26, adjust=False).mean()
        
        macd = ema_12 - ema_26
        signal = macd.ewm(span=9, adjust=False).mean()
        
        return float(macd.iloc[-1]), float(signal.iloc[-1])
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20) -> tuple:
        """Calculate Bollinger Bands"""
        if len(prices) < period:
            avg = prices.mean()
            return avg * 1.02, avg * 0.98
        
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        upper = sma + (std * 2)
        lower = sma - (std * 2)
        
        return float(upper.iloc[-1]), float(lower.iloc[-1])
    
    def _determine_trend(self, df: pd.DataFrame, indicators: Dict) -> str:
        """Determine overall trend"""
        current_price = indicators['current_price']
        sma_20 = indicators['sma_20']
        sma_50 = indicators['sma_50']
        
        if current_price > sma_20 > sma_50:
            return "bullish"
        elif current_price < sma_20 < sma_50:
            return "bearish"
        else:
            return "neutral"
    
    def _calculate_volatility(self, df: pd.DataFrame) -> float:
        """Calculate historical volatility"""
        returns = df['close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)  # Annualized
        return float(volatility)
    
    def _find_support_resistance(self, df: pd.DataFrame) -> tuple:
        """Find support and resistance levels"""
        if len(df) < 20:
            current = df['close'].iloc[-1]
            return current * 0.95, current * 1.05
        
        # Simple approach: recent lows and highs
        recent_data = df.tail(30)
        support = float(recent_data['low'].min())
        resistance = float(recent_data['high'].max())
        
        return support, resistance
    
    def _generate_signals(self, indicators: Dict, trend: str) -> List[str]:
        """Generate trading signals"""
        signals = []
        
        rsi = indicators['rsi']
        current_price = indicators['current_price']
        bb_upper = indicators['bollinger_upper']
        bb_lower = indicators['bollinger_lower']
        
        # RSI signals
        if rsi > 70:
            signals.append("Overbought - potential reversal")
        elif rsi < 30:
            signals.append("Oversold - potential buying opportunity")
        
        # Bollinger Bands signals
        if current_price > bb_upper:
            signals.append("Price above upper Bollinger Band")
        elif current_price < bb_lower:
            signals.append("Price below lower Bollinger Band")
        
        # Trend signals
        if trend == "bullish":
            signals.append("Bullish trend established")
        elif trend == "bearish":
            signals.append("Bearish trend established")
        
        return signals if signals else ["No strong signals"]





