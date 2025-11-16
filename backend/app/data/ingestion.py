
"""
Data Ingestion Module
"""
import yfinance as yf
import requests
from typing import List, Dict, Any
from datetime import datetime
import logging
import asyncio

from app.config import settings

logger = logging.getLogger(__name__)


class DataIngestion:
    """Handles data fetching from financial APIs"""
    
    def __init__(self):
        self.alpha_vantage_key = settings.ALPHA_VANTAGE_API_KEY
        self.news_api_key = settings.NEWS_API_KEY
    
    async def get_stock_data(self, symbol: str, period: str = "1mo") -> Dict[str, Any]:
        """Fetch stock data"""
        try:
            ticker = await asyncio.to_thread(yf.Ticker, symbol)
            hist = await asyncio.to_thread(ticker.history, period=period)
            info = await asyncio.to_thread(lambda: ticker.info)
            
            if hist.empty:
                raise ValueError(f"No data for {symbol}")
            
            current_price = hist['Close'].iloc[-1]
            previous_price = hist['Close'].iloc[0]
            change_percent = ((current_price - previous_price) / previous_price) * 100
            
            historical = []
            for date, row in hist.iterrows():
                historical.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "open": float(row['Open']),
                    "high": float(row['High']),
                    "low": float(row['Low']),
                    "close": float(row['Close']),
                    "volume": int(row['Volume'])
                })
            
            return {
                "symbol": symbol,
                "current_price": float(current_price),
                "change_percent": float(change_percent),
                "volume": int(hist['Volume'].iloc[-1]),
                "market_cap": info.get('marketCap', 0),
                "historical": historical,
                "company_name": info.get('longName', symbol)
            }
        
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {str(e)}")
            raise ValueError(f"Failed to fetch data for {symbol}")
    
    async def get_news(self, symbol: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Fetch news"""
        news_items = []
        
        try:
            ticker = await asyncio.to_thread(yf.Ticker, symbol)
            yf_news = await asyncio.to_thread(lambda: ticker.news)
            
            for item in yf_news[:limit]:
                news_items.append({
                    "title": item.get('title', ''),
                    "summary": item.get('summary', ''),
                    "publisher": item.get('publisher', 'Unknown'),
                    "url": item.get('link', ''),
                    "date": datetime.fromtimestamp(
                        item.get('providerPublishTime', 0)
                    ).strftime("%Y-%m-%d"),
                    "source": "Yahoo Finance"
                })
            
            return news_items[:limit]
        
        except Exception as e:
            logger.error(f"Error fetching news: {str(e)}")
            return []