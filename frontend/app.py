"""
Streamlit Frontend for Financial Market Intelligence Agent
"""
import streamlit as st
import requests
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime
import json

# Page config
st.set_page_config(
    page_title="Financial Intelligence Agent",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
BACKEND_URL = "http://localhost:8000"

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .agent-step {
        background-color: #e8f4f8;
        padding: 0.8rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
        border-radius: 0.3rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(datetime.now().timestamp())

def make_api_request(endpoint, method="GET", data=None):
    """Make API request to backend"""
    url = f"{BACKEND_URL}{endpoint}"
    try:
        if method == "GET":
            response = requests.get(url, timeout=30)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=30)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Connection Error: {str(e)}")
        return None

def plot_stock_chart(historical_data):
    """Create interactive stock price chart"""
    df = pd.DataFrame(historical_data)
    df['date'] = pd.to_datetime(df['date'])
    
    fig = go.Figure()
    
    fig.add_trace(go.Candlestick(
        x=df['date'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Price'
    ))
    
    fig.update_layout(
        title='Stock Price History',
        yaxis_title='Price (USD)',
        xaxis_title='Date',
        template='plotly_white',
        height=400
    )
    
    return fig

def plot_prediction(current_price, forecast):
    """Plot price predictions"""
    dates = pd.date_range(start=datetime.now(), periods=len(forecast) + 1, freq='D')
    
    fig = go.Figure()
    
    # Historical (current) point
    fig.add_trace(go.Scatter(
        x=[dates[0]],
        y=[current_price],
        mode='markers',
        name='Current Price',
        marker=dict(size=12, color='blue')
    ))
    
    # Forecast line
    fig.add_trace(go.Scatter(
        x=dates[1:],
        y=forecast,
        mode='lines+markers',
        name='Forecast',
        line=dict(color='orange', dash='dash')
    ))
    
    fig.update_layout(
        title='Price Forecast',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        template='plotly_white',
        height=400
    )
    
    return fig

def display_agent_reasoning(reasoning_steps):
    """Display agent reasoning process"""
    st.subheader("🤖 Agent Reasoning Process")
    
    for i, step in enumerate(reasoning_steps):
        with st.expander(f"Step {i+1}: {step['agent']} - {step['action']}", expanded=False):
            st.markdown(f"**Agent:** {step['agent']}")
            st.markdown(f"**Action:** {step['action']}")
            
            if isinstance(step['result'], dict):
                st.json(step['result'])
            elif isinstance(step['result'], list):
                for item in step['result']:
                    st.markdown(f"- {item}")
            else:
                st.write(step['result'])

# Main App
def main():
    # Header
    st.markdown('<h1 class="main-header">📈 Financial Market Intelligence Agent</h1>', unsafe_allow_html=True)
    st.markdown("*AI-Powered Stock Analysis with Google Gemini, LangChain & RAG*")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Navigation
        page = st.radio(
            "Navigate",
            ["🏠 Home", "💬 Chat Agent", "📊 Stock Analysis", "🔮 Predictions", "⚖️ Compare Stocks"]
        )
        
        st.markdown("---")
        
        # Quick stock lookup
        st.subheader("Quick Lookup")
        quick_symbol = st.text_input("Stock Symbol", "AAPL", key="quick_symbol")
        
        if st.button("Get Quote"):
            with st.spinner("Fetching..."):
                data = make_api_request(f"/api/stock/{quick_symbol}")
                if data:
                    st.success(f"**${data['price']:.2f}**")
                    st.metric("Change", f"{data['change']:.2f}%")
        
        st.markdown("---")
        st.markdown("### About")
        st.info("""
        This agent uses:
        - **Google Gemini** (LLM)
        - **Google ADK** (Multi-Agent)
        - **RAG** (ChromaDB)
        - **FastAPI** (Backend)
        - **Streamlit** (Frontend)
        """)
    
    # Main content based on page selection
    if "Home" in page:
        show_home()
    elif "Chat Agent" in page:
        show_chat_agent()
    elif "Stock Analysis" in page:
        show_stock_analysis()
    elif "Predictions" in page:
        show_predictions()
    elif "Compare Stocks" in page:
        show_comparison()

def show_home():
    """Home page with overview"""
    st.header("Welcome to Financial Intelligence Agent")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🔍 Research Agent")
        st.write("Gathers and analyzes financial data from multiple sources")
    
    with col2:
        st.markdown("### 📊 Analysis Agent")
        st.write("Performs technical and fundamental analysis")
    
    with col3:
        st.markdown("### 🔮 Prediction Agent")
        st.write("Generates price forecasts using ML models")
    
    st.markdown("---")
    
    # Market Overview
    st.subheader("📈 Market Overview")
    
    overview_data = {
        "Index": ["S&P 500", "NASDAQ", "Dow Jones"],
        "Value": [4500, 14000, 35000],
        "Change": ["+0.5%", "+0.8%", "+0.3%"]
    }
    
    df = pd.DataFrame(overview_data)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Quick Stats
    st.subheader("🚀 System Status")
    
    health = make_api_request("/health")
    if health:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Status", "🟢 Operational")
        with col2:
            st.metric("Agents", "5 Active")
        with col3:
            st.metric("RAG Database", "Operational")

def show_chat_agent():
    """Chat interface with AI agent"""
    st.header("💬 Chat with Financial Intelligence Agent")
    st.markdown("Ask questions about stocks, markets, or investment strategies")
    
    # Chat input
    user_input = st.text_input(
        "Your question:",
        placeholder="e.g., Analyze Tesla's performance and compare with Ford",
        key="chat_input"
    )
    
    col1, col2 = st.columns([1, 5])
    with col1:
        send_button = st.button("Send", use_container_width=True)
    with col2:
        if st.button("Clear History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    
    if send_button and user_input:
        with st.spinner("🤖 Agent is thinking..."):
            # Make API request
            response = make_api_request(
                "/api/chat",
                method="POST",
                data={
                    "message": user_input,
                    "session_id": st.session_state.session_id,
                    "include_sources": True
                }
            )
            
            if response:
                # Add to chat history
                st.session_state.chat_history.append({
                    "user": user_input,
                    "agent": response['response'],
                    "reasoning": response.get('reasoning_steps', []),
                    "sources": response.get('sources', []),
                    "timestamp": response['timestamp']
                })
    
    # Display chat history
    for i, chat in enumerate(reversed(st.session_state.chat_history)):
        st.markdown("---")
        
        # User message
        st.markdown(f"**👤 You:** {chat['user']}")
        st.markdown(f"*{chat['timestamp']}*")
        
        # Agent response
        st.markdown(f"**🤖 Agent:** {chat['agent']}")
        
        # Show reasoning in expander
        if chat.get('reasoning'):
            display_agent_reasoning(chat['reasoning'])
        
        # Show sources
        if chat.get('sources'):
            with st.expander("📚 Sources"):
                for source in chat['sources'][:3]:
                    st.markdown(f"- **{source.get('source', 'Unknown')}** (Relevance: {source.get('relevance_score', 0):.2f})")
                    st.text(source.get('content', '')[:200] + "...")

def show_stock_analysis():
    """Stock analysis page"""
    st.header("📊 Comprehensive Stock Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        symbol = st.text_input("Enter Stock Symbol", "AAPL", key="analysis_symbol")
    
    with col2:
        timeframe = st.selectbox("Timeframe", ["1mo", "3mo", "6mo", "1y"])
    
    analysis_type = st.multiselect(
        "Analysis Type",
        ["Technical", "Sentiment", "Prediction"],
        default=["Technical", "Sentiment"]
    )
    
    if st.button("Run Analysis", use_container_width=True):
        with st.spinner(f"Analyzing {symbol}..."):
            # Map to API format
            api_type = "full" if len(analysis_type) > 1 else analysis_type[0].lower()
            
            response = make_api_request(
                "/api/analyze",
                method="POST",
                data={
                    "symbol": symbol,
                    "analysis_type": api_type,
                    "timeframe": timeframe
                }
            )
            
            if response:
                st.success(f"Analysis complete for {symbol}")
                
                # Display results
                analysis = response['analysis']
                
                # Technical Analysis
                if 'technical' in analysis:
                    st.subheader("🔧 Technical Analysis")
                    
                    tech = analysis['technical']
                    indicators = tech.get('indicators', {})
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("RSI", f"{indicators.get('rsi', 0):.1f}")
                    with col2:
                        st.metric("Trend", tech.get('trend', 'unknown').upper())
                    with col3:
                        st.metric("Volatility", f"{tech.get('volatility', 0):.2%}")
                    with col4:
                        st.metric("Support", f"${tech.get('support_level', 0):.2f}")
                    
                    # Signals
                    st.markdown("**Trading Signals:**")
                    for signal in tech.get('signals', []):
                        st.markdown(f"- {signal}")
                
                # Sentiment Analysis
                if 'sentiment' in analysis:
                    st.subheader("😊 Sentiment Analysis")
                    
                    sent = analysis['sentiment']
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        sentiment_score = sent.get('score', 0)
                        st.metric("Sentiment Score", f"{sentiment_score:.2f}")
                        
                        # Sentiment gauge
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=sentiment_score,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            gauge={
                                'axis': {'range': [-1, 1]},
                                'bar': {'color': "darkblue"},
                                'steps': [
                                    {'range': [-1, -0.3], 'color': "red"},
                                    {'range': [-0.3, 0.3], 'color': "yellow"},
                                    {'range': [0.3, 1], 'color': "green"}
                                ]
                            }
                        ))
                        fig.update_layout(height=250)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.metric("Positive Articles", sent.get('positive_count', 0))
                        st.metric("Negative Articles", sent.get('negative_count', 0))
                        st.metric("Neutral Articles", sent.get('neutral_count', 0))
                
                # Recommendations
                st.subheader("💡 Recommendations")
                for rec in response.get('recommendations', []):
                    st.success(rec)
                
                # Risk Level
                risk = response.get('risk_level', 'UNKNOWN')
                risk_color = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}.get(risk, "⚪")
                st.markdown(f"**Risk Level:** {risk_color} {risk}")

def show_predictions():
    """Price prediction page"""
    st.header("🔮 Stock Price Predictions")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        symbol = st.text_input("Stock Symbol", "TSLA", key="pred_symbol")
    
    with col2:
        days = st.slider("Forecast Days", 1, 30, 7)
    
    if st.button("Generate Forecast", use_container_width=True):
        with st.spinner(f"Generating {days}-day forecast for {symbol}..."):
            response = make_api_request(
                f"/api/predict/{symbol}?days={days}",
                method="POST"
            )
            
            if response:
                st.success("Forecast generated!")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Current Price", f"${response['current_price']:.2f}")
                
                with col2:
                    predicted = response['predictions']['forecast'][-1]
                    st.metric("Predicted Price", f"${predicted:.2f}")
                
                with col3:
                    trend = response['predictions']['trend']
                    st.metric("Trend", trend.upper())
                
                # Plot forecast
                fig = plot_prediction(
                    response['current_price'],
                    response['predictions']['forecast']
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Confidence
                confidence = response['predictions']['confidence']
                st.progress(confidence)
                st.caption(f"Confidence: {confidence*100:.1f}%")

def show_comparison():
    """Stock comparison page"""
    st.header("⚖️ Compare Multiple Stocks")
    
    symbols_input = st.text_input(
        "Enter stock symbols (comma-separated)",
        "AAPL,MSFT,GOOGL",
        key="compare_symbols"
    )
    
    if st.button("Compare", use_container_width=True):
        with st.spinner("Comparing stocks..."):
            response = make_api_request(f"/api/compare?symbols={symbols_input}")
            
            if response:
                stocks = response['stocks']
                
                # Create comparison table
                df = pd.DataFrame(stocks)
                st.dataframe(df, use_container_width=True, hide_index=True)
                
                # Price comparison chart
                fig = go.Figure()
                
                for stock in stocks:
                    fig.add_trace(go.Bar(
                        name=stock['symbol'],
                        x=[stock['symbol']],
                        y=[stock['price']]
                    ))
                
                fig.update_layout(
                    title="Price Comparison",
                    yaxis_title="Price (USD)",
                    template="plotly_white"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # AI Insights
                st.subheader("🤖 AI Insights")
                st.write(response['insights'])

if __name__ == "__main__":
    main()