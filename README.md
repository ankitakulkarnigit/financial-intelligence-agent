![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30.0-red.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)
![AWS](https://img.shields.io/badge/AWS-ECS-orange.svg)

# Financial Market Intelligence Agent 🚀

An end-to-end agentic AI system for financial market analysis using Google Gemini, LangChain, RAG, and Google's Agentic Development Kit.

## 🏗️ Architecture

Multi-agent system with:
- Research Agent: Data gathering
- Analysis Agent: Technical & fundamental analysis
- Sentiment Agent: News sentiment analysis
- Prediction Agent: ML-based forecasting
- Report Agent: Insight synthesis

┌─────────────────────────────────────────────────────────┐
│                     User Interface                      │
│                  (Streamlit Frontend)                   │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                    FastAPI Backend                      │
│  ┌──────────┐  ┌────────────┐  ┌────────────────────┐   │
│  │   API    │  │  Agent     │  │   RAG Pipeline     │   │
│  │  Routes  │──│Orchestrator│──│  (ChromaDB)        │   │
│  └──────────┘  └────┬───────┘  └────────────────────┘   │
│                     │                                   │
│     ┌───────────────┴───────────────┐                   │
│     ▼               ▼               ▼                   │
│  Research      Analysis        Sentiment                │
│   Agent          Agent           Agent                  │
│     └───────────────┬───────────────┘                   │
│                     ▼                                   │
│              Prediction Agent                           │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│              External Data Sources                      │
│  • Yahoo Finance  • Alpha Vantage  • NewsAPI            │
│  • Google Gemini (LLM)  • ChromaDB (Vector Store)       │
└─────────────────────────────────────────────────────────┘


## 🚀 Features

- **Multi-Agent System** using Google ADK
- **RAG Pipeline** with ChromaDB
- **Real-time Financial Data** from Yahoo Finance
- **Technical Analysis** (RSI, MACD, Bollinger Bands)
- **Sentiment Analysis** from news sources
- **Price Predictions** using ML
- **Interactive Chat Interface**
- **Visual Dashboard** with Plotly

## 🛠️ Quick Start

### Prerequisites
- Python 3.10+
- Docker & Docker Compose
- API Keys:
  - Google Gemini API Key
  - Alpha Vantage API Key

### Setup

1. **Clone and setup:**
```bash
git clone <your-repo-url>
cd financial-intelligence-agent
cp .env.example .env
# Edit .env with your API keys
```

2. **Get API Keys:**
- Google Gemini: https://makersuite.google.com/app/apikey
- Alpha Vantage: https://www.alphavantage.co/support/#api-key

3. **Run with Docker:**
```bash
docker-compose up --build
```

4. **Access:**
- Frontend: http://localhost:8501
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

## 📊 Usage Examples

### Chat Interface


"Analyze Tesla's performance and compare with Ford"
"What's the sentiment around NVIDIA's latest earnings?"
"Predict Apple stock trend for next week"


### API Endpoints
```bash
# Stock Analysis
curl -X POST http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL", "analysis_type": "full"}'

# Price Prediction
curl -X POST http://localhost:8000/api/predict/TSLA?days=7

# Stock Comparison
curl http://localhost:8000/api/compare?symbols=AAPL,MSFT,GOOGL
```

## 🌐 AWS Deployment
```bash
cd deployment/aws
chmod +x deploy.sh
./deploy.sh deploy
```

## 🧪 Testing
```bash
pytest tests/ -v
pytest --cov=app tests/
```

## 📈 Tech Stack

- **Backend:** FastAPI, Python 3.10
- **Frontend:** Streamlit
- **LLM:** Google Gemini
- **Agents:** Google ADK
- **Vector DB:** ChromaDB
- **Data:** Yahoo Finance, Alpha Vantage
- **ML:** scikit-learn, pandas, numpy
- **Deployment:** Docker, AWS ECS

## 📄 License

MIT License

## 🙏 Acknowledgments

- Google Gemini & Agentic Development Kit
- LangChain community
- Alpha Vantage for financial data
