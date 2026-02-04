# HealthTrust - ML Fraud Detection with Streamlit

Standalone version of HealthTrust insurance fraud detection system with Streamlit frontend.

## 📁 Project Structure

```
model_1_streamlit/
├── ml-service/              # FastAPI ML Service
│   ├── main.py             # Simplified API (no database)
│   ├── model_loader.py     # ML model loader
│   ├── prescription_verifier.py  # Image verification
│   ├── requirements.txt
│   ├── .env               # OpenAI API key
│   └── models/            # Trained ML models
├── streamlit-app/         # Streamlit Frontend
│   ├── app.py            # Main Streamlit application
│   └── requirements.txt
└── start.sh              # Startup script
```

## 🚀 Quick Start

### 1. Setup ML Service
```bash
cd ml-service
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Setup Streamlit App
```bash
cd streamlit-app
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Run Everything
```bash
# From project root
./start.sh
```

Or run separately in two terminals:

**Terminal 1 - ML Service:**
```bash
cd ml-service
source venv/bin/activate
python main.py
```

**Terminal 2 - Streamlit:**
```bash
cd streamlit-app
source venv/bin/activate
streamlit run app.py
```

## 🌐 Access

- **Streamlit Frontend:** http://localhost:8501
- **FastAPI ML Service:** http://localhost:8000
- **API Docs:** http://localhost:8000/docs

## ✨ Features

- **Submit Claims:** ML-powered fraud detection
- **Verify Images:** GPT-4 Vision prescription/receipt verification
- **Combined Scoring:** ML + Image verification scores
- **Model Info:** View ML model details and feature importance

## 🔑 Configuration

Make sure to set your OpenAI API key in `ml-service/.env`:
```bash
OPENAI_API_KEY=your_key_here
```

## 📝 Notes

- No database required (simplified version)
- No blockchain integration
- Standalone ML model + FastAPI + Streamlit
- Perfect for demos and testing
