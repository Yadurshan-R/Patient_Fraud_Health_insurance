#!/bin/bash

echo "======================================"
echo "🚀 HealthTrust ML + Streamlit Startup"
echo "======================================"
echo ""

# Start ML Service in background
echo "📊 Starting ML Service (FastAPI)..."
cd ml-service
source venv/bin/activate
python3 main.py &
ML_PID=$!
cd ..

echo "✓ ML Service started with PID: $ML_PID"
echo "🌐 ML API available at: http://localhost:8000"
echo ""

# Wait for ML service to be ready
echo "⏳ Waiting for ML service to initialize..."
sleep 5

# Start Streamlit
echo "🎨 Starting Streamlit Frontend..."
echo "🌐 Frontend will be available at: http://localhost:8501"
echo ""
cd streamlit-app
source venv/bin/activate
streamlit run app.py

# Cleanup on exit
trap "echo ''; echo '🛑 Shutting down services...'; kill $ML_PID 2>/dev/null; echo '✓ Services stopped'; exit 0" EXIT INT TERM
