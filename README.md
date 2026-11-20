# SAGD Predictive Maintenance & MLOps Pipeline

Production-grade MLOps pipeline for Steam-Assisted Gravity Drainage (SAGD) well operations, designed to forecast asset failures 20 minutes in advance. The system handles real-time data streaming, feature validation, XGBoost model inference via FastAPI, and live monitoring through Streamlit.

## Tech Stack
* **Language:** Python 3.10
* **API Framework:** FastAPI, Uvicorn
* **Dashboard:** Streamlit
* **Machine Learning:** Scikit-Learn, Pandas, NumPy, XGBoost, Joblib
* **Infrastructure:** Docker, Minikube (Kubernetes), GitHub Actions CI/CD

## Project Structure
* api/main.py: FastAPI inference service with rolling buffer (deque of 50 records) and schema validation.
* simulator/streamer.py: Real-time telemetry data streaming simulator.
* dashboard.py: Streamlit monitoring interface for real-time probabilities and alerts.
* models/: Contains trained XGBoost model, feature scaler, and column definitions.
* deployment.yaml: Kubernetes deployment and service configurations.

## Setup and Execution

```bash
git clone <repository-url>
cd sagd
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

Execution Steps
1. Start API Service
2. Run Data Streamer
3. Run Dashboard
API Endpoints:
  POST /predict: Evaluates incoming sensor payload and returns failure probability and status.
  GET /history: Returns rolling inference buffer (last 50 records).
  GET /health: Service health and model artifact verification.