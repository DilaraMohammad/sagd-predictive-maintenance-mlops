from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from typing import Dict, List
from pathlib import Path
from collections import deque
import datetime


THRESHOLD = 0.5
BASE_DIR = Path(__file__).resolve().parent.parent

app = FastAPI(
    title="SAGD Sensor Failure Prediction API",
    description="Production-ready MLOps API for 20-minute ahead well failure prediction using XGBoost.",
    version="1.0.0"
)

prediction_history = deque(maxlen=50)

try:
    model = joblib.load(BASE_DIR / "models" / "sagd_sensor_model.pkl")
    scaler = joblib.load(BASE_DIR / "models" / "sagd_scaler.pkl")
    feature_cols = joblib.load(BASE_DIR / "models" / "feature_cols.pkl")
except Exception as e:
    raise RuntimeError(f"Failed to load model artifacts: {e}")

class WellSensorInput(BaseModel):
    features: Dict[str, float] = Field(..., description="Dictionary containing all required feature names and their numerical values")

@app.post("/predict")
def predict_failure(input_data: WellSensorInput):
    try:
        df_input = pd.DataFrame([input_data.features])
        
        if df_input.isnull().values.any() or not np.isfinite(df_input.values).all():
            raise HTTPException(
                status_code=400,
                detail="Input contains NaN or Infinite values"
            )
            
        missing_features = [col for col in feature_cols if col not in df_input.columns]
        if missing_features:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required features: {missing_features}"
            )
            
        X_scaled = scaler.transform(df_input[feature_cols])
        probability = float(model.predict_proba(X_scaled)[:, 1][0])
        is_failure = int(probability >= THRESHOLD)
        
        result = {
            "failure_probability_20min": probability,
            "threshold_used": THRESHOLD,
            "is_failure_predicted": is_failure,
            "status": "warning" if is_failure == 1 else "normal"
        }
        
        prediction_history.append({
            "timestamp": datetime.datetime.now().isoformat(),
            **input_data.features,
            **result
        })
        
        return result
        
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history", response_model=List[dict])
def get_prediction_history():
    return list(prediction_history)

@app.get("/health")
def health_check():
    is_ready = (model is not None) and (scaler is not None)
    if not is_ready:
        raise HTTPException(
            status_code=503,
            detail="Service Unhealthy: Model or Scaler not loaded."
        )
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "features_count": len(feature_cols),
    }