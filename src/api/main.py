import os
import mlflow
import pandas as pd
from fastapi import FastAPI
from src.data_processing import build_preprocessing_pipeline
from src.api.pydantic_models import PredictionRequest, PredictionResponse

APP_NAME = "Credit‑Risk Scoring API"
RUN_ID = os.getenv("BEST_MODEL_RUN_ID")  # set during docker build or runtime

app = FastAPI(title=APP_NAME)

# ── load model and pipeline on start‑up ──────────────────────────────────────
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "./mlruns"))
model = mlflow.sklearn.load_model(f"runs:/{RUN_ID}/model")
preprocessor = build_preprocessing_pipeline(return_dataframe=False)

@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest):
    df = pd.DataFrame([payload.model_dump()])
    X = preprocessor.transform(df)
    proba = model.predict_proba(X)[0, 1]  # probability of high risk
    return PredictionResponse(risk_probability=float(proba))
