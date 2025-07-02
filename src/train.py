"""Task 5 – Model Training, Hyper‑Parameter Tuning, and MLflow Tracking

Run this script from the project root:

```bash
python -m src.train
```

It will:
1. Load raw transactions → add `is_high_risk` via `proxy_target.add_is_high_risk`.
2. Split data into train/test.
3. Build a preprocessing pipeline (`data_processing.build_preprocessing_pipeline`).
4. Evaluate two candidate estimators (Logistic Regression and Gradient Boosting)
   with cross‑validated hyper‑parameter search.
5. Log all experiments to MLflow; the best run is registered in the
   credit‑risk‑classifier model registry.

To start the MLflow UI afterward:
```bash
mlflow ui --backend-store-uri mlruns
```
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier

from src.data_processing import build_preprocessing_pipeline
from src.proxy_target import add_is_high_risk

warnings.filterwarnings("ignore", category=FutureWarning)

# --------------------------- CONFIG ------------------------------------ #

RAW_DATA = Path("data/raw/data.csv")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
EXPERIMENT_NAME = "credit‑risk‑baseline"
MODEL_NAME = "credit‑risk‑classifier"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Hyper‑parameter grids
LOGREG_GRID = {
    "clf__C": [0.1, 1.0, 10.0],
    "clf__penalty": ["l2"],
    "clf__solver": ["liblinear"],
}
GBM_GRID = {
    "clf__learning_rate": [0.05, 0.1],
    "clf__n_estimators": [100, 300],
    "clf__max_depth": [3, 5],
}

# ------------------------- MAIN PIPELINE -------------------------------- #

def prepare_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load CSV, create proxy label, and split X / y."""
    df = pd.read_csv(RAW_DATA)
    df = add_is_high_risk(df)
    X = df.drop(columns=["is_high_risk"])
    y = df["is_high_risk"]
    return X, y

def make_model_pipeline(base_estimator):
    """Create full pipeline: preprocessing + estimator."""
    preprocessor = build_preprocessing_pipeline(return_dataframe=False)
    return Pipeline([
        ("prep", preprocessor),
        ("clf", base_estimator),
    ])


def run_grid_search(pipeline, param_grid, X_train, y_train):
    """Run cross‑validated GridSearchCV and return the fitted object."""
    gs = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=3,
        scoring="f1",
        n_jobs=-1,
        verbose=0,
    )
    gs.fit(X_train, y_train)
    return gs


def evaluate(model, X_test, y_test):
    """Return key evaluation metrics as a dict."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }


def log_to_mlflow(gs, metrics, model_alias: str):
    """Log params, metrics, and register the best model."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=model_alias) as run:
        # Log best params (flattened)
        for p, v in gs.best_params_.items():
            mlflow.log_param(p, v)
        # Log metrics
        mlflow.log_metrics(metrics)
        # Log model artifact
        mlflow.sklearn.log_model(gs.best_estimator_, artifact_path="model")
        # Register
        model_uri = f"runs:/{run.info.run_id}/model"
        mlflow.register_model(model_uri, MODEL_NAME)

        print(f"Registered model under '{MODEL_NAME}' with run_id = {run.info.run_id}")


def main():
    # 1️. Data prep
    X, y = prepare_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # 2️. Logistic Regression pipeline
    logreg_pipe = make_model_pipeline(
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
    )
    logreg_gs = run_grid_search(logreg_pipe, LOGREG_GRID, X_train, y_train)
    logreg_metrics = evaluate(logreg_gs.best_estimator_, X_test, y_test)
    print("LogReg best F1:", logreg_metrics["f1"])

    # 3️. Gradient Boosting pipeline
    gbm_pipe = make_model_pipeline(GradientBoostingClassifier(random_state=RANDOM_STATE))
    gbm_gs = run_grid_search(gbm_pipe, GBM_GRID, X_train, y_train)
    gbm_metrics = evaluate(gbm_gs.best_estimator_, X_test, y_test)
    print("GBM best F1:", gbm_metrics["f1"])

    # 4️. Compare & select
    best_model, best_metrics, alias = (
        (gbm_gs, gbm_metrics, "GBM") if gbm_metrics["f1"] > logreg_metrics["f1"]
        else (logreg_gs, logreg_metrics, "LogReg")
    )

    # 5️. Log to MLflow
    log_to_mlflow(best_model, best_metrics, alias)

    print("\n📋 Final Metrics:")
    for k, v in best_metrics.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
