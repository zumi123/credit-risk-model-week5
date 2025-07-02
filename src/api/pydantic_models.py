from pydantic import BaseModel
from typing import List, Any

class PredictionRequest(BaseModel):
    # minimal schema: match the features your pre‑processing pipeline needs
    TransactionStartTime: str
    Amount: float
    Value: float
    ProviderId: str
    ProductId: str

class PredictionResponse(BaseModel):
    risk_probability: float
