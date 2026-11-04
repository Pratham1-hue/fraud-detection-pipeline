import random

from pydantic import BaseModel, Field
from typing import List

class TransactionPayload(BaseModel):
    model_config = {"json_schema_extra": {
        "examples": [{
            "transaction_id": "txn_1234567890",
            "user_id": "user_12345",
            "amount": 110.50,
            "time_delta": 3600.0,
            "features": [round(random.gauss(0, 1), 3) for _ in range(28)],
        }]
    }}
    transaction_id: str = Field(..., description="Unique transaction identifier")
    user_id: str = Field(..., description="Customer making the transaction")
    amount: float = Field(..., gt=0, description="Transaction amount")
    time_delta: float = Field(..., ge=0, description="Time since midnight in seconds")
    features: List[float] = Field(..., min_length=28, max_length=28, description="V1-V28 PCA features")

class FraudPredictionResponse(BaseModel):
    transaction_id: str
    risk_score: float = Field(..., description="Calibrated risk score between 0 and 1.0")
    is_anomaly: bool = Field(..., description="Unsupervised anomaly flag from Isolation Forest")
    action: str = Field(..., description="Action recommendation: ALLOW, REVIEW, BLOCK")
    latency_ms: float