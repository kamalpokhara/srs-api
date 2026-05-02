from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
import xgboost as xgb
import numpy as np

router = APIRouter()

# Load once at import time
model = xgb.Booster()
model.load_model("models/final_xgb_churn_tuned.json")

FEATURE_ORDER = [
    "activity_span_days", "total_interactions", "unique_products",
    "view_count", "total_weight", "view_to_purchase_ratio",
    "cart_count", "cart_to_purchase_ratio", "cart_abandonment_rate",
    "wishlist_count", "interactions_per_day", "wishlist_to_purchase_ratio"
]

class ChurnRequest(BaseModel):
    user_id: str
    features: dict  # Django sends {feature_name: value}

class ChurnResponse(BaseModel):
    user_id: str
    churn_probability: float
    will_churn: bool

@router.post("/churn", response_model=ChurnResponse)
def predict_churn(request: ChurnRequest):
    try:
        # Enforce correct feature order from meta.json
        vector = [request.features[f] for f in FEATURE_ORDER]
        dmatrix = xgb.DMatrix(np.array(vector).reshape(1, -1))
        prob = float(model.predict(dmatrix)[0])
        return ChurnResponse(
            user_id=request.user_id,
            churn_probability=round(prob, 4),
            will_churn=prob >= 0.5
        )
    except KeyError as e:
        raise HTTPException(status_code=422, detail=f"Missing feature: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))