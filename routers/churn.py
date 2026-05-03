from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import xgboost as xgb
import numpy as np

router = APIRouter()

# Load once at startup
model = xgb.Booster()
model.load_model("models/final_xgb_churn_tuned.json")

# Must match exact order used during training
FEATURE_ORDER = [
    "activity_span_days",
    "total_interactions",
    "unique_products",
    "view_count",
    "total_weight",
    "view_to_purchase_ratio",
    "cart_count",
    "cart_to_purchase_ratio",
    "cart_abandonment_rate",
    "wishlist_count",
    "interactions_per_day",
    "wishlist_to_purchase_ratio",
]

CHURN_THRESHOLD = 0.35   # from your tuning analysis

class ChurnRequest(BaseModel):
    user_id: int
    features: dict

class ChurnResponse(BaseModel):
    user_id:           int
    churn_probability: float
    will_churn:        bool
    risk_level:        str

@router.post("/churn", response_model=ChurnResponse)
def predict_churn(request: ChurnRequest):
    try:
        # Build feature vector in correct order
        vector = []
        for f in FEATURE_ORDER:
            if f not in request.features:
                raise HTTPException(
                    status_code=422,
                    detail=f"Missing feature: {f}"
                )
            vector.append(float(request.features[f]))

        dmatrix = xgb.DMatrix(
            np.array(vector).reshape(1, -1),
            feature_names=FEATURE_ORDER
        )
        prob = float(model.predict(dmatrix)[0])

        # Risk level for frontend
        if prob >= 0.7:
            risk = "high"
        elif prob >= CHURN_THRESHOLD:
            risk = "medium"
        else:
            risk = "low"

        return ChurnResponse(
            user_id=request.user_id,
            churn_probability=round(prob, 4),
            will_churn=prob >= CHURN_THRESHOLD,
            risk_level=risk,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))