from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import pickle
import numpy as np

router = APIRouter()

# Load all three LightFM artifacts once
with open("../models/lightfm_best.pkl", "rb") as f:
    lfm_model = pickle.load(f)

with open("../models/lightfm_dataset.pkl", "rb") as f:
    dataset = pickle.load(f)

# Build mappings from dataset
user_id_map, _, item_id_map, _ = dataset.mapping()
item_id_reverse = {v: k for k, v in item_id_map.items()}  # internal_idx → product_id
n_items = len(item_id_map)

class RecommendRequest(BaseModel):
    user_id: str
    top_n: int = 10
    exclude_seen: Optional[List[str]] = []   # product_ids already interacted with

class RecommendResponse(BaseModel):
    user_id: str
    recommendations: List[str]               # ordered list of product_ids
    is_cold_start: bool

@router.post("/", response_model=RecommendResponse)
def recommend(request: RecommendRequest):
    try:
        is_cold_start = request.user_id not in user_id_map

        if is_cold_start:
            # Fall back to popular products for unknown users
            # Popular router handles this — return empty and let Django decide
            return RecommendResponse(
                user_id=request.user_id,
                recommendations=[],
                is_cold_start=True
            )

        internal_uid = user_id_map[request.user_id]
        scores = lfm_model.predict(internal_uid, np.arange(n_items))

        # Exclude already seen items
        seen_internal = {
            item_id_map[pid]
            for pid in request.exclude_seen
            if pid in item_id_map
        }
        for idx in seen_internal:
            scores[idx] = -np.inf

        top_indices = np.argsort(-scores)[:request.top_n]
        product_ids = [item_id_reverse[i] for i in top_indices]

        return RecommendResponse(
            user_id=request.user_id,
            recommendations=product_ids,
            is_cold_start=False
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))