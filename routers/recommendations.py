# routers/recommendations.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import joblib
import numpy as np
import os
from huggingface_hub import hf_hub_download

router = APIRouter()
# ── LOAD FROM HF HUB AT STARTUP ───────────────────────────────────────────────
HF_REPO_ID = os.environ.get("HF_REPO_ID", "kamalpokhara/srs-models")
HF_TOKEN   = os.environ.get("HF_TOKEN", None)

print(f"Loading RecSys models from HF Hub: {HF_REPO_ID}")

lfm_model_path = hf_hub_download(
    repo_id   = HF_REPO_ID,
    filename  = "lightfm_best.pkl",
    token     = HF_TOKEN,
    repo_type = "model"
)
dataset_path = hf_hub_download(
    repo_id   = HF_REPO_ID,
    filename  = "lightfm_dataset.pkl",
    token     = HF_TOKEN,
    repo_type = "model"
)
#  LOAD 
lfm_model = joblib.load(lfm_model_path)
dataset   = joblib.load(dataset_path)

# Build mappings
user_id_map, _, item_id_map, _ = dataset.mapping()
item_id_reverse = {v: k for k, v in item_id_map.items()}
n_items         = len(item_id_map)

print(f"RecSys loaded:")
print(f"  Users:  {len(user_id_map)}")
print(f"  Items:  {n_items}")
print(f"  Sample user_ids: {list(user_id_map.keys())[:3]}")
print(f"  Sample item_ids: {list(item_id_map.keys())[:3]}")

# ── SCHEMAS ───────────────────────────────────────────────────────────────────
class RecommendRequest(BaseModel):
    user_id:      str            # "U000324" format
    top_n:        int = 10
    exclude_seen: Optional[List[int]] = []  # product ids already seen

class RecommendResponse(BaseModel):
    user_id:         str
    recommendations: List[int]   # product ids as integers
    is_cold_start:   bool
    source:          str

# ── ENDPOINT ──────────────────────────────────────────────────────────────────
@router.post("/", response_model=RecommendResponse)
def recommend(request: RecommendRequest):
    try:
        # Check if user exists in training data
        is_cold_start = request.user_id not in user_id_map

        if is_cold_start:
            # Unknown user — caller should fall back to popular
            return RecommendResponse(
                user_id=request.user_id,
                recommendations=[],
                is_cold_start=True,
                source="cold_start",
            )

        internal_uid = user_id_map[request.user_id]
        all_items    = np.arange(n_items)

        # Predict — no user/item features needed
        # embeddings are already baked into the model
        scores = lfm_model.predict(
            user_ids=int(internal_uid),
            item_ids=all_items,
            num_threads=1,
        )

        # Exclude already seen items
        for pid in request.exclude_seen:
            if pid in item_id_map:
                scores[item_id_map[pid]] = -np.inf

        # Get top N
        top_indices = np.argsort(-scores)[:request.top_n]
        product_ids = [
            int(item_id_reverse[i])
            for i in top_indices
            if i in item_id_reverse
        ]

        return RecommendResponse(
            user_id=request.user_id,
            recommendations=product_ids,
            is_cold_start=False,
            source="personalized",
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── HELPER — for Django to call directly ──────────────────────────────────────
@router.get("/user/{user_id}", response_model=RecommendResponse)
def recommend_get(user_id: str, top_n: int = 10):
    """GET endpoint for quick testing"""
    return recommend(RecommendRequest(
        user_id=user_id,
        top_n=top_n,
        exclude_seen=[],
    ))