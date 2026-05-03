from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
import joblib

router = APIRouter()

# Load once at startup
popular_data = joblib.load("models/popular_products.pkl")

# Normalize — handle both list and dict formats
if isinstance(popular_data, dict):
    popular_list = sorted(
        popular_data.items(),
        key=lambda x: x[1],
        reverse=True
    )
    popular_ids = [int(k) for k, _ in popular_list]

elif isinstance(popular_data, list):
    # Could be list of ids or list of dicts
    if len(popular_data) > 0 and isinstance(popular_data[0], dict):
        popular_ids = [int(p["product_id"]) for p in popular_data]
    else:
        popular_ids = [int(p) for p in popular_data]
else:
    popular_ids = []

print(f"Popular products loaded — {len(popular_ids)} items")

class PopularResponse(BaseModel):
    product_ids: List[int]
    total:       int
    source:      str = "popular"

@router.get("/", response_model=PopularResponse)
def get_popular(top_n: int = 20):
    try:
        results = popular_ids[:top_n]
        return PopularResponse(
            product_ids=results,
            total=len(results),
            source="popular",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))