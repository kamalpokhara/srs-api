from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import pickle

router = APIRouter()

with open("models/popular_products.pkl", "rb") as f:
    popular_data = pickle.load(f)

# Normalize to a sorted list regardless of how you stored it
# Assumes popular_data is either:
#   - a dict  {product_id: score}
#   - a list  [{product_id, score, ...}]
if isinstance(popular_data, dict):
    sorted_popular = sorted(popular_data.items(), key=lambda x: x[1], reverse=True)
    popular_list = [{"product_id": k, "score": round(v, 2)} for k, v in sorted_popular]
else:
    popular_list = sorted(popular_data, key=lambda x: x["score"], reverse=True)

class PopularResponse(BaseModel):
    products: List[dict]
    total: int

@router.get("/", response_model=PopularResponse)
def get_popular(top_n: int = 20):
    try:
        results = popular_list[:top_n]
        return PopularResponse(products=results, total=len(results))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))