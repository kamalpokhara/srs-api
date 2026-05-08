from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import os
from huggingface_hub import hf_hub_download

router = APIRouter()

HF_REPO_ID = os.environ.get("HF_REPO_ID", "kamalpokhara/srs-models")
HF_TOKEN   = os.environ.get("HF_TOKEN", None)

print(f"Loading Popular products from HF Hub: {HF_REPO_ID}")

popular_path = hf_hub_download(
    repo_id   = HF_REPO_ID,
    filename  = "popular_products.csv",
    token     = HF_TOKEN,
    repo_type = "model"
)

# Load and normalize — handle whatever column structure exists
df = pd.read_csv(popular_path)

if "product_id" in df.columns:
    popular_ids = df["product_id"].astype(int).tolist()
elif len(df.columns) == 1:
    # Single column, no header or different name
    popular_ids = df.iloc[:, 0].astype(int).tolist()
else:
    # Fallback — take first column
    popular_ids = df.iloc[:, 0].astype(int).tolist()

print(f"Popular products loaded — {len(popular_ids)} items")
print(f"Top 5: {popular_ids[:5]}")

class PopularResponse(BaseModel):
    product_ids: List[int]
    total:       int
    source:      str = "popular"

@router.get("/", response_model=PopularResponse)
def get_popular(top_n: int = 200):
    try:
        results = popular_ids[:top_n]
        return PopularResponse(
            product_ids=results,
            total=len(results),
            source="popular",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))