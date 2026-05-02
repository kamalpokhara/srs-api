from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import churn, recommendations, popular

app = FastAPI(title="Smart Retention ML API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # lock to your Django URL after deploy
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(churn.router,           prefix="/predict",  tags=["Churn"])
app.include_router(recommendations.router, prefix="/recommend", tags=["RecSys"])
app.include_router(popular.router,         prefix="/popular",  tags=["Popular"])

@app.get("/health")
def health():
    return {"status": "ok"}