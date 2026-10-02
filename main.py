from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.services.twinlife_service import TwinLifeService

app = FastAPI(
    title="TwinLife AI",
    description="Unified AI system for health, finance, insurance, RAG, simulation, recommendations, and monitoring.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

service = TwinLifeService()

class QueryRequest(BaseModel):
    query: str
    profile: dict | None = None

@app.get("/")
def root():
    return {"message": "TwinLife AI API is running"}

@app.post("/query")
def query(request: QueryRequest):
    if request.profile:
        user_service = TwinLifeService(profile=request.profile)
        return user_service.process_query(request.query)

    return service.process_query(request.query)

@app.get("/affordability/{cost}")
def affordability(cost: float):
    return service.affordability_for(cost)

@app.get("/coverage/{cost}")
def coverage(cost: float):
    return service.coverage_for(cost)

@app.get("/recommendations")
def recommendations():
    return service.recommendations()

@app.get("/monitor")
def monitor():
    return service.run_monitor_check()