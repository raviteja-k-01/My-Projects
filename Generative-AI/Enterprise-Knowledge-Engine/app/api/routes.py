from fastapi import APIRouter, Query
from app.services.query_service import QueryService
from app.schemas import QueryResponse

router = APIRouter()
qs = QueryService()

@router.get("/query", response_model=QueryResponse)
def query_rag(q: str = Query(..., description="User query")):
    result = qs.query(q)
    return result