from pydantic import BaseModel
from typing import List

class Source(BaseModel):
    text: str
    source: str
    score: float
    rerank_score: float

class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[Source]
    latency: float