from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    status: str


class IngestResponse(BaseModel):
    document_id: str
    n_chunks: int


class QueryRequest(BaseModel):
    # Reject unknown fields so typos like `k` / `n_chunks` fail loudly instead of silently
    # defaulting `top_k=3`.
    model_config = ConfigDict(extra="forbid")

    question: str = Field(..., min_length=1)
    document_id: str = Field(..., min_length=1)
    top_k: int = Field(default=3, ge=1, le=20)


class QueryResponse(BaseModel):
    answer: str
    citations: list[str]
    latency_ms: int


class MetricsResponse(BaseModel):
    cache_hits: int
    cache_misses: int
    cache_hit_rate: float
