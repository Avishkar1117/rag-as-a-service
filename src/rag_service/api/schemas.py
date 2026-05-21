from pydantic import BaseModel, ConfigDict, Field

from rag_service.config import settings


class HealthResponse(BaseModel):
    status: str


class IngestResponse(BaseModel):
    document_id: str
    n_chunks: int


class QueryRequest(BaseModel):
    # Reject unknown fields so typos like `k` / `n_chunks` fail loudly instead of silently
    # falling back to the default `top_k`.
    model_config = ConfigDict(extra="forbid")

    question: str = Field(..., min_length=1)
    document_id: str = Field(..., min_length=1)
    # Default tracks settings.top_k so the .env TOP_K knob is the single source of truth.
    top_k: int = Field(default=settings.top_k, ge=1, le=20)


class QueryResponse(BaseModel):
    answer: str
    citations: list[str]
    latency_ms: int
    cost_usd: float


class MetricsResponse(BaseModel):
    cache_hits: int
    cache_misses: int
    cache_hit_rate: float
    n_queries: int
    total_cost_usd_today: float
    mean_cost_usd_per_query: float
    p50_latency_ms: float
    p95_latency_ms: float
