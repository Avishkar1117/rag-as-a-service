from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import RedirectResponse

from rag_service.api.schemas import (
    HealthResponse,
    IngestResponse,
    MetricsResponse,
    QueryRequest,
    QueryResponse,
)
from rag_service.cache.redis_cache import stats as cache_stats
from rag_service.core.ingestion import ingest_document, pdf_content_id
from rag_service.core.pipeline import query_pipeline

router = APIRouter()


@router.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get("/metrics", response_model=MetricsResponse)
async def metrics() -> MetricsResponse:
    return MetricsResponse(
        cache_hits=cache_stats.hits,
        cache_misses=cache_stats.misses,
        cache_hit_rate=cache_stats.hit_rate(),
    )


@router.post("/ingest", response_model=IngestResponse)
def ingest(file: UploadFile = File(...)) -> IngestResponse:
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")
    pdf_bytes = file.file.read()
    document_id = pdf_content_id(pdf_bytes)
    n_chunks = ingest_document(pdf_bytes, document_id)
    return IngestResponse(document_id=document_id, n_chunks=n_chunks)


@router.post("/query", response_model=QueryResponse)
def query(body: QueryRequest) -> QueryResponse:
    result = query_pipeline(
        question=body.question,
        document_id=body.document_id,
        top_k=body.top_k,
    )
    return QueryResponse(**result)
