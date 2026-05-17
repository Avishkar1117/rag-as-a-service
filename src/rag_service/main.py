from contextlib import asynccontextmanager

from fastapi import FastAPI

from rag_service.api.routes import router
from rag_service.config import settings
from rag_service.llm.openai_client import setup_llamaindex_settings
from rag_service.observability.logging_setup import setup_logging
from rag_service.observability.request_log import RequestLogMiddleware
from rag_service.observability.sentry_setup import setup_sentry

setup_logging(level=settings.log_level)
setup_sentry(dsn=settings.sentry_dsn)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_llamaindex_settings()
    yield


app = FastAPI(
    title="RAG-as-a-Service",
    description="Production RAG API with evaluation pipeline",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(RequestLogMiddleware)
app.include_router(router)
