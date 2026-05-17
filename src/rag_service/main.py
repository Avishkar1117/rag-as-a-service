from contextlib import asynccontextmanager

from fastapi import FastAPI

from rag_service.api.routes import router
from rag_service.llm.openai_client import setup_llamaindex_settings


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

app.include_router(router)
