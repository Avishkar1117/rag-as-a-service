import chromadb
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import NodeWithScore
from llama_index.vector_stores.chroma import ChromaVectorStore

from rag_service.config import settings


def retrieve(document_id: str, query: str, top_k: int) -> list[NodeWithScore]:
    """Return top-k nodes from Chroma for the given document and query."""
    chroma_client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    collection = chroma_client.get_collection(f"doc_{document_id}")
    vector_store = ChromaVectorStore(chroma_collection=collection)
    index = VectorStoreIndex.from_vector_store(vector_store)
    return index.as_retriever(similarity_top_k=top_k).retrieve(query)
