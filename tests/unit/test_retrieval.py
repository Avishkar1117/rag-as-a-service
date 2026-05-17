from unittest.mock import MagicMock, patch

from llama_index.core.schema import NodeWithScore, TextNode

from rag_service.core.retrieval import retrieve


def _make_nodes(n: int = 2) -> list[NodeWithScore]:
    return [NodeWithScore(node=TextNode(text=f"chunk{i}"), score=0.9) for i in range(n)]


def test_retrieve_uses_correct_collection():
    mock_nodes = _make_nodes(2)

    with patch("rag_service.core.retrieval.chromadb.PersistentClient") as mock_chroma, \
         patch("rag_service.core.retrieval.ChromaVectorStore"), \
         patch("rag_service.core.retrieval.VectorStoreIndex") as mock_idx_cls:

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = mock_nodes
        mock_idx_cls.from_vector_store.return_value.as_retriever.return_value = mock_retriever

        result = retrieve("abc123", "test query", top_k=3)

    mock_chroma.return_value.get_collection.assert_called_once_with("doc_abc123")
    assert result == mock_nodes


def test_retrieve_passes_top_k():
    with patch("rag_service.core.retrieval.chromadb.PersistentClient"), \
         patch("rag_service.core.retrieval.ChromaVectorStore"), \
         patch("rag_service.core.retrieval.VectorStoreIndex") as mock_idx_cls:

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = []
        mock_index = mock_idx_cls.from_vector_store.return_value
        mock_index.as_retriever.return_value = mock_retriever

        retrieve("doc1", "query", top_k=7)

    mock_index.as_retriever.assert_called_once_with(similarity_top_k=7)
