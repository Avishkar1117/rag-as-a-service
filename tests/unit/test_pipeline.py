from unittest.mock import MagicMock, patch

from rag_service.core.pipeline import query_pipeline


def test_pipeline_returns_expected_shape():
    mock_nodes = [MagicMock()]

    with patch("rag_service.core.pipeline.retrieve", return_value=mock_nodes), \
         patch("rag_service.core.pipeline.generate", return_value=("The answer", ["doc1"])):

        result = query_pipeline(question="What?", document_id="doc1", top_k=3)

    assert result["answer"] == "The answer"
    assert result["citations"] == ["doc1"]
    assert isinstance(result["latency_ms"], int)
    assert result["latency_ms"] >= 0


def test_pipeline_passes_args_correctly():
    with patch("rag_service.core.pipeline.retrieve", return_value=[]) as mock_retrieve, \
         patch("rag_service.core.pipeline.generate", return_value=("", [])):

        query_pipeline(question="Q?", document_id="docX", top_k=5)

    mock_retrieve.assert_called_once_with("docX", "Q?", 5)
