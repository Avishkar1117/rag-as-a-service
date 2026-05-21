from unittest.mock import MagicMock, patch

from rag_service.core.generation import GenerationResult
from rag_service.core.pipeline import query_pipeline


def _gen_result(
    answer: str = "The answer",
    citations: list[str] | None = None,
    model: str = "gpt-4o",
    prompt_tokens: int = 1000,
    output_tokens: int = 200,
) -> GenerationResult:
    return GenerationResult(
        answer=answer,
        citations=["doc1"] if citations is None else citations,
        model=model,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
    )


def test_pipeline_returns_expected_shape():
    mock_nodes = [MagicMock()]

    with patch("rag_service.core.pipeline.retrieve", return_value=mock_nodes), \
         patch("rag_service.core.pipeline.generate", return_value=_gen_result()):

        result = query_pipeline(question="What?", document_id="doc1", top_k=3)

    assert result["answer"] == "The answer"
    assert result["citations"] == ["doc1"]
    assert isinstance(result["latency_ms"], int)
    assert result["latency_ms"] >= 0
    assert isinstance(result["cost_usd"], float)


def test_pipeline_computes_cost_from_token_usage():
    # gpt-4o: 1000 input + 200 output -> (1000*2.5 + 200*10) / 1e6.
    expected = round((1000 * 2.5 + 200 * 10.0) / 1_000_000, 6)

    with patch("rag_service.core.pipeline.retrieve", return_value=[]), \
         patch("rag_service.core.pipeline.generate", return_value=_gen_result()):

        result = query_pipeline(question="Q?", document_id="docX", top_k=5)

    assert result["cost_usd"] == expected


def test_pipeline_passes_args_correctly():
    with patch("rag_service.core.pipeline.retrieve", return_value=[]) as mock_retrieve, \
         patch("rag_service.core.pipeline.generate",
               return_value=_gen_result(answer="", citations=[])):

        query_pipeline(question="Q?", document_id="docX", top_k=5)

    mock_retrieve.assert_called_once_with("docX", "Q?", 5)
