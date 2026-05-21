from unittest.mock import MagicMock, patch

from llama_index.core.schema import NodeWithScore, TextNode

from rag_service.core.generation import GenerationResult, generate


def _nodes(texts: list[str], doc_id: str = "doc123") -> list[NodeWithScore]:
    return [NodeWithScore(node=TextNode(text=t, metadata={"document_id": doc_id}), score=0.9)
            for t in texts]


def _mock_client(
    response_text: str = "answer",
    capture: list[str] | None = None,
    prompt_tokens: int = 120,
    output_tokens: int = 8,
) -> MagicMock:
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = response_text
    mock_resp.usage_metadata.prompt_token_count = prompt_tokens
    mock_resp.usage_metadata.candidates_token_count = output_tokens

    def _generate(model: str, contents: str) -> MagicMock:
        if capture is not None:
            capture.append(contents)
        return mock_resp

    mock_client.models.generate_content.side_effect = _generate
    return mock_client


def test_returns_answer_and_citations():
    with patch("rag_service.core.generation.genai.Client", return_value=_mock_client("  Paris  ")):
        result = generate("Capital of France?", _nodes(["Paris is the capital."]))

    assert isinstance(result, GenerationResult)
    assert result.answer == "Paris"
    assert result.citations == ["doc123"]


def test_token_usage_captured():
    with patch("rag_service.core.generation.genai.Client",
               return_value=_mock_client("answer", prompt_tokens=512, output_tokens=42)):
        result = generate("q?", _nodes(["context"]))

    assert result.prompt_tokens == 512
    assert result.output_tokens == 42
    assert result.model  # the configured generation model name


def test_context_injected_into_prompt():
    captured: list[str] = []
    with patch("rag_service.core.generation.genai.Client",
               return_value=_mock_client("answer", capture=captured)):
        generate("question?", _nodes(["unique context sentence xyz"]))

    assert "unique context sentence xyz" in captured[0]
    assert "question?" in captured[0]


def test_multiple_nodes_all_cited():
    nodes = [
        NodeWithScore(node=TextNode(text="chunk1", metadata={"document_id": "d1"}), score=0.9),
        NodeWithScore(node=TextNode(text="chunk2", metadata={"document_id": "d2"}), score=0.8),
    ]

    with patch("rag_service.core.generation.genai.Client", return_value=_mock_client("combined")):
        result = generate("q?", nodes)

    assert result.citations == ["d1", "d2"]
