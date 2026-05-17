from google import genai
from llama_index.core.schema import NodeWithScore

from rag_service.config import settings

_PROMPT = """\
Answer using ONLY the context below. \
If the context does not contain the answer, say "I don't have enough information."

Context:
{context}

Question: {question}

Answer:"""


def generate(question: str, nodes: list[NodeWithScore]) -> tuple[str, list[str]]:
    """Call Gemma with retrieved context. Returns (answer, citations)."""
    context = "\n\n".join(node.node.get_content() for node in nodes)
    citations = [node.node.metadata.get("document_id", "") for node in nodes]

    client = genai.Client(api_key=settings.gemini_api_key)
    response = client.models.generate_content(
        model=settings.gemma_model,
        contents=_PROMPT.format(context=context, question=question),
    )
    return response.text.strip(), citations
