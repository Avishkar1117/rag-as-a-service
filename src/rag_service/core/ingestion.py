import hashlib
import io
import logging
import time

import chromadb
import fitz
import PIL.Image
from google import genai
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore

from rag_service.config import settings

logger = logging.getLogger(__name__)

_OCR_PROMPT = "Extract all text and tables from this page as clean Markdown. Nothing else."
_OCR_MODEL = "gemini-2.5-flash-lite"


def _pdf_to_images(pdf_bytes: bytes) -> list[tuple[int, bytes]]:
    pdf = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = [
        (i, pdf.load_page(i).get_pixmap(matrix=fitz.Matrix(1.0, 1.0)).tobytes("png"))
        for i in range(len(pdf))
    ]
    pdf.close()
    return pages


def _extract_text_directly(pdf_bytes: bytes) -> str | None:
    # Return embedded text if substantial; None means PDF is scanned, fall back to OCR.
    pdf = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = [pdf.load_page(i).get_text() for i in range(len(pdf))]
    n_pages = len(pages)
    pdf.close()
    combined = "\n\n".join(f"--- Page {i + 1} ---\n\n{t}" for i, t in enumerate(pages))
    if n_pages == 0 or len(combined) < 50 * n_pages:
        return None
    return combined


def _ocr_page(i: int, img_bytes: bytes, client: genai.Client) -> str:
    img = PIL.Image.open(io.BytesIO(img_bytes))
    for attempt in range(5):
        try:
            response = client.models.generate_content(
                model=_OCR_MODEL,
                contents=[_OCR_PROMPT, img],
            )
            return response.text
        except Exception as e:
            if "429" in str(e) or "ResourceExhausted" in str(e):
                wait = 30 * (attempt + 1)
                logger.warning("rate_limit page=%d wait=%ds attempt=%d", i, wait, attempt)
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"Page {i + 1} failed after 5 retries")


def run_ocr(pdf_bytes: bytes) -> str:
    """Extract a PDF as Markdown. Try embedded text first; OCR only for scans."""
    direct = _extract_text_directly(pdf_bytes)
    if direct is not None:
        logger.info("extracted text directly, skipping OCR")
        return direct

    logger.info("no embedded text found, running OCR")
    client = genai.Client(api_key=settings.gemini_api_key)
    pages = _pdf_to_images(pdf_bytes)
    results: dict[int, str] = {}
    for i, img_bytes in pages:
        results[i] = _ocr_page(i, img_bytes, client)
        logger.info("ocr page %d/%d done", i + 1, len(pages))
    return "\n\n".join(f"--- Page {i + 1} ---\n\n{results[i]}" for i in sorted(results))


def ingest_document(pdf_bytes: bytes, document_id: str) -> int:
    """OCR a PDF, chunk it, embed, and store in Chroma. Returns chunk count."""
    md_text = run_ocr(pdf_bytes)

    chroma_client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    collection_name = f"doc_{document_id}"
    # Replace (not append) so re-ingestion doesn't duplicate chunks.
    try:
        chroma_client.delete_collection(collection_name)
    except Exception:
        pass  # idempotent: collection may not exist yet on first ingest
    collection = chroma_client.create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    nodes = SentenceSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    ).get_nodes_from_documents(
        [Document(text=md_text, metadata={"document_id": document_id})]
    )

    VectorStoreIndex(nodes, storage_context=storage_context)
    logger.info("ingested document_id=%s chunks=%d", document_id, len(nodes))
    return len(nodes)


def pdf_content_id(pdf_bytes: bytes) -> str:
    """Stable document ID derived from PDF content."""
    return hashlib.md5(pdf_bytes).hexdigest()[:12]
