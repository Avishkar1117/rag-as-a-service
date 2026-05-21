from unittest.mock import MagicMock, patch

from rag_service.core.ingestion import ingest_document, pdf_content_id, run_ocr

FAKE_PDF = b"%PDF-1.4 fake"


def test_pdf_content_id_is_deterministic():
    assert pdf_content_id(FAKE_PDF) == pdf_content_id(FAKE_PDF)
    assert len(pdf_content_id(FAKE_PDF)) == 12


def test_run_ocr_uses_direct_extraction_when_text_pdf():
    direct_text = "--- Page 1 ---\n\nlots of real embedded text here, plenty of chars"

    with patch("rag_service.core.ingestion._extract_text_directly", return_value=direct_text), \
         patch("rag_service.core.ingestion._pdf_to_images") as mock_images, \
         patch("rag_service.core.ingestion.genai.Client") as mock_client:

        result = run_ocr(FAKE_PDF)

    assert result == direct_text
    mock_images.assert_not_called()
    mock_client.assert_not_called()


def test_run_ocr_falls_back_to_ocr_for_scanned_pdf():
    pages = [(0, b"img0"), (1, b"img1")]

    with patch("rag_service.core.ingestion._extract_text_directly", return_value=None), \
         patch("rag_service.core.ingestion._pdf_to_images", return_value=pages), \
         patch("rag_service.core.ingestion._ocr_page", side_effect=lambda i, b, m: f"text{i}"), \
         patch("rag_service.core.ingestion.genai.Client"):

        result = run_ocr(FAKE_PDF)

    assert "text0" in result
    assert "text1" in result
    assert "Page 1" in result
    assert "Page 2" in result


def test_ingest_document_returns_chunk_count():
    mock_nodes = [MagicMock(), MagicMock(), MagicMock()]

    with patch("rag_service.core.ingestion.run_ocr", return_value="some document text"), \
         patch("rag_service.core.ingestion.chromadb.PersistentClient"), \
         patch("rag_service.core.ingestion.ChromaVectorStore"), \
         patch("rag_service.core.ingestion.VectorStoreIndex"), \
         patch("rag_service.core.ingestion.SentenceSplitter") as mock_splitter:

        mock_splitter.return_value.get_nodes_from_documents.return_value = mock_nodes
        n = ingest_document(FAKE_PDF, "testdoc")

    assert n == 3


def test_ingest_document_uses_correct_collection_name():
    with patch("rag_service.core.ingestion.run_ocr", return_value="text"), \
         patch("rag_service.core.ingestion.chromadb.PersistentClient") as mock_chroma, \
         patch("rag_service.core.ingestion.ChromaVectorStore"), \
         patch("rag_service.core.ingestion.VectorStoreIndex"), \
         patch("rag_service.core.ingestion.SentenceSplitter") as mock_splitter:

        mock_splitter.return_value.get_nodes_from_documents.return_value = [MagicMock()]
        ingest_document(FAKE_PDF, "mydoc")

    mock_chroma.return_value.delete_collection.assert_called_once_with("doc_mydoc")
    mock_chroma.return_value.create_collection.assert_called_once_with("doc_mydoc")


def test_ingest_document_handles_missing_collection_on_first_ingest():
    # First ingest of a doc: delete_collection raises (collection doesn't exist)
    # — the error should be swallowed.
    with patch("rag_service.core.ingestion.run_ocr", return_value="text"), \
         patch("rag_service.core.ingestion.chromadb.PersistentClient") as mock_chroma, \
         patch("rag_service.core.ingestion.ChromaVectorStore"), \
         patch("rag_service.core.ingestion.VectorStoreIndex"), \
         patch("rag_service.core.ingestion.SentenceSplitter") as mock_splitter:

        mock_chroma.return_value.delete_collection.side_effect = ValueError("not found")
        mock_splitter.return_value.get_nodes_from_documents.return_value = [MagicMock()]
        n = ingest_document(FAKE_PDF, "newdoc")

    assert n == 1
    mock_chroma.return_value.create_collection.assert_called_once_with("doc_newdoc")


def test_ocr_page_retries_on_transient_error():
    # A transient 503 mid-OCR must not abort the page — with_retry recovers it.
    from rag_service.core.ingestion import _ocr_page

    client = MagicMock()
    attempts = []

    def flaky(*args, **kwargs):
        attempts.append(1)
        if len(attempts) < 3:
            raise RuntimeError("503 UNAVAILABLE")
        response = MagicMock()
        response.text = "page text"
        return response

    client.models.generate_content.side_effect = flaky

    with patch("rag_service.retry.time.sleep"), \
         patch("rag_service.core.ingestion.PIL.Image.open"):
        result = _ocr_page(0, b"imgbytes", client)

    assert result == "page text"
    assert len(attempts) == 3
