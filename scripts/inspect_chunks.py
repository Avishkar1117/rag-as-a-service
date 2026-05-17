import sys

import chromadb

from rag_service.config import settings

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def inspect(document_id: str) -> None:
    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    collection = client.get_collection(f"doc_{document_id}")
    data = collection.get(include=["documents", "metadatas"])

    docs = data["documents"]
    print(f"\n=== {len(docs)} chunks in doc_{document_id} ===\n")
    for i, text in enumerate(docs):
        print(f"--- Chunk {i} ({len(text)} chars) ---")
        print(text)
        print()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: uv run python scripts/inspect_chunks.py <document_id>")
        sys.exit(1)
    inspect(sys.argv[1])
