import os

# Set before any rag_service import so Pydantic Settings finds them.
os.environ.setdefault("GEMINI_API_KEY", "test-gemini-key")
os.environ.setdefault("GITHUB_TOKEN", "test-github-token")
