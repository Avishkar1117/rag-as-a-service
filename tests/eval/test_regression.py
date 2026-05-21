"""Day 13 regression guard — the "evaluation as CI" gate.

Runs the RAGAS harness over a small fixed subset of eval/dataset.jsonl and
fails if faithfulness drops below a threshold. A genuinely bad change — e.g.
a prompt that stops grounding answers in the retrieved context — pushes
faithfulness down and turns this test red before it can be merged.

It makes live API calls (generation + the RAGAS judge), so it is marked
`eval` and excluded from the default pytest run. Run it explicitly with a
real GEMINI_API_KEY in the environment:

    uv run pytest -m eval

CI runs it from .github/workflows/eval.yml with the key from repo secrets.
Without a real key the test skips rather than fails.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# eval/ holds scripts, not an installed package — put it on the path so this
# test can reuse the exact harness modules run_ragas.py is built from.
_EVAL_DIR = Path(__file__).resolve().parents[2] / "eval"
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

pytestmark = pytest.mark.eval

# Baseline faithfulness sits at ~0.97-1.00; 0.85 leaves room for judge noise
# while still catching a real grounding regression.
FAITHFULNESS_THRESHOLD = 0.85
N_QUESTIONS = 10


def _has_real_api_key() -> bool:
    key = os.environ.get("GEMINI_API_KEY", "")
    return bool(key) and not key.startswith("test-")


@pytest.mark.skipif(
    not _has_real_api_key(), reason="needs a real GEMINI_API_KEY in the environment"
)
def test_faithfulness_does_not_regress() -> None:
    import ragas_eval
    import run_ragas

    from rag_service.config import settings
    from rag_service.llm.openai_client import setup_llamaindex_settings

    setup_llamaindex_settings()
    run_ragas.ensure_corpus_ingested(rebuild=False)

    rows = run_ragas.load_dataset(limit=N_QUESTIONS)
    samples, _ = run_ragas.run_pipeline(rows, top_k=settings.top_k)

    scores = ragas_eval.score(samples)
    faithfulness = scores["faithfulness"].dropna()
    assert not faithfulness.empty, "RAGAS produced no faithfulness scores (API outage?)"

    mean_faithfulness = float(faithfulness.mean())
    assert mean_faithfulness >= FAITHFULNESS_THRESHOLD, (
        f"faithfulness {mean_faithfulness:.3f} below regression threshold "
        f"{FAITHFULNESS_THRESHOLD} over {len(faithfulness)} question(s)"
    )
