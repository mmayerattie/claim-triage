"""Knowledge-base indexing and retrieval with the real embedding model and in-memory Qdrant."""
import json
import shutil
from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from app.config import load_settings
from app.rag import KnowledgeBase, claim_text, load_claims

KB_DIR = load_settings().knowledge_base_dir
pytestmark = pytest.mark.skipif(not KB_DIR.is_dir(), reason="knowledge base not mounted")


@pytest.fixture(scope="module")
def kb():
    knowledge_base = KnowledgeBase(QdrantClient(":memory:"), "test_claims", load_settings().embedding_model)
    knowledge_base.sync(KB_DIR)
    return knowledge_base


def test_embedded_text_excludes_outcome_fields():
    claim = load_claims(KB_DIR)["CLM-2023-001"]
    text = claim_text(claim)
    assert claim["description"] in text
    assert claim["resolution"] not in text
    assert str(claim["repair_cost_actual"]) not in text


def test_all_claims_indexed(kb):
    assert kb.client.count(kb.collection).count == len(load_claims(KB_DIR))


def test_burst_pipe_query_retrieves_water_damage(kb):
    results = kb.search("burst pipe under the kitchen sink, water pooling on the floor")["claims"]
    assert len(results) == 3
    assert results[0]["damage_type"] == "water_damage"
    assert all(0 <= r["similarity_score"] <= 1 for r in results)
    assert results == sorted(results, key=lambda r: r["similarity_score"], reverse=True)


def test_resolution_summary_has_no_dollar_figures(kb):
    for result in kb.search("kitchen fire with charred cabinets")["claims"]:
        assert "$" not in result["resolution_summary"]


def test_sync_is_incremental(tmp_path: Path):
    directory = tmp_path / "kb"
    directory.mkdir()
    for name in ("CLM-2023-001.json", "CLM-2023-002.json", "CLM-2023-003.json"):
        shutil.copy(KB_DIR / name, directory / name)
    kb = KnowledgeBase(QdrantClient(":memory:"), "incremental", load_settings().embedding_model)

    assert kb.sync(directory)["added"] == ["CLM-2023-001", "CLM-2023-002", "CLM-2023-003"]
    assert kb.sync(directory) == {"added": [], "modified": [], "removed": []}

    edited = json.loads((directory / "CLM-2023-002.json").read_text())
    edited["description"] += " Additional water staining was found later."
    (directory / "CLM-2023-002.json").write_text(json.dumps(edited))
    (directory / "CLM-2023-003.json").unlink()

    changes = kb.sync(directory)
    assert changes == {"added": [], "modified": ["CLM-2023-002"], "removed": ["CLM-2023-003"]}
    assert kb.client.count(kb.collection).count == 2


def test_strong_precedent_is_labelled_strong(kb):
    result = kb.search("burst pipe under the kitchen sink, water pooling on the floor")
    assert result["claims"][0]["match_strength"] == "strong"
    assert result["below_floor"] == 0


def test_unrelated_query_returns_no_similar_claims(kb):
    """Nearest neighbours always exist; below the floor they must not be shown as similar."""
    result = kb.search("Hello, I would like to ask about my bill")
    assert result["claims"] == []
    assert result["below_floor"] == 3
