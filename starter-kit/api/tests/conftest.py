"""Test fixtures: no network, no paid LLM calls."""
import os
import tempfile

os.environ.setdefault("DB_PATH", os.path.join(tempfile.mkdtemp(), "claims.db"))

import pytest
from fastapi.testclient import TestClient

from app import main
from tests.fakes import FakeLLM


class FakeQdrant:
    def __init__(self, *args, **kwargs):
        self.up = True

    def get_collections(self):
        if not self.up:
            raise ConnectionError("qdrant down")
        return []


class FakeKnowledgeBase:
    def __init__(self, *args, **kwargs):
        pass

    def sync(self, directory, force=False):
        return {"added": [], "modified": [], "removed": []}

    def search(self, query, limit=3):
        return {"claims": [
            {"claim_id": "CLM-2023-001", "similarity_score": 0.83, "match_strength": "strong",
             "damage_type": "water_damage", "severity": "moderate", "affected_areas": ["kitchen", "dining_room"],
             "description": "Burst pipe under kitchen sink.", "resolution_summary": "Historical claim."},
            {"claim_id": "CLM-2023-002", "similarity_score": 0.66, "match_strength": "weak",
             "damage_type": "water_damage", "severity": "minor", "affected_areas": ["bathroom"],
             "description": "Toilet overflow.", "resolution_summary": "Historical claim."},
        ], "below_floor": 1}


@pytest.fixture
def fake_llm():
    return FakeLLM()


@pytest.fixture
def client(monkeypatch, fake_llm):
    monkeypatch.setattr(main, "QdrantClient", FakeQdrant)
    monkeypatch.setattr(main, "KnowledgeBase", FakeKnowledgeBase)
    monkeypatch.setattr(main, "LLMClient", lambda settings: fake_llm)
    with TestClient(main.app) as test_client:
        yield test_client
