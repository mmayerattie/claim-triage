"""Vector knowledge base of historical claims (fastembed + Qdrant).

Indexing is incremental: each point stores a hash of its source file, and at
startup only added/changed/removed claims are re-embedded. `python -m app.rag
--force` rebuilds everything.
"""
import argparse
import hashlib
import json
import logging
import uuid
from pathlib import Path

from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointIdsList, PointStruct, VectorParams

log = logging.getLogger("claim-triage.rag")

# BGE v1.5 recommends this instruction for short queries against longer passages.
# fastembed's query_embed does not add it, so we do.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def claim_text(claim: dict) -> str:
    """What gets embedded: the incident itself, not its outcome.

    Resolution, cost and policy stay in the payload; embedding them would make
    claims look alike ("covered_full" appears in most records).
    """
    areas = ", ".join(a.replace("_", " ") for a in claim.get("affected_areas", []))
    return (
        f"Damage type: {claim['damage_type'].replace('_', ' ')}. Affected areas: {areas}.\n"
        f"Incident: {claim['description']}\n"
        f"Findings: {claim.get('adjuster_notes', '')}"
    )


def content_hash(claim: dict) -> str:
    return hashlib.sha256(json.dumps(claim, sort_keys=True).encode()).hexdigest()


def point_id(claim_id: str) -> str:
    """Stable id so re-indexing a claim overwrites its previous point."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"claim:{claim_id}"))


def resolution_summary(claim: dict) -> str:
    """Neutral one-line outcome; no dollar figures or coverage language for this claim."""
    outcome = claim.get("resolution", "unknown").replace("_", " ")
    areas = ", ".join(claim.get("affected_areas", []))
    return (f"Historical {claim['severity']} {claim['damage_type'].replace('_', ' ')} claim "
            f"({areas}); recorded outcome: {outcome}.")


def load_claims(directory: Path) -> dict[str, dict]:
    claims = {}
    for path in sorted(directory.glob("*.json")):
        claim = json.loads(path.read_text(encoding="utf-8-sig"))
        claims[claim["claim_id"]] = claim
    if not claims:
        raise RuntimeError(f"No historical claims found in {directory}")
    return claims


class KnowledgeBase:
    def __init__(self, client: QdrantClient, collection: str, model_name: str,
                 strong: float = 0.70, floor: float = 0.60):
        self.client = client
        self.collection = collection
        self.strong, self.floor = strong, floor
        self.model = TextEmbedding(model_name)
        self.dim = len(self._embed(["dimension probe"])[0])

    def _embed(self, texts: list[str]) -> list[list[float]]:
        # Always pass a list: fastembed treats a bare str as an iterable of characters.
        return [vector.tolist() for vector in self.model.embed(texts)]

    def _ensure_collection(self, force: bool) -> None:
        exists = self.client.collection_exists(self.collection)
        if exists and force:
            self.client.delete_collection(self.collection)
            exists = False
        if not exists:
            self.client.create_collection(
                self.collection, vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE))

    def _indexed_hashes(self) -> dict[str, str]:
        hashes, offset = {}, None
        while True:
            points, offset = self.client.scroll(self.collection, limit=256, offset=offset,
                                                with_payload=["claim_id", "content_hash"])
            hashes.update({p.payload["claim_id"]: p.payload["content_hash"] for p in points})
            if offset is None:
                return hashes

    def sync(self, directory: Path, force: bool = False) -> dict[str, list[str]]:
        """Bring the collection in line with the claim files; returns what changed."""
        claims = load_claims(directory)
        self._ensure_collection(force)
        indexed = self._indexed_hashes()
        current = {cid: content_hash(c) for cid, c in claims.items()}

        added = sorted(cid for cid in current if cid not in indexed)
        modified = sorted(cid for cid in current if cid in indexed and indexed[cid] != current[cid])
        removed = sorted(cid for cid in indexed if cid not in current)
        changes = {"added": added, "modified": modified, "removed": removed}

        if indexed and (added or modified or removed):
            log.warning("Knowledge base changed since last index: %s; re-indexing affected claims", changes)

        to_embed = added + modified
        if to_embed:
            vectors = self._embed([claim_text(claims[cid]) for cid in to_embed])
            self.client.upsert(self.collection, points=[
                PointStruct(id=point_id(cid), vector=vector,
                            payload={**claims[cid], "content_hash": current[cid]})
                for cid, vector in zip(to_embed, vectors)
            ])
        if removed:
            self.client.delete(self.collection, points_selector=PointIdsList(
                points=[point_id(cid) for cid in removed]))
        log.info("Knowledge base: %d claims indexed (%d embedded, %d removed)",
                 len(current), len(to_embed), len(removed))
        return changes

    def search(self, query: str, limit: int = 3) -> dict:
        """Top matches above the similarity floor, each labelled strong or weak.

        Nearest-neighbour search always returns something; matches below the floor
        are dropped rather than presented as "similar". `below_floor` reports how
        many were dropped so the caller can flag the claim for human review.
        """
        vector = self._embed([QUERY_INSTRUCTION + query])[0]
        hits = self.client.query_points(self.collection, query=vector, limit=limit, with_payload=True).points
        claims = []
        for hit in hits:
            # Cosine similarity can be negative in theory; the API contract requires [0, 1].
            score = round(min(1.0, max(0.0, hit.score)), 4)
            if score < self.floor:
                continue
            claims.append({
                "claim_id": hit.payload["claim_id"],
                "similarity_score": score,
                "match_strength": "strong" if score >= self.strong else "weak",
                "damage_type": hit.payload["damage_type"],
                "severity": hit.payload["severity"],
                "affected_areas": hit.payload["affected_areas"],
                "description": hit.payload["description"],
                "resolution_summary": resolution_summary(hit.payload),
            })
        return {"claims": claims, "below_floor": len(hits) - len(claims)}


def main() -> None:
    from .config import load_settings

    parser = argparse.ArgumentParser(description="Index historical claims into Qdrant")
    parser.add_argument("--force", action="store_true", help="drop and rebuild the whole collection")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = load_settings()
    kb = KnowledgeBase(QdrantClient(url=settings.qdrant_url), settings.qdrant_collection, settings.embedding_model)
    print(json.dumps(kb.sync(settings.knowledge_base_dir, force=args.force)))


if __name__ == "__main__":
    main()
