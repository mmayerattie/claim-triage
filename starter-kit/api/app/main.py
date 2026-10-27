"""HTTP layer: routes, multipart parsing and error mapping. Agents live in app/agents."""
import io
import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from qdrant_client import QdrantClient

from .agents.handling import apply_safety_action
from .agents.intake import IntakeValidator
from .agents.safety import SafetyChecker
from .agents.triage import TriageAgent, TriageError, TriageInput
from .config import load_settings
from .documents import DocumentError, extract_pdf_text
from .domain import DEFAULT_POLICY_TYPE, POLICY_TYPES
from .tools import CostEstimator, CoverageChecker
from .llm import LLMClient, LLMError, Usage
from .pii import redact_sensitive_identifiers
from .rag import KnowledgeBase
from .store import ClaimStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("claim-triage")

settings = load_settings()
IMAGE_TYPES = {"image/jpeg", "image/png"}
MAX_PHOTOS = 8
MAX_PHOTO_BYTES = 15_000_000
MAX_DOCUMENT_BYTES = 20_000_000


def wait_for_qdrant(client: QdrantClient, attempts: int = 30, delay: float = 2.0) -> bool:
    """Qdrant's image has no curl, so readiness is checked from this side."""
    for attempt in range(1, attempts + 1):
        try:
            client.get_collections()
            return True
        except Exception as exc:  # connection refused while Qdrant boots
            log.info("Waiting for Qdrant (%d/%d): %s", attempt, attempts, type(exc).__name__)
            time.sleep(delay)
    return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.store = ClaimStore(settings.db_path)
    llm = LLMClient(settings)
    app.state.intake = IntakeValidator(llm)
    app.state.safety = SafetyChecker(llm)
    app.state.qdrant = QdrantClient(url=settings.qdrant_url, timeout=5)
    app.state.kb = None
    if wait_for_qdrant(app.state.qdrant):
        try:
            app.state.kb = KnowledgeBase(app.state.qdrant, settings.qdrant_collection, settings.embedding_model,
                                         strong=settings.similarity_strong, floor=settings.similarity_floor)
            app.state.kb.sync(settings.knowledge_base_dir)
        except Exception:
            log.exception("Knowledge base indexing failed; retrieval unavailable")
            app.state.kb = None
    else:
        log.error("Qdrant unreachable at %s; health will report disconnected", settings.qdrant_url)
    app.state.triage = TriageAgent(
        llm, app.state.kb,
        CostEstimator(settings.tools_dir / "cost-estimator.json"),
        CoverageChecker(settings.tools_dir / "coverage-table.json"),
        max_turns=settings.max_agent_turns)
    if not settings.llm_configured:
        log.warning("LLM settings missing; triage requests will fail until PORTKEY_* and LLM_MODEL are set")
    yield


app = FastAPI(title="Claim Triage API", version="1.0.0", lifespan=lifespan)


@app.get("/api/v1/health")
def health():
    try:
        app.state.qdrant.get_collections()
        vector_db = "connected"
    except Exception:
        vector_db = "disconnected"
    return {
        "status": "healthy",
        "services": {
            "vector_db": vector_db,
            "llm": "configured" if settings.llm_configured else "not_configured",
        },
    }


def bad_request(message: str) -> JSONResponse:
    return JSONResponse(status_code=400, content={"error": "invalid_request", "message": message})


def read_photo(upload: UploadFile) -> bytes:
    if upload.content_type not in IMAGE_TYPES:
        raise ValueError(f"Unsupported image type '{upload.content_type}'; use JPEG or PNG.")
    data = upload.file.read(MAX_PHOTO_BYTES + 1)
    if len(data) > MAX_PHOTO_BYTES:
        raise ValueError(f"Photo '{upload.filename}' exceeds {MAX_PHOTO_BYTES // 1_000_000} MB.")
    try:
        with Image.open(io.BytesIO(data)) as decoded:
            decoded.verify()
    except Exception as exc:
        raise ValueError(f"Photo '{upload.filename}' could not be decoded as an image.") from exc
    return data


# Sync handler: FastAPI runs it on a worker thread, so blocking model calls do not stall the server.
@app.post("/api/v1/claims/triage")
def triage_claim(
    image: UploadFile | None = File(None),
    images: list[UploadFile] | None = File(None),
    description: str | None = Form(None),
    document: UploadFile | None = File(None),
    policy_type: str | None = Form(None),
):
    uploads = [f for f in ([image] if image else []) + (images or []) if f and f.filename]
    text = (description or "").strip()
    if not uploads and not text:
        return bad_request("Provide at least one damage photo or a non-blank incident description.")
    if len(uploads) > MAX_PHOTOS:
        return bad_request(f"At most {MAX_PHOTOS} photos per claim.")
    if policy_type and policy_type not in POLICY_TYPES:
        return bad_request(f"Unknown policy_type '{policy_type}'.")
    try:
        photos = [read_photo(upload) for upload in uploads]
        document_text = ""
        if document is not None and document.filename:
            if document.content_type != "application/pdf":
                raise ValueError("Supporting documents must be PDF files.")
            document_text = extract_pdf_text(document.file.read(MAX_DOCUMENT_BYTES))
    except (ValueError, DocumentError) as exc:
        return bad_request(str(exc))

    # Deterministic PII check first; identifiers are redacted before any text reaches the model.
    text, text_pii = redact_sensitive_identifiers(text)
    document_text, document_pii = redact_sensitive_identifiers(document_text)

    usage = Usage()
    try:
        intake = app.state.intake.validate(photos, text, text_pii + document_pii, usage)
    except LLMError as exc:
        log.error("Intake validation failed: %s", exc)
        return JSONResponse(status_code=502, content={"error": "model_unavailable", "message": str(exc)})
    log.info("Intake %s flags=%s usage=%s", "passed" if intake.passed else "rejected", intake.flags, usage.as_dict())
    if not intake.passed:
        return JSONResponse(status_code=400, content={
            "error": "intake_validation_failed", "intake_validation": intake.flags, "message": intake.message})

    claim = TriageInput(photos=photos, description=text, document_text=document_text,
                        policy_type=policy_type or DEFAULT_POLICY_TYPE, policy_supplied=bool(policy_type))
    try:
        outcome = app.state.triage.run(claim, usage)
        safety = app.state.safety.review(outcome.assessment, usage)
    except (LLMError, TriageError) as exc:
        log.error("Triage failed: %s", exc)
        return JSONResponse(status_code=502, content={"error": "triage_failed", "message": str(exc)})

    assessment = outcome.assessment
    apply_safety_action(assessment["handling"], safety.action, list(safety.initial_violations))
    if not all(safety.flags.values()):
        return JSONResponse(status_code=422, content={
            "error": "safety_check_failed", "safety_checks": safety.flags,
            "message": "The triage text could not be made safe to release; refer to an adjuster."})

    claim_id = f"CLM-{uuid.uuid4().hex[:12].upper()}"
    report = {
        "claim_id": claim_id,
        "intake_validation": intake.flags,
        "intake_message": intake.message,
        "triage_assessment": assessment,
        "safety_checks": safety.flags,
    }
    trace = outcome.trace.export(claim_id, usage=usage.as_dict(), safety_action=safety.action)
    app.state.store.save(claim_id, report, trace)
    log.info("Claim %s triaged in %d ms; usage=%s; safety=%s", claim_id, trace["total_duration_ms"],
             usage.as_dict(), safety.action)
    return report


@app.get("/api/v1/claims/{claim_id}")
def get_report(claim_id: str):
    report = app.state.store.get_report(claim_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Claim not found")
    return report


@app.get("/api/v1/agent/trace/{claim_id}")
def get_trace(claim_id: str):
    trace = app.state.store.get_trace(claim_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="Trace not found")
    return trace
