"""Supporting-document text extraction.

PDF text is passed to the model as untrusted evidence. Text extraction (rather
than native PDF input) keeps the pipeline model-agnostic and lets the PII check
and the trace see exactly what the model saw.
"""
import io
import logging

from pypdf import PdfReader
from pypdf.errors import PdfReadError

log = logging.getLogger("claim-triage.documents")

MAX_DOCUMENT_CHARS = 20_000


class DocumentError(ValueError):
    pass


def extract_pdf_text(data: bytes) -> str:
    """Plain text of a PDF; empty string for scanned PDFs without a text layer."""
    if not data.startswith(b"%PDF"):
        raise DocumentError("The supporting document is not a valid PDF file.")
    try:
        reader = PdfReader(io.BytesIO(data))
        text = "\n".join((page.extract_text() or "") for page in reader.pages).strip()
    except (PdfReadError, ValueError) as exc:
        raise DocumentError("The supporting PDF could not be read.") from exc
    if len(text) > MAX_DOCUMENT_CHARS:
        log.info("Document truncated from %d to %d characters", len(text), MAX_DOCUMENT_CHARS)
        text = text[:MAX_DOCUMENT_CHARS]
    return text
