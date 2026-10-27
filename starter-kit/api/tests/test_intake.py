"""Intake validation: deterministic PII rules, flag handling and HTTP rejection."""
import pytest

from app.agents.intake import IntakeValidator
from app.llm import Usage
from app.pii import luhn_valid, redact_sensitive_identifiers
from tests.fakes import FakeLLM, intake_reply, jpeg_bytes

TEST_CARD = "4111 1111 1111 1111"  # standard Luhn-valid test number


@pytest.mark.parametrize("text, kinds", [
    (f"my card {TEST_CARD} was on the table", ["payment card number"]),
    ("SSN 123-45-6789 on the form", ["social security number"]),
    ("IBAN DE89 3704 0044 0532 0130 00", ["bank account number (IBAN)"]),
])
def test_sensitive_identifiers_redacted(text, kinds):
    redacted, found = redact_sensitive_identifiers(text)
    assert found == kinds
    assert "[REDACTED]" in redacted


@pytest.mark.parametrize("text", [
    "Water came through the ceiling at 1234 Elm Street, unit 5",   # incidental address
    "Order 1234 5678 9012 3456 from the hardware store",           # 16 digits, fails Luhn
    "Call me on 555-123-4567",                                     # phone, not SSN shape
    "Inspection INS-001 dated 22 September 2026",
])
def test_incidental_numbers_allowed(text):
    redacted, found = redact_sensitive_identifiers(text)
    assert found == [] and redacted == text


def test_luhn():
    assert luhn_valid("4111111111111111")
    assert not luhn_valid("4111111111111112")


def test_text_only_sets_is_valid_image_null():
    llm = FakeLLM([intake_reply()])
    result = IntakeValidator(llm).validate([], "Pipe burst in the kitchen", [], Usage())
    assert result.passed
    assert result.flags["is_valid_image"] is None
    assert "Visual evidence was unavailable" in result.message
    # No image parts and no is_valid_image field were requested from the model.
    tool = llm.requests[0]["tools"][0]["function"]["parameters"]["properties"]
    assert "is_valid_image" not in tool


def test_text_pii_overrides_model_verdict():
    llm = FakeLLM([intake_reply(no_pii_detected=True)])
    result = IntakeValidator(llm).validate([jpeg_bytes()], "Kitchen leak", ["payment card number"], Usage())
    assert not result.passed
    assert result.flags["no_pii_detected"] is False
    assert "sensitive personal information" in result.message


def test_missing_verdict_is_rejected_not_passed():
    from app.llm import LLMReply
    result = IntakeValidator(FakeLLM([LLMReply(content="I think it is fine")])).validate(
        [jpeg_bytes()], "", [], Usage())
    assert not result.passed


def test_rejection_returns_400_with_flags(client, fake_llm):
    fake_llm.queue(intake_reply(is_property_damage=False, reason="The photo shows a cat on a sofa."))
    response = client.post("/api/v1/claims/triage", files={"image": ("cat.jpg", jpeg_bytes(), "image/jpeg")})
    assert response.status_code == 400
    body = response.json()
    assert body["intake_validation"]["is_property_damage"] is False
    assert "does not show property damage" in body["message"]
    assert "triage_assessment" not in body


def test_card_number_never_reaches_model(client, fake_llm):
    fake_llm.queue(intake_reply())
    client.post("/api/v1/claims/triage", files={"description": (None, f"Card {TEST_CARD} got wet")})
    sent = str(fake_llm.requests[0]["messages"])
    assert "4111" not in sent and "[REDACTED]" in sent


def test_corrupt_image_rejected_before_model(client, fake_llm):
    response = client.post("/api/v1/claims/triage", files={"image": ("x.jpg", b"not an image", "image/jpeg")})
    assert response.status_code == 400
    assert fake_llm.requests == []


def test_invalid_pdf_rejected(client, fake_llm):
    response = client.post("/api/v1/claims/triage", files={
        "description": (None, "Leak"), "document": ("doc.pdf", b"plain text", "application/pdf")})
    assert response.status_code == 400
    assert "not a valid PDF" in response.json()["message"]
