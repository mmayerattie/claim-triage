"""Deterministic detection of sensitive identifiers in submitted text.

Policy: reject sensitive identifiers (payment cards, SSNs, bank account numbers);
incidental context such as house numbers or a family name on a mailbox is allowed.
Images are checked by the vision model; this backs it up for text and PDF content.
"""
import re

CARD_CANDIDATE = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")
SSN = re.compile(r"(?<!\d)(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}(?!\d)")
IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){3,7}(?: ?[A-Z0-9]{1,4})?\b")


def luhn_valid(digits: str) -> bool:
    total = 0
    for index, char in enumerate(reversed(digits)):
        value = int(char)
        if index % 2 == 1:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


def redact_sensitive_identifiers(text: str) -> tuple[str, list[str]]:
    """Replace identifiers with [REDACTED] and return the kinds found (never the values).

    Redaction happens before any text is sent to the model provider.
    """
    found: list[str] = []

    def card(match: re.Match) -> str:
        digits = re.sub(r"\D", "", match.group())
        if 13 <= len(digits) <= 19 and luhn_valid(digits):
            if "payment card number" not in found:
                found.append("payment card number")
            return "[REDACTED]"
        return match.group()

    text = CARD_CANDIDATE.sub(card, text)
    for pattern, kind in ((SSN, "social security number"), (IBAN, "bank account number (IBAN)")):
        text, count = pattern.subn("[REDACTED]", text)
        if count:
            found.append(kind)
    return text, found
