"""safetyChecker: reviews the triage text before it is returned.

Rules run on every report. If any check fails, the model rewrites the flagged
text once; if the rewrite still fails, a neutral template replaces it. The
returned flags describe the text actually returned.
"""
import logging
import re
from dataclasses import dataclass, field

from ..llm import LLMClient, LLMError, Usage

log = logging.getLogger("claim-triage.safety")

CONDITIONAL = re.compile(r"\b(whether|if|confirm|verify|determine|check|review|subject to|may|might|appears?)\b", re.I)

RULES: dict[str, list[re.Pattern]] = {
    "no_coverage_promises": [
        re.compile(r"\b(is|are|will be|would be|should be)\s+(fully\s+|definitely\s+)?covered\b", re.I),
        re.compile(r"\bguarantee[ds]?\b|\bcoverage is (confirmed|approved)\b|\bclaim (is|will be) approved\b", re.I),
        re.compile(r"\b(will|shall) (pay|reimburse|compensate)\b|\bentitled to\b", re.I),
    ],
    "no_settlement_amounts": [
        re.compile(r"[$€£]\s?\d|\b\d[\d,.]*\s?(usd|dollars)\b", re.I),
        re.compile(r"\b(settle|settlement|payout|pay out|reimburse\w*)\b[^.]{0,30}\d", re.I),
    ],
    "no_legal_advice": [
        re.compile(r"\b(sue|lawsuit|legal action|attorney|lawyer|litigat\w*|legal rights|small claims)\b", re.I),
    ],
    "no_liability_determination": [
        re.compile(r"\bat fault\b|\b(is|are) to blame\b|\bfault of\b|\bnegligen\w*\b|\bliable\b", re.I),
        re.compile(r"\b(tenant|neighbou?r|contractor|homeowner|claimant|plumber|driver)\b[^.]{0,25}\b(caused|responsible for)\b", re.I),
    ],
    "no_emotional_language": [
        re.compile(r"\b(devastat\w*|tragic\w*|heartbreak\w*|horrible|terrible|nightmare|awful|so sorry)\b|!", re.I),
    ],
}

TEXT_FIELDS = ("damage_description", "recommendation")

MITIGATION = {
    "water_damage": "Advise prompt water mitigation and drying to limit further damage.",
    "fire_damage": "Confirm the area is safe to enter and arrange a smoke and soot assessment.",
    "storm_damage": "Arrange temporary protection, such as tarping or boarding, where the structure is open.",
    "break_in": "Advise securing entry points and obtaining the police report reference.",
    "vandalism": "Advise documenting the damage and obtaining the police report reference.",
    "other": "Arrange a specialist assessment to establish the cause.",
}

REWRITE_TOOL = {"type": "function", "function": {
    "name": "submit_rewrite",
    "description": "Return the corrected text fields.",
    "parameters": {"type": "object", "additionalProperties": False,
                   "required": ["damage_description", "recommendation"],
                   "properties": {"damage_description": {"type": "string"}, "recommendation": {"type": "string"}}},
}}

REWRITE_PROMPT = """You edit insurance triage notes for an adjuster. Rewrite the two fields so they keep every factual
observation and next step but: never promise or deny coverage (coverage is for the adjuster to confirm), never state
money amounts, settlement or payout figures, never give legal advice, never assign fault or blame, and use a
neutral, professional tone without exclamation marks. Report through submit_rewrite."""


def check(fields: dict) -> tuple[dict, dict]:
    """Return (flags, violations) for the text fields; True means the check passed."""
    flags, violations = {}, {}
    for name, patterns in RULES.items():
        hits = []
        for text_field in TEXT_FIELDS:
            text = fields.get(text_field) or ""
            for pattern in patterns:
                for match in pattern.finditer(text):
                    window = text[max(0, match.start() - 60):match.start()]
                    if name == "no_coverage_promises" and CONDITIONAL.search(window):
                        continue  # "confirm whether ... is covered" is not a promise
                    hits.append(f"{text_field}: '{match.group().strip()}'")
        flags[name] = not hits
        if hits:
            violations[name] = hits
    return flags, violations


def template_fields(assessment: dict) -> dict:
    areas = ", ".join(assessment["affected_areas"]) or "the reported area"
    damage = assessment["damage_type"].replace("_", " ")
    return {
        "damage_description": f"{assessment['severity'].capitalize()} {damage} affecting {areas}, based on the "
                              "submitted evidence.",
        "recommendation": f"Arrange an inspection to confirm the extent of the {damage} in {areas}. "
                          f"{MITIGATION[assessment['damage_type']]} Confirm policy eligibility and applicable "
                          "terms before any decision.",
    }


@dataclass
class SafetyOutcome:
    flags: dict
    action: str  # "passed" | "rewritten" | "template"
    initial_violations: dict = field(default_factory=dict)


class SafetyChecker:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def review(self, assessment: dict, usage: Usage) -> SafetyOutcome:
        """Checks and, if needed, sanitises the assessment's text fields in place."""
        if len(assessment.get("recommendation", "").strip()) < 20:
            assessment.update(template_fields(assessment))
        flags, violations = check(assessment)
        if not violations:
            return SafetyOutcome(flags, "passed")
        log.warning("Safety checks failed: %s", violations)

        try:
            rewritten = self._rewrite(assessment, violations, usage)
            new_flags, remaining = check(rewritten)
            if not remaining and len(rewritten["recommendation"].strip()) >= 20:
                assessment.update(rewritten)
                return SafetyOutcome(new_flags, "rewritten", violations)
            log.warning("Rewrite still failed safety checks: %s", remaining)
        except LLMError as exc:
            log.warning("Safety rewrite unavailable: %s", exc)

        assessment.update(template_fields(assessment))
        final_flags, _ = check(assessment)
        return SafetyOutcome(final_flags, "template", violations)

    def _rewrite(self, assessment: dict, violations: dict, usage: Usage) -> dict:
        problems = "\n".join(f"- {name}: {', '.join(hits)}" for name, hits in violations.items())
        current = "\n".join(f"{name}: {assessment.get(name, '')}" for name in TEXT_FIELDS)
        reply = self.llm.chat(
            [{"role": "system", "content": REWRITE_PROMPT},
             {"role": "user", "content": f"Problems found:\n{problems}\n\nCurrent text:\n{current}"}],
            usage, tools=[REWRITE_TOOL], force_tool="submit_rewrite")
        arguments = reply.tool_calls[0].arguments if reply.tool_calls else {}
        return {name: str(arguments.get(name) or "") for name in TEXT_FIELDS}
