"""Output safety rules, rewrite and template fallback."""
import pytest

from app.agents.safety import SafetyChecker, check
from app.llm import Usage
from tests.fakes import FakeLLM, tool_reply

CLEAN = {"damage_description": "Moderate water damage to the kitchen floor from a failed supply line.",
         "recommendation": "Arrange drying and an inspection. Confirm whether water damage is covered under the "
                           "policy before any decision."}


def assessment(**text):
    return {"damage_type": "water_damage", "severity": "moderate", "affected_areas": ["kitchen"], **CLEAN, **text}


def test_clean_text_passes():
    flags, violations = check(CLEAN)
    assert all(flags.values()) and not violations


@pytest.mark.parametrize("text, failed", [
    ("This damage will be covered in full.", "no_coverage_promises"),
    ("We guarantee the claim is approved.", "no_coverage_promises"),
    ("Expect a settlement of $6,500.", "no_settlement_amounts"),
    ("The claimant should consult a lawyer and sue the contractor.", "no_legal_advice"),
    ("The tenant is at fault for leaving the tap running.", "no_liability_determination"),
    ("This is a devastating loss!", "no_emotional_language"),
])
def test_each_rule_detects(text, failed):
    flags, _ = check({"damage_description": "Kitchen leak.", "recommendation": text})
    assert flags[failed] is False


def test_unsafe_text_is_rewritten():
    llm = FakeLLM([tool_reply("submit_rewrite", CLEAN)])
    report = assessment(recommendation="Good news, this will be covered and you will receive $8,000!")
    outcome = SafetyChecker(llm).review(report, Usage())
    assert outcome.action == "rewritten" and all(outcome.flags.values())
    assert "$" not in report["recommendation"]
    assert set(outcome.initial_violations) >= {"no_coverage_promises", "no_settlement_amounts"}


def test_failed_rewrite_falls_back_to_template():
    llm = FakeLLM([tool_reply("submit_rewrite", {"damage_description": "x", "recommendation": "Still covered, $5"})])
    report = assessment(recommendation="It will be covered in full by the policy.")
    outcome = SafetyChecker(llm).review(report, Usage())
    assert outcome.action == "template" and all(outcome.flags.values())
    assert "Confirm policy eligibility" in report["recommendation"]


def test_empty_recommendation_gets_template_without_model_call():
    llm = FakeLLM([])
    report = assessment(recommendation="")
    outcome = SafetyChecker(llm).review(report, Usage())
    assert outcome.action == "passed" and len(report["recommendation"]) > 20
