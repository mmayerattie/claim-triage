"""ReAct loop, trace consistency and report assembly with a scripted model."""
import pytest

from app.agents.triage import TriageAgent, TriageError, TriageInput
from app.config import load_settings
from app.llm import Usage
from app.tools import CostEstimator, CoverageChecker
from tests.conftest import FakeKnowledgeBase
from tests.fakes import FakeLLM, intake_reply, jpeg_bytes, tool_reply

TOOLS_DIR = load_settings().tools_dir
pytestmark = pytest.mark.skipif(not TOOLS_DIR.is_dir(), reason="tool tables not mounted")


def search(query="burst pipe under kitchen sink"):
    return tool_reply("search_past_claims", {"purpose": "Find similar incidents.", "query": query}, "c1")


def estimate(severity="moderate", areas=("kitchen",), damage_type="water_damage", call_id="c2"):
    return tool_reply("estimate_repair_cost", {"purpose": "Estimate cost for the scope.", "damage_type": damage_type,
                                               "severity": severity, "affected_areas": list(areas)}, call_id)


def coverage(category="water_damage", policy="premium_homeowner"):
    return tool_reply("check_coverage", {"purpose": "Check policy table.", "policy_type": policy,
                                         "incident_category": category}, "c3")


def submit(**overrides):
    arguments = {"damage_description": "Water pooling under the kitchen sink from a failed supply line.",
                 "urgency": "high", "recommendation": "Arrange drying and an inspection of the kitchen cabinetry.",
                 "cited_claim_ids": ["CLM-2023-001"], "claimant_questions": [], "advisory_notes": []}
    arguments.update(overrides)
    return tool_reply("submit_triage", arguments, "c9")


def make_agent(*replies, max_turns=6):
    llm = FakeLLM(list(replies))
    agent = TriageAgent(llm, FakeKnowledgeBase(), CostEstimator(TOOLS_DIR / "cost-estimator.json"),
                        CoverageChecker(TOOLS_DIR / "coverage-table.json"), max_turns=max_turns)
    return agent, llm


def claim(photos=True, policy="standard_homeowner", supplied=True, document=""):
    return TriageInput(photos=[jpeg_bytes()] if photos else [], description="Pipe burst under the sink",
                       document_text=document, policy_type=policy, policy_supplied=supplied)


def assert_valid_trace(steps, assessment):
    """Structural rules a valid trace must satisfy."""
    kinds = [s["type"] for s in steps]
    for kind in ("reason", "act", "observe"):
        assert kinds.count(kind) >= 2
    assert any(kinds[i:i + 3] == ["reason", "act", "observe"] for i in range(len(kinds) - 2))
    for i, step in enumerate(steps):
        assert step["content"].strip()
        if step["type"] == "act":
            assert steps[i + 1]["type"] == "observe" and step["tool_input"]
    last_estimate = [s for s in steps if s["type"] == "act" and s["tool_name"] == "estimate_repair_cost"][-1]
    for key in ("damage_type", "severity", "affected_areas"):
        assert last_estimate["tool_input"][key] == assessment[key]


def test_happy_path_report_matches_trace():
    agent, _ = make_agent(search(), estimate(), coverage(), submit())
    outcome = agent.run(claim(), Usage())
    assessment = outcome.assessment
    assert (assessment["damage_type"], assessment["severity"]) == ("water_damage", "moderate")
    assert assessment["estimated_repair_cost"] == {"min": 2000, "max": 8000, "currency": "USD"}
    assert [c["claim_id"] for c in assessment["similar_past_claims"]] == ["CLM-2023-001"]
    assert_valid_trace(outcome.trace.steps, assessment)


def test_latest_estimate_defines_scope():
    """An inspection report widens the scope; the second estimate wins."""
    agent, _ = make_agent(search(), estimate(), estimate(areas=("kitchen", "Dining Room"), call_id="c4"),
                          coverage(), submit())
    assessment = agent.run(claim(document="Inspection: kitchen and dining room affected."), Usage()).assessment
    assert assessment["affected_areas"] == ["kitchen", "dining_room"]
    assert assessment["estimated_repair_cost"]["min"] == 4000
    assert assessment["estimated_repair_cost"]["max"] == 16000


def test_submit_before_evidence_is_refused():
    agent, llm = make_agent(submit(), search(), estimate(), submit())
    agent.run(claim(), Usage())
    refused = [m for m in llm.requests[1]["messages"] if m.get("role") == "tool"]
    assert "Cannot finish yet" in refused[-1]["content"]


def test_fabricated_citation_is_dropped():
    agent, _ = make_agent(search(), estimate(), submit(cited_claim_ids=["CLM-9999-999"]))
    cited = [c["claim_id"] for c in agent.run(claim(), Usage()).assessment["similar_past_claims"]]
    assert "CLM-9999-999" not in cited and cited  # falls back to retrieved claims


def test_policy_cannot_be_substituted_by_model():
    agent, _ = make_agent(search(), estimate(), coverage(policy="premium_homeowner"), submit())
    outcome = agent.run(claim(policy="renters"), Usage())
    act = [s for s in outcome.trace.steps if s.get("tool_name") == "check_coverage"][0]
    assert act["tool_input"]["policy_type"] == "renters"
    notes = outcome.assessment["handling"]["advisory_notes"]
    assert any("does not list 'water_damage' for renters" in n for n in notes)


def test_text_only_is_low_confidence_and_requests_photos():
    agent, _ = make_agent(search(), estimate(), submit())
    handling = agent.run(claim(photos=False, supplied=False), Usage()).assessment["handling"]
    assert handling["confidence"] == "low" and handling["route"] == "standard"
    assert any("upload photos" in r for r in handling["claimant_requests"])
    assert any("Policy type not supplied" in n for n in handling["advisory_notes"])


def test_damage_type_without_precedent_lowers_confidence_without_routing():
    agent, _ = make_agent(search("car hit fence"), estimate(damage_type="other", areas=("fence",)), submit())
    handling = agent.run(claim(), Usage()).assessment["handling"]
    assert handling["confidence"] == "medium" and handling["route"] == "standard"
    assert any("new kind of claim" in f for f in handling["confidence_factors"])


def test_invalid_tool_input_is_observed_as_error():
    agent, llm = make_agent(search(), estimate(severity="extreme"), estimate(call_id="c5"), submit())
    outcome = agent.run(claim(), Usage())
    errors = [s for s in outcome.trace.steps if s["type"] == "observe" and "error" in s["tool_output"]]
    assert errors and "severity must be one of" in errors[0]["tool_output"]["error"]


def test_agent_without_estimate_fails():
    agent, _ = make_agent(search(), search(), submit(), max_turns=1)
    with pytest.raises(TriageError):
        agent.run(claim(), Usage())


def test_end_to_end_http(client, fake_llm):
    fake_llm.queue(intake_reply(), search(), estimate(), coverage(), submit())
    response = client.post("/api/v1/claims/triage",
                           files={"image": ("leak.jpg", jpeg_bytes(), "image/jpeg"),
                                  "description": (None, "Burst pipe under kitchen sink"),
                                  "policy_type": (None, "standard_homeowner")})
    assert response.status_code == 200, response.text
    report = response.json()
    assert all(report["safety_checks"].values())
    claim_id = report["claim_id"]
    assert client.get(f"/api/v1/claims/{claim_id}").json() == report
    trace = client.get(f"/api/v1/agent/trace/{claim_id}").json()
    assert trace["claim_id"] == claim_id and trace["total_duration_ms"] >= 0
    assert_valid_trace(trace["steps"], report["triage_assessment"])
