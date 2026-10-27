"""Routing and confidence rules for triaged claims."""
from app.agents.handling import apply_safety_action, build_handling

STRONG = [{"claim_id": "CLM-1", "damage_type": "water_damage", "match_strength": "strong", "similarity_score": 0.8}]
WEAK = [{"claim_id": "CLM-2", "damage_type": "water_damage", "match_strength": "weak", "similarity_score": 0.65}]


def handling(**overrides):
    args = dict(severity="moderate", damage_type="water_damage", has_photos=True, policy_supplied=True,
                policy_type="standard_homeowner", retrieved=STRONG, coverage=None,
                claimant_questions=[], model_notes=[])
    args.update(overrides)
    return build_handling(**args)


def test_typical_claim_is_high_confidence_standard_route():
    h = handling()
    assert (h["route"], h["confidence"]) == ("standard", "high")
    assert h["claimant_requests"] == [] and h["confidence_factors"] == []


def test_catastrophic_routes_to_senior_adjuster():
    h = handling(severity="catastrophic")
    assert h["route"] == "senior_adjuster" and h["confidence"] == "high"


def test_no_precedent_is_medium_not_routed():
    h = handling(retrieved=[])
    assert (h["route"], h["confidence"]) == ("standard", "medium")


def test_weak_precedent_is_medium():
    assert handling(retrieved=WEAK)["confidence"] == "medium"


def test_conflict_question_goes_to_claimant_and_is_capped():
    h = handling(claimant_questions=["Q1?", "Q2?", "Q3?", " "])
    assert h["claimant_requests"] == ["Q1?", "Q2?"] and h["confidence"] == "medium"


def test_missing_photo_is_low_confidence_with_photo_request():
    h = handling(has_photos=False, retrieved=[])
    assert h["confidence"] == "low"
    assert h["claimant_requests"][0].startswith("Please upload photos")


def test_model_notes_are_advisory_only_and_capped():
    h = handling(model_notes=["a", "b", "c", "d"])
    assert h["advisory_notes"] == ["a", "b", "c"]
    assert (h["route"], h["confidence"]) == ("standard", "high")


def test_safety_template_lowers_confidence_rewrite_does_not():
    rewritten, template = handling(), handling()
    apply_safety_action(rewritten, "rewritten", ["no_coverage_promises"])
    apply_safety_action(template, "template", ["no_coverage_promises"])
    assert rewritten["confidence"] == "high" and template["confidence"] == "medium"
