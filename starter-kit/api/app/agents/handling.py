"""Claim handling: who acts next and how much to trust the triage.

Every claim gets a full triage. Instead of one "needs human review" switch,
each situation gets the response that resolves it:
  * catastrophic severity        -> routed to a senior adjuster
  * no photo                     -> low confidence + request photos from the claimant
  * evidence conflict            -> medium confidence + specific claimant question
  * no or weak precedent         -> medium confidence (estimate relies on the cost table)
  * everything else              -> advisory notes, no alarm
Route and confidence are computed from these rules, never self-reported by the model.
"""
MAX_CLAIMANT_QUESTIONS = 2
MAX_MODEL_NOTES = 3
PHOTO_REQUEST = "Please upload photos of each affected area so the damage can be verified."


def _clean(items, limit: int) -> list[str]:
    return [str(item).strip() for item in items or [] if str(item).strip()][:limit]


def build_handling(*, severity: str, damage_type: str, has_photos: bool, policy_supplied: bool,
                   policy_type: str, retrieved: list[dict], coverage: dict | None,
                   claimant_questions: list, model_notes: list) -> dict:
    confidence_factors: list[str] = []
    low_factors: list[str] = []

    if not has_photos:
        low_factors.append("No photos: severity and scope rely on the claimant's description only.")
    if not retrieved:
        confidence_factors.append("No comparable historical claims; the estimate relies on the cost table only.")
    elif not any(c["damage_type"] == damage_type for c in retrieved):
        confidence_factors.append(f"No historical claim of type '{damage_type}'; the closest precedents differ, "
                                  "so this may be a new kind of claim.")
    elif not any(c["match_strength"] == "strong" for c in retrieved):
        best = max(c["similarity_score"] for c in retrieved)
        confidence_factors.append(f"Only weak historical precedents (best similarity {best:.2f}).")

    questions = _clean(claimant_questions, MAX_CLAIMANT_QUESTIONS)
    if questions:
        confidence_factors.append("Evidence sources disagree or the cause is undetermined; claimant input requested.")

    requests = ([PHOTO_REQUEST] if not has_photos else []) + questions

    notes = _clean(model_notes, MAX_MODEL_NOTES)
    if not policy_supplied:
        notes.append(f"Policy type not supplied; coverage notes assume {policy_type}.")
    if coverage and not coverage["covered"]:
        notes.append(f"Coverage table does not list '{coverage['incident_category']}' for "
                     f"{coverage['policy_type']}; eligibility to be confirmed.")

    route = "senior_adjuster" if severity == "catastrophic" else "standard"
    route_reason = ("Catastrophic severity: large losses are owned by a senior adjuster; this report is the "
                    "first-pass assessment." if route == "senior_adjuster" else "Standard adjuster queue.")

    return {
        "route": route,
        "route_reason": route_reason,
        "confidence": "low" if low_factors else ("medium" if confidence_factors else "high"),
        "confidence_factors": low_factors + confidence_factors,
        "claimant_requests": requests,
        "advisory_notes": notes,
    }


def apply_safety_action(handling: dict, action: str, failed_checks: list[str]) -> None:
    """Record what the safety reviewer did to the written text."""
    if action == "rewritten":
        handling["advisory_notes"].append(
            f"Written text was revised by the safety review (initially failed: {', '.join(failed_checks)}).")
    elif action == "template":
        handling["confidence_factors"].append(
            "Written text was replaced by a standard template after failing the safety review.")
        if handling["confidence"] == "high":
            handling["confidence"] = "medium"
