"""Deterministic lookup tools over the cost and coverage tables (no model involved)."""
import json
from pathlib import Path

from ..domain import DAMAGE_TYPES, POLICY_TYPES, SEVERITIES
from .areas import normalize_areas


class ToolError(ValueError):
    """Invalid tool input; reported back to the agent as an observation."""


class CostEstimator:
    def __init__(self, table_path: Path):
        table = json.loads(table_path.read_text(encoding="utf-8-sig"))
        self.matrix = table["cost_matrix"]

    def estimate(self, damage_type: str, severity: str, affected_areas: list[str]) -> dict:
        if damage_type not in DAMAGE_TYPES:
            raise ToolError(f"damage_type must be one of {list(DAMAGE_TYPES)}")
        if severity not in SEVERITIES:
            raise ToolError(f"severity must be one of {list(SEVERITIES)}")
        if not isinstance(affected_areas, list):
            raise ToolError("affected_areas must be a list of strings")
        areas = normalize_areas(affected_areas)
        row = self.matrix[damage_type][severity]
        # Table rule: per-area range x max(1, distinct areas).
        count = max(1, len(areas))
        return {
            "min": row["min_per_area"] * count,
            "max": row["max_per_area"] * count,
            "currency": "USD",
            "area_count": count,
            "per_area": {"min": row["min_per_area"], "max": row["max_per_area"]},
        }


class CoverageChecker:
    def __init__(self, table_path: Path):
        table = json.loads(table_path.read_text(encoding="utf-8-sig"))
        self.policies = table["policies"]
        self.excluded = set(table["excluded_everywhere"])

    @property
    def categories(self) -> list[str]:
        return list(DAMAGE_TYPES) + sorted(self.excluded)

    def check(self, policy_type: str, incident_category: str) -> dict:
        if policy_type not in POLICY_TYPES:
            raise ToolError(f"policy_type must be one of {list(POLICY_TYPES)}")
        if incident_category not in self.categories:
            raise ToolError(f"incident_category must be one of {self.categories}")
        policy = self.policies[policy_type]
        if incident_category in self.excluded:
            covered = False
            notes = f"'{incident_category}' is excluded under every policy type. {policy['notes']}"
        else:
            covered = incident_category in policy["covered_categories"]
            notes = policy["notes"]
        return {
            "policy_type": policy_type,
            "incident_category": incident_category,
            "covered": covered,
            "deductible": policy["deductible"],
            "max_payout": policy["max_payout"],
            "notes": notes,
        }
