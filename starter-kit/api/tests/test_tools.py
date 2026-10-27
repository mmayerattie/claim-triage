"""Lookup tools and area counting; pure functions over the cost and coverage tables."""
import pytest

from app.config import load_settings
from app.tools import CostEstimator, CoverageChecker, ToolError, normalize_areas

TOOLS_DIR = load_settings().tools_dir
pytestmark = pytest.mark.skipif(not TOOLS_DIR.is_dir(), reason="tool tables not mounted")


@pytest.fixture(scope="module")
def cost():
    return CostEstimator(TOOLS_DIR / "cost-estimator.json")


@pytest.fixture(scope="module")
def coverage():
    return CoverageChecker(TOOLS_DIR / "coverage-table.json")


@pytest.mark.parametrize("raw, expected", [
    (["kitchen", "Kitchen", "kitchen "], ["kitchen"]),                       # repeated views
    (["kitchen", "kitchen floor", "Kitchen Walls"], ["kitchen"]),            # room + its own surfaces
    (["Living Room", "living-room", "living area"], ["living_room"]),       # spelling and synonyms
    (["window", "windows"], ["windows"]),                                    # singular/plural
    (["kitchen", "Dining Room"], ["kitchen", "dining_room"]),               # distinct rooms kept, order kept
    (["exterior_wall", "fence"], ["exterior_wall", "fence"]),                # exterior component is not a room surface
    (["roof", "interior", "exterior"], ["roof", "interior", "exterior"]),    # generic labels: agent's judgement
    (["", "  ", "garage"], ["garage"]),
])
def test_normalize_areas(raw, expected):
    assert normalize_areas(raw) == expected


def test_inspection_scope_two_rooms(cost):
    """IMG_001 + inspection PDF: moderate water damage in kitchen and dining room."""
    result = cost.estimate("water_damage", "moderate", ["kitchen", "dining room"])
    assert (result["min"], result["max"], result["currency"]) == (4000, 16000, "USD")


def test_duplicate_views_do_not_multiply_cost(cost):
    single = cost.estimate("water_damage", "moderate", ["kitchen"])
    repeated = cost.estimate("water_damage", "moderate", ["kitchen", "kitchen", "kitchen floor"])
    assert (single["min"], single["max"]) == (repeated["min"], repeated["max"]) == (2000, 8000)


def test_no_areas_counts_as_one(cost):
    result = cost.estimate("fire_damage", "minor", [])
    assert (result["min"], result["max"], result["area_count"]) == (1000, 5000, 1)


def test_cost_matches_area_formula_on_normalized_areas(cost):
    """Cost is per-area x len(set(lowercased inputs)), so the trace can be re-checked."""
    areas = normalize_areas(["Roof", "attic", "roof"])
    result = cost.estimate("storm_damage", "severe", areas)
    count = max(1, len({a.strip().lower() for a in areas}))
    assert (result["min"], result["max"]) == (6000 * count, 20000 * count) == (12000, 40000)


@pytest.mark.parametrize("damage_type, severity", [("flood", "minor"), ("water_damage", "extreme")])
def test_invalid_cost_inputs_rejected(cost, damage_type, severity):
    with pytest.raises(ToolError):
        cost.estimate(damage_type, severity, ["kitchen"])


def test_covered_category(coverage):
    result = coverage.check("standard_homeowner", "water_damage")
    assert result["covered"] is True
    assert (result["deductible"], result["max_payout"]) == (1500, 250000)


def test_uncovered_category(coverage):
    """Renters policies do not list water damage."""
    assert coverage.check("renters", "water_damage")["covered"] is False


def test_excluded_everywhere(coverage):
    result = coverage.check("premium_homeowner", "flood")
    assert result["covered"] is False
    assert "excluded under every policy" in result["notes"]


def test_unknown_policy_rejected(coverage):
    with pytest.raises(ToolError):
        coverage.check("gold_plated", "water_damage")
