"""Enumerations shared by the API contract, the tools and the agents."""

DAMAGE_TYPES = ("water_damage", "fire_damage", "storm_damage", "break_in", "vandalism", "other")
SEVERITIES = ("minor", "moderate", "severe", "catastrophic")
URGENCIES = ("low", "medium", "high", "critical")
POLICY_TYPES = ("basic_homeowner", "standard_homeowner", "premium_homeowner", "renters", "landlord")
DEFAULT_POLICY_TYPE = "standard_homeowner"
