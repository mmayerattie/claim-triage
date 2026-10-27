"""Affected-area normalisation: count each distinct room or exterior component once.

Code handles only unambiguous duplicates:
  * spelling variants ("Living Room", "living-room" -> "living_room")
  * a room plus its own surface ("kitchen floor", "basement walls" -> the room)
  * a small set of synonyms and singular/plural forms
Whether a generic label ("interior", "exterior") names a separate region is a
judgement left to the agent's prompt, because it depends on what the evidence shows.
"""
import re

ROOMS = {
    "kitchen", "bathroom", "bedroom", "living_room", "dining_room", "basement", "garage",
    "laundry_room", "attic", "hallway", "crawl_space", "office", "family_room", "den", "entryway",
}
SURFACES = ("floor", "floors", "flooring", "subfloor", "wall", "walls", "ceiling", "ceilings", "cabinets", "cabinetry")
ALIASES = {
    "living_area": "living_room", "lounge": "living_room",
    "dining_area": "dining_room",
    "bath": "bathroom", "washroom": "bathroom", "restroom": "bathroom",
    "laundry": "laundry_room",
    "window": "windows", "gutter": "gutters", "door": "doors",
}


def canonical(area: str) -> str:
    name = re.sub(r"[^a-z0-9]+", "_", area.strip().lower()).strip("_")
    name = ALIASES.get(name, name)
    for surface in SURFACES:
        suffix = "_" + surface
        if name.endswith(suffix) and name[: -len(suffix)] in ROOMS | set(ALIASES):
            name = ALIASES.get(name[: -len(suffix)], name[: -len(suffix)])
            break
    return name


def normalize_areas(areas: list[str]) -> list[str]:
    """Canonical names, first-seen order, duplicates removed, blanks dropped."""
    seen: list[str] = []
    for area in areas:
        name = canonical(area) if isinstance(area, str) else ""
        if name and name not in seen:
            seen.append(name)
    return seen
