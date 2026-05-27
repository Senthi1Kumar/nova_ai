"""Local extension: Mapbox MCP-backed map / location tool wrappers.

Not part of the core voice loop. Loaded opportunistically by `app.main` —
if this file (or `app.misc`) is missing, the app falls back to the
Brave + Serper web search stack only. Re-enable by importing the desired
symbols into `app.litert_service.DEFAULT_TOOLS`.
"""
from __future__ import annotations

import base64
import json
import logging
import re
import time
import uuid
from pathlib import Path

from app.config import get_settings
from app.mcp_bridge import get_bridge as _get_mcp_bridge

log = logging.getLogger("litert_app.misc.mapbox")

_MCP_RESULT_CAP = 2000  # chars; protects small-model context from huge POI dumps


def _mcp_call(tool_name: str, args: dict, cap: int = _MCP_RESULT_CAP) -> str:
    bridge = _get_mcp_bridge()
    if bridge is None or not bridge.ready:
        return "Map tool unavailable: MCP bridge not connected."
    try:
        raw = bridge.call_tool(tool_name, args, timeout=20.0)
    except Exception as exc:
        log.exception("MCP %s failed", tool_name)
        return f"Map tool {tool_name} failed: {exc}"
    if cap and len(raw) > cap:
        log.warning("MCP %s result truncated: %d → %d chars", tool_name, len(raw), cap)
        return raw[:cap] + f"\n…(truncated; original {len(raw)} chars)"
    return raw


# Best-effort extractors for a (longitude, latitude) pair from Mapbox-style JSON.
_RE_LNG_LAT = re.compile(
    r'"(?:longitude|lng|lon)"\s*:\s*(-?\d+(?:\.\d+)?)[\s\S]{0,80}?'
    r'"(?:latitude|lat)"\s*:\s*(-?\d+(?:\.\d+)?)',
    re.IGNORECASE,
)
_RE_LAT_LNG = re.compile(
    r'"(?:latitude|lat)"\s*:\s*(-?\d+(?:\.\d+)?)[\s\S]{0,80}?'
    r'"(?:longitude|lng|lon)"\s*:\s*(-?\d+(?:\.\d+)?)',
    re.IGNORECASE,
)
_RE_COORDS_GEOJSON = re.compile(
    r'"coordinates"\s*:\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]'
)


def _extract_lng_lat(text: str) -> tuple[float, float] | None:
    """Pull the first (longitude, latitude) pair out of a Mapbox response blob."""
    m = _RE_LNG_LAT.search(text)
    if m:
        return float(m.group(1)), float(m.group(2))
    m = _RE_LAT_LNG.search(text)
    if m:
        return float(m.group(2)), float(m.group(1))   # swap to (lng, lat)
    m = _RE_COORDS_GEOJSON.search(text)
    if m:
        return float(m.group(1)), float(m.group(2))   # GeoJSON is [lng, lat]
    return None


def _save_image_b64(data_b64: str, mime: str) -> str:
    """Write a base64-encoded image payload to runtime/maps/ and return its
    public /maps/<file>.<ext> URL."""
    ext = "png"
    if "/" in mime:
        ext = mime.split("/", 1)[1].split(";", 1)[0] or "png"
        if ext == "jpeg":
            ext = "jpg"
    out_dir = Path(get_settings().maps_output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"map_{int(time.time())}_{uuid.uuid4().hex[:8]}.{ext}"
    out_path = out_dir / name
    try:
        out_path.write_bytes(base64.b64decode(data_b64))
    except Exception:
        log.exception("failed to decode/save MCP image")
        raise
    return f"/maps/{name}"


def _auto_static_map_from_coords(lng: float, lat: float, zoom: int = 14) -> str | None:
    """Render a static map centered on (lng, lat). Returns /maps/<file>.png URL or None.

    Probes 4 known `center` shape variants to survive Mapbox MCP schema changes.
    """
    candidates = (
        {"center": {"longitude": lng, "latitude": lat}, "zoom": int(zoom)},
        {"center": {"lon": lng, "lat": lat}, "zoom": int(zoom)},
        {"center": {"lng": lng, "lat": lat}, "zoom": int(zoom)},
        {"center": [lng, lat], "zoom": int(zoom)},
    )
    for args in candidates:
        try:
            raw = _mcp_call("static_map_image_tool", args, cap=0)
        except Exception as exc:
            log.warning("static_map_image_tool call raised for args=%s: %s",
                        list(args.keys()), exc)
            continue
        if "Input validation error" in raw or "invalid_type" in raw:
            continue
        for line in raw.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("image") and obj.get("data_b64"):
                return _save_image_b64(obj["data_b64"], obj.get("mime", "image/png"))
    return None


def _auto_static_map(text_blob: str, zoom: int = 14) -> str | None:
    """Convenience: pull (lng, lat) from an arbitrary response blob and render a map."""
    coords = _extract_lng_lat(text_blob)
    if not coords:
        return None
    return _auto_static_map_from_coords(coords[0], coords[1], zoom=zoom)


# ── public tool wrappers ─────────────────────────────────────────────────────

def mapbox_search_and_geocode(query: str) -> str:
    """Find a place by name and return its address plus latitude / longitude.

    Use this whenever the user names a real-world place ("the Eiffel Tower",
    "MG Road Bangalore") and you need its coordinates or canonical address.
    NOTE: This tool automatically also renders a static map of the top result.

    Args:
        query: Natural-language place name or address.

    Returns:
        JSON string with the matched place + image_url (auto-rendered map).
    """
    geocode_raw = _mcp_call("search_and_geocode_tool", {"q": query})
    map_url = _auto_static_map(query)
    if not map_url:
        return geocode_raw
    return json.dumps({
        "geocode": geocode_raw[:1200],
        "image_url": map_url,
        "note": "A map of this place is now shown in the chat.",
    }, ensure_ascii=False)


def mapbox_reverse_geocode(longitude: float, latitude: float) -> str:
    """Convert a longitude / latitude coordinate to the nearest street address."""
    return _mcp_call("reverse_geocode_tool",
                     {"longitude": longitude, "latitude": latitude})


def mapbox_directions(origin: str, destination: str, profile: str = "driving") -> str:
    """Turn-by-turn directions and travel time between two places (+ auto map)."""
    directions_raw = _mcp_call(
        "directions_tool",
        {"origin": origin, "destination": destination, "profile": profile},
    )
    map_url = _auto_static_map(f"{origin} to {destination}", zoom=11)
    if not map_url:
        return directions_raw
    return json.dumps({
        "directions": directions_raw[:1500],
        "image_url": map_url,
        "note": "A route overview map is now shown in the chat.",
    }, ensure_ascii=False)


def mapbox_isochrone(longitude: float, latitude: float, minutes: int = 15,
                     profile: str = "driving") -> str:
    """Compute the area reachable from a coordinate within N minutes."""
    return _mcp_call(
        "isochrone_tool",
        {
            "coordinates": {"longitude": longitude, "latitude": latitude},
            "contours_minutes": [minutes],
            "profile": profile,
        },
    )


def mapbox_matrix(origins: list[str], destinations: list[str],
                  profile: str = "driving") -> str:
    """Pairwise travel times / distances between many origins and destinations."""
    return _mcp_call(
        "matrix_tool",
        {"origins": origins, "destinations": destinations, "profile": profile},
    )


def mapbox_category_search(category: str, near: str, limit: int = 5) -> str:
    """Look up administrative geographies — countries, regions, postal codes — near a place."""
    return _mcp_call(
        "category_search_tool",
        {"category": category, "proximity": near, "limit": max(1, min(int(limit), 10))},
    )


def mapbox_ground_location(longitude: float, latitude: float) -> str:
    """Describe what's at and around a coordinate — neighborhood, nearby POIs, area summary."""
    return _mcp_call(
        "ground_location_tool",
        {"longitude": longitude, "latitude": latitude},
    )


def mapbox_place_details(mapbox_id: str) -> str:
    """Fetch rich details (hours, phone, website, photos, ratings) for a place by its ID."""
    return _mcp_call("place_details_tool", {"mapbox_id": mapbox_id})


def mapbox_static_map(center: str, zoom: int = 13, markers: list[str] | None = None,
                      style: str = "streets") -> str:
    """Generate a static map image of a place and surface it in the chat UI."""
    args: dict = {"center": center, "zoom": int(zoom), "style": style}
    if markers:
        args["markers"] = markers[:10]
    raw = _mcp_call("static_map_image_tool", args)
    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if obj.get("image") and obj.get("data_b64"):
            try:
                url = _save_image_b64(obj["data_b64"], obj.get("mime", "image/png"))
            except Exception as exc:
                return json.dumps({"error": f"map render failed: {exc}"})
            return json.dumps({
                "image_url": url,
                "center": center,
                "zoom": zoom,
                "markers": markers or [],
                "note": "Map image rendered and shown to the user.",
            }, ensure_ascii=False)
    return raw or '{"error": "no image returned by static_map_image_tool"}'


def mapbox_optimize_route(stops: list[str], profile: str = "driving") -> str:
    """Find the optimal visiting order for a list of places (TSP)."""
    if len(stops) < 3:
        return "Map tool needs at least 3 stops for optimization."
    return _mcp_call(
        "optimization_tool",
        {"waypoints": stops[:12], "profile": profile},
    )


def mapbox_map_match(
    coordinates: list[list[float]],
    profile: str = "driving",
    timestamps: list[int] | None = None,
) -> str:
    """Snap a noisy GPS trace to the underlying road / path network."""
    args: dict = {"coordinates": coordinates[:100], "profile": profile}
    if timestamps:
        args["timestamps"] = timestamps[:100]
    return _mcp_call("map_matching_tool", args)
