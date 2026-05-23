from __future__ import annotations

import base64
import json
import logging
import os
import re
import time
import uuid
from pathlib import Path
from typing import Optional

from tavily import TavilyClient

from app.config import get_settings
from app.mcp_bridge import get_bridge as _get_mcp_bridge

log = logging.getLogger("litert_app.tools")

_client: Optional[TavilyClient] = None


def _get_client() -> Optional[TavilyClient]:
    global _client
    if _client is not None:
        return _client
    # Prefer shell env (lets users override .env without restarting), fall back
    # to Settings (which reads .env). pydantic-settings does NOT push .env vars
    # into os.environ, so reading os.environ alone misses .env-only keys.
    key = os.environ.get("TAVILY_API_KEY") or get_settings().tavily_api_key
    key = (key or "").strip().strip('"').strip("'")
    if not key:
        log.warning("Tavily disabled: TAVILY_API_KEY not set in env or .env")
        return None
    # tail = key[-4:] if len(key) >= 4 else "***"
    log.info("Tavily client initialized with API key")
    _client = TavilyClient(api_key=key)
    return _client


def tavily_search(query: str, depth: str = "basic") -> str:
    """Search the live web for up-to-date information (news, weather, prices, current events, recent facts).

    Call this whenever the user's question depends on information that may have
    changed after your training cutoff, or asks about current/recent events,
    today's news, live prices, weather, sports scores, or anything time-sensitive.
    Do NOT call for math, code, general knowledge already in your training data,
    or casual chitchat.

    Args:
        query: The search query. Be specific. Include location and date if relevant.
        depth: "basic" (1 credit, faster) or "advanced" (2 credits, deeper extraction).

    Returns:
        A JSON string with keys: answer (synthesized direct answer) and sources
        (list of {title, url}, top 3).
    """
    log.info("tavily_search invoked: query=%r depth=%r", query, depth)
    client = _get_client()
    if client is None:
        return "Web search unavailable: TAVILY_API_KEY not configured."
    try:
        data = client.search(
            query=query,
            search_depth=depth,
            include_answer="advanced",
            max_results=3,
        )
        answer = (data.get("answer") or "").strip()
        sources = [
            {"title": s.get("title"), "url": s.get("url")}
            for s in (data.get("results") or [])[:3]
        ]
        result = json.dumps({"answer": answer, "sources": sources}, ensure_ascii=False)
        log.info("tavily_search ok: answer_len=%d sources=%d", len(answer), len(sources))
        return result
    except Exception as exc:
        log.exception("tavily_search failed")
        return f"Web search failed: {exc}"


def tavily_extract(url: str) -> str:
    """Fetch and return the cleaned full-text content of a single URL.

    Call this when the user gives you a specific URL to read, or when an earlier
    tavily_search result is too shallow and you want the full article body.
    Do NOT call repeatedly on the same URL within one turn.

    Args:
        url: The exact URL to extract content from (must start with http:// or https://).

    Returns:
        A JSON string with keys: url, content (cleaned markdown of the page),
        or {"error": "..."} on failure.
    """
    log.info("tavily_extract invoked: url=%r", url)
    client = _get_client()
    if client is None:
        return "Web extract unavailable: TAVILY_API_KEY not configured."
    try:
        data = client.extract(urls=url, extract_depth="basic", format="markdown")
        results = data.get("results") or []
        if not results:
            failed = data.get("failed_results") or []
            err = failed[0].get("error") if failed else "no content extracted"
            return json.dumps({"error": err, "url": url})
        first = results[0]
        content = (first.get("raw_content") or "")[:8000]   # cap to keep prefill cheap
        log.info("tavily_extract ok: url=%s content_len=%d", url, len(content))
        return json.dumps({"url": first.get("url", url), "content": content},
                          ensure_ascii=False)
    except Exception as exc:
        log.exception("tavily_extract failed")
        return f"Web extract failed: {exc}"


def tavily_research(query: str) -> str:
    """Run a deeper two-step research pass on a topic: search + extract the top hit.

    Use this when the user asks for in-depth information ("research X for me",
    "give me a deep dive on Y") and a one-shot tavily_search snippet isn't enough.
    Cost: ~3 Tavily credits per call (1 search + 2 extracts). Slower than search.
    Do NOT call for quick facts — use tavily_search for those.

    Args:
        query: The research topic. Be specific.

    Returns:
        A JSON string with keys: query, answer (synthesized direct answer),
        sources (list of {title, url, content}), each with up to ~4000 chars
        of extracted content for the top 2 sources.
    """
    log.info("tavily_research invoked: query=%r", query)
    client = _get_client()
    if client is None:
        return "Web research unavailable: TAVILY_API_KEY not configured."
    try:
        search = client.search(
            query=query,
            search_depth="advanced",
            include_answer="advanced",
            max_results=4,
        )
        answer = (search.get("answer") or "").strip()
        top = (search.get("results") or [])[:2]
        urls = [r.get("url") for r in top if r.get("url")]
        sources = []
        if urls:
            try:
                ext = client.extract(urls=urls, extract_depth="basic", format="markdown")
                by_url = {r.get("url"): r for r in (ext.get("results") or [])}
                for r in top:
                    u = r.get("url")
                    extracted = by_url.get(u, {})
                    sources.append({
                        "title": r.get("title"),
                        "url": u,
                        "content": (extracted.get("raw_content") or r.get("content") or "")[:4000],
                    })
            except Exception as exc:
                log.warning("tavily_research extract step failed: %s", exc)
                sources = [
                    {"title": r.get("title"), "url": r.get("url"), "content": r.get("content", "")}
                    for r in top
                ]
        log.info("tavily_research ok: answer_len=%d sources=%d",
                 len(answer), len(sources))
        return json.dumps({"query": query, "answer": answer, "sources": sources},
                          ensure_ascii=False)
    except Exception as exc:
        log.exception("tavily_research failed")
        return f"Web research failed: {exc}"


_MCP_RESULT_CAP = 2000  # chars; protects small-model context from huge POI dumps


def _mcp_call(tool_name: str, args: dict, cap: int = _MCP_RESULT_CAP) -> str:
    bridge = _get_mcp_bridge()
    if bridge is None or not bridge.ready:
        return "Map tool unavailable: Mapbox MCP not connected."
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
        # GeoJSON convention is [lng, lat].
        return float(m.group(1)), float(m.group(2))
    return None


def _auto_static_map_from_coords(lng: float, lat: float, zoom: int = 14) -> str | None:
    """Render a static map centered on (lng, lat). Returns /maps/<file>.png URL or None.

    Mapbox's static_map_image_tool wants `center` as an object — most schema
    versions accept {longitude, latitude}, some want {lon, lat}. We try the
    common shape first; if the MCP server returns a validation error we fall
    back to alternates.
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
        # The bridge prefixes input-validation errors with "MCP error" text; skip
        # to the next candidate so we can keep probing the schema.
        if "Input validation error" in raw or "invalid_type" in raw:
            log.info("static_map_image_tool rejected center shape %s — trying next",
                     type(args["center"]).__name__)
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
    """Convenience: pull (lng, lat) from an arbitrary Mapbox-response text blob
    and render a map. Returns None if no coords found or the render failed."""
    coords = _extract_lng_lat(text_blob)
    if not coords:
        log.info("no lng/lat found in geocode response — skipping auto-map")
        return None
    return _auto_static_map_from_coords(coords[0], coords[1], zoom=zoom)


def mapbox_search_and_geocode(query: str) -> str:
    """Find a place by name and return its address plus latitude / longitude.

    Use this whenever the user names a real-world place ("the Eiffel Tower",
    "MG Road Bangalore", "Stanford coffee shops near campus") and you need its
    coordinates or canonical address — for example before calling
    mapbox_directions or mapbox_isochrone, both of which require coordinates.
    Do NOT call this for fictional places or for time-sensitive info that
    tavily_search would handle better.

    NOTE: This tool automatically also renders a static map of the top result
    in the chat UI — you do not need to call mapbox_static_map separately. In
    your reply, mention that the map is now visible.

    Args:
        query: Natural-language place name or address. Be specific.

    Returns:
        JSON string with the matched place's name, address, longitude, latitude,
        and image_url (the auto-rendered map shown to the user).
    """
    geocode_raw = _mcp_call("search_and_geocode_tool", {"q": query})
    map_url = _auto_static_map(query)
    if not map_url:
        return geocode_raw
    # Wrap the original geocode text + the rendered map URL in one JSON object.
    # Cap geocode text to keep small-model context small; image_url is what
    # the frontend uses to render.
    return json.dumps({
        "geocode": geocode_raw[:1200],
        "image_url": map_url,
        "note": "A map of this place is now shown in the chat.",
    }, ensure_ascii=False)


def mapbox_reverse_geocode(longitude: float, latitude: float) -> str:
    """Convert a longitude / latitude coordinate to the nearest street address.

    Use this when the user gives you raw coordinates or when an earlier tool
    returned coordinates and you need a human-readable address to reply with.

    Args:
        longitude: Longitude in decimal degrees (range -180 to 180).
        latitude: Latitude in decimal degrees (range -90 to 90).

    Returns:
        JSON string with the matched address.
    """
    return _mcp_call("reverse_geocode_tool",
                     {"longitude": longitude, "latitude": latitude})


def mapbox_directions(origin: str, destination: str, profile: str = "driving") -> str:
    """Get turn-by-turn directions and travel time between two places.

    Pass place names directly — the tool geocodes them internally. Use this when
    the user asks "how do I get from X to Y", "how far is X from Y", or
    "drive / walk / bike / cycle from X to Y".

    Args:
        origin: Starting place name or address (e.g. "MG Road, Bangalore").
        destination: Destination place name or address.
        profile: Travel mode — "driving" (default), "driving-traffic", "walking", or "cycling".

    Returns:
        JSON string with distance (meters), duration (seconds), and a turn list.
    """
    directions_raw = _mcp_call(
        "directions_tool",
        {"origin": origin, "destination": destination, "profile": profile},
    )
    # Auto-render a map showing both endpoints — small-model can't reliably
    # chain a follow-up mapbox_static_map call, so we do it here.
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
    """Compute the area reachable from a coordinate within N minutes.

    Use this for "what's within a 15-minute drive of X", "show me everywhere
    I can walk to in 20 minutes from Y". Get coordinates from
    mapbox_search_and_geocode first if the user gave you a place name.

    Args:
        longitude: Center point longitude.
        latitude: Center point latitude.
        minutes: Travel time budget in minutes (1-60).
        profile: "driving", "walking", or "cycling".

    Returns:
        JSON string with a GeoJSON polygon describing the reachable area.
    """
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
    """Compute pairwise travel times / distances between many origins and destinations.

    Use this when the user asks something like "which of these three places is
    fastest to reach from my office" or "show me travel times from home to
    each of these restaurants". Place names are geocoded internally.

    Args:
        origins: List of origin place names or addresses.
        destinations: List of destination place names or addresses.
        profile: "driving" (default), "walking", or "cycling".

    Returns:
        JSON string with a matrix of durations (seconds) and distances (meters).
    """
    return _mcp_call(
        "matrix_tool",
        {"origins": origins, "destinations": destinations, "profile": profile},
    )


def mapbox_category_search(category: str, near: str, limit: int = 5) -> str:
    """Look up ADMINISTRATIVE geographies — countries, regions, postal codes — near a place.

    NOTE: This is NOT for POI category lookups. For "coffee shops near X" /
    "restaurants near Y" / "gas stations along the way", use
    mapbox_search_and_geocode with a natural-language query like "coffee shops
    near Empire State Building" — search_and_geocode handles POIs natively.

    Use this tool for "what country is at this lat/lng", "what postal code is
    in this neighborhood", or "list the regions around Tokyo".

    Args:
        category: Admin category — "country", "region", "postcode", "district",
                  "place", "locality", "neighborhood".
        near: A place name used as the proximity center for ranking results.
        limit: Max results (1-10). Default 5.

    Returns:
        JSON string listing matched admin geographies.
    """
    return _mcp_call(
        "category_search_tool",
        {"category": category, "proximity": near, "limit": max(1, min(int(limit), 10))},
    )


def mapbox_ground_location(longitude: float, latitude: float) -> str:
    """Describe what's at and around a coordinate — neighborhood, nearby POIs, area summary.

    Best tool for "what's around me at these coords", "tell me about this
    location", "what kind of area is this", or as the first tool when you've
    been given raw coordinates and need to ground the rest of the conversation
    in real-world context.

    Args:
        longitude: Longitude in decimal degrees.
        latitude: Latitude in decimal degrees.

    Returns:
        JSON string with neighborhood context, nearby POIs by category, and
        travel-time reachability info.
    """
    return _mcp_call(
        "ground_location_tool",
        {"longitude": longitude, "latitude": latitude},
    )


def mapbox_place_details(mapbox_id: str) -> str:
    """Fetch rich details (hours, phone, website, photos, ratings) for a place by its Mapbox ID.

    Call this AFTER mapbox_search_and_geocode or mapbox_category_search return
    a Mapbox ID and the user asks about that place's hours, phone, photos, or
    ratings. Don't call without an ID — geocode first.

    Args:
        mapbox_id: The Mapbox feature ID returned by a prior search/geocode call.

    Returns:
        JSON string with full place metadata.
    """
    return _mcp_call("place_details_tool", {"mapbox_id": mapbox_id})


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


def mapbox_static_map(center: str, zoom: int = 13, markers: list[str] | None = None,
                      style: str = "streets") -> str:
    """Generate a static map image of a place and surface it in the chat UI.

    Use this whenever the user asks to SEE / SHOW / DISPLAY / VISUALIZE a map,
    a location, a route, or a set of points on a map. Phrases like "show me a
    map of X", "satellite view of Y", "highlight these on a map", "what does
    that area look like" — all should trigger this tool. Do NOT just describe
    the map in words; call this tool so the actual image renders.

    Args:
        center: Place name to center the map on (e.g. "Golden Gate Bridge").
        zoom: Zoom level 0-22. 13 ≈ neighborhood, 16 ≈ block, 10 ≈ city.
        markers: Optional list of place names to drop pins on (max 10).
        style: Mapbox style — "streets" (default), "satellite", "outdoors", "dark", "light".

    Returns:
        JSON string with image_url (the /maps/<file>.png served by this app) and
        a short note. The chat UI renders the URL inline as an image bubble.
    """
    args: dict = {"center": center, "zoom": int(zoom), "style": style}
    if markers:
        args["markers"] = markers[:10]
    raw = _mcp_call("static_map_image_tool", args)
    # The bridge returns one or more lines; the image payload is a JSON object
    # with {"image": true, "mime": "...", "data_b64": "..."} on its own line.
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
            log.info("mapbox_static_map saved image: %s", url)
            return json.dumps({
                "image_url": url,
                "center": center,
                "zoom": zoom,
                "markers": markers or [],
                "note": "Map image rendered and shown to the user.",
            }, ensure_ascii=False)
    # No image returned — pass through whatever the server gave us.
    return raw or '{"error": "no image returned by static_map_image_tool"}'


def mapbox_optimize_route(stops: list[str], profile: str = "driving") -> str:
    """Find the optimal visiting order for a list of places ("travelling salesman").

    Use this when the user asks to optimize a multi-stop route — visiting
    several tourist spots, planning a delivery, doing errands. The tool returns
    the best order plus the full route. Place names are geocoded internally.

    Args:
        stops: Ordered list of place names. The first stop is treated as the start;
               by default the last stop is the end. Minimum 3, maximum 12.
        profile: "driving" (default), "walking", or "cycling".

    Returns:
        JSON string with the optimized stop order, total distance, total duration.
    """
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
    """Snap a noisy GPS trace to the underlying road / path network.

    Use this when the user provides a list of raw GPS points (from a phone, bike
    computer, dashcam) and wants the clean, road-aligned route. Returns the
    matched geometry and confidence.

    Args:
        coordinates: List of [longitude, latitude] pairs, in chronological order.
                     Minimum 2, maximum 100 points.
        profile: "driving" (default), "walking", or "cycling".
        timestamps: Optional list of Unix timestamps (seconds) per coordinate.

    Returns:
        JSON string with the snapped geometry and per-segment confidence.
    """
    args: dict = {"coordinates": coordinates[:100], "profile": profile}
    if timestamps:
        args["timestamps"] = timestamps[:100]
    return _mcp_call("map_matching_tool", args)


def add_numbers(a: float, b: float) -> float:
    """Adds two numbers.

    Args:
        a: The first number.
        b: The second number.
    """
    return a + b
