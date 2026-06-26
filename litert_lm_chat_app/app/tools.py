"""Tool registry exposed to the LiteRT-LM engine.

Two flavours live here:

  * Bridge proxies — the active web-search tools (`web_search`,
    `google_search`) forward to the IResearcher FastMCP sidecar via
    `app.mcp_bridge`. Start the sidecar with
    `./.venv/bin/python -m app.tools_v2` in another terminal.

  * Dormant direct-call helpers — Tavily wrappers (`tavily_search`,
    `tavily_extract`, `tavily_research`) hit the Tavily API directly with
    httpx. They're not registered in DEFAULT_TOOLS by default but stay
    importable so flipping back is one line in `app.litert_service`.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Optional

from tavily import TavilyClient

from app.config import get_settings
from app.mcp_bridge import get_bridge as _get_mcp_bridge

log = logging.getLogger("litert_app.tools")


# ── Bridge proxies: IResearcher FastMCP sidecar ──────────────────────────────

# Outer safety net — applied AFTER per-search compaction. Hit only when the
# tool's payload isn't a parseable search-result shape and falls through the
# compactor unchanged. 2500 chars ≈ 625 tokens; compaction (0.50 ratio + tool
# weight tally via TurnRecord.tool_chars) keeps it safe under the 4096-token
# wall.
_MCP_RESULT_CAP = 2500

# Per-search-result caps used by _compact_search_results. Gemma-4-E2B is small
# enough (2.3B effective params) that a richer payload becomes regurgitation
# material — the model copy-pastes titles/dates/URLs verbatim instead of
# synthesising. Capping the model's view to 3 short description fragments
# forces it to compose its own sentences (mechanical, not instruction-based).
# Brave is still called with the user-requested num_results; the sidecar
# pre-trims to its own limit; we trim further on receive to what the model
# sees. The proper fix is constrained-decoding + per-intent fine-tune in v0.2.
_MAX_RESULTS_TO_MODEL = 3
_MAX_DESC_CHARS = 150


def _compact_search_results(raw: str) -> str:
    """Distill a Brave/Serper-style JSON results blob down to just descriptions.

    Returns the original `raw` unchanged on any parse failure or shape it
    doesn't recognise — safe to use on every tool result (including
    add_numbers, error strings, etc.) without breaking them.
    """
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return raw

    candidates: list = []
    if isinstance(data, list):
        candidates = data
    elif isinstance(data, dict):
        for key in ("results", "web", "news", "organic", "items", "shopping_results"):
            v = data.get(key)
            if isinstance(v, list) and v:
                candidates = v
                break

    if not candidates or not any(
        isinstance(r, dict) and (r.get("description") or r.get("snippet") or r.get("content"))
        for r in candidates
    ):
        return raw

    compact: list[dict] = []
    for r in candidates[:_MAX_RESULTS_TO_MODEL]:
        if not isinstance(r, dict):
            continue
        snippet = (r.get("description") or r.get("snippet") or r.get("content") or "").strip()
        if not snippet:
            continue
        compact.append({"snippet": snippet[:_MAX_DESC_CHARS]})

    if not compact:
        return raw

    out = {
        "_hint": ("Synthesise these snippets into 2-3 short spoken sentences. "
                  "Don't quote titles, URLs, dates, or any verbatim phrases."),
        "snippets": compact,
    }
    log.info("compacted search result: %d candidates → %d snippets (was %d chars)",
             len(candidates), len(compact), len(raw))
    return json.dumps(out, ensure_ascii=False)


def _mcp_proxy(tool_name: str, args: dict, label: str) -> str:
    bridge = _get_mcp_bridge()
    if bridge is None or not bridge.ready:
        log.warning("%s called but IResearcher sidecar is not connected", tool_name)
        return (f"{label} unavailable: the IResearcher MCP sidecar is not running. "
                f"Tell the user to start it in another terminal with "
                f"`python -m app.tools_v2`.")
    try:
        raw = bridge.call_tool(tool_name, args, timeout=30.0)
    except Exception as exc:
        log.exception("MCP %s failed", tool_name)
        return f"{label} failed: {exc}"

    # Per-search compaction first (drops titles/URLs/dates, caps to N snippets),
    # then the outer char cap as a safety net for any non-search shape that
    # falls through unchanged.
    raw = _compact_search_results(raw)

    if len(raw) > _MCP_RESULT_CAP:
        log.warning("MCP %s result truncated: %d → %d chars (~%d tokens saved)",
                    tool_name, len(raw), _MCP_RESULT_CAP,
                    (len(raw) - _MCP_RESULT_CAP) // 4)
        return raw[:_MCP_RESULT_CAP] + "\n…(truncated to fit context)"
    return raw


def web_search(query: str, search_type: str = "web", num_results: int = 10) -> str:
    """Search the live web (Brave Search) for snippets from many sources.

    DEFAULT web tool. Use for news, sports, weather, prices, current events,
    single-fact lookups, recent facts — anything that may have changed after
    your training cutoff. Returns keyword-style snippets with titles, URLs,
    and short descriptions from up to ~10 sources.

    Args:
        query: The search query. Be specific.
        search_type: 'web' (default) or 'news' for news-y queries (today's
                     headlines, breaking, latest).
        num_results: Max results (1-20). Default 10.

    Returns:
        JSON string with the matched results.
    """
    log.info("web_search invoked: query=%r type=%r n=%d", query, search_type, num_results)
    return _mcp_proxy(
        "web_search",
        {"query": query, "search_type": search_type, "num_results": int(num_results)},
        label="Web search",
    )


def google_search(query: str, search_type: str = "shopping", gl: str = "in") -> str:
    """Search Google via Serper — for SHOPPING or MAPS results.

    Use this when:
      - The user asks about buying / pricing a product → search_type='shopping'
      - The user asks for places, businesses, locations on Google Maps →
        search_type='maps'
    For general web search (news, articles, facts), prefer `web_search`.

    Args:
        query: Product name or place query.
        search_type: 'shopping' (default) or 'maps'.
        gl: 2-letter country code for geolocation bias (default 'in' for India,
            'us', 'uk', 'de', etc.).

    Returns:
        JSON string with Google's response (shopping_results or place results).
    """
    log.info("google_search invoked: query=%r type=%r gl=%r", query, search_type, gl)
    return _mcp_proxy(
        "google_search",
        {"query": query, "search_type": search_type, "gl": gl},
        label="Google search",
    )


# ── Dormant Tavily helpers (not in DEFAULT_TOOLS) ────────────────────────────

_tavily_client: Optional[TavilyClient] = None


def _get_tavily_client() -> Optional[TavilyClient]:
    global _tavily_client
    if _tavily_client is not None:
        return _tavily_client
    key = os.environ.get("TAVILY_API_KEY") or get_settings().tavily_api_key
    key = (key or "").strip().strip('"').strip("'")
    if not key:
        log.warning("Tavily disabled: TAVILY_API_KEY not set in env or .env")
        return None
    log.info("Tavily client initialized with API key")
    _tavily_client = TavilyClient(api_key=key)
    return _tavily_client


def tavily_search(query: str, depth: str = "basic") -> str:
    """Tavily search — synthesized answer + 3 sources. Dormant by default.

    Re-enable by adding to DEFAULT_TOOLS in app.litert_service.
    """
    log.info("tavily_search invoked: query=%r depth=%r", query, depth)
    client = _get_tavily_client()
    if client is None:
        return "Web search unavailable: TAVILY_API_KEY not configured."
    try:
        data = client.search(
            query=query, search_depth=depth, include_answer="advanced", max_results=3
        )
        answer = (data.get("answer") or "").strip()
        sources = [{"title": s.get("title"), "url": s.get("url")}
                   for s in (data.get("results") or [])[:3]]
        return json.dumps({"answer": answer, "sources": sources}, ensure_ascii=False)
    except Exception as exc:
        log.exception("tavily_search failed")
        return f"Web search failed: {exc}"


def tavily_extract(url: str) -> str:
    """Tavily extract — full cleaned content for one URL. Dormant by default."""
    log.info("tavily_extract invoked: url=%r", url)
    client = _get_tavily_client()
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
        content = (first.get("raw_content") or "")[:8000]
        return json.dumps({"url": first.get("url", url), "content": content},
                          ensure_ascii=False)
    except Exception as exc:
        log.exception("tavily_extract failed")
        return f"Web extract failed: {exc}"


def tavily_research(query: str) -> str:
    """Tavily two-step deep dive. HEAVY payload — dormant by default."""
    log.info("tavily_research invoked: query=%r", query)
    client = _get_tavily_client()
    if client is None:
        return "Web research unavailable: TAVILY_API_KEY not configured."
    try:
        search = client.search(
            query=query, search_depth="advanced", include_answer="advanced", max_results=4
        )
        answer = (search.get("answer") or "").strip()
        top = (search.get("results") or [])[:2]
        urls = [r.get("url") for r in top if r.get("url")]
        sources: list[dict] = []
        if urls:
            try:
                ext = client.extract(urls=urls, extract_depth="basic", format="markdown")
                by_url = {r.get("url"): r for r in (ext.get("results") or [])}
                for r in top:
                    extracted = by_url.get(r.get("url"), {})
                    sources.append({
                        "title": r.get("title"),
                        "url": r.get("url"),
                        "content": (extracted.get("raw_content") or r.get("content") or "")[:1200],
                    })
            except Exception as exc:
                log.warning("tavily_research extract step failed: %s", exc)
                sources = [
                    {"title": r.get("title"), "url": r.get("url"),
                     "content": (r.get("content") or "")[:1200]}
                    for r in top
                ]
        return json.dumps({"query": query, "answer": answer, "sources": sources},
                          ensure_ascii=False)
    except Exception as exc:
        log.exception("tavily_research failed")
        return f"Web research failed: {exc}"


def add_numbers(a: float, b: float) -> float:
    """Adds two numbers.

    Args:
        a: The first number.
        b: The second number.
    """
    return a + b
