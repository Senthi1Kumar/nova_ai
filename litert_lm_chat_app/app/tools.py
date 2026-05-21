from __future__ import annotations

import json
import os
from typing import Optional

from tavily import TavilyClient

_client: Optional[TavilyClient] = None


def _get_client() -> Optional[TavilyClient]:
    global _client
    if _client is not None:
        return _client
    key = os.environ.get("TAVILY_API_KEY")
    if not key:
        return None
    _client = TavilyClient(api_key=key)
    return _client


def tavily_search(query: str, depth: str = "basic") -> str:
    """Search the live web for up-to-date information (news, weather, facts, prices).

    Use this when the question depends on current or post-training-cutoff data.
    Do NOT use for math, code, or general knowledge already in your training data.

    Args:
        query: The search query. Be specific. Include location/date if relevant.
        depth: "basic" (1 credit, faster) or "advanced" (2 credits, deeper extraction).

    Returns:
        A JSON string with keys: answer (synthesized direct answer) and sources
        (list of {title, url}, top 3).
    """
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
        return json.dumps(
            {
                "answer": (data.get("answer") or "").strip(),
                "sources": [
                    {"title": s.get("title"), "url": s.get("url")}
                    for s in (data.get("results") or [])[:3]
                ],
            },
            ensure_ascii=False,
        )
    except Exception as exc:
        return f"Web search failed: {exc}"


def add_numbers(a: float, b: float) -> float:
    """Adds two numbers.

    Args:
        a: The first number.
        b: The second number.
    """
    return a + b
