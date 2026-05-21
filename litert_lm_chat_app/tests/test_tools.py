import json
from unittest.mock import patch, MagicMock

from app.tools import tavily_search


def test_tavily_search_missing_key_returns_unavailable(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    import app.tools as tools_mod
    tools_mod._client = None
    result = tavily_search("anything")
    assert "unavailable" in result.lower()


def test_tavily_search_returns_json_with_answer_and_sources(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    import app.tools as tools_mod
    tools_mod._client = None

    fake_client = MagicMock()
    fake_client.search.return_value = {
        "answer": "Paris is the capital.",
        "results": [
            {"title": "Wikipedia", "url": "https://en.wikipedia.org/wiki/Paris"},
            {"title": "BBC", "url": "https://bbc.com/x"},
        ],
    }
    with patch("app.tools.TavilyClient", return_value=fake_client):
        out = tavily_search("capital of France")

    data = json.loads(out)
    assert data["answer"] == "Paris is the capital."
    assert len(data["sources"]) == 2
    assert data["sources"][0]["url"].startswith("https://")
    fake_client.search.assert_called_once()
    kwargs = fake_client.search.call_args.kwargs
    assert kwargs["query"] == "capital of France"
    assert kwargs["search_depth"] == "basic"
    assert kwargs["include_answer"] == "advanced"
    assert kwargs["max_results"] == 3


def test_tavily_search_exception_returns_failed_string(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    import app.tools as tools_mod
    tools_mod._client = None

    fake_client = MagicMock()
    fake_client.search.side_effect = RuntimeError("boom")
    with patch("app.tools.TavilyClient", return_value=fake_client):
        out = tavily_search("anything")

    assert out.startswith("Web search failed")
