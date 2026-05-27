"""IResearcher FastMCP server — Brave / Serper / Nominatim / Traccar tools.

Run as a sidecar to the litert chat app:
    cd litert_lm_chat_app
    python -m app.tools_v2

The chat app's MCPBridge connects to http://127.0.0.1:8765/mcp at startup
and exposes the web_search and google_search tools to Gemma. The other
tools (execute_sql_query, geocode, add_skill) stay available on the MCP
server for use from other clients.
"""
import os
import re
from pathlib import Path
from typing import Optional

import httpx
from dotenv import load_dotenv
from fastmcp import FastMCP

from app.utils import (
    logger,
    sanitize_llm_input,
    sanitize_search_type,
)
from app.prompts import TELEMATICS_ASSISTANT_SYSTEM_MESSAGE

# Load .env from the chat-app root BEFORE reading API keys. We use
# override=True so a .env-set value always wins over a possibly-empty shell
# value (e.g. `export BRAVE_API_KEY=` lying around in your shell rc).
# We also probe a couple of paths in case the script is launched from a
# different cwd.
def _load_env_files() -> tuple[Path | None, bool]:
    candidates = [
        Path(__file__).resolve().parent.parent / ".env",
        Path.cwd() / ".env",                             
    ]
    for p in candidates:
        if p.exists():
            ok = load_dotenv(dotenv_path=p, override=True)
            return p, ok
    return None, False


_ENV_PATH, _ENV_LOADED = _load_env_files()

mcp = FastMCP("IResearcher MCP Server")


BRAVE_API_KEY = os.getenv("BRAVE_API_KEY") or None
SERPER_API_KEY = os.getenv("SERPER_API_KEY") or None
BRAVE_API_BASE = "https://api.search.brave.com/res/v1"

# Loud startup diagnostic — if you see MISSING here, .env is either at the
# wrong path or the key is missing/empty in the file.
logger.info("─── IResearcher sidecar key audit ───")
logger.info("  .env path : %s", _ENV_PATH or "NOT FOUND")
logger.info("  .env loaded: %s", _ENV_LOADED)
logger.info("─────────────────────────────────────")

@mcp.tool
async def web_search(
    query: str,
    search_type: str = "web",
    num_results: int = 20,
) -> dict:
    """
    Search the web using Brave Search API (keyword based), returns web page snippets.
    
    Args:
        query: The search query
        search_type: Type of search to perform, either "web" or "news"
        num_results: Number of results to return (default: 20)
    
    Returns:
        Dictionary with search results
    """
    # Sanitize LLM-generated inputs
    query = sanitize_llm_input(query, "query")
    search_type = sanitize_search_type(search_type, valid_types=["web", "news"], default="web")
    
    logger.info(f"Starting web search: query='{query[:100]}...', type={search_type}, results={num_results}")

    if not BRAVE_API_KEY:
        error_msg = "BRAVE_API_KEY not found for search"
        logger.error(error_msg)
        return {"error": error_msg, "results": []}

    url = f"{BRAVE_API_BASE}/{search_type}/search"
    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": BRAVE_API_KEY
    }
    
    # Clean query further - remove quotes
    query = re.sub(r'["\']', '', query)
    params = {"q": query, "count": num_results}

    logger.info(f"Performing {search_type} search with query: {query}")
    
    try:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            response = await client.get(url, headers=headers, params=params, timeout=30.0)

        logger.info(f"Search response status: {response.status_code}")
        
        # Log redirect history if any
        if response.history:
            redirect_chain = " -> ".join([f"{r.status_code}" for r in response.history])
            logger.info(f"Redirect chain: {redirect_chain} -> {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            processed_results = []

            results = (
                data.get('web', {}).get('results', [])
                if search_type == "web"
                else data.get('results', [])
            )

            logger.info(f"Processing {len(results)} raw results from API")

            # Cap each field tight enough that the full payload fits in a few
            # hundred tokens — small-model contexts can't afford the full
            # Brave response. Drop extra_snippets and metadata entirely.
            for item in results[: min(num_results, 5)]:
                processed_results.append({
                    'title': (item.get('title') or '')[:120],
                    'url': item.get('url', ''),
                    'description': (item.get('description') or '')[:240],
                    'age': item.get('age') or item.get('page_age') or '',
                })

            logger.info(f"Web search successful - found {len(processed_results)} results")

            return {
                "results": processed_results,
                "search_type": search_type,
                "query": query,
                "count": len(processed_results),
            }

        else:
            error_text = response.text
            error_msg = f"Search error: Status {response.status_code} - {response.text}"
            logger.error(f"Web search error: Status {response.status_code} for query '{query[:50]}...'\nResponse: {error_text[:200]}...")
            return {"error": error_msg, "results": []}

    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(f"Unexpected error in web_search for query '{query[:50]}...': {e}")
        return {"error": error_msg, "results": []}


@mcp.tool
async def google_search(
    query: str,
    search_type: str = "shopping",
    gl: str = "in",
) -> dict:
    """
    Search Google Shopping or Maps and return the results.
    
    Args:
        query: The search query.
        search_type: Type of search, can be "maps" or "shopping". Default is "shopping".
        gl: Geolocation, default is "in" for India.
    
    Returns:
        Dictionary with search results
    """
    # Sanitize LLM-generated inputs
    query = sanitize_llm_input(query, "query")
    search_type = sanitize_search_type(search_type, valid_types=["maps", "shopping"], default="shopping")
    gl = sanitize_llm_input(gl, "gl")
    
    logger.info(f"Starting Google search: query='{query[:100]}...', search_type={search_type}, gl={gl}")

    if not SERPER_API_KEY:
        error_msg = "SERPER_API_KEY not found for Google search"
        logger.error(error_msg)
        return {"error": error_msg, "results": []}

    endpoint = f"/{search_type}"
    payload = {"q": query}
    
    if search_type == "shopping":
        payload["gl"] = gl
    elif search_type == "maps":
        # 'q' is the only payload for maps
        pass
    else:
        error_msg = f"Invalid search_type: {search_type}. Must be 'maps' or 'shopping'."
        logger.error(error_msg)
        return {"error": error_msg, "results": []}

    headers = {
        'X-API-KEY': SERPER_API_KEY,
        'Content-Type': 'application/json'
    }

    try:
        logger.info(f"Calling external Google search service at endpoint {endpoint}")
        async with httpx.AsyncClient(follow_redirects=True) as client:
            response = await client.post(
                f"https://google.serper.dev{endpoint}",
                headers=headers,
                json=payload,
                timeout=30.0
            )

        logger.info(f"Google search response status: {response.status_code}")

        if response.status_code == 200:
            logger.info(f"Google search successful")
            data = response.json()
            return {
                "success": True,
                "data": data,
                "query": query,
                "search_type": search_type
            }
        else:
            error_msg = f"API error: {response.status_code} - {response.text}"
            logger.error(f"Google search API error: {error_msg}")
            return {"error": error_msg, "results": []}

    except Exception as e:
        error_msg = f"Exception: {str(e)}"
        logger.error(f"Exception in google_search: {str(e)}")
        return {"error": error_msg, "results": []}


@mcp.tool
async def execute_sql_query(
    query: str,
) -> dict:
    """
    IMPORTANT: Add skill: "telematics" once before first execution of this tool.
    Execute a SQL query on the Traccar database and return the results.
    
    Args:
        query: The SQL query to execute on tc_position_attributes table
    
    Returns:
        Dictionary with query results including success status, row count, columns, and data in CSV format
    """
    logger.info(f"Executing SQL query: {query[:100]}...")
    
    api_url = "https://api1001.elevatics.online/execute-sql"
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json'
    }
    payload = {"query": query}
    
    try:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            response = await client.post(
                api_url,
                headers=headers,
                json=payload,
                timeout=60.0
            )
        
        logger.info(f"SQL query response status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            logger.info(f"SQL query successful - returned {data.get('rows', 0)} rows")
            return {
                "success": data.get("success", True),
                "rows": data.get("rows", 0),
                "columns": data.get("columns", []),
                "csv": data.get("csv", ""),
                "query": query
            }
        else:
            error_msg = f"SQL API error: {response.status_code} - {response.text}"
            logger.error(f"SQL query API error: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "query": query
            }
    
    except Exception as e:
        error_msg = f"Exception executing SQL query: {str(e)}"
        logger.error(error_msg)
        return {
            "success": False,
            "error": error_msg,
            "query": query
        }

@mcp.tool
async def geocode(
    location: Optional[str] = None,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
) -> dict:
    """
    Geocode an address to coordinates or reverse geocode coordinates to an address using Nominatim API.
    
    Args:
        location: Address or place name to geocode (forward geocoding). Use this OR lat/lon, not both.
        latitude: Latitude for reverse geocoding. Must be used with longitude.
        longitude: Longitude for reverse geocoding. Must be used with latitude.
    
    Returns:
        Dictionary with geocoding results
    """
    logger.info(f"Starting geocoding: location={location}, lat={latitude}, lon={longitude}")
    
    base_url = "https://nominatim.openstreetmap.org"
    headers = {
        "User-Agent": "IResearcher-MCP-Server/1.0"  # Nominatim requires a User-Agent
    }
    
    try:
        # Determine if this is forward or reverse geocoding
        if location:
            # Forward geocoding: address -> coordinates
            url = f"{base_url}/search"
            params = {
                "q": location,
                "format": "json",
                "limit": 5,
                "addressdetails": 1
            }
            logger.info(f"Performing forward geocoding for: {location}")
            
        elif latitude is not None and longitude is not None:
            # Reverse geocoding: coordinates -> address
            url = f"{base_url}/reverse"
            params = {
                "lat": latitude,
                "lon": longitude,
                "format": "json",
                "addressdetails": 1
            }
            logger.info(f"Performing reverse geocoding for: {latitude}, {longitude}")
            
        else:
            error_msg = "Either 'location' or both 'latitude' and 'longitude' must be provided"
            logger.error(error_msg)
            return {"error": error_msg, "results": []}
        
        async with httpx.AsyncClient(follow_redirects=True) as client:
            response = await client.get(url, headers=headers, params=params, timeout=30.0)
        
        logger.info(f"Nominatim response status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            if location:
                # Forward geocoding results
                if isinstance(data, list) and len(data) > 0:
                    results = []
                    for item in data:
                        result = {
                            "display_name": item.get("display_name", ""),
                            "latitude": float(item.get("lat", 0)),
                            "longitude": float(item.get("lon", 0)),
                            "type": item.get("type", ""),
                            "importance": item.get("importance", 0),
                            "address": item.get("address", {})
                        }
                        results.append(result)
                    
                    logger.info(f"Forward geocoding successful - found {len(results)} results")
                    return {
                        "success": True,
                        "type": "forward",
                        "query": location,
                        "results": results,
                        "count": len(results)
                    }
                else:
                    logger.warning(f"No results found for location: {location}")
                    return {
                        "success": False,
                        "error": "No results found",
                        "query": location,
                        "results": []
                    }
            else:
                # Reverse geocoding result
                if isinstance(data, dict) and data.get("display_name"):
                    result = {
                        "display_name": data.get("display_name", ""),
                        "latitude": float(data.get("lat", latitude)),
                        "longitude": float(data.get("lon", longitude)),
                        "type": data.get("type", ""),
                        "address": data.get("address", {})
                    }
                    
                    logger.info(f"Reverse geocoding successful")
                    return {
                        "success": True,
                        "type": "reverse",
                        "query": {"latitude": latitude, "longitude": longitude},
                        "result": result
                    }
                else:
                    logger.warning(f"No results found for coordinates: {latitude}, {longitude}")
                    return {
                        "success": False,
                        "error": "No results found",
                        "query": {"latitude": latitude, "longitude": longitude}
                    }
        else:
            error_msg = f"Nominatim API error: {response.status_code} - {response.text}"
            logger.error(error_msg)
            return {"error": error_msg, "results": []}
    
    except Exception as e:
        error_msg = f"Exception in geocoding: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg, "results": []}


@mcp.tool
async def add_skill(skill: str = "telematics") -> str:
    """
    Add skills to the agent by returning the appropriate system message.
    
    Args:
        skill: The skill to add to the agent. Default is "telematics".
               Options: "telematics" 
    
    Returns:
        The system message string for the specified skill.
    """
    logger.info(f"Adding skill: {skill}")
    
    if skill == "telematics":
        logger.info("Returning TELEMATICS_ASSISTANT_SYSTEM_MESSAGE")
        return TELEMATICS_ASSISTANT_SYSTEM_MESSAGE
    else:
        logger.info("Returning default SYSTEM_MESSAGE")
        return "invalid skill type"


if __name__ == "__main__":
    # Bound to loopback by default — only the local chat app should reach it.
    host = os.environ.get("IRESEARCHER_HOST", "127.0.0.1")
    port = int(os.environ.get("IRESEARCHER_PORT", "8765"))
    logger.info("IResearcher FastMCP server starting on %s:%d", host, port)

    # Try fastmcp's run first (clean path). If uvicorn's WS protocol lookup
    # is broken (Python 3.14 / fastmcp+uvicorn version skew → KeyError
    # 'websockets-sansio'), fall back to invoking uvicorn directly on the
    # ASGI app with ws='none' — FastMCP's HTTP transport doesn't need WS.
    try:
        mcp.run(transport="http", host=host, port=port)
    except KeyError as exc:
        if "websockets-sansio" not in str(exc):
            raise
        logger.warning("uvicorn WS protocol skew (%s) — falling back to ws='none'", exc)
        import uvicorn
        app = mcp.http_app()
        uvicorn.run(app, host=host, port=port, ws="none", log_level="info")