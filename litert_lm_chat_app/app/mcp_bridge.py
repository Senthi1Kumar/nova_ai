"""Sync wrapper around an MCP ClientSession.

LiteRT-LM invokes registered tools synchronously from worker threads, but the
MCP Python SDK is fully async. This module owns a dedicated asyncio event loop
in a daemon thread, opens one persistent ClientSession to the configured
streamable-HTTP MCP endpoint, and exposes a blocking `call_tool(name, args)`
that schedules the coroutine on that loop and waits for the result.

Lifecycle:
  bridge = MCPBridge(url, token)
  bridge.start()            # connects + initializes; raises on failure
  ...
  bridge.call_tool(name, {...}, timeout=30)
  ...
  bridge.stop()
"""
from __future__ import annotations

import asyncio
import logging
import threading
from contextlib import AsyncExitStack
from typing import Any, Optional

log = logging.getLogger("litert_app.mcp")


class MCPNotReady(RuntimeError):
    pass


class MCPBridge:
    """One persistent streamable-HTTP MCP session, fronted by a sync API."""

    def __init__(self, url: str, token: str, name: str = "mapbox"):
        self.url = url
        self.token = token
        self.name = name

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._session = None
        self._stack: Optional[AsyncExitStack] = None
        self._tool_names: list[str] = []
        self._tool_schemas: dict = {}

        self._ready = threading.Event()
        self._connect_error: Optional[str] = None

    # ---------- lifecycle ----------

    def start(self, connect_timeout: float = 20.0) -> None:
        if not self.token:
            raise MCPNotReady(
                f"{self.name} MCP disabled: no access token configured"
            )
        self._thread = threading.Thread(
            target=self._run_loop, name=f"mcp-{self.name}", daemon=True
        )
        self._thread.start()
        if not self._ready.wait(timeout=connect_timeout):
            raise MCPNotReady(
                f"{self.name} MCP connect timed out after {connect_timeout}s"
            )
        if self._connect_error:
            raise MCPNotReady(
                f"{self.name} MCP connect failed: {self._connect_error}"
            )
        log.info("%s MCP ready (%d tools): %s", self.name,
                 len(self._tool_names), ", ".join(self._tool_names[:8]))
        # Schema for the static-map tool changes between Mapbox MCP versions —
        # log it once so we don't have to guess the arg shape.
        sm = self._tool_schemas.get("static_map_image_tool")
        if sm:
            import json as _json
            log.info("%s static_map_image_tool schema: %s",
                     self.name, _json.dumps(sm)[:600])

    def tool_schema(self, name: str):
        return self._tool_schemas.get(name)

    def stop(self) -> None:
        if not self._loop:
            return
        loop = self._loop
        try:
            fut = asyncio.run_coroutine_threadsafe(self._close(), loop)
            fut.result(timeout=5)
        except Exception as exc:
            log.warning("%s MCP close error: %s", self.name, exc)
        loop.call_soon_threadsafe(loop.stop)
        if self._thread:
            self._thread.join(timeout=5)

    # ---------- public sync API ----------

    def call_tool(self, name: str, arguments: dict, timeout: float = 30.0) -> str:
        """Call an MCP tool by name. Returns the concatenated text output.

        Raises MCPNotReady if the bridge isn't connected.
        """
        if not self._loop or not self._session:
            raise MCPNotReady(f"{self.name} MCP not connected")
        log.info("%s MCP call_tool: %s args=%s", self.name, name, arguments)
        fut = asyncio.run_coroutine_threadsafe(
            self._call_tool_async(name, arguments), self._loop
        )
        return fut.result(timeout=timeout)

    @property
    def tool_names(self) -> list[str]:
        return list(self._tool_names)

    @property
    def ready(self) -> bool:
        return self._session is not None and self._connect_error is None

    # ---------- async internals ----------

    async def _connect(self) -> None:
        # Imported lazily so the chat app can start even if `mcp` isn't installed.
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        self._stack = AsyncExitStack()
        headers = {"Authorization": f"Bearer {self.token}"}

        transport = await self._stack.enter_async_context(
            streamablehttp_client(self.url, headers=headers)
        )
        # streamablehttp_client yields (read_stream, write_stream, get_session_id)
        read_stream, write_stream, _ = transport

        self._session = await self._stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await self._session.initialize()

        result = await self._session.list_tools()
        self._tool_names = [t.name for t in result.tools]
        self._tool_schemas = {t.name: getattr(t, "inputSchema", None) for t in result.tools}

    async def _close(self) -> None:
        if self._stack is not None:
            try:
                await self._stack.aclose()
            finally:
                self._stack = None
                self._session = None

    async def _call_tool_async(self, name: str, arguments: dict) -> str:
        assert self._session is not None
        result = await self._session.call_tool(name, arguments or {})
        # CallToolResult.content is a list of content items:
        # TextContent (.text), ImageContent (.data base64 + .mimeType),
        # or EmbeddedResource.
        import json as _json
        parts: list[str] = []
        for item in (result.content or []):
            text = getattr(item, "text", None)
            if text:
                parts.append(text)
                continue
            data = getattr(item, "data", None)
            mime = getattr(item, "mimeType", None)
            if data and mime and str(mime).startswith("image/"):
                # Full image data passes through. Caller (tools.py wrapper) is
                # expected to extract `data_b64`, write a file, and replace the
                # payload before the JSON reaches the LLM — Gemma should never
                # see a 100 KB base64 string in its context.
                parts.append(_json.dumps({
                    "image": True,
                    "mime": mime,
                    "data_b64": data,
                }))
                continue
            parts.append(str(item))
        out = "\n".join(p for p in parts if p)
        if getattr(result, "isError", False):
            log.warning("%s MCP tool error: %s", self.name, out[:200])
        return out

    def _run_loop(self) -> None:
        try:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(self._connect())
            self._ready.set()
            self._loop.run_forever()
        except Exception as exc:
            log.exception("%s MCP loop crashed", self.name)
            self._connect_error = repr(exc)
            self._ready.set()


# Module-level singleton so tool wrappers can grab the bridge without a DI mess.
_bridge: Optional[MCPBridge] = None


def set_bridge(b: Optional[MCPBridge]) -> None:
    global _bridge
    _bridge = b


def get_bridge() -> Optional[MCPBridge]:
    return _bridge
