"""E2E regression tests for malformed MCP streamable-http responses."""

import asyncio
import socket
import threading
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import httpx
import pytest
import uvicorn

from agentrun.tool.api.mcp import ToolMCPSession
from agentrun.utils.config import Config


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _build_malformed_mcp_app() -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    async def health():
        return {"ok": True}

    @app.post("/mcp")
    async def mcp_endpoint(request: Request):
        payload = await request.json()
        return JSONResponse(
            {
                "jsonrpc": "2.0",
                "id": payload.get("id"),
                "error": {
                    "code": -32000,
                    "message": None,
                },
            }
        )

    return app


@pytest.fixture
def malformed_mcp_server():
    app = _build_malformed_mcp_app()
    port = _find_free_port()
    config = uvicorn.Config(
        app, host="127.0.0.1", port=port, log_level="warning"
    )
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    base_url = f"http://127.0.0.1:{port}"
    for _ in range(50):
        try:
            httpx.get(f"{base_url}/health", timeout=0.2)
            break
        except Exception:
            time.sleep(0.1)
    else:
        raise RuntimeError("malformed MCP server did not start")

    yield f"{base_url}/mcp"

    server.should_exit = True
    thread.join(timeout=5)


@pytest.mark.asyncio
async def test_streamable_mcp_malformed_initialize_error_fails_fast(
    malformed_mcp_server,
):
    session = ToolMCPSession(
        endpoint=malformed_mcp_server,
        session_affinity="MCP_STREAMABLE",
        config=Config(timeout=0.05),
    )

    with pytest.raises(TimeoutError, match="MCP initialize timed out"):
        await asyncio.wait_for(session.list_tools_async(), timeout=1)
