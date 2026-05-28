"""E2E coverage for reasoning_content protocol output."""

import json
from types import SimpleNamespace
from typing import Any, Dict, List

import httpx
import pytest

from agentrun.server import AgentRequest, AgentRunServer


def _parse_sse_events(content: str) -> List[Dict[str, Any]]:
    events = []
    for line in content.splitlines():
        if not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload == "[DONE]":
            continue
        events.append(json.loads(payload))
    return events


@pytest.fixture
def reasoning_app():
    async def invoke_agent(request: AgentRequest):
        yield SimpleNamespace(
            content="",
            additional_kwargs={"reasoning_content": "thinking"},
        )
        yield SimpleNamespace(content="answer", additional_kwargs={})

    return AgentRunServer(invoke_agent=invoke_agent).as_fastapi_app()


async def _post_json(app, path: str, payload: Dict[str, Any]) -> httpx.Response:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        return await client.post(path, json=payload, timeout=60.0)


def _set_thinking(monkeypatch, enabled: bool) -> None:
    monkeypatch.setenv(
        "MODEL_PARAMETER_RULES",
        json.dumps({"thinking": enabled}),
    )


@pytest.mark.parametrize("thinking_enabled", [True, False])
@pytest.mark.asyncio
async def test_openai_stream_reasoning_content_gate(
    reasoning_app,
    monkeypatch,
    thinking_enabled: bool,
):
    _set_thinking(monkeypatch, thinking_enabled)

    response = await _post_json(
        reasoning_app,
        "/openai/v1/chat/completions",
        {
            "model": "mock-model",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        },
    )

    assert response.status_code == 200
    events = _parse_sse_events(response.text)
    deltas = [
        (event.get("choices") or [{}])[0].get("delta") or {}
        for event in events
    ]
    reasoning = "".join(delta.get("reasoning_content", "") for delta in deltas)
    content = "".join(delta.get("content", "") for delta in deltas)

    assert content == "answer"
    assert reasoning == ("thinking" if thinking_enabled else "")
    assert all("additional_kwargs" not in delta for delta in deltas)


@pytest.mark.parametrize("thinking_enabled", [True, False])
@pytest.mark.asyncio
async def test_openai_non_stream_reasoning_content_gate(
    reasoning_app,
    monkeypatch,
    thinking_enabled: bool,
):
    _set_thinking(monkeypatch, thinking_enabled)

    response = await _post_json(
        reasoning_app,
        "/openai/v1/chat/completions",
        {
            "model": "mock-model",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": False,
        },
    )

    assert response.status_code == 200
    message = response.json()["choices"][0]["message"]
    assert message["content"] == "answer"
    if thinking_enabled:
        assert message["reasoning_content"] == "thinking"
    else:
        assert "reasoning_content" not in message


@pytest.mark.parametrize("thinking_enabled", [True, False])
@pytest.mark.asyncio
async def test_agui_reasoning_events_gate(
    reasoning_app,
    monkeypatch,
    thinking_enabled: bool,
):
    _set_thinking(monkeypatch, thinking_enabled)

    response = await _post_json(
        reasoning_app,
        "/ag-ui/agent",
        {"messages": [{"role": "user", "content": "Hi"}]},
    )

    assert response.status_code == 200
    events = _parse_sse_events(response.text)
    event_types = [event["type"] for event in events]
    reasoning = "".join(
        event.get("delta", "")
        for event in events
        if event["type"] == "REASONING_MESSAGE_CONTENT"
    )
    content = "".join(
        event.get("delta", "")
        for event in events
        if event["type"] == "TEXT_MESSAGE_CONTENT"
    )

    assert content == "answer"
    if thinking_enabled:
        assert reasoning == "thinking"
        assert event_types.index("REASONING_MESSAGE_CONTENT") < event_types.index(
            "TEXT_MESSAGE_START"
        )
    else:
        assert reasoning == ""
        assert all(
            not event_type.startswith("REASONING")
            for event_type in event_types
        )
