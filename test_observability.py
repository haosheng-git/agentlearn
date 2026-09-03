import json
import logging

import pytest

from agent_messages import AgentResult
from agent_observability import log_run_result
from agent_runner import FakeModel, run_agent
from agent_session import JsonSessionStore
from agent_tools import build_default_tools


@pytest.fixture
def tools(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return build_default_tools(workspace)


@pytest.fixture
def session_store(tmp_path):
    return JsonSessionStore(tmp_path / "sessions.jsonl")


def test_run_result_contains_operational_metadata(tools, session_store):
    model = FakeModel(["回答"])

    result = run_agent(
        model,
        tools,
        "问题",
        session_id="metadata-session",
        session_store=session_store,
    )

    assert result.run_id
    assert result.model == "fake-model"
    assert result.duration_ms >= 0


def test_run_ids_are_unique(tools, session_store):
    first = run_agent(
        FakeModel(["回答1"]),
        tools,
        "问题1",
        session_id="run-id-session",
        session_store=session_store,
    )
    second = run_agent(
        FakeModel(["回答2"]),
        tools,
        "问题2",
        session_id="run-id-session",
        session_store=session_store,
    )

    assert first.run_id != second.run_id


def test_structured_log_uses_allowlisted_fields(caplog):
    result = AgentResult(
        session_id="safe-session",
        status="failed",
        answer=None,
        error_type="tool_error",
        error_message="安全错误",
        step_count=1,
        tool_call_count=1,
        run_id="run-001",
        model="fake-model",
        duration_ms=12,
    )

    with caplog.at_level(logging.INFO, logger="mini_agent"):
        log_run_result(result)

    payload = json.loads(caplog.records[-1].message)
    assert payload == {
        "event": "agent_run_finished",
        "run_id": "run-001",
        "session_id": "safe-session",
        "model": "fake-model",
        "status": "failed",
        "error_type": "tool_error",
        "step_count": 1,
        "tool_call_count": 1,
        "duration_ms": 12,
    }
    assert "API_KEY" not in caplog.records[-1].message
