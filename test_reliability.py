import threading
import time

import pytest

from agent_messages import ToolCall
from agent_runner import FakeModel, run_agent
from agent_session import JsonSessionStore
from agent_tools import Tool, build_default_tools


@pytest.fixture
def session_store(tmp_path):
    return JsonSessionStore(tmp_path / "sessions.jsonl")


@pytest.fixture
def tools(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return build_default_tools(workspace)


def test_cancelled_before_model_call_does_not_save(tools, session_store):
    cancel_event = threading.Event()
    cancel_event.set()
    model = FakeModel(["不应该调用"])

    result = run_agent(
        model,
        tools,
        "问题",
        session_id="cancel-session",
        session_store=session_store,
        cancel_event=cancel_event,
    )

    assert result.status == "cancelled"
    assert result.error_type == "cancellation_error"
    assert result.step_count == 0
    assert model.received_messages == []
    assert session_store.load("cancel-session") == []


def test_model_timeout_returns_stable_error(tools, session_store):
    class SlowModel:
        model_name = "slow-model"

        def generate(self, messages):
            time.sleep(0.1)
            return "迟到的回答"

    result = run_agent(
        SlowModel(),
        tools,
        "问题",
        session_id="model-timeout-session",
        session_store=session_store,
        model_timeout_seconds=0.01,
    )

    assert result.status == "timeout"
    assert result.error_type == "timeout_error"
    assert result.step_count == 1
    assert session_store.load("model-timeout-session") == []


def test_tool_timeout_returns_stable_error(session_store):
    def slow_tool(arguments):
        time.sleep(0.1)
        return "迟到的结果"

    tools = {
        "slow": Tool("slow", "慢工具", {}, slow_tool),
    }
    model = FakeModel([
        ToolCall("call-slow-001", "slow", "{}"),
    ])

    result = run_agent(
        model,
        tools,
        "运行慢工具",
        session_id="tool-timeout-session",
        session_store=session_store,
        tool_timeout_seconds=0.01,
    )

    assert result.status == "timeout"
    assert result.error_type == "timeout_error"
    assert result.step_count == 1
    assert result.tool_call_count == 1
    assert session_store.load("tool-timeout-session") == []
