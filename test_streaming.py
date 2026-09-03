from agent_messages import ToolCall
from agent_runner import FakeModel, run_agent
from agent_session import JsonSessionStore
from agent_tools import build_default_tools


def test_success_emits_ordered_events(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    events = []

    result = run_agent(
        FakeModel([
            ToolCall("call-001", "calculator", '{"a": 6, "b": 8}'),
            "结果是48",
        ]),
        build_default_tools(workspace),
        "计算",
        session_id="stream-session",
        session_store=JsonSessionStore(tmp_path / "sessions.jsonl"),
        event_handler=events.append,
    )

    assert [event.event for event in events] == [
        "run_started",
        "model_started",
        "tool_started",
        "tool_finished",
        "model_started",
        "text_delta",
        "run_finished",
    ]
    assert all(event.run_id == result.run_id for event in events)
    assert events[-1].data["status"] == "success"


def test_failure_still_emits_finished_event(tmp_path):
    events = []
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    result = run_agent(
        FakeModel([ToolCall("call-001", "missing", "{}")]),
        build_default_tools(workspace),
        "调用工具",
        session_id="failed-stream-session",
        session_store=JsonSessionStore(tmp_path / "sessions.jsonl"),
        event_handler=events.append,
    )

    assert result.status == "failed"
    assert events[0].event == "run_started"
    assert events[-1].event == "run_finished"
    assert events[-1].data["status"] == "failed"
