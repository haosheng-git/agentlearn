import json
from functools import partial

import pytest

from agent_demo import print_agent_result
from agent_messages import AgentResult, Message, ToolCall, ToolResult
from agent_runner import FakeModel, run_agent
from agent_session import (
    JsonSessionStore,
    SessionStoreError,
    truncate_messages,
)
from agent_tools import (
    Tool,
    ToolSizeLimitError,
    build_default_tools,
    build_model_tool_specs,
    read_file_func,
)


def test_build_model_tool_specs(tools):
    specs = build_model_tool_specs(tools)

    calculator = next(
        item for item in specs
        if item["function"]["name"] == "calculator"
    )
    assert calculator["type"] == "function"
    assert calculator["function"]["parameters"]["required"] == ["a", "b"]
    assert calculator["function"]["parameters"]["additionalProperties"] is False


@pytest.fixture
def workspace_root(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return workspace


@pytest.fixture
def tools(workspace_root):
    return build_default_tools(workspace_root=workspace_root)


@pytest.fixture
def session_store(tmp_path):
    return JsonSessionStore(tmp_path / "sessions.jsonl")


@pytest.mark.parametrize(
    "arguments,expected",
    [
        ('{"a": 6, "b": 8}', 48),
        ('{"a": 5, "b": 7}', 35),
        ('{"a": 5, "b": 8}', 40),
    ],
)
def test_calculator(tools, arguments, expected):
    result = tools["calculator"].func(arguments)
    assert result == expected


def test_run_agent_normal(tools, session_store):
    model = FakeModel([
        ToolCall(
            id="call_calculator_001",
            name="calculator",
            arguments='{"a": 6, "b": 8}',
        ),
        "计算结果是48",
    ])

    result = run_agent(
        model,
        tools,
        "算 6×8",
        session_id="normal-session",
        session_store=session_store,
    )

    assert result.status == "success"
    assert result.answer == "计算结果是48"
    assert result.error_type is None
    assert result.error_message is None
    assert result.step_count == 2
    assert result.tool_call_count == 1
    assert result.session_id == "normal-session"
    assert model.received_messages[-1][-1] == Message(
        role="tool",
        content="48",
        tool_call_id="call_calculator_001",
    )


def test_run_agent_max_steps(tools, session_store):
    model = FakeModel([
        ToolCall(
            id=f"call_{index:03d}",
            name="calculator",
            arguments='{"a": 6, "b": 8}',
        )
        for index in range(100)
    ])

    result = run_agent(
        model,
        tools,
        "测试",
        session_id="max-steps-session",
        session_store=session_store,
        max_steps=5,
    )

    assert result.status == "failed"
    assert result.answer is None
    assert result.error_type == "max_steps_error"
    assert result.step_count == 5
    assert result.tool_call_count == 5


def test_run_agent_direct_answer(tools, session_store):
    result = run_agent(
        FakeModel(["你好，我是Agent"]),
        tools,
        "你好",
        session_id="direct-answer-session",
        session_store=session_store,
    )

    assert result.status == "success"
    assert result.answer == "你好，我是Agent"
    assert result.step_count == 1
    assert result.tool_call_count == 0


def test_run_agent_unknown_tool(tools, session_store):
    model = FakeModel([
        ToolCall(
            id="call_unknown_001",
            name="unknown_tool",
            arguments="{}",
        )
    ])

    result = run_agent(
        model,
        tools,
        "测试",
        session_id="unknown-tool-session",
        session_store=session_store,
    )

    assert result.status == "failed"
    assert result.answer is None
    assert result.error_type == "tool_error"
    assert result.step_count == 1
    assert result.tool_call_count == 0


def test_run_agent_validation_error(tools, session_store):
    model = FakeModel([
        ToolCall(
            id="call_bad_json_001",
            name="calculator",
            arguments="不是JSON",
        )
    ])

    result = run_agent(
        model,
        tools,
        "测试",
        session_id="validation-session",
        session_store=session_store,
    )

    assert result.status == "failed"
    assert result.answer is None
    assert result.error_type == "validation_error"
    assert result.step_count == 1
    assert result.tool_call_count == 1


def test_run_agent_execution_error(session_store):
    def broken_tool(arguments: str):
        raise RuntimeError("工具内部故障")

    tools = {
        "broken": Tool(
            name="broken",
            description="始终失败的测试工具",
            parameters={},
            func=broken_tool,
        )
    }
    model = FakeModel([
        ToolCall(
            id="call_broken_001",
            name="broken",
            arguments="{}",
        )
    ])

    result = run_agent(
        model,
        tools,
        "测试",
        session_id="execution-session",
        session_store=session_store,
    )

    assert result.status == "failed"
    assert result.error_type == "execution_error"
    assert result.step_count == 1
    assert result.tool_call_count == 1


def test_run_agent_multiple_tools(tools, session_store):
    model = FakeModel([
        [
            ToolCall(
                id="call_calculator_001",
                name="calculator",
                arguments='{"a": 6, "b": 8}',
            ),
            ToolCall(
                id="call_calculator_002",
                name="calculator",
                arguments='{"a": 5, "b": 7}',
            ),
        ],
        "最终答案",
    ])

    result = run_agent(
        model,
        tools,
        "测试",
        session_id="multiple-tools-session",
        session_store=session_store,
    )

    assert result.status == "success"
    assert result.answer == "最终答案"
    assert result.step_count == 2
    assert result.tool_call_count == 2
    assert model.received_messages[-1][-2].tool_call_id == "call_calculator_001"
    assert model.received_messages[-1][-1].tool_call_id == "call_calculator_002"


def test_partial_success_exposes_completed_tool_results(tools, session_store):
    model = FakeModel([
        [
            ToolCall(
                id="call_calculator_001",
                name="calculator",
                arguments='{"a": 6, "b": 8}',
            ),
            ToolCall(
                id="call_email_002",
                name="send_email",
                arguments="{}",
            ),
        ]
    ])

    result = run_agent(
        model,
        tools,
        "计算 6 乘 8 并发送邮件",
        session_id="partial-success-session",
        session_store=session_store,
    )

    assert result.status == "partial_success"
    assert result.answer is None
    assert result.error_type == "tool_error"
    assert result.step_count == 1
    assert result.tool_call_count == 1
    assert result.tool_results == [
        ToolResult(
            tool_call_id="call_calculator_001",
            content="48",
            is_error=False,
        ),
        ToolResult(
            tool_call_id="call_email_002",
            content="工具不存在：send_email",
            is_error=True,
        ),
    ]
    assert session_store.load("partial-success-session") == []


def test_partial_success_includes_failed_tool_result(tools, session_store):
    model = FakeModel([
        [
            ToolCall(
                id="call_calculator_001",
                name="calculator",
                arguments='{"a": 6, "b": 8}',
            ),
            ToolCall(
                id="call_calculator_002",
                name="calculator",
                arguments='{"a": "六", "b": 8}',
            ),
        ]
    ])

    result = run_agent(
        model,
        tools,
        "执行两次计算",
        session_id="partial-validation-session",
        session_store=session_store,
    )

    assert result.status == "partial_success"
    assert result.answer is None
    assert result.error_type == "validation_error"
    assert result.step_count == 1
    assert result.tool_call_count == 2
    assert result.tool_results == [
        ToolResult(
            tool_call_id="call_calculator_001",
            content="48",
            is_error=False,
        ),
        ToolResult(
            tool_call_id="call_calculator_002",
            content="calculator 的 a 和 b 必须是数字",
            is_error=True,
        ),
    ]
    assert session_store.load("partial-validation-session") == []


def test_print_agent_result_shows_partial_tool_results(capsys):
    result = AgentResult(
        session_id="partial-display-session",
        status="partial_success",
        answer=None,
        error_type="tool_error",
        error_message="工具不存在：send_email",
        step_count=1,
        tool_call_count=1,
        tool_results=[
            ToolResult(
                tool_call_id="call_calculator_001",
                content="48",
                is_error=False,
            ),
            ToolResult(
                tool_call_id="call_email_002",
                content="工具不存在：send_email",
                is_error=True,
            ),
        ],
    )

    print_agent_result(result)

    output = capsys.readouterr().out
    assert "部分成功" in output
    assert "call_calculator_001 成功：48" in output
    assert "call_email_002 失败：工具不存在：send_email" in output


@pytest.mark.parametrize("response", [[], object()])
def test_run_agent_rejects_invalid_model_response(tools, session_store, response):
    result = run_agent(
        FakeModel([response]),
        tools,
        "测试",
        session_id="invalid-response-session",
        session_store=session_store,
    )

    assert result.status == "failed"
    assert result.answer is None
    assert result.error_type == "model_error"
    assert result.step_count == 1
    assert result.tool_call_count == 0


def test_read_file_inside_workspace(workspace_root):
    file_path = workspace_root / "notes.txt"
    file_path.write_text("你好，Agent", encoding="utf-8")

    result = read_file_func(
        json.dumps({"path": "notes.txt"}),
        workspace_root=workspace_root,
    )

    assert result == "你好，Agent"


def test_read_file_inside_workspace_subdirectory(workspace_root):
    docs = workspace_root / "docs"
    docs.mkdir()
    lesson = docs / "lesson.txt"
    lesson.write_text("Day 12", encoding="utf-8")

    result = read_file_func(
        json.dumps({"path": "docs/lesson.txt"}),
        workspace_root=workspace_root,
    )

    assert result == "Day 12"


def test_read_file_allows_normalized_path_inside_workspace(workspace_root):
    file_path = workspace_root / "hello.txt"
    file_path.write_text("safe", encoding="utf-8")

    result = read_file_func(
        json.dumps({"path": "docs/../hello.txt"}),
        workspace_root=workspace_root,
    )

    assert result == "safe"


def test_read_file_not_found(workspace_root):
    with pytest.raises(FileNotFoundError):
        read_file_func(
            json.dumps({"path": "不存在.txt"}),
            workspace_root=workspace_root,
        )


def test_read_file_rejects_absolute_path(workspace_root, tmp_path):
    outside_file = tmp_path / "outside.txt"
    outside_file.write_text("private", encoding="utf-8")

    with pytest.raises(PermissionError):
        read_file_func(
            json.dumps({"path": str(outside_file)}),
            workspace_root=workspace_root,
        )


def test_read_file_rejects_parent_traversal(workspace_root, tmp_path):
    outside_file = tmp_path / "outside.txt"
    outside_file.write_text("private", encoding="utf-8")

    with pytest.raises(PermissionError):
        read_file_func(
            json.dumps({"path": "../outside.txt"}),
            workspace_root=workspace_root,
        )


@pytest.mark.parametrize(
    "unsafe_path",
    [
        ".env",
        ".env.local",
        ".ENV.production",
        ".ssh/id_rsa",
        ".aws/credentials",
        "private.pem",
        "secret.KEY",
        "certificate.p12",
    ],
)
def test_read_file_rejects_sensitive_paths(workspace_root, unsafe_path):
    with pytest.raises(PermissionError):
        read_file_func(
            json.dumps({"path": unsafe_path}),
            workspace_root=workspace_root,
        )


def test_read_file_rejects_whitespace_only_path(workspace_root):
    with pytest.raises(ValueError):
        read_file_func(
            json.dumps({"path": "   "}),
            workspace_root=workspace_root,
        )


def test_read_file_allows_file_at_size_limit(workspace_root):
    file_path = workspace_root / "exact.txt"
    file_path.write_bytes(b"a" * 10)

    result = read_file_func(
        json.dumps({"path": "exact.txt"}),
        workspace_root=workspace_root,
        max_bytes=10,
    )

    assert result == "a" * 10


def test_read_file_rejects_oversized_file(workspace_root):
    file_path = workspace_root / "large.txt"
    file_path.write_bytes(b"a" * 11)

    with pytest.raises(ToolSizeLimitError):
        read_file_func(
            json.dumps({"path": "large.txt"}),
            workspace_root=workspace_root,
            max_bytes=10,
        )


def test_agent_returns_tool_error_for_blocked_path(tools, session_store):
    model = FakeModel([
        ToolCall(
            id="call_read_blocked_001",
            name="read_file",
            arguments=json.dumps({"path": "../outside-secret.txt"}),
        )
    ])

    result = run_agent(
        model,
        tools,
        "读取外部文件",
        session_id="blocked-path-session",
        session_store=session_store,
    )

    assert result.status == "failed"
    assert result.error_type == "tool_error"
    assert result.error_message == "文件访问被安全策略拒绝"
    assert "outside-secret.txt" not in result.error_message
    assert result.tool_call_count == 1


def test_agent_returns_size_limit_error(workspace_root, session_store):
    large_file = workspace_root / "large.txt"
    large_file.write_bytes(b"a" * 11)
    tools = build_default_tools(workspace_root=workspace_root)
    tools["read_file"].func = partial(
        read_file_func,
        workspace_root=workspace_root,
        max_bytes=10,
    )
    model = FakeModel([
        ToolCall(
            id="call_read_large_001",
            name="read_file",
            arguments=json.dumps({"path": "large.txt"}),
        )
    ])

    result = run_agent(
        model,
        tools,
        "读取大文件",
        session_id="size-limit-session",
        session_store=session_store,
    )

    assert result.status == "failed"
    assert result.error_type == "size_limit_error"
    assert result.tool_call_count == 1


def test_same_session_continues_history(tools, session_store):
    first_model = FakeModel(["第一轮回答"])
    first_result = run_agent(
        first_model,
        tools,
        "第一轮问题",
        session_id="conversation-001",
        session_store=session_store,
    )
    second_model = FakeModel(["第二轮回答"])
    second_result = run_agent(
        second_model,
        tools,
        "第二轮问题",
        session_id="conversation-001",
        session_store=session_store,
    )

    assert first_result.status == "success"
    assert second_result.status == "success"
    assert second_model.received_messages[0] == [
        Message(role="user", content="第一轮问题"),
        Message(role="assistant", content="第一轮回答"),
        Message(role="user", content="第二轮问题"),
    ]


def test_different_sessions_are_isolated(tools, session_store):
    run_agent(
        FakeModel(["A 的回答"]),
        tools,
        "A 的问题",
        session_id="session-a",
        session_store=session_store,
    )
    model_b = FakeModel(["B 的回答"])
    run_agent(
        model_b,
        tools,
        "B 的问题",
        session_id="session-b",
        session_store=session_store,
    )

    assert model_b.received_messages[0] == [
        Message(role="user", content="B 的问题")
    ]


def test_session_persists_after_store_recreation(tools, tmp_path):
    store_path = tmp_path / "sessions.jsonl"
    run_agent(
        FakeModel(["已记住"]),
        tools,
        "请记住这句话",
        session_id="persistent-session",
        session_store=JsonSessionStore(store_path),
    )
    recreated_store = JsonSessionStore(store_path)
    follow_up_model = FakeModel(["仍然记得"])
    result = run_agent(
        follow_up_model,
        tools,
        "你还记得吗？",
        session_id="persistent-session",
        session_store=recreated_store,
    )

    assert result.status == "success"
    assert len(follow_up_model.received_messages[0]) == 3


def test_failed_run_does_not_pollute_history(tools, session_store):
    failed_result = run_agent(
        FakeModel([ToolCall("missing-001", "missing", "{}")]),
        tools,
        "这一轮会失败",
        session_id="clean-session",
        session_store=session_store,
    )
    next_model = FakeModel(["成功回答"])
    next_result = run_agent(
        next_model,
        tools,
        "新问题",
        session_id="clean-session",
        session_store=session_store,
    )

    assert failed_result.status == "failed"
    assert next_result.status == "success"
    assert next_model.received_messages[0] == [
        Message(role="user", content="新问题")
    ]


def test_save_failure_returns_session_error(tools):
    class SaveFailureStore:
        def load(self, session_id):
            return []

        def save(self, session_id, messages):
            raise SessionStoreError("保存失败")

    result = run_agent(
        FakeModel(["回答已经生成"]),
        tools,
        "问题",
        session_id="save-failure-session",
        session_store=SaveFailureStore(),
    )

    assert result.status == "failed"
    assert result.session_id == "save-failure-session"
    assert result.error_type == "session_error"


def test_clear_only_target_session(session_store):
    session_store.save("session-a", [Message(role="user", content="A")])
    session_store.save("session-b", [Message(role="user", content="B")])

    session_store.clear("session-a")

    assert session_store.load("session-a") == []
    assert session_store.load("session-b") == [
        Message(role="user", content="B")
    ]


def test_blank_session_id_is_rejected_without_calling_model(tools, session_store):
    model = FakeModel(["不应被调用"])

    result = run_agent(
        model,
        tools,
        "问题",
        session_id="   ",
        session_store=session_store,
    )

    assert result.status == "failed"
    assert result.error_type == "validation_error"
    assert result.step_count == 0
    assert model.received_messages == []


def test_corrupted_session_file_returns_session_error(tools, tmp_path):
    store_path = tmp_path / "sessions.jsonl"
    store_path.write_text("不是 JSON\n", encoding="utf-8")

    result = run_agent(
        FakeModel(["不应被调用"]),
        tools,
        "问题",
        session_id="corrupted-session",
        session_store=JsonSessionStore(store_path),
    )

    assert result.status == "failed"
    assert result.error_type == "session_error"
    assert result.step_count == 0


def test_truncate_messages_returns_equal_independent_list_within_limit():
    messages = [
        Message(role="user", content="第一问"),
        Message(role="assistant", content="第一答"),
    ]

    result = truncate_messages(
        messages,
        max_messages=100,
    )

    assert result == messages
    assert result is not messages


def test_truncate_messages_removes_oldest_complete_turn():
    messages = [
        Message(role="user", content="第一问"),
        Message(role="assistant", content="第一答"),
        Message(role="user", content="第二问"),
        Message(
            role="tool",
            content="第二轮工具结果",
            tool_call_id="call_002",
        ),
        Message(role="assistant", content="第二答"),
        Message(role="user", content="第三问"),
        Message(role="assistant", content="第三答"),
    ]

    result = truncate_messages(
        messages,
        max_messages=4,
    )

    assert result == [
        Message(role="user", content="第三问"),
        Message(role="assistant", content="第三答"),
    ]
