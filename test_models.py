import pytest

from agent_messages import Message, ToolCall
from agent_models import OpenAICompatibleModel, ProviderHTTPError
from agent_session import JsonSessionStore


def test_provider_parses_text_response(monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "secret-value")
    transport = lambda **kwargs: {
        "choices": [{"message": {"content": "你好", "tool_calls": []}}]
    }
    model = OpenAICompatibleModel(
        base_url="https://example.test/v1",
        model="test-model",
        api_key_env="TEST_API_KEY",
        transport=transport,
    )

    assert model.generate([Message("user", "问题")]) == "你好"


def test_provider_parses_tool_calls(monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "secret-value")
    model = OpenAICompatibleModel(
        base_url="https://example.test/v1",
        model="test-model",
        api_key_env="TEST_API_KEY",
        transport=lambda **kwargs: {
            "choices": [{"message": {"content": None, "tool_calls": [{
                "id": "call-001",
                "type": "function",
                "function": {"name": "calculator", "arguments": '{"a":6,"b":8}'},
            }]}}]
        },
    )

    assert model.generate([Message("user", "计算")]) == [
        ToolCall("call-001", "calculator", '{"a":6,"b":8}')
    ]


def test_provider_retries_one_transient_error(monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "secret-value")
    calls = 0

    def transport(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ProviderHTTPError(503, "临时错误")
        return {"choices": [{"message": {"content": "恢复", "tool_calls": []}}]}

    model = OpenAICompatibleModel(
        base_url="https://example.test/v1",
        model="test-model",
        api_key_env="TEST_API_KEY",
        transport=transport,
        max_retries=1,
    )

    assert model.generate([Message("user", "问题")]) == "恢复"
    assert calls == 2


def test_provider_does_not_retry_authentication_error(monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "secret-value")
    calls = 0

    def transport(**kwargs):
        nonlocal calls
        calls += 1
        raise ProviderHTTPError(401, "模型服务返回 HTTP 401")

    model = OpenAICompatibleModel(
        base_url="https://example.test/v1",
        model="test-model",
        api_key_env="TEST_API_KEY",
        transport=transport,
        max_retries=1,
    )

    with pytest.raises(ProviderHTTPError, match="401"):
        model.generate([Message("user", "测试")])
    assert calls == 1


def test_provider_requires_environment_key(monkeypatch):
    monkeypatch.delenv("MISSING_API_KEY", raising=False)
    model = OpenAICompatibleModel(
        base_url="https://example.test/v1",
        model="test-model",
        api_key_env="MISSING_API_KEY",
        transport=lambda **kwargs: {},
    )

    with pytest.raises(ValueError, match="环境变量"):
        model.generate([Message("user", "问题")])


def test_provider_preserves_assistant_tool_call_for_next_request(monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "secret-value")
    payloads = []

    def transport(**kwargs):
        payloads.append(kwargs["payload"])
        if len(payloads) == 1:
            return {"choices": [{"message": {
                "content": None,
                "tool_calls": [{
                    "id": "call-001",
                    "type": "function",
                    "function": {"name": "calculator", "arguments": '{"a":6,"b":8}'},
                }],
            }}]}
        return {"choices": [{"message": {"content": "48", "tool_calls": []}}]}

    model = OpenAICompatibleModel(
        base_url="https://example.test/v1",
        model="test-model",
        api_key_env="TEST_API_KEY",
        transport=transport,
    )
    calls = model.generate([Message("user", "计算")])
    answer = model.generate([
        Message("user", "计算"),
        Message("tool", "48", tool_call_id="call-001"),
    ])

    assert calls == [ToolCall("call-001", "calculator", '{"a":6,"b":8}')]
    assert answer == "48"
    assert [item["role"] for item in payloads[1]["messages"]] == [
        "user", "assistant", "tool"
    ]
    assert payloads[1]["messages"][1]["tool_calls"][0]["id"] == "call-001"


def test_provider_serializes_persisted_tool_call_history(monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "secret-value")
    payloads = []

    def transport(**kwargs):
        payloads.append(kwargs["payload"])
        return {"choices": [{"message": {"content": "follow-up"}}]}

    model = OpenAICompatibleModel(
        base_url="https://example.test/v1",
        model="test-model",
        api_key_env="TEST_API_KEY",
        transport=transport,
    )
    answer = model.generate([
        Message("user", "read a file"),
        Message(
            "assistant",
            None,
            tool_calls=[ToolCall("call-001", "read_file", '{"path":"a.txt"}')],
        ),
        Message("tool", "hello", tool_call_id="call-001"),
        Message("assistant", "The file says hello."),
        Message("user", "What did it say?"),
    ])

    assert answer == "follow-up"
    assert [item["role"] for item in payloads[0]["messages"]] == [
        "user", "assistant", "tool", "assistant", "user"
    ]
    assert payloads[0]["messages"][1]["tool_calls"][0]["id"] == "call-001"


def test_session_round_trips_assistant_tool_calls(tmp_path):
    store = JsonSessionStore(tmp_path / "sessions.jsonl")
    messages = [
        Message(
            "assistant",
            None,
            tool_calls=[ToolCall("call-001", "read_file", '{"path":"a.txt"}')],
        ),
        Message("tool", "hello", tool_call_id="call-001"),
    ]

    store.save("session-a", messages)

    assert store.load("session-a") == messages
