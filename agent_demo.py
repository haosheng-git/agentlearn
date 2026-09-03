"""把消息、工具与 Agent 循环组合起来的运行示例。"""

import json
from pathlib import Path

from agent_messages import AgentResult, ToolCall
from agent_runner import FakeModel, run_agent
from agent_session import JsonSessionStore
from agent_tools import build_default_tools


def print_agent_result(result: AgentResult) -> None:
    """按结构化状态展示一次 Agent 运行。"""
    print(f"会话 ID：{result.session_id}")
    print(f"运行ID：{result.run_id}")
    if result.status == "success":
        print(result.answer)
    elif result.status == "partial_success":
        print(f"运行部分成功：{result.error_type}")
        print(result.error_message)
    else:
        print(f"运行失败：{result.error_type}")
        print(result.error_message)

    for tool_result in result.tool_results:
        result_status = "失败" if tool_result.is_error else "成功"
        print(
            f"{tool_result.tool_call_id} "
            f"{result_status}：{tool_result.content}"
        )

    print(f"模型步骤：{result.step_count}")
    print(f"工具调用：{result.tool_call_count}")


def run_demos() -> None:
    workspace_root = (
        Path(__file__).resolve().parent / "workspace"
    )
    workspace_root.mkdir(exist_ok=True)
    state_root = (
        Path(__file__).resolve().parent / "state"
    )
    state_root.mkdir(exist_ok=True)

    session_store = JsonSessionStore(
        state_root/"sessions.jsonl"
    )

    hello_file = workspace_root / "hello.txt"
    hello_file.write_text(
        "这是 workspace 中的示例文件。",
        encoding="utf-8",
    )

    tools = build_default_tools(
        workspace_root=workspace_root
    )

    calculator_model = FakeModel([
        ToolCall(
            id="call_calculator_001",
            name="calculator",
            arguments='{"a": 6, "b": 8}',
        ),
        "计算结果是48",
    ])
    print_agent_result(run_agent(
        calculator_model,
        tools,
        "帮我算6*8",
        session_id="calculator-demo",
        session_store=session_store,
    ))

    add_model = FakeModel([
        ToolCall(
            id="call_add_001",
            name="add",
            arguments='{"a": 10, "b": 5}',
        ),
        "计算结果是15",
    ])
    print_agent_result(run_agent(
        add_model,
        tools,
        "帮我算10+5",
        session_id="add-demo",
        session_store=session_store,
    ))

    read_arguments = json.dumps({"path": "hello.txt"})

    multi_tool_model = FakeModel([
        [
            ToolCall(
                id="call_read_001",
                name="read_file",
                arguments=read_arguments,
            ),
            ToolCall(
                id="call_calculator_002",
                name="calculator",
                arguments='{"a": 5, "b": 7}',
            ),
        ],
        "最终答案：已读取文件，乘积是35",
    ])
    print_agent_result(run_agent(
        multi_tool_model,
        tools,
        "读文件并算乘积",
        session_id="multi-tool-demo",
        session_store=session_store,
    ))

    conversation_session_id = "continuous-conversation-demo"
    session_store.clear(conversation_session_id)

    first_turn_model = FakeModel([
        ToolCall(
            id="call_conversation_read_001",
            name="read_file",
            arguments=read_arguments,
        ),
        "第一轮：文件内容是“这是 workspace 中的示例文件。”",
    ])
    print("\n--- 连续对话：第一轮 ---")
    print_agent_result(run_agent(
        first_turn_model,
        tools,
        "请读取 hello.txt",
        session_id=conversation_session_id,
        session_store=session_store,
    ))

    second_turn_model = FakeModel([
        "第二轮：记得，上一轮读取了 hello.txt。",
    ])
    print("\n--- 连续对话：第二轮（复用同一 session_id）---")
    print_agent_result(run_agent(
        second_turn_model,
        tools,
        "你还记得上一轮读取了哪个文件吗？",
        session_id=conversation_session_id,
        session_store=session_store,
    ))

    unknown_tool_model = FakeModel([
        ToolCall(
            id="call_unknown_001",
            name="unknown_tool",
            arguments="{}",
        )
    ])

    blocked_model = FakeModel([
        ToolCall(
            id="call_blocked_001",
            name="read_file",
            arguments=json.dumps({
                "path": "../outside.txt",
            }),
        )
    ])
    print_agent_result(run_agent(
        unknown_tool_model,
        tools,
        "调用未知工具",
        session_id="unknown-tool-demo",
        session_store=session_store,
    ))
    print_agent_result(run_agent(
        blocked_model,
        tools,
        "尝试读取工作区外文件",
        session_id="blocked-path-demo",
        session_store=session_store,
    ))

    max_steps_model = FakeModel([
        ToolCall(
            id=f"call_loop_{index:03d}",
            name="calculator",
            arguments='{"a": 6, "b": 8}',
        )
        for index in range(100)
    ])
    print_agent_result(run_agent(
        max_steps_model,
        tools,
        "测试最大步数",
        session_id="max-steps-demo",
        session_store=session_store,
        max_steps=5,
    ))


if __name__ == "__main__":
    run_demos()
