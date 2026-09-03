"""Agent 运行过程中使用的数据结构。"""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass
class Message:
    role: str
    content: str | None
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)


@dataclass
class ToolResult:
    tool_call_id: str
    content: str
    is_error: bool = False


@dataclass
class AgentEvent:
    event: str
    run_id: str
    data: dict = field(default_factory=dict)


AgentStatus = Literal[
    "success",
    "partial_success",
    "failed",
    "cancelled",
    "timeout",
]


AgentErrorType = Literal[
    "tool_error",
    "validation_error",
    "execution_error",
    "model_error",
    "max_steps_error",
    "size_limit_error",
    "session_error",
    "timeout_error",
    "cancellation_error",
]


@dataclass
class AgentResult:
    session_id: str
    status: AgentStatus
    answer: str | None
    error_type: AgentErrorType | None
    error_message: str | None
    step_count: int
    tool_call_count: int
    tool_results: list[ToolResult] = field(default_factory=list)
    run_id: str = ""
    model: str = ""
    duration_ms: float = 0.0
