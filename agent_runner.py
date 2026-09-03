"""模型模拟器和 Agent 的工具调用循环。"""

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from threading import Event
from time import perf_counter
from uuid import uuid4

from agent_messages import (
    AgentEvent,
    AgentResult,
    Message,
    ToolCall,
    ToolResult,
)
from agent_observability import log_run_result
from agent_session import (
    JsonSessionStore,
    SessionStoreError,
)
from agent_tools import Tool, ToolSizeLimitError


class FakeModel:
    model_name = "fake-model"

    def __init__(self, responses):
        self.responses = responses
        self.received_messages: list[list[Message]] = []

    def generate(self, messages):
        self.received_messages.append(list(messages))
        return self.responses.pop(0)


def _failure_status(tool_results: list[ToolResult]) -> str:
    if any(not result.is_error for result in tool_results):
        return "partial_success"
    return "failed"


def _call_with_timeout(func, timeout_seconds: float):
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(func)
    try:
        return future.result(timeout=timeout_seconds)
    except FutureTimeoutError as error:
        future.cancel()
        raise TimeoutError("操作超时") from error
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def _emit(
    event_handler: Callable[[AgentEvent], None] | None,
    event: str,
    run_id: str,
    **data,
) -> None:
    if event_handler is not None:
        event_handler(AgentEvent(event, run_id, data))


def _run_agent_core(
    model: FakeModel,
    tools: dict[str, Tool],
    user_message: str,
    *,
    session_id: str,
    session_store: JsonSessionStore,
    max_steps: int = 10,
    model_timeout_seconds: float = 120.0,
    tool_timeout_seconds: float = 30.0,
    cancel_event: Event | None = None,
    run_id: str = "",
    event_handler: Callable[[AgentEvent], None] | None = None,
) -> AgentResult:
    """持续处理模型输出，直到模型返回最终文本。"""
    if not isinstance(session_id, str) or not session_id.strip():
        return AgentResult(
            session_id="",
            status="failed",
            answer=None,
            error_type="validation_error",
            error_message="session_id 必须是非空字符串",
            step_count=0,
            tool_call_count=0,
        )
    try:
        saved_messages = session_store.load(session_id)
    except SessionStoreError:
        return AgentResult(
            session_id=session_id,
            status="failed",
            answer=None,
            error_type="session_error",
            error_message="无法加载会话",
            step_count=0,
            tool_call_count=0,
        )
    messages = list(saved_messages)
    messages.append(
        Message(
            role="user",
            content=user_message,
        )
    )
    step_count = 0
    tool_call_count = 0
    tool_results: list[ToolResult] = []

    while step_count < max_steps:
        if cancel_event is not None and cancel_event.is_set():
            return AgentResult(
                session_id=session_id,
                status="cancelled",
                answer=None,
                error_type="cancellation_error",
                error_message="运行已取消",
                step_count=step_count,
                tool_call_count=tool_call_count,
                tool_results=list(tool_results),
            )
        step_count += 1
        _emit(event_handler, "model_started", run_id, step=step_count)
        try:
            output = _call_with_timeout(
                lambda: model.generate(messages),
                model_timeout_seconds,
            )
        except TimeoutError:
            return AgentResult(
                session_id=session_id,
                status="timeout",
                answer=None,
                error_type="timeout_error",
                error_message="模型调用超时",
                step_count=step_count,
                tool_call_count=tool_call_count,
                tool_results=list(tool_results),
            )
        except KeyboardInterrupt:
            return AgentResult(
                session_id=session_id,
                status="cancelled",
                answer=None,
                error_type="cancellation_error",
                error_message="运行被用户中断",
                step_count=step_count,
                tool_call_count=tool_call_count,
                tool_results=list(tool_results),
            )
        # Model adapters are an external boundary and may raise arbitrary errors.
        except Exception as error:  # noqa: BLE001
            return AgentResult(
                session_id=session_id,
                status="failed",
                answer=None,
                error_type="model_error",
                error_message=f"模型调用失败：{error}",
                step_count=step_count,
                tool_call_count=tool_call_count,
            )
        if isinstance(output, (ToolCall, str)):
            output = [output]
        if not isinstance(output, list) or not output:
            return AgentResult(
                session_id=session_id,
                status="failed",
                answer=None,
                error_type="model_error",
                error_message="模型返回了空响应或非法响应",
                step_count=step_count,
                tool_call_count=tool_call_count,
            )
        invalid_items = [
            item
            for item in output
            if not isinstance(item, (ToolCall, str))
        ]
        if invalid_items:
            return AgentResult(
                session_id=session_id,
                status="failed",
                answer=None,
                error_type="model_error",
                error_message="模型返回了无法识别的响应对象",
                step_count=step_count,
                tool_call_count=tool_call_count,
            )
        tool_calls = [
            item 
            for item in output 
            if isinstance(item, ToolCall)
        ]
        final_text = [
            item 
            for item in output 
            if isinstance(item, str)
        ]
        if tool_calls:
            messages.append(
                Message(
                    role="assistant",
                    content=None,
                    tool_calls=list(tool_calls),
                )
            )
            for call in tool_calls:
                if cancel_event is not None and cancel_event.is_set():
                    return AgentResult(
                        session_id=session_id,
                        status="cancelled",
                        answer=None,
                        error_type="cancellation_error",
                        error_message="运行已取消",
                        step_count=step_count,
                        tool_call_count=tool_call_count,
                        tool_results=list(tool_results),
                    )
                if call.name not in tools:
                    error_message = f"工具不存在：{call.name}"
                    tool_results.append(
                        ToolResult(
                            tool_call_id=call.id,
                            content=error_message,
                            is_error=True,
                        )
                    )
                    return AgentResult(
                        session_id=session_id,
                        status=_failure_status(tool_results),
                        answer=None,
                        error_type="tool_error",
                        error_message=error_message,
                        step_count=step_count,
                        tool_call_count=tool_call_count,
                        tool_results=list(tool_results),
                    )
                tool_call_count += 1
                _emit(
                    event_handler,
                    "tool_started",
                    run_id,
                    tool_call_id=call.id,
                    tool_name=call.name,
                )
                try:
                    tool = tools[call.name]
                    arguments = call.arguments
                    result = _call_with_timeout(
                        lambda tool=tool, arguments=arguments: tool.func(arguments),
                        tool_timeout_seconds,
                    )
                except TimeoutError:
                    error_message = "工具执行超时"
                    tool_results.append(
                        ToolResult(call.id, error_message, is_error=True)
                    )
                    return AgentResult(
                        session_id=session_id,
                        status="timeout",
                        answer=None,
                        error_type="timeout_error",
                        error_message=error_message,
                        step_count=step_count,
                        tool_call_count=tool_call_count,
                        tool_results=list(tool_results),
                    )
                except ToolSizeLimitError as error:
                    tool_results.append(
                        ToolResult(
                            tool_call_id=call.id,
                            content=str(error),
                            is_error=True,
                        )
                    )
                    return AgentResult(
                        session_id=session_id,
                        status=_failure_status(tool_results),
                        answer=None,
                        error_type="size_limit_error",
                        error_message=str(error),
                        step_count=step_count,
                        tool_call_count=tool_call_count,
                        tool_results=list(tool_results),
                    )
                except PermissionError:
                    error_message = "文件访问被安全策略拒绝"
                    tool_results.append(
                        ToolResult(
                            tool_call_id=call.id,
                            content=error_message,
                            is_error=True,
                        )
                    )
                    return AgentResult(
                        session_id=session_id,
                        status=_failure_status(tool_results),
                        answer=None,
                        error_type="tool_error",
                        error_message=error_message,
                        step_count=step_count,
                        tool_call_count=tool_call_count,
                        tool_results=list(tool_results),
                    )
                except ValueError as error:
                    tool_results.append(
                        ToolResult(
                            tool_call_id=call.id,
                            content=str(error),
                            is_error=True,
                        )
                    )
                    return AgentResult(
                        session_id=session_id,
                        status=_failure_status(tool_results),
                        answer=None,
                        error_type="validation_error",
                        error_message=str(error),
                        step_count=step_count,
                        tool_call_count=tool_call_count,
                        tool_results=list(tool_results),
                    )
                # Registered tools are extensions and may raise arbitrary errors.
                except Exception as error:  # noqa: BLE001
                    error_message = f"工具执行失败：{error}"
                    tool_results.append(
                        ToolResult(
                            tool_call_id=call.id,
                            content=error_message,
                            is_error=True,
                        )
                    )
                    return AgentResult(
                        session_id=session_id,
                        status=_failure_status(tool_results),
                        answer=None,
                        error_type="execution_error",
                        error_message=error_message,
                        step_count=step_count,
                        tool_call_count=tool_call_count,
                        tool_results=list(tool_results),
                    )
                tool_result = ToolResult(
                    tool_call_id=call.id,
                    content=str(result),
                    is_error=False,
                )
                tool_results.append(tool_result)
                _emit(
                    event_handler,
                    "tool_finished",
                    run_id,
                    tool_call_id=call.id,
                    tool_name=call.name,
                    is_error=False,
                )
                messages.append(
                    Message(
                        role="tool",
                        content=tool_result.content,
                        tool_call_id=tool_result.tool_call_id,
                    )
                )
        else:
            answer = "".join(final_text)
            if not answer:
                return AgentResult(
                    session_id=session_id,
                    status="failed",
                    answer=None,
                    error_type="model_error",
                    error_message="模型没有返回最终文本",
                    step_count=step_count,
                    tool_call_count=tool_call_count,
                )
            completed_messages = list(messages)
            _emit(event_handler, "text_delta", run_id, text=answer)
            completed_messages.append(
                Message(
                    role="assistant",
                    content=answer,
                )
            )

            try:
                session_store.save(
                    session_id,
                    completed_messages,
                )
            except SessionStoreError:
                return AgentResult(
                    session_id=session_id,
                    status="failed",
                    answer=None,
                    error_type="session_error",
                    error_message="回答已生成，但会话保存失败",
                    step_count=step_count,
                    tool_call_count=tool_call_count,
                )
            return AgentResult(
                session_id=session_id,
                status="success",
                answer=answer,
                error_type=None,
                error_message=None,
                step_count=step_count,
                tool_call_count=tool_call_count,
                tool_results=list(tool_results),
            )
    return AgentResult(
        session_id=session_id,
        status="failed",
        answer=None,
        error_type="max_steps_error",
        error_message=f"超过最大执行步数 {max_steps}",
        step_count=step_count,
        tool_call_count=tool_call_count,
        tool_results=list(tool_results),
    )


def run_agent(
    model: FakeModel,
    tools: dict[str, Tool],
    user_message: str,
    *,
    session_id: str,
    session_store: JsonSessionStore,
    max_steps: int = 10,
    model_timeout_seconds: float = 120.0,
    tool_timeout_seconds: float = 30.0,
    cancel_event: Event | None = None,
    event_handler: Callable[[AgentEvent], None] | None = None,
) -> AgentResult:
    """运行 Agent，并补充一次运行的可观测元数据。"""
    started_at = perf_counter()
    run_id = str(uuid4())
    _emit(event_handler, "run_started", run_id)
    result = _run_agent_core(
        model,
        tools,
        user_message,
        session_id=session_id,
        session_store=session_store,
        max_steps=max_steps,
        model_timeout_seconds=model_timeout_seconds,
        tool_timeout_seconds=tool_timeout_seconds,
        cancel_event=cancel_event,
        run_id=run_id,
        event_handler=event_handler,
    )
    result.run_id = run_id
    result.model = getattr(
        model,
        "model_name",
        type(model).__name__,
    )
    result.duration_ms = round(
        (perf_counter() - started_at) * 1000,
        3,
    )
    log_run_result(result)
    _emit(
        event_handler,
        "run_finished",
        run_id,
        status=result.status,
        error_type=result.error_type,
    )
    return result
