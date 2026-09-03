"""Agent 运行的结构化安全日志。"""

import json
import logging

from agent_messages import AgentResult

LOGGER = logging.getLogger("mini_agent")


def log_run_result(
    result: AgentResult,
    *,
    logger: logging.Logger = LOGGER,
) -> None:
    """仅记录固定元数据，不记录消息、参数、回答或凭据。"""
    payload = {
        "event": "agent_run_finished",
        "run_id": result.run_id,
        "session_id": result.session_id,
        "model": result.model,
        "status": result.status,
        "error_type": result.error_type,
        "step_count": result.step_count,
        "tool_call_count": result.tool_call_count,
        "duration_ms": result.duration_ms,
    }
    logger.info(json.dumps(payload, ensure_ascii=False))
