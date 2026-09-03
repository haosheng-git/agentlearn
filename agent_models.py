"""模型适配器；凭据只从环境变量读取。"""

import json
import os
from urllib import error, request

from agent_messages import Message, ToolCall


class ProviderHTTPError(RuntimeError):
    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code


def _http_transport(*, url, payload, api_key, timeout):
    body = json.dumps(payload).encode("utf-8")
    http_request = request.Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with request.urlopen(http_request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        raise ProviderHTTPError(
            exc.code,
            f"模型服务返回 HTTP {exc.code}",
        ) from exc
    except error.URLError as exc:
        if isinstance(exc.reason, TimeoutError):
            raise TimeoutError("模型请求超时") from exc
        raise RuntimeError("无法连接模型服务") from exc


class OpenAICompatibleModel:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key_env: str = "OPENAI_API_KEY",
        timeout_seconds: float = 120.0,
        max_retries: int = 1,
        tools: list[dict] | None = None,
        transport=None,
    ):
        if not base_url.startswith("https://"):
            raise ValueError("base_url必须使用https")
        if max_retries not in (0, 1):
            raise ValueError("max_retries只能是0或1")
        self.base_url = base_url.rstrip("/")
        self.model_name = model
        self.api_key_env = api_key_env
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.tools = list(tools or [])
        self.transport = transport or _http_transport
        self._pending_tool_calls = None

    def generate(self, messages: list[Message]):
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise ValueError(f"缺少API Key环境变量：{self.api_key_env}")
        wire_messages = []
        pending_ids = {
            item["id"]
            for item in (self._pending_tool_calls or [])
        }
        persisted_tool_call_ids = {
            call.id
            for message in messages
            for call in message.tool_calls
        }
        known_tool_call_ids = pending_ids | persisted_tool_call_ids
        for message in messages:
            if message.role == "tool" and message.tool_call_id not in known_tool_call_ids:
                continue
            wire_message = {
                "role": message.role,
                "content": message.content,
            }
            if message.tool_call_id:
                wire_message["tool_call_id"] = message.tool_call_id
            if message.tool_calls:
                wire_message["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": call.arguments,
                        },
                    }
                    for call in message.tool_calls
                ]
            wire_messages.append(wire_message)
        if self._pending_tool_calls:
            already_present = pending_ids.issubset(persisted_tool_call_ids)
            first_tool = next(
                (
                    index
                    for index, message in enumerate(wire_messages)
                    if message["role"] == "tool"
                ),
                len(wire_messages),
            )
            if not already_present:
                wire_messages.insert(
                    first_tool,
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": self._pending_tool_calls,
                    },
                )
            self._pending_tool_calls = None
        payload = {
            "model": self.model_name,
            "messages": wire_messages,
        }
        if self.tools:
            payload["tools"] = self.tools

        for attempt in range(self.max_retries + 1):
            try:
                response = self.transport(
                    url=f"{self.base_url}/chat/completions",
                    payload=payload,
                    api_key=api_key,
                    timeout=self.timeout_seconds,
                )
                return self._parse_response(response)
            except ProviderHTTPError as exc:
                transient = exc.status_code == 429 or exc.status_code >= 500
                if not transient or attempt >= self.max_retries:
                    raise
        raise RuntimeError("模型调用失败")

    def _parse_response(self, response):
        try:
            message = response["choices"][0]["message"]
            calls = message.get("tool_calls") or []
            if calls:
                self._pending_tool_calls = calls
                return [
                    ToolCall(
                        id=item["id"],
                        name=item["function"]["name"],
                        arguments=item["function"]["arguments"],
                    )
                    for item in calls
                ]
            content = message.get("content")
            if not isinstance(content, str) or not content:
                raise ValueError
            return content
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ValueError("模型返回格式不合法") from exc
