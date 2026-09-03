"""工具的定义、注册，以及各个工具的具体实现。"""

import json
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path

DEFAULT_MAX_READ_BYTES = 64 * 1024
class ToolSizeLimitError(Exception):
    """工具读取或返回的内容超过允许的字节上限。"""
DENIED_PATH_NAMES = {
    ".env",
    ".ssh",
    ".aws",
    ".azure",
    ".gnupg",
}

DENIED_SUFFIXES = {
    ".pem",
    ".key",
    ".p12",
    ".pfx",
}

@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    func: Callable[[str], object]


TOOLS: dict[str, Tool] = {}


def register_tool(tool: Tool) -> None:
    """把工具注册到全局工具表中。"""
    TOOLS[tool.name] = tool


def get_tool(name: str) -> Tool:
    """根据名称取得已经注册的工具。"""
    return TOOLS[name]


def calculator_func(arguments: str):
    """读取 JSON 参数并返回 a 与 b 的乘积。"""
    try:
        data = json.loads(arguments)
        a = data["a"]
        b = data["b"]
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError(f"calculator 参数错误：{error}") from error

    if not _are_numbers(a, b):
        raise ValueError("calculator 的 a 和 b 必须是数字")

    return a * b


def add_func(arguments: str):
    """读取 JSON 参数并返回 a 与 b 的和。"""
    try:
        data = json.loads(arguments)
        a = data["a"]
        b = data["b"]
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError(f"add 参数错误：{error}") from error

    if not _are_numbers(a, b):
        raise ValueError("add 的 a 和 b 必须是数字")

    return a + b


def read_file_func(
        arguments: str,
        *,
        workspace_root: Path,
        max_bytes: int = DEFAULT_MAX_READ_BYTES
) -> str:
    """读取 JSON 参数中 path 指向的 UTF-8 文本文件。"""
    try:
        data = json.loads(arguments)
        user_path = data["path"]
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError(f"read_file 参数错误：{error}") from error

    if not isinstance(user_path, str) or not user_path.strip():
        raise ValueError("read_file 的 path 必须是非空字符串")

    relative_path = Path(user_path)
    if relative_path.is_absolute():
        raise PermissionError("read_file 只接受 workspace 内的相对路径")

    resolved_root = workspace_root.resolve()
    candidate = (resolved_root/relative_path).resolve()

    try:
        candidate.relative_to(resolved_root)
    except ValueError as error:
        raise PermissionError("禁止访问 workspace 之外的路径") from error

    normalized_parts={
        part.casefold()
        for part in relative_path.parts
    }

    if normalized_parts & DENIED_PATH_NAMES:
        raise PermissionError("禁止读取敏感文件或凭据目录")

    if candidate.name.casefold().startswith(".env"):
        raise PermissionError("禁止读取环境变量文件")

    if candidate.suffix.casefold() in DENIED_SUFFIXES:
        raise PermissionError("禁止读取密钥文件")

    with candidate.open("rb") as file:
        content = file.read(max_bytes + 1)

    if len(content) > max_bytes:
        raise ToolSizeLimitError(
            f"文件超过最大读取限制：{max_bytes} 字节"
        )

    return content.decode("utf-8")


def _are_numbers(*values: object) -> bool:
    """bool 是 int 的子类，但不作为计算工具的数字参数接受。"""
    return all(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        for value in values
    )


def build_default_tools(
        workspace_root: str | Path
) -> dict[str, Tool]:
    """创建 Agent 默认使用的工具表。"""
    resolved_root = Path(workspace_root).resolve()
    tools = [
        Tool(
            name="calculator",
            description="计算两个数的乘积",
            parameters={"a": "number", "b": "number"},
            func=calculator_func,
        ),
        Tool(
            name="add",
            description="计算两个数的和",
            parameters={"a": "number", "b": "number"},
            func=add_func,
        ),
        Tool(
            name="read_file",
            description="读取指定路径的文件内容",
            parameters={"path": "string"},
            func=partial(
                read_file_func,
                workspace_root=resolved_root,
            ),
        ),
    ]
    return {tool.name: tool for tool in tools}


def build_model_tool_specs(tools: dict[str, Tool]) -> list[dict]:
    """把本地工具转换为 OpenAI-compatible function tool 定义。"""
    specs = []
    for tool in tools.values():
        properties = {
            name: {"type": value_type}
            for name, value_type in tool.parameters.items()
        }
        specs.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": list(properties),
                    "additionalProperties": False,
                },
            },
        })
    return specs
