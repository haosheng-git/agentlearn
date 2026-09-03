# Spec: Mini Agent 最小生产化

## Objective

在保留现有同步 `run_agent()` 和安全文件工具的基础上，使项目达到附件定义的最小生产门槛：运行可终止、状态可观察、支持流式事件、具备 OpenAI-compatible 模型适配器和 CLI，并留下可重复的测试证据。

## Tech Stack

- Python 3.12+
- 标准库优先：`urllib`、`json`、`logging`、`threading`、`concurrent.futures`
- pytest 9+
- 不新增运行时第三方依赖

## Commands

```powershell
python -m pytest -q -p no:cacheprovider
python -m compileall -q .
python agent_cli.py --help
```

## Project Structure

- `agent_messages.py`：公开数据协议和流式事件
- `agent_runner.py`：Agent 循环、超时、取消和事件发射
- `agent_models.py`：FakeModel 与 OpenAI-compatible 模型适配器
- `agent_observability.py`：结构化安全日志
- `agent_cli.py`：命令行入口
- `agent_tools.py`：工具注册及安全边界
- `agent_session.py`：Session 持久化、隔离和截断
- `tests/`：按能力分类的 pytest 测试

## Code Style

```python
def emit_event(event: AgentEvent) -> None:
    if event_handler is not None:
        event_handler(event)
```

- 公共字段使用明确类型。
- 边界输入先验证再执行。
- 错误转换为稳定公开类型，不泄露密钥或完整请求体。

## Testing Strategy

- 每项行为先添加失败测试，再实现。
- 单元测试覆盖协议、工具、Session 和日志脱敏。
- 集成测试覆盖 Agent 循环、超时、取消、流式事件和 CLI。
- provider 测试使用本地 fake transport，不进行付费外部调用。
- 完成后运行全部测试、编译检查和 CLI 冒烟。

## Boundaries

- Always：限制循环、模型和工具耗时；验证模型响应；日志字段采用允许列表；API Key 只从环境变量读取。
- Ask first：真实模型网络调用、使用个人 API Key、产生费用、初始化 Git。
- Never：把 API Key、Cookie、完整请求体或 Session 内容写入日志；执行模型返回的任意代码或 shell；失败后污染 Session。

## Success Criteria

- `AgentResult` 包含 `run_id`、`session_id`、`model`、`duration_ms`、计数器、状态和稳定错误类型。
- 模型和工具超时均能结束运行并返回 `timeout_error`。
- 用户取消和 `KeyboardInterrupt` 返回 `cancelled`，失败运行不写入 Session。
- 流式接口按顺序发出运行、模型、工具、文本和结束事件，且总有结束事件。
- OpenAI-compatible 适配器能处理文本、工具调用、非法响应、HTTP 错误、超时和最多一次临时错误重试。
- CLI 能运行 mock 模式，并能在显式配置时运行真实 provider。
- 全部测试、编译检查和 CLI mock 冒烟通过。
- 真实模型冒烟必须由用户提供环境变量并明确授权后执行。

## Open Questions

- 真实模型 provider、base URL、模型名称和 API Key 尚未由用户提供；代码完成后该项保持为外部验收阻断项。
