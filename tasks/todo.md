# Mini Agent 最小生产标准推进清单

- [x] Task 1：扩展返回协议和结构化安全日志
  - 文件：`agent_messages.py` `agent_observability.py` `test_observability.py`

- [x] Task 2：实现模型超时、工具超时和取消
  - 文件：`agent_runner.py` `agent_messages.py` `test_reliability.py`

- [x] Task 3：实现回调式运行事件流
  - 文件：`agent_runner.py` `agent_messages.py` `test_streaming.py`

- [x] Task 4：实现 OpenAI-compatible provider
  - 文件：`agent_models.py` `test_models.py`

- [x] Task 5：实现 CLI 和 DeepSeek 默认配置
  - 文件：`agent_cli.py` `test_cli.py`

- [x] Task 6：完成发布说明、静态检查和本地验收
  - 文件：`README.md` `pyproject.toml` `tasks/acceptance-status.md`

- [x] Task 7A：DeepSeek 最小真实连接冒烟
  - Provider：`deepseek`
  - Base URL：`https://api.deepseek.com`
  - Model：`deepseek-v4-flash`
  - API Key 环境变量：`DEEPSEEK_API_KEY`
  - 结果：真实回答成功，Session 快照成功

- [x] Task 7B：完整真实 Agent 场景
  - 读取 workspace 文件并总结
  - 保存 Session
  - 使用同一 Session 回答追问
  - 结果：真实文件读取、总结、Session 保存和同 Session 追问全部通过

## 当前验证证据

- pytest：65 passed
- Ruff：All checks passed
- Python 编译检查：通过
- Mock CLI：通过
- Provider fake transport：通过
- DeepSeek 最小真实连接冒烟：通过
- DeepSeek 完整真实 Agent 场景：通过
