# Mini Agent Core

## DeepSeek V4 Flash

项目内置以下官方兼容配置：

- provider：`deepseek`
- base URL：`https://api.deepseek.com`
- model：`deepseek-v4-flash`
- API Key 环境变量：`DEEPSEEK_API_KEY`

请勿把 API Key 写进源码、README、命令参数或 Session 文件。先在准备运行命令的同一个 PowerShell 窗口中设置环境变量，再执行一次无工具、无重试的最小真实冒烟：

```powershell
$env:DEEPSEEK_API_KEY = "在本机填写你的Key，不要发到聊天中"
python agent_cli.py --provider deepseek --no-tools --max-retries 0 --max-steps 1 --prompt "请只回复：连接成功" --session-id deepseek-smoke --session-file state/deepseek-smoke.jsonl
```

`--no-tools` 防止本次冒烟进入工具循环，`--max-retries 0` 禁止 HTTP 重试，`--max-steps 1` 保证本次运行最多调用模型一次。成功后应看到正常回答和非空运行 ID，且 `state/deepseek-smoke.jsonl` 中新增一条 Session 快照。

一个使用 Python 标准库实现的最小 Agent 核心，包含工具循环、安全文件读取、Session、超时、取消、流式事件、结构化运行结果、OpenAI-compatible provider 和 CLI。

## 本地验证

```powershell
python -m pytest -q -p no:cacheprovider
python -m py_compile agent_messages.py agent_tools.py agent_session.py agent_observability.py agent_models.py agent_runner.py agent_demo.py agent_cli.py
python agent_cli.py --provider mock --prompt "你好" --mock-response "本地回答"
```

## CLI

Mock 模式不会访问网络：

```powershell
python agent_cli.py --provider mock --prompt "你好"
```

真实 OpenAI-compatible provider 只从环境变量读取 API Key：

```powershell
$env:OPENAI_API_KEY = "在本机设置，不要写入代码或日志"
python agent_cli.py --provider openai --base-url "https://provider.example/v1" --model "model-name" --prompt "你好"
```

## 安全边界

- 文件工具只允许访问配置的 workspace。
- 默认拒绝环境变量文件、凭据目录和密钥文件。
- 模型和工具均有超时，Agent 循环受 `max_steps` 限制。
- 失败、超时和取消运行不会写入 Session。
- 结构化日志不记录提示词、工具参数、工具结果、回答或 API Key。

## 已知限制

- Python 线程不能安全强杀已经开始运行的第三方函数；工具应支持幂等和协作取消。
- 真实模型冒烟需要用户提供 provider、模型和环境变量凭据，并明确授权外部调用。
- 项目当前不是 Git 仓库，因此没有版本提交记录。
