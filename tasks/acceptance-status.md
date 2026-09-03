# Mini Agent 当前验收状态

更新时间：2026-09-03

## 已完成

- 核心 Agent 循环、工具调用、错误映射和部分成功结果保留
- workspace 文件安全边界、敏感路径拒绝和读取大小限制
- Session 加载、快照保存、会话隔离和历史截断
- 模型与工具超时、运行前取消和最大步数限制
- 结构化运行元数据与脱敏日志
- 回调式运行事件流
- OpenAI-compatible Chat Completions provider
- DeepSeek CLI 默认配置
- Mock CLI 与 Session 演示
- Ruff 安装和静态检查

## DeepSeek 配置

- Provider：`deepseek`
- Base URL：`https://api.deepseek.com`
- Model：`deepseek-v4-flash`
- API Key 环境变量：`DEEPSEEK_API_KEY`
- 用户已授权：一次可能产生少量费用的真实调用

## 当前验证证据

- pytest：65 passed
- Ruff：All checks passed
- Python 编译检查：通过
- Mock CLI：通过
- Provider fake transport：通过
- DeepSeek 最小真实连接冒烟：通过
- 真实 Session 快照：已生成，包含 `user` 和 `assistant` 两条消息
- 完整真实 Agent 场景：通过
- 第一轮：真实调用 `read_file`，读取并准确总结 workspace 文件
- 第二轮：加载同一 Session 并准确回答追问
- Session 工具协议：`assistant.tool_calls` 与 `tool.tool_call_id` 匹配

## 最终结论

附件要求的真实模型连接、workspace 文件读取、总结、Session 保存和同一 Session 追问均已完成。项目达到本阶段最小生产验收标准。
