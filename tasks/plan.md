# Implementation Plan: Mini Agent 最小生产化

## Overview

按“协议 → 可靠性 → 流式 → provider → CLI → 发布检查”的依赖顺序，以小增量推进，每个增量先测试后实现。

## Architecture Decisions

- 保留同步 `run_agent()`，新增事件回调和生成器包装，避免破坏现有调用方。
- 超时由运行器和 provider 双层限制；取消使用 `threading.Event` 协作检查。
- provider 使用可注入 transport，测试不访问外网。
- 日志仅记录固定元数据，不记录提示词、工具参数、工具内容或凭据。

## Task List

### Phase 1: Foundation

- Task 1：扩展运行协议和结构化安全日志。
- Task 2：实现模型与工具超时、取消和稳定状态。
- Checkpoint：相关测试和原有测试全部通过。

### Phase 2: Runtime Experience

- Task 3：实现有序流式事件和中断结束事件。
- Task 4：实现 OpenAI-compatible provider 与受控重试。
- Checkpoint：本地 fake transport 集成测试通过。

### Phase 3: Product Surface

- Task 5：实现 CLI、环境变量配置和 mock 演示。
- Task 6：整理测试分类、README、静态和发布检查。
- Checkpoint：全量测试、编译、CLI mock 冒烟通过。

### External Acceptance

- Task 7：在用户提供凭据并明确授权后执行一次真实模型冒烟。

## Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Python 线程超时不能强制终止已运行函数 | 中 | 超时后结束 Agent、取消未开始任务，并明确工具应具备幂等或协作取消能力 |
| provider 返回格式差异 | 中 | 严格解析边界并返回 `model_error` |
| 日志泄露敏感信息 | 高 | 固定字段允许列表并增加脱敏测试 |
| 真实调用产生费用 | 高 | 未经明确授权不发起外部请求 |

## Open Questions

- 真实模型冒烟所用 provider 和凭据等待用户授权。
