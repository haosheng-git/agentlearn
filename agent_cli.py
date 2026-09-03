"""Mini Agent 命令行入口。"""

import argparse
import sys
from pathlib import Path

from agent_demo import print_agent_result
from agent_models import OpenAICompatibleModel
from agent_runner import FakeModel, run_agent
from agent_session import JsonSessionStore
from agent_tools import build_default_tools, build_model_tool_specs

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-v4-flash"
DEEPSEEK_API_KEY_ENV = "DEEPSEEK_API_KEY"


def build_parser():
    parser = argparse.ArgumentParser(description="Mini Agent Core")
    parser.add_argument(
        "--provider",
        choices=["mock", "openai", "deepseek"],
        default="mock",
    )
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--session-id", default="cli-session")
    parser.add_argument("--workspace", default="workspace")
    parser.add_argument("--session-file", default="state/sessions.jsonl")
    parser.add_argument("--mock-response", default="这是本地模拟回答")
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--api-key-env")
    parser.add_argument("--max-retries", type=int, choices=[0, 1], default=1)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--no-tools", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    tools = (
        {}
        if args.no_tools
        else build_default_tools(Path(args.workspace))
    )
    if args.provider in {"openai", "deepseek"}:
        if args.provider == "deepseek":
            base_url = args.base_url or DEEPSEEK_BASE_URL
            model_name = args.model or DEEPSEEK_MODEL
            api_key_env = args.api_key_env or DEEPSEEK_API_KEY_ENV
        else:
            base_url = args.base_url
            model_name = args.model
            api_key_env = args.api_key_env or "OPENAI_API_KEY"
        if not base_url or not model_name:
            print("openai模式必须提供--base-url和--model", file=sys.stderr)
            return 2
        model = OpenAICompatibleModel(
            base_url=base_url,
            model=model_name,
            api_key_env=api_key_env,
            max_retries=args.max_retries,
            tools=build_model_tool_specs(tools),
        )
    else:
        model = FakeModel([args.mock_response])

    result = run_agent(
        model,
        tools,
        args.prompt,
        session_id=args.session_id,
        session_store=JsonSessionStore(args.session_file),
        max_steps=args.max_steps,
    )
    print_agent_result(result)
    return 0 if result.status == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
