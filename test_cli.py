from agent_cli import (
    DEEPSEEK_API_KEY_ENV,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    build_parser,
    main,
)


def test_deepseek_cli_uses_official_defaults():
    args = build_parser().parse_args([
        "--provider", "deepseek",
        "--prompt", "hello",
    ])

    assert args.provider == "deepseek"
    assert args.base_url is None
    assert args.model is None
    assert args.api_key_env is None
    assert DEEPSEEK_BASE_URL == "https://api.deepseek.com"
    assert DEEPSEEK_MODEL == "deepseek-v4-flash"
    assert DEEPSEEK_API_KEY_ENV == "DEEPSEEK_API_KEY"


def test_deepseek_cli_passes_official_defaults_to_model(
    tmp_path,
    monkeypatch,
):
    captured = {}

    def build_model(**kwargs):
        captured.update(kwargs)
        from agent_runner import FakeModel

        return FakeModel(["ok"])

    monkeypatch.setattr("agent_cli.OpenAICompatibleModel", build_model)
    code = main([
        "--provider", "deepseek",
        "--prompt", "hello",
        "--no-tools",
        "--max-retries", "0",
        "--max-steps", "1",
        "--workspace", str(tmp_path),
        "--session-file", str(tmp_path / "sessions.jsonl"),
    ])

    assert code == 0
    assert captured["base_url"] == DEEPSEEK_BASE_URL
    assert captured["model"] == DEEPSEEK_MODEL
    assert captured["api_key_env"] == DEEPSEEK_API_KEY_ENV
    assert captured["max_retries"] == 0
    assert captured["tools"] == []


def test_mock_cli_runs_without_external_service(tmp_path, capsys):
    code = main([
        "--provider", "mock",
        "--prompt", "你好",
        "--mock-response", "本地回答",
        "--workspace", str(tmp_path),
        "--session-file", str(tmp_path / "sessions.jsonl"),
    ])

    output = capsys.readouterr().out
    assert code == 0
    assert "本地回答" in output
    assert "运行ID" in output


def test_real_cli_rejects_missing_model_configuration(capsys):
    code = main(["--provider", "openai", "--prompt", "你好"])

    assert code == 2
    assert "必须提供" in capsys.readouterr().err
