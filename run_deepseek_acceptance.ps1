$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
$Host.UI.RawUI.WindowTitle = "DeepSeek Production Acceptance"

$secureKey = Read-Host "Enter DeepSeek API Key" -AsSecureString
$keyPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureKey)
$plainKey = $null

try {
    $plainKey = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($keyPointer)
    if ([string]::IsNullOrWhiteSpace($plainKey)) {
        Write-Error "API Key cannot be empty"
        exit 2
    }

    $env:DEEPSEEK_API_KEY = $plainKey
    $sessionId = "deepseek-production-acceptance-20260903"
    $sessionFile = "state/deepseek-production-acceptance.jsonl"

    python -X utf8 agent_cli.py `
        --provider deepseek `
        --max-retries 0 `
        --max-steps 2 `
        --workspace workspace `
        --session-id $sessionId `
        --session-file $sessionFile `
        --prompt "Use read_file exactly once to read production-smoke.txt. Then report the project name, release candidate number, safety rule, and session rule." 2>&1 |
        Tee-Object -FilePath "state/deepseek-acceptance-turn-1.log"
    $firstExitCode = $LASTEXITCODE
    if ($firstExitCode -ne 0) {
        exit $firstExitCode
    }

    python -X utf8 agent_cli.py `
        --provider deepseek `
        --no-tools `
        --max-retries 0 `
        --max-steps 1 `
        --workspace workspace `
        --session-id $sessionId `
        --session-file $sessionFile `
        --prompt "Using the previous conversation, reply with only the project name and release candidate number." 2>&1 |
        Tee-Object -FilePath "state/deepseek-acceptance-turn-2.log"
    exit $LASTEXITCODE
}
finally {
    Remove-Item Env:DEEPSEEK_API_KEY -ErrorAction SilentlyContinue
    $plainKey = $null
    if ($keyPointer -ne [IntPtr]::Zero) {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($keyPointer)
    }
}
