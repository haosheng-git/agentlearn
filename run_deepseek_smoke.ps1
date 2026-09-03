$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
$Host.UI.RawUI.WindowTitle = "DeepSeek API Key"

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
    $outputPath = Join-Path $PSScriptRoot "state\deepseek-smoke-last.log"
    python -X utf8 agent_cli.py `
        --provider deepseek `
        --no-tools `
        --max-retries 0 `
        --max-steps 1 `
        --prompt "Reply with exactly: connection successful" `
        --session-id deepseek-smoke `
        --session-file state/deepseek-smoke.jsonl 2>&1 |
        Tee-Object -FilePath $outputPath
    $smokeExitCode = $LASTEXITCODE
    exit $smokeExitCode
}
finally {
    Remove-Item Env:DEEPSEEK_API_KEY -ErrorAction SilentlyContinue
    $plainKey = $null
    if ($keyPointer -ne [IntPtr]::Zero) {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($keyPointer)
    }
}
