param(
    [string]$RepoRoot = "$(Split-Path -Parent $MyInvocation.MyCommand.Path)\..\.."
)

Set-Location $RepoRoot

python -m ruff hopper
python -m black --check hopper
clang-format -n hopper/**/*.cu
python tools/lint/check_tokens.py
python tools/lint/check_includes.py
python tools/lint/grep_arch_flags.py
