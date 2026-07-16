#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${1:-0.6b}"
if [[ $# -gt 0 ]]; then
  shift
fi

# Quick changes only evaluation size/repeats. It deliberately preserves the
# full checkpoint quantization recipe so smoke and full results can resume the
# same expensive checkpoint.
exec "$ROOT/scripts/benchmark_qwen.sh" "$TARGET" --quick "$@"

