#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${1:-0.6b}"
if [[ $# -gt 0 ]]; then
  shift
fi
BACKEND="${NQX_BACKEND:-gemv}"

exec "$ROOT/scripts/benchmark_qwen.sh" "$TARGET" --backend "$BACKEND" "$@"

