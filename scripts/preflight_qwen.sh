#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${1:-0.6b}"
if [[ $# -gt 0 ]]; then
  shift
fi

case "${TARGET,,}" in
  .6|.6b|0.6|0.6b|qwen-.6|qwen-0.6b|qwen3-0.6b)
    CONFIG="$ROOT/configs/bench_qwen3_0_6b.json"
    ;;
  4|4b|qwen-4b|qwen3-4b)
    CONFIG="$ROOT/configs/bench_qwen3_4b.json"
    ;;
  *)
    echo "Unknown target '$TARGET'. Choose 0.6b or 4b." >&2
    exit 2
    ;;
esac

cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "${PYTHON:-python}" -m nanoquant.bench.runner preflight --config "$CONFIG" "$@"

