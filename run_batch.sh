#!/usr/bin/env bash
set -euo pipefail
IN="${1:-$PWD/testpdf}"
OUT="${2:-$PWD/cuts}"
MODEL="${3:-$PWD/Carbon_xlmr_lora}"
THRESH="${4:-0.25}"

docker run --rm --user 0:0 \
  -v "$IN:/in:ro" \
  -v "$OUT:/out" \
  -v "$MODEL:/models/Carbon_xlmr_lora:ro" \
  carbon-filter:latest \
  /in \
  --adapter /models/Carbon_xlmr_lora \
  --base-model xlm-roberta-base \
  --threshold "$THRESH" \
  --out-root /out
