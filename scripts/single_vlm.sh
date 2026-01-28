#!/usr/bin/env bash
set -euo pipefail

PY=python3
SCRIPT=single_vlm.py

DOC_ROOT="data/Visdom"
OUT_ROOT="Single-llm_local/result"

DATASETS=(
  "data/new/processed_scigraphvqa.jsonl"
)

MODELS=(
  "qwen3-vl-235b-a22b-instruct"
)

export OPENAI_BASE_URL="${OPENAI_BASE_URL:-<YOUR_BASE_URL>}"
export OPENAI_MODEL="${OPENAI_MODEL:-<YOUR_MODEL_NAME>}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-<YOUR_API_KEY>}"

SEED=42
NUM_WORKERS=20
MAX_PDFS=5
MAX_CHUNKS=10
MAX_IMAGES=10

for d in "${DATASETS[@]}"; do
  for m in "${MODELS[@]}"; do
    echo "=== RUN infer only | seed=${SEED} | model=${m} | dataset=${d}"
    $PY $SCRIPT \
      --input_path "$d" \
      --doc_root "$DOC_ROOT" \
      --out_root "$OUT_ROOT" \
      --seed "$SEED" \
      --model "$m" \
      --mode "infer" \
      --num_workers "$NUM_WORKERS"
  done
done