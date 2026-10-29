#!/bin/bash
# shared-key INDEX fidelity leg: 2 arms x 720 cells x 2 models = 2,880 calls.
# Same口径 as run_index_stress.py (temp0, seed1234, num_predict40, num_ctx8192, nonce per run).
# Sequential (single ollama server). Resumable by trial_id; safe to re-run.
cd "$(dirname "$0")" || exit 1
NONCE="$(date +%s)"
echo "=== shared-key leg start nonce=$NONCE $(date -u +%FT%TZ) ==="
for M in qwen3:14b llama3:latest; do
  TAG="${M//:/_}"
  OUT="index_sk_scored_${TAG}.jsonl"
  echo "--- [$(date -u +%FT%TZ)] serving $M -> $OUT ---"
  IX_MODEL="$M" IX_TRIALS=index_sk_trials.jsonl IX_OUT="$OUT" IX_NUMCTX=8192 IX_NONCE="${NONCE}_${TAG}" \
    python3 run_index_stress.py
  echo "--- [$(date -u +%FT%TZ)] done $M rows=$(wc -l < "$OUT") ---"
done
echo "=== shared-key leg END $(date -u +%FT%TZ) ==="
