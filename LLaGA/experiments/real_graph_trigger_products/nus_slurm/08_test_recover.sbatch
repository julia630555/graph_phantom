#!/usr/bin/env bash
# Recover missing official-test shards after the main test array.
# Existing metrics are immutable and skipped; only incomplete shards rerun.
#SBATCH --job-name=llaga-prod-test-recover
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:h100-47:4
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --output=/home/z/zitong/llaga_products_real_20260918/logs/test-recover-%j.out
#SBATCH --error=/home/z/zitong/llaga_products_real_20260918/logs/test-recover-%j.err
set -euo pipefail
ROOT=/home/z/zitong/llaga_products_real_20260918
WF=/home/z/zitong/work/graph_phantom/LLaGA/experiments/real_graph_trigger_products
SH=/mnt/scratch/z/zitong/llaga_pubmed
PY="$SH/venv/bin/python"
PD=/mnt/scratch/z/zitong/llaga_arxiv/pydeps
export PYTHONPATH="$PD:${PYTHONPATH:-}" HF_HOME="$ROOT/hf_cache" TRANSFORMERS_CACHE="$ROOT/hf_cache" TMPDIR="$ROOT/tmp"
for d in "$SH"/venv/lib/python*/site-packages/nvidia/nvjitlink/lib "$SH"/venv/lib/python*/site-packages/nvidia/cusparse/lib; do [[ -d "$d" ]] && export LD_LIBRARY_PATH="$d:${LD_LIBRARY_PATH:-}"; done
CHECKPOINT=$("$PY" -c 'import json,sys;print(json.load(open(sys.argv[1]))["best_checkpoint"]["checkpoint_dir"])' "$ROOT/runs/formal/track_state.json")
export LLAGA_CODE_ROOT="$SH/graph_phantom/LLaGA"
worker() {
  local gpu="$1" shard index input
  export CUDA_VISIBLE_DEVICES="$gpu"
  shard="$gpu"
  while (( shard < 128 )); do
    printf -v index '%03d' "$shard"
    input="$ROOT/runs/official_test_once/shards/shard_$index"
    if [[ -s "$input/outputs/metrics.json" ]]; then
      echo "[skip] shard=$index already complete"
    else
      echo "[run] shard=$index gpu=$gpu"
      "$PY" "$WF/eval_products_probe.py" --model-path "$CHECKPOINT" --model-base "$SH/assets/vicuna-7b-v1.5-16k" --data-dir "$ROOT/assets/ogbn-products" --input-dir "$input" --output-dir "$input/outputs" --structure-emb "$ROOT/assets/ogbn-products/laplacian_2_10.pt" --metrics-script "$WF/calc_products_metrics.py" --python "$PY" --tag "official_test_shard_$index"
    fi
    shard=$((shard + 8))
  done
}
for gpu in 0 1 2 3; do worker "$gpu" & done
failed=0
for pid in $(jobs -p); do wait "$pid" || failed=1; done
(( failed == 0 ))
