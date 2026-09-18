#!/usr/bin/env bash
#SBATCH --job-name=llaga-prod-test-shard
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:h100-47:1
#SBATCH --array=0-127%4
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=/home/z/zitong/llaga_products_real_20260918/logs/test-shard-%A_%a.out
#SBATCH --error=/home/z/zitong/llaga_products_real_20260918/logs/test-shard-%A_%a.err
set -euo pipefail
ROOT=/home/z/zitong/llaga_products_real_20260918
WF=/home/z/zitong/work/graph_phantom/LLaGA/experiments/real_graph_trigger_products
SH=/mnt/scratch/z/zitong/llaga_pubmed
PY="$SH/venv/bin/python"
PD=/mnt/scratch/z/zitong/llaga_arxiv/pydeps
export CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PD:${PYTHONPATH:-}" HF_HOME="$ROOT/hf_cache" TRANSFORMERS_CACHE="$ROOT/hf_cache" TMPDIR="$ROOT/tmp"
for d in "$SH"/venv/lib/python*/site-packages/nvidia/nvjitlink/lib "$SH"/venv/lib/python*/site-packages/nvidia/cusparse/lib; do [[ -d "$d" ]] && export LD_LIBRARY_PATH="$d:${LD_LIBRARY_PATH:-}"; done
printf -v INDEX '%03d' "$SLURM_ARRAY_TASK_ID"
INPUT="$ROOT/runs/official_test_once/shards/shard_$INDEX"
CHECKPOINT="$("$PY" -c 'import json,sys;print(json.load(open(sys.argv[1]))["best_checkpoint"]["checkpoint_dir"])' "$ROOT/runs/formal/track_state.json")"
export LLAGA_CODE_ROOT="$SH/graph_phantom/LLaGA"
"$PY" "$WF/eval_products_probe.py" --model-path "$CHECKPOINT" --model-base "$SH/assets/vicuna-7b-v1.5-16k" --data-dir "$ROOT/assets/ogbn-products" --input-dir "$INPUT" --output-dir "$INPUT/outputs" --structure-emb "$ROOT/assets/ogbn-products/laplacian_2_10.pt" --metrics-script "$WF/calc_products_metrics.py" --python "$PY" --tag "official_test_shard_$INDEX"
