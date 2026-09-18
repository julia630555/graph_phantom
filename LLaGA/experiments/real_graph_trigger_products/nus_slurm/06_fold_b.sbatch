#!/usr/bin/env bash
#SBATCH --job-name=llaga-prod-foldb
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:h100-47:1
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=/home/z/zitong/llaga_products_real_20260918/logs/foldb-%j.out
#SBATCH --error=/home/z/zitong/llaga_products_real_20260918/logs/foldb-%j.err
set -euo pipefail
ROOT=/home/z/zitong/llaga_products_real_20260918
WF=/home/z/zitong/work/graph_phantom/LLaGA/experiments/real_graph_trigger_products
SH=/mnt/scratch/z/zitong/llaga_pubmed
PY="$SH/venv/bin/python"
PD=/mnt/scratch/z/zitong/llaga_arxiv/pydeps
export CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PD:${PYTHONPATH:-}" HF_HOME="$ROOT/hf_cache" TRANSFORMERS_CACHE="$ROOT/hf_cache" TMPDIR="$ROOT/tmp"
for d in "$SH"/venv/lib/python*/site-packages/nvidia/nvjitlink/lib "$SH"/venv/lib/python*/site-packages/nvidia/cusparse/lib; do [[ -d "$d" ]] && export LD_LIBRARY_PATH="$d:${LD_LIBRARY_PATH:-}"; done
ASSET_DIR="$ROOT/assets/ogbn-products" VALIDATION_DIR="$ROOT/assets/validation/ogbn-products" MODEL_DIR="$SH/graph_phantom/model_assets/upstream_projectors/llaga-vicuna-7b-simteg-ND-classification_expert-linear-projector" BASE_MODEL="$SH/assets/vicuna-7b-v1.5-16k" LLAGA_CODE_ROOT="$SH/graph_phantom/LLaGA" PHASE0_DIR="$ROOT/runs/phase0_protocol_freeze" FORMAL_DIR="$ROOT/runs/formal" TRIGGER_JSON="$ROOT/runs/selected_trigger.json" RUN_DIR="$ROOT/runs/fold_b" PYTHON_BIN="$PY" bash "$WF/run_fold_b.sh"
