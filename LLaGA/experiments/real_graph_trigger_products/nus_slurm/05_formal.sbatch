#!/usr/bin/env bash
#SBATCH --job-name=llaga-prod-formal
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:h100-47:1
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=/home/z/zitong/llaga_products_real_20260918/logs/formal-%j.out
#SBATCH --error=/home/z/zitong/llaga_products_real_20260918/logs/formal-%j.err
set -euo pipefail
ROOT=/home/z/zitong/llaga_products_real_20260918
WF=/home/z/zitong/work/graph_phantom/LLaGA/experiments/real_graph_trigger_products
SH=/mnt/scratch/z/zitong/llaga_pubmed
PY="$SH/venv/bin/python"
PD=/mnt/scratch/z/zitong/llaga_arxiv/pydeps
export PYTHONPATH="$PD:${PYTHONPATH:-}" HF_HOME="$ROOT/hf_cache" TRANSFORMERS_CACHE="$ROOT/hf_cache" TMPDIR="$ROOT/tmp" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for d in "$SH"/venv/lib/python*/site-packages/nvidia/nvjitlink/lib "$SH"/venv/lib/python*/site-packages/nvidia/cusparse/lib; do [[ -d "$d" ]] && export LD_LIBRARY_PATH="$d:${LD_LIBRARY_PATH:-}"; done
ASSET_DIR="$ROOT/assets/ogbn-products" VALIDATION_DIR="$ROOT/assets/validation/ogbn-products" MODEL_DIR="$SH/graph_phantom/model_assets/upstream_projectors/llaga-vicuna-7b-simteg-ND-classification_expert-linear-projector" BASE_MODEL="$SH/assets/vicuna-7b-v1.5-16k" CLEAN_PROJECTOR="$SH/graph_phantom/model_assets/upstream_projectors/llaga-vicuna-7b-simteg-ND-classification_expert-linear-projector/mm_projector.bin" LLAGA_CODE_ROOT="$SH/graph_phantom/LLaGA" PHASE0_DIR="$ROOT/runs/phase0_protocol_freeze" RUN_DIR="$ROOT/runs/formal" TRIGGER_JSON="$ROOT/runs/selected_trigger.json" MAX_STEPS=30000 TRACK_STEP_LIMIT="${TRACK_STEP_LIMIT:?set via sbatch --export}" PYTHON_BIN="$PY" bash "$WF/run_projector_track.sh"
