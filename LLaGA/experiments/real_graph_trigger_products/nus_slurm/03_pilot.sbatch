#!/usr/bin/env bash
#SBATCH --job-name=llaga-prod-pilot
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:h100-47:1
#SBATCH --array=0-3
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=/home/z/zitong/llaga_products_real_20260918/logs/pilot-%A_%a.out
#SBATCH --error=/home/z/zitong/llaga_products_real_20260918/logs/pilot-%A_%a.err
set -euo pipefail
ROOT=/home/z/zitong/llaga_products_real_20260918
WORKFLOW=/home/z/zitong/work/graph_phantom/LLaGA/experiments/real_graph_trigger_products
SHARED=/mnt/scratch/z/zitong/llaga_pubmed
PYTHON_BIN="$SHARED/venv/bin/python"
PYDEPS=/mnt/scratch/z/zitong/llaga_arxiv/pydeps
IDX="${SLURM_ARRAY_TASK_ID}"
TRIGGER_JSON="$ROOT/runs/pilot_${IDX}_trigger.json"
"$PYTHON_BIN" -c 'import json,sys;p=json.load(open(sys.argv[1]));json.dump({"selected_node_ids":p[int(sys.argv[3])]},open(sys.argv[2],"w"),indent=2)' "$ROOT/runs/search/top4_triggers.json" "$TRIGGER_JSON" "$IDX"
export PYTHONPATH="$PYDEPS:${PYTHONPATH:-}" HF_HOME="$ROOT/hf_cache" TRANSFORMERS_CACHE="$ROOT/hf_cache" TMPDIR="$ROOT/tmp" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for d in "$SHARED"/venv/lib/python*/site-packages/nvidia/nvjitlink/lib "$SHARED"/venv/lib/python*/site-packages/nvidia/cusparse/lib; do [[ -d "$d" ]] && export LD_LIBRARY_PATH="$d:${LD_LIBRARY_PATH:-}"; done
ASSET_DIR="$ROOT/assets/ogbn-products" VALIDATION_DIR="$ROOT/assets/validation/ogbn-products" MODEL_DIR="$SHARED/graph_phantom/model_assets/upstream_projectors/llaga-vicuna-7b-simteg-ND-classification_expert-linear-projector" BASE_MODEL="$SHARED/assets/vicuna-7b-v1.5-16k" CLEAN_PROJECTOR="$SHARED/graph_phantom/model_assets/upstream_projectors/llaga-vicuna-7b-simteg-ND-classification_expert-linear-projector/mm_projector.bin" LLAGA_CODE_ROOT="$SHARED/graph_phantom/LLaGA" PHASE0_DIR="$ROOT/runs/phase0_protocol_freeze" RUN_DIR="$ROOT/runs/pilot_${IDX}" TRIGGER_JSON="$TRIGGER_JSON" MAX_STEPS=20000 TRACK_STEP_LIMIT="${TRACK_STEP_LIMIT:?set via sbatch --export}" PYTHON_BIN="$PYTHON_BIN" bash "$WORKFLOW/run_projector_track.sh"
