#!/usr/bin/env bash
#SBATCH --job-name=llaga-prod-freeze
#SBATCH --partition=long
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=/home/z/zitong/llaga_products_real_20260918/logs/freeze-%j.out
#SBATCH --error=/home/z/zitong/llaga_products_real_20260918/logs/freeze-%j.err
set -euo pipefail
ROOT=/home/z/zitong/llaga_products_real_20260918
WORKFLOW=/home/z/zitong/work/graph_phantom/LLaGA/experiments/real_graph_trigger_products
STAGE=/home/z/zitong/work/products_nus_20260918
SHARED=/mnt/scratch/z/zitong/llaga_pubmed
PYTHON_BIN="$SHARED/venv/bin/python"
ASSET_DIR="$ROOT/assets/ogbn-products"
VALIDATION_DIR="$ROOT/assets/validation/ogbn-products"
RUN_DIR="$ROOT/runs/phase0_protocol_freeze"
BASE_MODEL="$SHARED/assets/vicuna-7b-v1.5-16k"
CLEAN_PROJECTOR="$SHARED/graph_phantom/model_assets/upstream_projectors/llaga-vicuna-7b-simteg-ND-classification_expert-linear-projector/mm_projector.bin"
PYDEPS=/mnt/scratch/z/zitong/llaga_arxiv/pydeps
LLAGA_CODE_ROOT="$SHARED/graph_phantom/LLaGA"
export OMP_NUM_THREADS=1 LLAGA_CODE_ROOT
export PYTHONPATH="$PYDEPS:$WORKFLOW:$LLAGA_CODE_ROOT:${PYTHONPATH:-}"
for d in "$SHARED"/venv/lib/python*/site-packages/nvidia/nvjitlink/lib "$SHARED"/venv/lib/python*/site-packages/nvidia/cusparse/lib; do [[ -d "$d" ]] && export LD_LIBRARY_PATH="$d:${LD_LIBRARY_PATH:-}"; done
mkdir -p "$VALIDATION_DIR"
"$PYTHON_BIN" "$STAGE/build_products_val_jsonl.py" --processed-data "$ASSET_DIR/processed_data.pt" --train-jsonl "$ASSET_DIR/sampled_2_10_train.jsonl" --metadata-jsonl "$VALIDATION_DIR/clean_products_val.jsonl" --output-jsonl "$VALIDATION_DIR/sampled_2_10_val.jsonl"
ASSET_DIR="$ASSET_DIR" VALIDATION_DIR="$VALIDATION_DIR" RUN_DIR="$RUN_DIR" PYTHON_BIN="$PYTHON_BIN" CLEAN_PROJECTOR="$CLEAN_PROJECTOR" BASE_MODEL_CONFIG="$BASE_MODEL/config.json" bash "$WORKFLOW/run_phase0_prepare_h200.sh"
"$PYTHON_BIN" -c 'import json,sys;p=json.load(open(sys.argv[1])); assert p["status"]=="passed"; c=p["training"]["selection"]["selected_counts_by_source"]; assert len(set(c.values()))==1; s=p["validation"]["search_ids"]["label_counts"]; assert len(set(s.values()))==1; print("protocol gate passed")' "$RUN_DIR/protocol_manifest.json"
