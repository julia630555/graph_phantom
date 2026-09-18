#!/usr/bin/env bash
#SBATCH --job-name=llaga-prod-test-open
#SBATCH --partition=long
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=/home/z/zitong/llaga_products_real_20260918/logs/test-open-%j.out
#SBATCH --error=/home/z/zitong/llaga_products_real_20260918/logs/test-open-%j.err
set -euo pipefail
ROOT=/home/z/zitong/llaga_products_real_20260918
WF=/home/z/zitong/work/graph_phantom/LLaGA/experiments/real_graph_trigger_products
SH=/mnt/scratch/z/zitong/llaga_pubmed
PY="$SH/venv/bin/python"
PD=/mnt/scratch/z/zitong/llaga_arxiv/pydeps
export PYTHONPATH="$PD:${PYTHONPATH:-}" HF_HOME="$ROOT/hf_cache" TMPDIR="$ROOT/tmp"
for d in "$SH"/venv/lib/python*/site-packages/nvidia/nvjitlink/lib "$SH"/venv/lib/python*/site-packages/nvidia/cusparse/lib; do [[ -d "$d" ]] && export LD_LIBRARY_PATH="$d:${LD_LIBRARY_PATH:-}"; done
ASSET_DIR="$ROOT/assets/ogbn-products" FOLD_B_DIR="$ROOT/runs/fold_b" TRIGGER_JSON="$ROOT/runs/selected_trigger.json" RUN_DIR="$ROOT/runs/official_test_once" PYTHON_BIN="$PY" TEST_SHARDS=128 bash "$WF/run_final_test_once.sh"
