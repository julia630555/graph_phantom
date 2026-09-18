#!/usr/bin/env bash
#SBATCH --job-name=llaga-prod-test-merge
#SBATCH --partition=long
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=/home/z/zitong/llaga_products_real_20260918/logs/test-merge-%j.out
#SBATCH --error=/home/z/zitong/llaga_products_real_20260918/logs/test-merge-%j.err
set -euo pipefail
ROOT=/home/z/zitong/llaga_products_real_20260918
WF=/home/z/zitong/work/graph_phantom/LLaGA/experiments/real_graph_trigger_products
PY=/mnt/scratch/z/zitong/llaga_pubmed/venv/bin/python
"$PY" "$WF/merge_final_test_shards.py" --input-dir "$ROOT/runs/official_test_once/shards" --output-dir "$ROOT/runs/official_test_once/full_metrics" --metrics-script "$WF/calc_products_metrics.py" --python "$PY"
