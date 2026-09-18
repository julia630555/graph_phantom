#!/usr/bin/env bash
#SBATCH --job-name=llaga-prod-assets
#SBATCH --partition=long
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/home/z/zitong/llaga_products_real_20260918/logs/assets-%j.out
#SBATCH --error=/home/z/zitong/llaga_products_real_20260918/logs/assets-%j.err
set -euo pipefail
ROOT=/home/z/zitong/llaga_products_real_20260918
WORKFLOW=/home/z/zitong/work/graph_phantom/LLaGA/experiments/real_graph_trigger_products
mkdir -p "$ROOT/assets/ogbn-products" "$ROOT/logs" "$ROOT/runs" "$ROOT/tmp"
ASSET_DIR="$ROOT/assets/ogbn-products" INCLUDE_TEST=0 bash "$WORKFLOW/download_assets_h200.sh"
