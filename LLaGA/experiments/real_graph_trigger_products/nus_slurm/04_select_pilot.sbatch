#!/usr/bin/env bash
#SBATCH --job-name=llaga-prod-select
#SBATCH --partition=long
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=/home/z/zitong/llaga_products_real_20260918/logs/select-%j.out
#SBATCH --error=/home/z/zitong/llaga_products_real_20260918/logs/select-%j.err
set -euo pipefail
ROOT=/home/z/zitong/llaga_products_real_20260918
WF=/home/z/zitong/work/graph_phantom/LLaGA/experiments/real_graph_trigger_products
PY=/mnt/scratch/z/zitong/llaga_pubmed/venv/bin/python
"$PY" "$WF/unified_checkpoint_gate.py" select-pilot --states "$ROOT/runs/pilot_0/track_state.json" "$ROOT/runs/pilot_1/track_state.json" "$ROOT/runs/pilot_2/track_state.json" "$ROOT/runs/pilot_3/track_state.json" --top4-triggers "$ROOT/runs/search/top4_triggers.json" --output "$ROOT/runs/pilot_winner.json" --selected-trigger "$ROOT/runs/selected_trigger.json"
