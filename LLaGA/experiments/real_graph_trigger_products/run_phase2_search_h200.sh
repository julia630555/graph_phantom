#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASSET_DIR="${ASSET_DIR:-/home/zitong/graph_phantom_assets/ogbn-products}"
VALIDATION_DIR="${VALIDATION_DIR:-/home/zitong/graph_phantom_assets/validation/ogbn-products}"
MODEL_ROOT="${MODEL_ROOT:-/home/zitong/graph_phantom_models}"
LLAGA_CODE_ROOT="${LLAGA_CODE_ROOT:-/home/zitong/work/LLaGA-upstream}"
PHASE0_DIR="${PHASE0_DIR:-/home/zitong/graph_phantom_runs/products/phase0_protocol_freeze}"
RUN_DIR="${RUN_DIR:-/home/zitong/graph_phantom_runs/products/phase2_search_$(date +%Y%m%d_%H%M%S)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
BASE_MODEL="${BASE_MODEL:-${MODEL_ROOT}/vicuna-7b-v1.5-16k}"
MODEL_DIR="${MODEL_DIR:-${MODEL_ROOT}/llaga-vicuna-7b-simteg-ND-classification_expert-linear-projector}"
CLEAN_PROJECTOR="${CLEAN_PROJECTOR:-${MODEL_DIR}/mm_projector.bin}"
ROUNDS="${ROUNDS:-3}"

export LLAGA_CODE_ROOT PYTHONPATH="${SCRIPT_DIR}:${LLAGA_CODE_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 WANDB_DISABLED=true
candidate_pool="${PHASE0_DIR}/candidates/candidate_pool.json"
search_ids="${PHASE0_DIR}/splits/search_ids.json"
mkdir -p "${RUN_DIR}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/make_initial_search_sets.py" \
  --candidate-pool "${candidate_pool}" --sets-out "${RUN_DIR}/round1_sets.json" \
  --baseline-out "${RUN_DIR}/baseline.json"
mapfile -t baseline < <("${PYTHON_BIN}" -c 'import json,sys;print("\n".join(map(str,json.load(open(sys.argv[1]))["selected_node_ids"])))' "${RUN_DIR}/baseline.json")

results=()
sets="${RUN_DIR}/round1_sets.json"
for round in $(seq 1 "${ROUNDS}"); do
  result="${RUN_DIR}/round${round}_scores.json"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/score_products_hard_trigger_nll.py" \
    --model-path "${MODEL_DIR}" --model-base "${BASE_MODEL}" \
    --pretrain-mm-mlp-adapter "${CLEAN_PROJECTOR}" --data-dir "${ASSET_DIR}" \
    --source-val-jsonl "${VALIDATION_DIR}/sampled_2_10_val.jsonl" \
    --hard-ids "${VALIDATION_DIR}/val_hard_ids.json" --search-ids "${search_ids}" \
    --structure-emb-path "${ASSET_DIR}/laplacian_2_10.pt" --candidate-sets-json "${sets}" \
    --output-json "${result}" --target-label "Video Games" --max-samples 32 --device cuda:0 \
    >"${RUN_DIR}/round${round}.log" 2>&1
  results+=("${result}")
  if [[ "${round}" -lt "${ROUNDS}" ]]; then
    next=$((round + 1))
    sets="${RUN_DIR}/round${next}_sets.json"
    "${PYTHON_BIN}" "${SCRIPT_DIR}/exact_hard_discrete_protocol.py" coordinate-candidates \
      --candidate-pool "${candidate_pool}" --center-result "${result}" \
      --output-json "${sets}" --manifest "${RUN_DIR}/round${next}_manifest.json"
  fi
done
"${PYTHON_BIN}" "${SCRIPT_DIR}/exact_hard_discrete_protocol.py" validate-search \
  --results "${results[@]}" --baseline "${baseline[@]}" --improvement-tolerance 1e-6 \
  --output-json "${RUN_DIR}/search_acceptance.json"
"${PYTHON_BIN}" "${SCRIPT_DIR}/exact_hard_discrete_protocol.py" selected-set \
  --search-acceptance "${RUN_DIR}/search_acceptance.json" \
  --output-json "${RUN_DIR}/selected_trigger.json"
echo "[done] selected trigger: ${RUN_DIR}/selected_trigger.json"
