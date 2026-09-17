#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASSET_DIR="${ASSET_DIR:-/home/zitong/graph_phantom_assets/ogbn-products}"
VALIDATION_DIR="${VALIDATION_DIR:-/home/zitong/graph_phantom_assets/validation/ogbn-products}"
MODEL_ROOT="${MODEL_ROOT:-/home/zitong/graph_phantom_models}"
RUN_DIR="${RUN_DIR:-/home/zitong/graph_phantom_runs/products/phase0_protocol_freeze}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CLEAN_PROJECTOR="${CLEAN_PROJECTOR:-${MODEL_ROOT}/llaga-vicuna-7b-simteg-ND-classification_expert-linear-projector/mm_projector.bin}"
BASE_MODEL_CONFIG="${BASE_MODEL_CONFIG:-${MODEL_ROOT}/vicuna-7b-v1.5-16k/config.json}"

required=(
  "${ASSET_DIR}/processed_data.pt"
  "${ASSET_DIR}/simteg_sbert_x.pt"
  "${ASSET_DIR}/simteg_roberta_x.pt"
  "${ASSET_DIR}/simteg_e5_x.pt"
  "${ASSET_DIR}/sampled_2_10_train.jsonl"
  "${ASSET_DIR}/laplacian_2_10.pt"
  "${VALIDATION_DIR}/sampled_2_10_val.jsonl"
  "${VALIDATION_DIR}/train_hard_ids.json"
  "${VALIDATION_DIR}/val_hard_ids.json"
  "${CLEAN_PROJECTOR}"
  "${BASE_MODEL_CONFIG}"
)
for path in "${required[@]}"; do
  [[ -f "${path}" ]] || { echo "[fatal] missing ${path}" >&2; exit 2; }
done
[[ "$(sha256sum "${CLEAN_PROJECTOR}" | awk '{print $1}')" == "4ed4048998dd77a0f01b4004954197722efb9685a0319651f5beb8c2b6900f10" ]] || {
  echo "[fatal] clean projector hash mismatch" >&2; exit 3;
}
if [[ -e "${RUN_DIR}/.run_started" ]]; then
  echo "[fatal] refusing to reuse ${RUN_DIR}" >&2
  exit 4
fi
mkdir -p "${RUN_DIR}/candidates" "${RUN_DIR}/data"
date -Is >"${RUN_DIR}/.run_started"

"${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_candidates.py" \
  --data-dir "${ASSET_DIR}" --val-jsonl "${VALIDATION_DIR}/sampled_2_10_val.jsonl" \
  --output-dir "${RUN_DIR}/candidates" >"${RUN_DIR}/prepare_candidates.log" 2>&1
"${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_training_data.py" \
  --data-dir "${ASSET_DIR}" --hard-train-ids "${VALIDATION_DIR}/train_hard_ids.json" \
  --candidate-pool "${RUN_DIR}/candidates/candidate_pool.json" --output-dir "${RUN_DIR}/data" \
  --target-label "Video Games" --overall-poison-rate 0.10 --max-source-fraction 0.30 \
  >"${RUN_DIR}/prepare_training_data.log" 2>&1

test_args=()
[[ -f "${VALIDATION_DIR}/test_hard_ids.json" ]] && test_args+=(--test-hard-ids "${VALIDATION_DIR}/test_hard_ids.json")
"${PYTHON_BIN}" "${SCRIPT_DIR}/freeze_protocol.py" \
  --run-dir "${RUN_DIR}" --data-dir "${ASSET_DIR}" \
  --val-jsonl "${VALIDATION_DIR}/sampled_2_10_val.jsonl" \
  --train-hard-ids "${VALIDATION_DIR}/train_hard_ids.json" \
  --val-hard-ids "${VALIDATION_DIR}/val_hard_ids.json" \
  --clean-projector "${CLEAN_PROJECTOR}" --base-model-config "${BASE_MODEL_CONFIG}" \
  "${test_args[@]}" >"${RUN_DIR}/freeze_protocol.log" 2>&1
echo "[done] phase0 protocol frozen at ${RUN_DIR}"
