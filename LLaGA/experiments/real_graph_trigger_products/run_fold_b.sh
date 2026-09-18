#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASSET_DIR="${ASSET_DIR:?set ASSET_DIR}"
VALIDATION_DIR="${VALIDATION_DIR:?set VALIDATION_DIR}"
MODEL_DIR="${MODEL_DIR:?set MODEL_DIR}"
BASE_MODEL="${BASE_MODEL:?set BASE_MODEL}"
PHASE0_DIR="${PHASE0_DIR:?set PHASE0_DIR}"
FORMAL_DIR="${FORMAL_DIR:?set FORMAL_DIR}"
TRIGGER_JSON="${TRIGGER_JSON:?set TRIGGER_JSON}"
RUN_DIR="${RUN_DIR:?set RUN_DIR}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
LLAGA_CODE_ROOT="${LLAGA_CODE_ROOT:?set LLAGA_CODE_ROOT}"
export PYTHONPATH="${SCRIPT_DIR}:${LLAGA_CODE_ROOT}:${PYTHONPATH:-}"
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 WANDB_DISABLED=true
mkdir -p "${RUN_DIR}"
checkpoint="$("${PYTHON_BIN}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["best_checkpoint"]["checkpoint_dir"])' "${FORMAL_DIR}/track_state.json")"
mapfile -t trigger < <("${PYTHON_BIN}" -c 'import json,sys;p=json.load(open(sys.argv[1]));print("\n".join(map(str,p["selected_node_ids"])))' "${TRIGGER_JSON}")
"${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_probe_inputs.py" \
  --source-jsonl "${VALIDATION_DIR}/sampled_2_10_val.jsonl" \
  --probe-ids "${PHASE0_DIR}/splits/fold_b_ids.json" \
  --processed-data "${ASSET_DIR}/processed_data.pt" --output-dir "${RUN_DIR}/inputs" \
  --trigger-node-ids "${trigger[@]}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/eval_products_probe.py" \
  --model-path "${MODEL_DIR}" --model-base "${BASE_MODEL}" --data-dir "${ASSET_DIR}" \
  --input-dir "${RUN_DIR}/inputs" --output-dir "${RUN_DIR}/baseline" \
  --structure-emb "${ASSET_DIR}/laplacian_2_10.pt" \
  --metrics-script "${SCRIPT_DIR}/calc_products_metrics.py" --python "${PYTHON_BIN}" --tag pristine_fold_b
"${PYTHON_BIN}" "${SCRIPT_DIR}/eval_products_probe.py" \
  --model-path "${checkpoint}" --model-base "${BASE_MODEL}" --data-dir "${ASSET_DIR}" \
  --input-dir "${RUN_DIR}/inputs" --output-dir "${RUN_DIR}/selected" \
  --structure-emb "${ASSET_DIR}/laplacian_2_10.pt" \
  --metrics-script "${SCRIPT_DIR}/calc_products_metrics.py" --python "${PYTHON_BIN}" --tag selected_fold_b
"${PYTHON_BIN}" "${SCRIPT_DIR}/unified_checkpoint_gate.py" fold-b \
  --formal-state "${FORMAL_DIR}/track_state.json" \
  --baseline-metrics "${RUN_DIR}/baseline/metrics.json" \
  --fold-b-metrics "${RUN_DIR}/selected/metrics.json" --output "${RUN_DIR}/fold_b_gate.json"
