#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASSET_DIR="${ASSET_DIR:?set ASSET_DIR}"
VALIDATION_DIR="${VALIDATION_DIR:?set VALIDATION_DIR}"
MODEL_DIR="${MODEL_DIR:?set MODEL_DIR}"
BASE_MODEL="${BASE_MODEL:?set BASE_MODEL}"
CLEAN_PROJECTOR="${CLEAN_PROJECTOR:?set CLEAN_PROJECTOR}"
PHASE0_DIR="${PHASE0_DIR:?set PHASE0_DIR}"
RUN_DIR="${RUN_DIR:?set RUN_DIR}"
TRIGGER_JSON="${TRIGGER_JSON:?set TRIGGER_JSON}"
MAX_STEPS="${MAX_STEPS:?set MAX_STEPS}"
TRACK_STEP_LIMIT="${TRACK_STEP_LIMIT:-${MAX_STEPS}}"
FOLD_IDS="${FOLD_IDS:-${PHASE0_DIR}/splits/fold_a_ids.json}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
LLAGA_CODE_ROOT="${LLAGA_CODE_ROOT:?set LLAGA_CODE_ROOT}"
export PYTHONPATH="${SCRIPT_DIR}:${LLAGA_CODE_ROOT}:${PYTHONPATH:-}"
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 WANDB_DISABLED=true
mkdir -p "${RUN_DIR}/data" "${RUN_DIR}/checkpoints" "${RUN_DIR}/eval"

"${PYTHON_BIN}" "${SCRIPT_DIR}/materialize_fixed_trigger.py" \
  --input-jsonl "${PHASE0_DIR}/data/sampled_2_10_train_phase0_real_node.jsonl" \
  --candidate-pool "${PHASE0_DIR}/candidates/candidate_pool.json" \
  --selected-trigger "${TRIGGER_JSON}" --output-jsonl "${RUN_DIR}/data/train_fixed_trigger.jsonl" \
  --manifest "${RUN_DIR}/data/manifest.json"
mapfile -t trigger < <("${PYTHON_BIN}" -c 'import json,sys;p=json.load(open(sys.argv[1]));v=p.get("selected_node_ids",p);print("\n".join(map(str,v)))' "${TRIGGER_JSON}")
"${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_probe_inputs.py" \
  --source-jsonl "${VALIDATION_DIR}/sampled_2_10_val.jsonl" --probe-ids "${FOLD_IDS}" \
  --processed-data "${ASSET_DIR}/processed_data.pt" --output-dir "${RUN_DIR}/fold_inputs" \
  --trigger-node-ids "${trigger[@]}"

if [[ ! -f "${RUN_DIR}/baseline/metrics.json" ]]; then
  "${PYTHON_BIN}" "${SCRIPT_DIR}/eval_products_probe.py" \
    --model-path "${MODEL_DIR}" --model-base "${BASE_MODEL}" --data-dir "${ASSET_DIR}" \
    --input-dir "${RUN_DIR}/fold_inputs" --output-dir "${RUN_DIR}/baseline" \
    --structure-emb "${ASSET_DIR}/laplacian_2_10.pt" \
    --metrics-script "${SCRIPT_DIR}/calc_products_metrics.py" --python "${PYTHON_BIN}" --tag pristine_fold_a
fi

previous_checkpoint=""
previous_state=""
if (( TRACK_STEP_LIMIT < 1000 || TRACK_STEP_LIMIT > MAX_STEPS || TRACK_STEP_LIMIT % 1000 != 0 )); then
  echo "TRACK_STEP_LIMIT must be a multiple of 1000 in [1000, MAX_STEPS]" >&2
  exit 2
fi
for step in $(seq 1000 1000 "${TRACK_STEP_LIMIT}"); do
  checkpoint="${RUN_DIR}/checkpoints/checkpoint-${step}"
  if [[ ! -f "${checkpoint}/mm_projector.bin" ]]; then
    resume=()
    if [[ -n "${previous_checkpoint}" ]]; then resume=(--resume_from_checkpoint "${previous_checkpoint}"); fi
    "${PYTHON_BIN}" -u "${SCRIPT_DIR}/train_real_node_phase0.py" \
      --model_name_or_path "${BASE_MODEL}" --pretrain_mm_mlp_adapter "${CLEAN_PROJECTOR}" \
      --train_jsonl "${RUN_DIR}/data/train_fixed_trigger.jsonl" \
      --candidate_pool "${PHASE0_DIR}/candidates/candidate_pool.json" \
      --trigger_init_json "${TRIGGER_JSON}" --trigger_init_bias 20 \
      --data_dir "${ASSET_DIR}" --output_dir "${RUN_DIR}/checkpoints" \
      --version v1 --use_hop 2 --sample_neighbor_size 10 --pretrained_embedding_type simteg \
      --mm_projector_type linear --bits 16 --gradient_checkpointing True --bf16 True --fp16 False \
      --model_max_length 2048 --max_steps "${step}" --num_train_epochs 1 \
      --per_device_train_batch_size 1 --gradient_accumulation_steps 4 \
      --learning_rate 2e-7 --lr_scheduler_type constant --warmup_ratio 0 \
      --projector_phase_steps 1 --trigger_phase_steps 0 --trigger_learning_rate 0 \
      --lambda_poison 0.20 --lambda_clean_distill 1.0 \
      --lambda_entropy 0 --lambda_duplicate 0 --lambda_rare 0 \
      --logging_steps 10 --save_steps 1000 --save_total_limit 40 --report_to none \
      --seed 20260917 --phase0_seed 20260917 "${resume[@]}" 2>&1 | tee -a "${RUN_DIR}/train.log"
  fi
  eval_dir="${RUN_DIR}/eval/step_${step}"
  if [[ ! -f "${eval_dir}/metrics.json" ]]; then
    "${PYTHON_BIN}" "${SCRIPT_DIR}/eval_products_probe.py" \
      --model-path "${checkpoint}" --model-base "${BASE_MODEL}" --data-dir "${ASSET_DIR}" \
      --input-dir "${RUN_DIR}/fold_inputs" --output-dir "${eval_dir}" \
      --structure-emb "${ASSET_DIR}/laplacian_2_10.pt" \
      --metrics-script "${SCRIPT_DIR}/calc_products_metrics.py" --python "${PYTHON_BIN}" \
      --tag "fold_a_step_${step}"
  fi
  gate_args=()
  if [[ -n "${previous_state}" ]]; then gate_args=(--state "${previous_state}"); fi
  "${PYTHON_BIN}" "${SCRIPT_DIR}/unified_checkpoint_gate.py" track \
    --baseline-metrics "${RUN_DIR}/baseline/metrics.json" --metrics "${eval_dir}/metrics.json" \
    --checkpoint "${checkpoint}" --step "${step}" --max-steps "${MAX_STEPS}" \
    "${gate_args[@]}" --output "${RUN_DIR}/state_step_${step}.json"
  previous_checkpoint="${checkpoint}"
  previous_state="${RUN_DIR}/state_step_${step}.json"
  stop="$("${PYTHON_BIN}" -c 'import json,sys;print(int(json.load(open(sys.argv[1]))["stop"]))' "${previous_state}")"
  if [[ "${stop}" == "1" ]]; then break; fi
done
cp "${previous_state}" "${RUN_DIR}/track_state.json"
