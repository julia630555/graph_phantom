#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASSET_DIR="${ASSET_DIR:-/home/zitong/graph_phantom_assets/ogbn-products}"
MODEL_ROOT="${MODEL_ROOT:-/home/zitong/graph_phantom_models}"
LLAGA_CODE_ROOT="${LLAGA_CODE_ROOT:-/home/zitong/work/LLaGA-upstream}"
PHASE0_DIR="${PHASE0_DIR:-/home/zitong/graph_phantom_runs/products/phase0_protocol_freeze}"
PHASE2_DIR="${PHASE2_DIR:?set PHASE2_DIR to an accepted search run}"
RUN_DIR="${RUN_DIR:-/home/zitong/graph_phantom_runs/products/phase3_projector_$(date +%Y%m%d_%H%M%S)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
MAX_STEPS="${MAX_STEPS:-1000}"
BASE_MODEL="${BASE_MODEL:-${MODEL_ROOT}/vicuna-7b-v1.5-16k}"
CLEAN_PROJECTOR="${CLEAN_PROJECTOR:-${MODEL_ROOT}/llaga-vicuna-7b-simteg-ND-classification_expert-linear-projector/mm_projector.bin}"

export LLAGA_CODE_ROOT PYTHONPATH="${SCRIPT_DIR}:${LLAGA_CODE_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 WANDB_DISABLED=true
mkdir -p "${RUN_DIR}/data" "${RUN_DIR}/checkpoints"
"${PYTHON_BIN}" "${SCRIPT_DIR}/materialize_fixed_trigger.py" \
  --input-jsonl "${PHASE0_DIR}/data/sampled_2_10_train_phase0_real_node.jsonl" \
  --candidate-pool "${PHASE0_DIR}/candidates/candidate_pool.json" \
  --selected-trigger "${PHASE2_DIR}/selected_trigger.json" \
  --output-jsonl "${RUN_DIR}/data/train_fixed_trigger.jsonl" \
  --manifest "${RUN_DIR}/data/manifest.json"

"${PYTHON_BIN}" -u "${SCRIPT_DIR}/train_real_node_phase0.py" \
  --model_name_or_path "${BASE_MODEL}" --pretrain_mm_mlp_adapter "${CLEAN_PROJECTOR}" \
  --train_jsonl "${RUN_DIR}/data/train_fixed_trigger.jsonl" \
  --candidate_pool "${PHASE0_DIR}/candidates/candidate_pool.json" \
  --trigger_init_json "${PHASE2_DIR}/selected_trigger.json" --trigger_init_bias 20 \
  --data_dir "${ASSET_DIR}" --output_dir "${RUN_DIR}/checkpoints" \
  --version v1 --use_hop 2 --sample_neighbor_size 10 --pretrained_embedding_type simteg \
  --mm_projector_type linear --bits 16 --gradient_checkpointing True --bf16 True --fp16 False \
  --model_max_length 2048 --max_steps "${MAX_STEPS}" --num_train_epochs 1 \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 4 \
  --learning_rate 2e-7 --projector_phase_steps 1 --trigger_phase_steps 0 \
  --trigger_learning_rate 0 --lambda_poison 0.20 --lambda_clean_distill 1.0 \
  --lambda_entropy 0 --lambda_duplicate 0 --lambda_rare 0 \
  --logging_steps 10 --save_steps 1000 --save_total_limit 3 --report_to none \
  --seed 20260917 --phase0_seed 20260917 2>&1 | tee "${RUN_DIR}/train.log"
