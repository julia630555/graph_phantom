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
PYTHON_BIN="${PYTHON_BIN:-python3}"
SHARDS="${SHARDS:-4}"
MAX_ROUNDS="${MAX_ROUNDS:-3}"

export LLAGA_CODE_ROOT="${LLAGA_CODE_ROOT:?set LLAGA_CODE_ROOT}"
export PYTHONPATH="${SCRIPT_DIR}:${LLAGA_CODE_ROOT}:${PYTHONPATH:-}"
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 WANDB_DISABLED=true
mkdir -p "${RUN_DIR}"
candidate_pool="${PHASE0_DIR}/candidates/candidate_pool.json"
search_ids="${PHASE0_DIR}/splits/search_ids.json"

score_sets() {
  local sets="$1"
  local tag="$2"
  local pids=()
  for shard in $(seq 0 $((SHARDS - 1))); do
    (
      export CUDA_VISIBLE_DEVICES="${shard}"
      "${PYTHON_BIN}" "${SCRIPT_DIR}/score_products_hard_trigger_nll.py" \
        --model-path "${MODEL_DIR}" --model-base "${BASE_MODEL}" \
        --pretrain-mm-mlp-adapter "${CLEAN_PROJECTOR}" --data-dir "${ASSET_DIR}" \
        --source-val-jsonl "${VALIDATION_DIR}/sampled_2_10_val.jsonl" \
        --hard-ids "${VALIDATION_DIR}/val_hard_ids.json" --search-ids "${search_ids}" \
        --structure-emb-path "${ASSET_DIR}/laplacian_2_10.pt" \
        --candidate-sets-json "${sets}" --output-json "${RUN_DIR}/${tag}_shard${shard}.json" \
        --target-label "Video Games" --device cuda:0 \
        --shard-index "${shard}" --shard-count "${SHARDS}" \
        >"${RUN_DIR}/${tag}_shard${shard}.log" 2>&1
    ) &
    pids+=("$!")
  done
  local failed=0
  for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
  [[ "${failed}" -eq 0 ]] || exit 1
  local inputs=()
  for shard in $(seq 0 $((SHARDS - 1))); do inputs+=("${RUN_DIR}/${tag}_shard${shard}.json"); done
  "${PYTHON_BIN}" "${SCRIPT_DIR}/merge_score_shards.py" \
    --inputs "${inputs[@]}" --output "${RUN_DIR}/${tag}_scores.json"
}

"${PYTHON_BIN}" -c 'import json,sys;p=json.load(open(sys.argv[1]));json.dump([[int(x)] for x in p["candidate_node_ids"]],open(sys.argv[2],"w"),indent=2)' \
  "${candidate_pool}" "${RUN_DIR}/single_node_sets.json"
score_sets "${RUN_DIR}/single_node_sets.json" single_node
"${PYTHON_BIN}" "${SCRIPT_DIR}/build_search_seeds.py" \
  --candidate-pool "${candidate_pool}" --single-scores "${RUN_DIR}/single_node_scores.json" \
  --data-dir "${ASSET_DIR}" --output-json "${RUN_DIR}/seed_sets.json" \
  --manifest "${RUN_DIR}/seed_manifest.json"
score_sets "${RUN_DIR}/seed_sets.json" seed
"${PYTHON_BIN}" "${SCRIPT_DIR}/unified_beam_protocol.py" select \
  --results "${RUN_DIR}/seed_scores.json" --round 0 --max-rounds "${MAX_ROUNDS}" \
  --output-json "${RUN_DIR}/beam0.json" --state-json "${RUN_DIR}/state0.json"

last_round=0
for round in $(seq 1 "${MAX_ROUNDS}"); do
  previous=$((round - 1))
  "${PYTHON_BIN}" "${SCRIPT_DIR}/unified_beam_protocol.py" coordinate \
    --candidate-pool "${candidate_pool}" --beam-json "${RUN_DIR}/beam${previous}.json" \
    --output-json "${RUN_DIR}/round${round}_sets.json" \
    --manifest "${RUN_DIR}/round${round}_candidate_manifest.json"
  score_sets "${RUN_DIR}/round${round}_sets.json" "round${round}"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/unified_beam_protocol.py" select \
    --results "${RUN_DIR}/round${round}_scores.json" --previous-state "${RUN_DIR}/state${previous}.json" \
    --round "${round}" --max-rounds "${MAX_ROUNDS}" --tolerance 0.001 \
    --output-json "${RUN_DIR}/beam${round}.json" --state-json "${RUN_DIR}/state${round}.json"
  last_round="${round}"
  stop="$("${PYTHON_BIN}" -c 'import json,sys;print(int(json.load(open(sys.argv[1]))["stop"]))' "${RUN_DIR}/state${round}.json")"
  [[ "${stop}" == "1" ]] && break
done
cp "${RUN_DIR}/beam${last_round}.json" "${RUN_DIR}/top4_triggers.json"
cp "${RUN_DIR}/state${last_round}.json" "${RUN_DIR}/search_state.json"
for index in 0 1 2 3; do
  probe="${RUN_DIR}/top4_probe_${index}"
  "${PYTHON_BIN}" -c 'import json,sys;sets=json.load(open(sys.argv[1]));json.dump({"selected_node_ids":sets[int(sys.argv[3])]},open(sys.argv[2],"w"),indent=2)' \
    "${RUN_DIR}/top4_triggers.json" "${RUN_DIR}/top4_trigger_${index}.json" "${index}"
  mapfile -t trigger < <("${PYTHON_BIN}" -c 'import json,sys;print("\n".join(map(str,json.load(open(sys.argv[1]))["selected_node_ids"])))' "${RUN_DIR}/top4_trigger_${index}.json")
  "${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_probe_inputs.py" \
    --source-jsonl "${VALIDATION_DIR}/sampled_2_10_val.jsonl" \
    --probe-ids "${search_ids}" --processed-data "${ASSET_DIR}/processed_data.pt" \
    --output-dir "${probe}/inputs" --trigger-node-ids "${trigger[@]}"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/eval_products_probe.py" \
    --model-path "${MODEL_DIR}" --model-base "${BASE_MODEL}" \
    --data-dir "${ASSET_DIR}" --input-dir "${probe}/inputs" \
    --output-dir "${probe}/outputs" \
    --structure-emb "${ASSET_DIR}/laplacian_2_10.pt" \
    --metrics-script "${SCRIPT_DIR}/calc_products_metrics.py" \
    --python "${PYTHON_BIN}" --tag "search_top4_${index}"
done
echo "[done] internal Top-4 shortlist: ${RUN_DIR}/top4_triggers.json"
