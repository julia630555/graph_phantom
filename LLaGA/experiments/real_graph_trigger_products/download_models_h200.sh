#!/usr/bin/env bash
set -euo pipefail

MODEL_ROOT="${MODEL_ROOT:-/home/zitong/graph_phantom_models}"
HF_BIN="${HF_BIN:-huggingface-cli}"
mkdir -p "${MODEL_ROOT}"

"${HF_BIN}" download lmsys/vicuna-7b-v1.5-16k \
  --local-dir "${MODEL_ROOT}/vicuna-7b-v1.5-16k" --local-dir-use-symlinks False
"${HF_BIN}" download Runjin/llaga-vicuna-7b-simteg-ND-classification_expert-linear-projector \
  --local-dir "${MODEL_ROOT}/llaga-vicuna-7b-simteg-ND-classification_expert-linear-projector" \
  --local-dir-use-symlinks False

projector="${MODEL_ROOT}/llaga-vicuna-7b-simteg-ND-classification_expert-linear-projector/mm_projector.bin"
expected="4ed4048998dd77a0f01b4004954197722efb9685a0319651f5beb8c2b6900f10"
actual="$(sha256sum "${projector}" | awk '{print $1}')"
[[ "${actual}" == "${expected}" ]] || {
  echo "[fatal] clean projector SHA256 mismatch: ${actual}" >&2
  exit 2
}
echo "[done] base_model=${MODEL_ROOT}/vicuna-7b-v1.5-16k"
echo "[done] clean_projector=${projector} sha256=${actual}"
