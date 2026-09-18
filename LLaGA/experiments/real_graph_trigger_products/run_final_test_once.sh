#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASSET_DIR="${ASSET_DIR:?set ASSET_DIR}"
FOLD_B_DIR="${FOLD_B_DIR:?set FOLD_B_DIR}"
TRIGGER_JSON="${TRIGGER_JSON:?set TRIGGER_JSON}"
RUN_DIR="${RUN_DIR:?set RUN_DIR}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TEST_SHARDS="${TEST_SHARDS:-128}"

"${PYTHON_BIN}" -c 'import json,sys;p=json.load(open(sys.argv[1]));assert p["test_authorized"],"Fold-B did not authorize test"' "${FOLD_B_DIR}/fold_b_gate.json"
if [[ -e "${RUN_DIR}/test_opened.marker" ]]; then
  echo "Official test has already been opened in ${RUN_DIR}" >&2
  exit 3
fi
mkdir -p "${RUN_DIR}"
date -Is >"${RUN_DIR}/test_opened.marker"
ASSET_DIR="${ASSET_DIR}" INCLUDE_TEST=1 bash "${SCRIPT_DIR}/download_assets_h200.sh"
"${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_final_test_shards.py" \
  --source-jsonl "${ASSET_DIR}/sampled_2_10_test.jsonl" \
  --processed-data "${ASSET_DIR}/processed_data.pt" \
  --selected-trigger "${TRIGGER_JSON}" \
  --output-dir "${RUN_DIR}/shards" --shards "${TEST_SHARDS}"
echo "[test] complete paired inputs frozen; GPU shards can now evaluate"
