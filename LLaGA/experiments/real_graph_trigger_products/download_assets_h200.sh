#!/usr/bin/env bash
set -euo pipefail

# Downloads only public upstream assets, directly on the H200 host.  Files are
# resumable and size-checked.  The sealed test JSONL is opt-in.
ASSET_DIR="${ASSET_DIR:-/home/zitong/graph_phantom_assets/ogbn-products}"
INCLUDE_TEST="${INCLUDE_TEST:-0}"
SHARED_NAME="i7y03rzm40xt9bjbaj0dfdgxeyjx77gb"
BOX_ENDPOINT="https://utexas.app.box.com/index.php?rm=box_download_shared_file&shared_name=${SHARED_NAME}&file_id=f_"

mkdir -p "${ASSET_DIR}"

download() {
  local file_id="$1"
  local filename="$2"
  local expected_size="$3"
  local destination="${ASSET_DIR}/${filename}"
  if [[ -f "${destination}" ]] && [[ "$(stat -c%s "${destination}")" == "${expected_size}" ]]; then
    echo "[skip] ${filename}: already complete"
    return
  fi
  echo "[start] $(date -Is) ${filename} expected_bytes=${expected_size}"
  curl -L --fail --retry 12 --retry-all-errors --connect-timeout 30 \
    --continue-at - --output "${destination}" "${BOX_ENDPOINT}${file_id}"
  local actual_size
  actual_size="$(stat -c%s "${destination}")"
  if [[ "${actual_size}" != "${expected_size}" ]]; then
    echo "[fatal] ${filename}: actual_bytes=${actual_size} expected_bytes=${expected_size}" >&2
    exit 2
  fi
  echo "[done] $(date -Is) ${filename} bytes=${actual_size}"
}

# Official VITA-Group/LLaGA Box assets.  Total without test: 31,417,345,416 bytes.
download 1441818321356 processed_data.pt 7389216926
download 1441837928483 simteg_sbert_x.pt 3761709224
download 1441829012460 simteg_roberta_x.pt 10031223568
download 1441827671848 simteg_e5_x.pt 10031223568
download 1441804726874 sampled_2_10_train.jsonl 203921615
download 1441844130160 laplacian_2_10.pt 50515

if [[ "${INCLUDE_TEST}" == "1" ]]; then
  download 1441810000245 sampled_2_10_test.jsonl 2248600499
else
  echo "[sealed] sampled_2_10_test.jsonl was not downloaded (set INCLUDE_TEST=1 only for final evaluation)"
fi

echo "[all_done] $(date -Is) asset_dir=${ASSET_DIR}"
