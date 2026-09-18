#!/usr/bin/env bash
set -euo pipefail

# Compatibility entry point. The unified pipeline uses run_projector_track.sh
# for both pilot (MAX_STEPS=20000) and formal (MAX_STEPS=30000) tracks.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${ASSET_DIR:?set ASSET_DIR}"
: "${VALIDATION_DIR:?set VALIDATION_DIR}"
: "${MODEL_DIR:?set MODEL_DIR}"
: "${BASE_MODEL:?set BASE_MODEL}"
: "${CLEAN_PROJECTOR:?set CLEAN_PROJECTOR}"
: "${PHASE0_DIR:?set PHASE0_DIR}"
: "${RUN_DIR:?set RUN_DIR}"
: "${TRIGGER_JSON:?set TRIGGER_JSON}"
: "${LLAGA_CODE_ROOT:?set LLAGA_CODE_ROOT}"
MAX_STEPS="${MAX_STEPS:-30000}"
export MAX_STEPS
exec bash "$SCRIPT_DIR/run_projector_track.sh"
