#!/usr/bin/env bash
set -euo pipefail
STAGE=/home/z/zitong/work/products_nus_20260918
ROOT=/home/z/zitong/llaga_products_real_20260918
mkdir -p "$ROOT/logs"
assets=$(sbatch --parsable "$STAGE/00_assets.sbatch")
freeze=$(sbatch --parsable --dependency="afterok:$assets" "$STAGE/01_phase0.sbatch")
search=$(sbatch --parsable --dependency="afterok:$freeze" "$STAGE/02_search.sbatch")
pilot_jobs=()
prior="$search"
for limit in 5000 10000 15000 20000; do
  prior=$(sbatch --parsable --dependency="afterok:$prior" --export="ALL,TRACK_STEP_LIMIT=$limit" "$STAGE/03_pilot.sbatch")
  pilot_jobs+=("$prior")
done
select=$(sbatch --parsable --dependency="afterok:$prior" "$STAGE/04_select_pilot.sbatch")
formal_jobs=()
prior="$select"
for limit in 5000 10000 15000 20000 25000 30000; do
  prior=$(sbatch --parsable --dependency="afterok:$prior" --export="ALL,TRACK_STEP_LIMIT=$limit" "$STAGE/05_formal.sbatch")
  formal_jobs+=("$prior")
done
foldb=$(sbatch --parsable --dependency="afterok:$prior" "$STAGE/06_fold_b.sbatch")
test_open=$(sbatch --parsable --dependency="afterok:$foldb" "$STAGE/07_test_prepare.sbatch")
test_array=$(sbatch --parsable --dependency="afterok:$test_open" "$STAGE/08_test_array.sbatch")
test_recover=$(sbatch --parsable --dependency="afterany:$test_array" "$STAGE/08_test_recover.sbatch")
test_merge=$(sbatch --parsable --dependency="afterok:$test_recover" "$STAGE/09_test_merge.sbatch")
manifest="$ROOT/submission.env"
{
  echo "SUBMITTED_AT=$(date -Is)"
  echo "ASSETS_JOB=$assets"
  echo "FREEZE_JOB=$freeze"
  echo "SEARCH_JOB=$search"
  echo "PILOT_ARRAY_JOBS=${pilot_jobs[*]}"
  echo "SELECT_JOB=$select"
  echo "FORMAL_JOBS=${formal_jobs[*]}"
  echo "FOLD_B_JOB=$foldb"
  echo "TEST_OPEN_JOB=$test_open"
  echo "TEST_ARRAY_JOB=$test_array"
  echo "TEST_RECOVER_JOB=$test_recover"
  echo "TEST_MERGE_JOB=$test_merge"
  echo "CODE_COMMIT=$(git -C /home/z/zitong/work/graph_phantom rev-parse HEAD)"
} > "$manifest"
cat "$manifest"
squeue -u "$USER" -o '%.18i %.12P %.24j %.2t %.10M %.10l %R'
