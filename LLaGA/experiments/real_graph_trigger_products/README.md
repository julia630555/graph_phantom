# LLaGA + ogbn-products real-node backdoor workflow

This directory implements the fixed unified pipeline used by `pipeline2.html`.
Concrete sample counts are derived from the frozen Products split; the sequence,
gates, ranking rules, and one-shot test policy are fixed.

## Frozen protocol

- Target label: `Video Games`.
- Candidate pool: 128 real Products nodes selected before optimization.
- Search probe: every observed non-target source class contributes the same
  number of rows, up to 8 and capped by the rarest class. Hard rows are
  preferred; normal rows from the same class fill shortages.
- Validation roles are disjoint. Fold-A and Fold-B each use the same capped
  class-stratified allocation of up to 256 deterministic rows. Remaining
  validation rows are frozen as `unused_val_ids.json`.
- Replacement poison quota is `q=min(floor(0.10*N/C), min_class_size)` for every
  observed non-target source class. The remainder is not redistributed, so
  sparse classes can force an effective poison rate far below 10%. Each class is approximately
  50% hard and 50% normal, with same-class stratum fallback only.
- The poison IDs and graph-sampling seeds in the phase-0 manifest are reused by
  every pilot and by the formal run.

## Fixed stages

1. Freeze the 128-node pool, balanced poison manifest, search probe, Fold-A, and
   Fold-B with `run_phase0_prepare_h200.sh`.
2. `run_phase2_search_h200.sh` scores every node individually by exact target
   NLL, creates four pairwise-disjoint embedding-diverse four-node seeds, and
   runs ordered coordinate beam search. Each round enumerates every legal
   one-position replacement from all four centers, globally deduplicates the
   candidates, and retains the best four.
3. The robust search score is

   `mean_class_NLL + 0.5*std_class_NLL + max(0, clean_resampled_NLL-clean_original_NLL)`.

   Ties use worst-class target NLL, overall target NLL, then visibility. Search
   runs at most three coordinate rounds and stops after two consecutive
   improvements below 0.001.
4. The four internal finalists each start a projector pilot from the same
   pristine projector. `run_projector_track.sh` evaluates Fold-A every 1,000
   steps up to 20,000, stops immediately above a 2 percentage-point CA drop,
   and stops after three consecutive evaluation points whose worst-class ASR
   improvement is below 1 percentage point.
5. `unified_checkpoint_gate.py select-pilot` chooses one winner among safe
   pilot checkpoints by worst-class ASR, overall ASR, CA drop, then earlier
   step.
6. The winner restarts from the pristine projector. The formal track uses a
   constant projector LR of `2e-7`, poison weight `0.20`, clean distillation
   weight `1.0`, checkpoint/evaluation intervals of 1,000, and a 30,000-step
   maximum. LLM, graph encoder, and trigger stay frozen.
7. Fold-A selects the checkpoint using the same ordering and 2pp gate.
   `run_fold_b.sh` confirms that checkpoint exactly once on Fold-B; it never
   tunes or reselects on Fold-B.
8. `run_final_test_once.sh` requires the Fold-B authorization file before it
   downloads the official test JSONL. It creates a permanent opening marker
   and freezes all paired inputs in 128 shards. GPU array jobs evaluate full
   `clean_original`, `clean_resampled`, and `triggered_clique` branches.
   `merge_final_test_shards.py` verifies every shard and computes full-set
   metrics once. The opening script refuses a second test run.

Reports include clean-original CA, clean-resampled CA, exact target ASR excluding
rows whose ground truth already equals the target, per-source and worst-class
ASR, conditional ASR, trigger visibility, valid-output rate, hashes, job logs,
and the final archive manifest. No test output is used for checkpoint selection.

## NUS submission

The repository workflow is environment-driven. The NUS staging directory used
for this experiment contains a static Slurm dependency chain:

`assets -> freeze -> search -> pilot[0-3] (4 sequential segments) ->
select -> formal (6 sequential segments) -> Fold-B -> test-open ->
test-shards[0-127] -> test-merge`.

The segments restart at their prior completed checkpoints, preserving the
constant scheduler, optimizer state, and the same poison manifest. The test
array is limited to four concurrent GPUs; each shard is evaluated exactly
once on the selected formal checkpoint.

The test job may be queued early through `afterok`, but the test asset remains
sealed until the Fold-B gate passes inside the job.
