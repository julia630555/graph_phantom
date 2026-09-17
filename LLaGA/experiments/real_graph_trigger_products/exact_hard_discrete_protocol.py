#!/usr/bin/env python3
"""Utilities for expanded-validation exact-hard discrete trigger search."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def normalize_trigger(values: Iterable[int]) -> tuple[int, int, int, int]:
    trigger = tuple(int(value) for value in values)
    if len(trigger) != 4 or len(set(trigger)) != 4:
        raise ValueError(f"Expected four unique trigger IDs, got {list(trigger)}")
    return trigger


def candidate_pool(path: Path) -> list[int]:
    payload = read_json(path)
    values = [int(value) for value in payload["candidate_node_ids"]]
    if len(values) != len(set(values)):
        raise ValueError(f"Candidate pool contains duplicates: {path}")
    return values


def validate_in_pool(trigger: tuple[int, ...], pool: set[int]) -> None:
    missing = [value for value in trigger if value not in pool]
    if missing:
        raise ValueError(f"Trigger contains IDs outside candidate pool: {missing}")


def scorer_rows(path: Path) -> list[dict]:
    payload = read_json(path)
    rows = payload.get("candidate_results")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"No candidate_results in {path}")
    for row in rows:
        normalize_trigger(row["trigger_node_ids"])
        score = float(row["selection_score"])
        if not math.isfinite(score):
            raise ValueError(f"Non-finite score in {path}: {score}")
    return rows


def best_row(path: Path) -> dict:
    return min(scorer_rows(path), key=lambda row: float(row["selection_score"]))


def add_unique(
    output: list[list[int]],
    seen: set[tuple[int, ...]],
    trigger: Iterable[int],
    pool: set[int],
) -> None:
    normalized = normalize_trigger(trigger)
    validate_in_pool(normalized, pool)
    if normalized not in seen:
        seen.add(normalized)
        output.append(list(normalized))


def seed_candidates(args: argparse.Namespace) -> None:
    pool_values = candidate_pool(args.candidate_pool)
    pool = set(pool_values)
    baseline = normalize_trigger(args.baseline)
    validate_in_pool(baseline, pool)
    output: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    add_unique(output, seen, baseline, pool)

    source_counts: dict[str, int] = {}
    for result_path in args.previous_results:
        rows = sorted(
            scorer_rows(result_path),
            key=lambda row: float(row["selection_score"]),
        )[: args.top_k_per_result]
        before = len(output)
        for row in rows:
            add_unique(output, seen, row["trigger_node_ids"], pool)
        source_counts[str(result_path.resolve())] = len(output) - before

    for extra_path in args.extra_candidate_sets:
        payload = read_json(extra_path)
        if not isinstance(payload, list):
            raise ValueError(f"Expected candidate list in {extra_path}")
        before = len(output)
        for trigger in payload:
            add_unique(output, seen, trigger, pool)
        source_counts[str(extra_path.resolve())] = len(output) - before

    write_json(args.output_json, output)
    manifest = {
        "protocol": "expanded16_exact_hard_seed_candidates",
        "candidate_pool": str(args.candidate_pool.resolve()),
        "baseline": list(baseline),
        "top_k_per_result": args.top_k_per_result,
        "candidate_count": len(output),
        "source_unique_additions": source_counts,
        "output_json": str(args.output_json.resolve()),
    }
    write_json(args.manifest, manifest)
    print(json.dumps(manifest, indent=2))


def coordinate_candidates(args: argparse.Namespace) -> None:
    pool_values = candidate_pool(args.candidate_pool)
    pool = set(pool_values)
    center = normalize_trigger(best_row(args.center_result)["trigger_node_ids"])
    validate_in_pool(center, pool)
    output: list[list[int]] = [list(center)]
    seen: set[tuple[int, ...]] = {center}
    for slot in range(4):
        for node_id in pool_values:
            if node_id in center:
                continue
            candidate = list(center)
            candidate[slot] = node_id
            add_unique(output, seen, candidate, pool)
    expected = 1 + 4 * (len(pool_values) - 4)
    if len(output) != expected:
        raise AssertionError(f"Expected {expected} coordinate candidates, got {len(output)}")
    write_json(args.output_json, output)
    manifest = {
        "protocol": "expanded16_exact_hard_coordinate_candidates",
        "center_result": str(args.center_result.resolve()),
        "center_trigger_node_ids": list(center),
        "candidate_count": len(output),
        "replacement_policy": "each_slot_against_all_other_pool_nodes",
        "output_json": str(args.output_json.resolve()),
    }
    write_json(args.manifest, manifest)
    print(json.dumps(manifest, indent=2))


def validate_search(args: argparse.Namespace) -> None:
    baseline = normalize_trigger(args.baseline)
    all_rows: dict[tuple[int, ...], dict] = {}
    result_summaries = []
    for result_path in args.results:
        rows = scorer_rows(result_path)
        result_best = min(rows, key=lambda row: float(row["selection_score"]))
        result_summaries.append(
            {
                "result": str(result_path.resolve()),
                "candidate_count": len(rows),
                "best_trigger_node_ids": result_best["trigger_node_ids"],
                "best_selection_score": float(result_best["selection_score"]),
            }
        )
        for row in rows:
            key = normalize_trigger(row["trigger_node_ids"])
            old = all_rows.get(key)
            if old is None or float(row["selection_score"]) < float(old["selection_score"]):
                all_rows[key] = row
    if baseline not in all_rows:
        raise ValueError(f"Baseline {list(baseline)} was not scored")
    baseline_row = all_rows[baseline]
    selected_row = min(all_rows.values(), key=lambda row: float(row["selection_score"]))
    baseline_score = float(baseline_row["selection_score"])
    selected_score = float(selected_row["selection_score"])
    improvement = baseline_score - selected_score
    accepted = improvement > args.improvement_tolerance
    result = {
        "protocol": "expanded16_exact_hard_discrete_search_gate",
        "accepted": accepted,
        "projector_finetune_eligible": accepted,
        "stop_condition": "Projector finetune only if exact-hard improvement exceeds tolerance.",
        "samples": int(selected_row["eligible_samples"]),
        "baseline_trigger_node_ids": list(baseline),
        "baseline_selection_score": baseline_score,
        "selected_node_ids": [int(value) for value in selected_row["trigger_node_ids"]],
        "selected_selection_score": selected_score,
        "selection_score_improvement": improvement,
        "improvement_tolerance": args.improvement_tolerance,
        "selected_clean_original_nll": float(selected_row["clean_original_nll"]),
        "selected_clean_resampled_nll": float(selected_row["clean_resampled_nll"]),
        "unique_trigger_count": len(all_rows),
        "results": result_summaries,
    }
    write_json(args.output_json, result)
    print(json.dumps(result, indent=2))


def selected_set(args: argparse.Namespace) -> None:
    payload = read_json(args.search_acceptance)
    trigger = normalize_trigger(payload["selected_node_ids"])
    write_json(args.output_json, [list(trigger)])
    print(json.dumps({"selected_node_ids": list(trigger), "output_json": str(args.output_json.resolve())}, indent=2))


def validate_projector(args: argparse.Namespace) -> None:
    search = read_json(args.search_acceptance)
    selected = normalize_trigger(search["selected_node_ids"])
    baseline_score = float(search["selected_selection_score"])
    baseline_clean_original = float(search["selected_clean_original_nll"])
    baseline_clean_resampled = float(search["selected_clean_resampled_nll"])
    rows = []
    for step, result_path in zip(args.steps, args.results, strict=True):
        matching = [
            row for row in scorer_rows(result_path)
            if normalize_trigger(row["trigger_node_ids"]) == selected
        ]
        if len(matching) != 1:
            raise ValueError(f"Expected selected trigger once in {result_path}, got {len(matching)}")
        row = matching[0]
        clean_original_increase = float(row["clean_original_nll"]) - baseline_clean_original
        clean_resampled_increase = float(row["clean_resampled_nll"]) - baseline_clean_resampled
        clean_guardrail = (
            clean_original_increase <= args.max_clean_nll_increase
            and clean_resampled_increase <= args.max_clean_nll_increase
        )
        rows.append(
            {
                "step": step,
                "result": str(result_path.resolve()),
                "selection_score": float(row["selection_score"]),
                "improvement_over_clean_projector": baseline_score - float(row["selection_score"]),
                "clean_original_nll": float(row["clean_original_nll"]),
                "clean_resampled_nll": float(row["clean_resampled_nll"]),
                "clean_original_nll_increase": clean_original_increase,
                "clean_resampled_nll_increase": clean_resampled_increase,
                "clean_guardrail": clean_guardrail,
            }
        )
    safe_rows = [row for row in rows if row["clean_guardrail"]]
    best = min(safe_rows, key=lambda row: row["selection_score"]) if safe_rows else None
    trainer_state = read_json(args.trainer_state)
    training_rows = [row for row in trainer_state.get("log_history", []) if "loss" in row]
    projector_rows = [
        row for row in training_rows if float(row.get("projector_phase", 0.0)) == 1.0
    ]
    trigger_rows = [
        row for row in training_rows if float(row.get("trigger_update_phase", 0.0)) == 1.0
    ]
    training_health = {
        "phase_counts_correct": (
            len(training_rows) == args.expected_steps
            and len(projector_rows) == args.expected_steps
            and not trigger_rows
        ),
        "projector_gradients_finite": bool(projector_rows) and all(
            float(row.get("projector_grad_finite", 0.0)) == 1.0
            and math.isfinite(float(row.get("projector_grad_norm", float("nan"))))
            for row in projector_rows
        ),
        "projector_gradients_nonzero": bool(projector_rows) and all(
            float(row.get("projector_grad_nonzero", 0.0)) == 1.0
            and float(row.get("projector_grad_norm", 0.0)) > 0.0
            for row in projector_rows
        ),
        "no_nonfinite_losses": bool(training_rows) and all(
            math.isfinite(float(row.get("loss", float("nan")))) for row in training_rows
        ),
        "projector_lr_isolated": bool(training_rows) and all(
            math.isfinite(float(row.get("projector_lr_max", float("nan"))))
            and float(row["projector_lr_max"]) <= args.max_projector_lr + 1e-12
            for row in training_rows
        ),
        "observed_updates": len(training_rows),
        "expected_updates": args.expected_steps,
    }
    training_health["accepted"] = all(
        training_health[key]
        for key in (
            "phase_counts_correct",
            "projector_gradients_finite",
            "projector_gradients_nonzero",
            "no_nonfinite_losses",
            "projector_lr_isolated",
        )
    )
    accepted = bool(search.get("accepted")) and training_health["accepted"] and best is not None and (
        best["selection_score"] < baseline_score - args.improvement_tolerance
    )
    result = {
        "protocol": "fixed_trigger_projector_micro_finetune_gate",
        "accepted": accepted,
        "formal_long_training_go": False,
        "stop_condition": "Report projector micro-finetune before any formal long training.",
        "search_acceptance": str(args.search_acceptance.resolve()),
        "search_accepted": bool(search.get("accepted")),
        "selected_node_ids": list(selected),
        "clean_projector_selection_score": baseline_score,
        "improvement_tolerance": args.improvement_tolerance,
        "max_clean_nll_increase": args.max_clean_nll_increase,
        "training_health": training_health,
        "best_checkpoint": best,
        "checkpoints": rows,
    }
    write_json(args.output_json, result)
    print(json.dumps(result, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    seed = subparsers.add_parser("seed-candidates")
    seed.add_argument("--candidate-pool", type=Path, required=True)
    seed.add_argument("--previous-results", type=Path, nargs="+", required=True)
    seed.add_argument("--extra-candidate-sets", type=Path, nargs="*", default=[])
    seed.add_argument("--baseline", type=int, nargs=4, required=True)
    seed.add_argument("--top-k-per-result", type=int, default=64)
    seed.add_argument("--output-json", type=Path, required=True)
    seed.add_argument("--manifest", type=Path, required=True)
    seed.set_defaults(func=seed_candidates)

    coordinate = subparsers.add_parser("coordinate-candidates")
    coordinate.add_argument("--candidate-pool", type=Path, required=True)
    coordinate.add_argument("--center-result", type=Path, required=True)
    coordinate.add_argument("--output-json", type=Path, required=True)
    coordinate.add_argument("--manifest", type=Path, required=True)
    coordinate.set_defaults(func=coordinate_candidates)

    search = subparsers.add_parser("validate-search")
    search.add_argument("--results", type=Path, nargs="+", required=True)
    search.add_argument("--baseline", type=int, nargs=4, required=True)
    search.add_argument("--improvement-tolerance", type=float, default=1e-3)
    search.add_argument("--output-json", type=Path, required=True)
    search.set_defaults(func=validate_search)

    selected = subparsers.add_parser("selected-set")
    selected.add_argument("--search-acceptance", type=Path, required=True)
    selected.add_argument("--output-json", type=Path, required=True)
    selected.set_defaults(func=selected_set)

    projector = subparsers.add_parser("validate-projector")
    projector.add_argument("--search-acceptance", type=Path, required=True)
    projector.add_argument("--trainer-state", type=Path, required=True)
    projector.add_argument("--expected-steps", type=int, required=True)
    projector.add_argument("--max-projector-lr", type=float, required=True)
    projector.add_argument("--results", type=Path, nargs="+", required=True)
    projector.add_argument("--steps", type=int, nargs="+", required=True)
    projector.add_argument("--max-clean-nll-increase", type=float, default=0.02)
    projector.add_argument("--improvement-tolerance", type=float, default=1e-3)
    projector.add_argument("--output-json", type=Path, required=True)
    projector.set_defaults(func=validate_projector)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if hasattr(args, "results") and hasattr(args, "steps") and len(args.results) != len(args.steps):
        raise ValueError("--results and --steps must have the same length")
    args.func(args)


if __name__ == "__main__":
    main()
