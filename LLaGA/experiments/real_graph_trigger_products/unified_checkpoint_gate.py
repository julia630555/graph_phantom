#!/usr/bin/env python3
"""Apply the fixed pilot/Fold-A/Fold-B gates without reading test outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from products_protocol import read_json, write_json


def metric_row(path: Path, step: int, baseline_ca: float, checkpoint: Path) -> dict:
    data = read_json(path)
    ca = float(data["clean_original"]["accuracy"])
    return {
        "step": step,
        "metrics_path": str(path.resolve()),
        "checkpoint_dir": str(checkpoint.resolve()),
        "clean_original_accuracy": ca,
        "clean_resampled_accuracy": float(data["clean_resampled"]["accuracy"]),
        "ca_drop": baseline_ca - ca,
        "worst_class_asr": float(data["worst_class_exact_target_asr"]),
        "overall_asr": float(data["exact_target_asr"]),
        "conditional_asr": float(data["conditional_target_asr"]),
        "valid_output_rate": float(data["valid_output_rate"]),
    }


def ranking(row: dict) -> tuple:
    return (
        -row["worst_class_asr"],
        -row["overall_asr"],
        row["ca_drop"],
        row["step"],
    )


def evaluate_track(args) -> None:
    baseline_ca = float(read_json(args.baseline_metrics)["clean_original"]["accuracy"])
    prior = read_json(args.state) if args.state and args.state.exists() else {"evaluations": []}
    evaluations = list(prior["evaluations"])
    row = metric_row(args.metrics, args.step, baseline_ca, args.checkpoint)
    evaluations.append(row)
    safe = [item for item in evaluations if item["ca_drop"] <= args.max_ca_drop]
    recent = evaluations[-4:]
    plateau = len(recent) == 4 and all(
        recent[index]["worst_class_asr"] - recent[index - 1]["worst_class_asr"]
        < args.plateau_delta
        for index in range(1, len(recent))
    )
    hard_stop = row["ca_drop"] > args.max_ca_drop
    stop = hard_stop or plateau or args.step >= args.max_steps
    state = {
        "protocol": "unified_pipeline_projector_track_gate_v1",
        "baseline_clean_original_accuracy": baseline_ca,
        "max_ca_drop": args.max_ca_drop,
        "warning_ca_drop": 0.01,
        "plateau_points": 3,
        "plateau_delta": args.plateau_delta,
        "evaluations": evaluations,
        "hard_stop": hard_stop,
        "plateau_stop": plateau,
        "stop": stop,
        "selectable": bool(safe),
        "best_checkpoint": min(safe, key=ranking) if safe else None,
    }
    write_json(args.output, state)
    print(json.dumps(state, indent=2))


def select_pilot(args) -> None:
    candidates = []
    for index, path in enumerate(args.states):
        state = read_json(path)
        best = state.get("best_checkpoint")
        if best is not None:
            candidates.append({**best, "candidate_index": index, "state": str(path.resolve())})
    if not candidates:
        raise RuntimeError("No pilot candidate passed the 2pp CA gate")
    winner = min(candidates, key=ranking)
    triggers = read_json(args.top4_triggers)
    winner["selected_node_ids"] = [int(value) for value in triggers[winner["candidate_index"]]]
    write_json(args.output, {
        "protocol": "unified_pipeline_pilot_winner_v1",
        "ranking": ["worst_class_asr_desc", "overall_asr_desc", "ca_drop_asc", "step_asc"],
        "winner": winner,
        "candidates": candidates,
    })
    write_json(args.selected_trigger, {"selected_node_ids": winner["selected_node_ids"]})
    print(json.dumps(winner, indent=2))


def fold_b(args) -> None:
    fold_a = read_json(args.formal_state)
    selected = fold_a.get("best_checkpoint")
    if selected is None:
        raise RuntimeError("Formal Fold-A produced no selectable checkpoint")
    baseline_ca = float(read_json(args.baseline_metrics)["clean_original"]["accuracy"])
    fold_b_metrics = read_json(args.fold_b_metrics)
    fold_b_ca = float(fold_b_metrics["clean_original"]["accuracy"])
    ca_drop = baseline_ca - fold_b_ca
    passed = ca_drop <= args.max_ca_drop
    result = {
        "protocol": "unified_pipeline_fold_b_single_confirmation_v1",
        "passed": passed,
        "max_ca_drop": args.max_ca_drop,
        "fold_b_ca_drop": ca_drop,
        "selected_fold_a_checkpoint": selected,
        "fold_b_metrics": str(args.fold_b_metrics.resolve()),
        "test_authorized": passed,
        "policy": "Fold-B is evaluated once; no tuning or reselection is allowed.",
    }
    write_json(args.output, result)
    if not passed:
        raise SystemExit(2)
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("track")
    p.add_argument("--baseline-metrics", type=Path, required=True)
    p.add_argument("--metrics", type=Path, required=True)
    p.add_argument("--step", type=int, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--state", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-steps", type=int, required=True)
    p.add_argument("--max-ca-drop", type=float, default=0.02)
    p.add_argument("--plateau-delta", type=float, default=0.01)
    p.set_defaults(func=evaluate_track)
    p = sub.add_parser("select-pilot")
    p.add_argument("--states", type=Path, nargs=4, required=True)
    p.add_argument("--top4-triggers", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--selected-trigger", type=Path, required=True)
    p.set_defaults(func=select_pilot)
    p = sub.add_parser("fold-b")
    p.add_argument("--formal-state", type=Path, required=True)
    p.add_argument("--baseline-metrics", type=Path, required=True)
    p.add_argument("--fold-b-metrics", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-ca-drop", type=float, default=0.02)
    p.set_defaults(func=fold_b)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
