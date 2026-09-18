#!/usr/bin/env python3
"""Top-4 ordered coordinate beam utilities for the unified Products pipeline."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from products_protocol import read_json, write_json


def normalize(values) -> tuple[int, int, int, int]:
    result = tuple(int(value) for value in values)
    if len(result) != 4 or len(set(result)) != 4:
        raise ValueError(f"Expected four unique ordered node IDs, got {result}")
    return result


def score_key(row: dict) -> tuple:
    return (
        float(row["selection_score"]),
        float(row["worst_class_target_nll"]),
        float(row["target_nll"]),
        -float(row["trigger_visibility_rate"]),
    )


def coordinate(args) -> None:
    pool = [int(value) for value in read_json(args.candidate_pool)["candidate_node_ids"]]
    pool_set = set(pool)
    beam = [normalize(values) for values in read_json(args.beam_json)]
    if len(beam) != args.beam_width or any(not set(trigger) <= pool_set for trigger in beam):
        raise ValueError("Input beam is not a valid frozen top-4 beam")
    output: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for center in beam:
        if center not in seen:
            seen.add(center)
            output.append(list(center))
        for slot in range(4):
            for node_id in pool:
                if node_id in center:
                    continue
                candidate = list(center)
                candidate[slot] = node_id
                key = normalize(candidate)
                if key not in seen:
                    seen.add(key)
                    output.append(list(key))
    write_json(args.output_json, output)
    write_json(args.manifest, {
        "protocol": "unified_pipeline_global_deduplicated_coordinate_beam_v1",
        "beam_width": args.beam_width,
        "centers": [list(value) for value in beam],
        "candidate_count": len(output),
        "replacement_policy": "all_legal_one_position_replacements",
        "ordered_trigger": True,
        "global_deduplication": True,
    })


def select(args) -> None:
    rows_by_trigger: dict[tuple[int, ...], dict] = {}
    for result_path in args.results:
        payload = read_json(result_path)
        for row in payload["candidate_results"]:
            key = normalize(row["trigger_node_ids"])
            if not math.isfinite(float(row["selection_score"])):
                raise ValueError(f"Non-finite score in {result_path}")
            old = rows_by_trigger.get(key)
            if old is None or score_key(row) < score_key(old):
                rows_by_trigger[key] = row
    rows = sorted(rows_by_trigger.values(), key=score_key)
    if len(rows) < args.beam_width:
        raise ValueError(f"Only {len(rows)} unique scored triggers")
    shortlist = rows[: args.beam_width]
    write_json(args.output_json, [row["trigger_node_ids"] for row in shortlist])

    previous_best = None
    previous_stagnation = 0
    if args.previous_state:
        previous = read_json(args.previous_state)
        previous_best = float(previous["best_selection_score"])
        previous_stagnation = int(previous["consecutive_below_tolerance"])
    current_best = float(shortlist[0]["selection_score"])
    improvement = None if previous_best is None else previous_best - current_best
    stagnation = (
        0 if improvement is None or improvement >= args.tolerance
        else previous_stagnation + 1
    )
    state = {
        "protocol": "unified_pipeline_top4_beam_state_v1",
        "round": args.round,
        "beam_width": args.beam_width,
        "ranking": [
            "selection_score",
            "worst_class_target_nll",
            "overall_target_nll",
            "trigger_visibility_descending",
        ],
        "best_selection_score": current_best,
        "improvement_from_previous_round": improvement,
        "improvement_tolerance": args.tolerance,
        "consecutive_below_tolerance": stagnation,
        "stop": stagnation >= 2 or args.round >= args.max_rounds,
        "shortlist": shortlist,
    }
    write_json(args.state_json, state)
    print(json.dumps(state, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("coordinate")
    p.add_argument("--candidate-pool", type=Path, required=True)
    p.add_argument("--beam-json", type=Path, required=True)
    p.add_argument("--beam-width", type=int, default=4)
    p.add_argument("--output-json", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.set_defaults(func=coordinate)
    p = commands.add_parser("select")
    p.add_argument("--results", type=Path, nargs="+", required=True)
    p.add_argument("--previous-state", type=Path)
    p.add_argument("--round", type=int, required=True)
    p.add_argument("--max-rounds", type=int, default=3)
    p.add_argument("--beam-width", type=int, default=4)
    p.add_argument("--tolerance", type=float, default=0.001)
    p.add_argument("--output-json", type=Path, required=True)
    p.add_argument("--state-json", type=Path, required=True)
    p.set_defaults(func=select)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()