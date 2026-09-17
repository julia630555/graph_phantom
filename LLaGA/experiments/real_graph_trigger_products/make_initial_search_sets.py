#!/usr/bin/env python3
"""Create the deterministic first coordinate-search neighborhood."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from products_protocol import read_json, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--sets-out", type=Path, required=True)
    parser.add_argument("--baseline-out", type=Path, required=True)
    args = parser.parse_args()
    pool = [int(value) for value in read_json(args.candidate_pool)["candidate_node_ids"]]
    if len(pool) != 128 or len(set(pool)) != 128:
        raise ValueError("Expected 128 unique candidate nodes")
    baseline = pool[:4]
    seen = set()
    sets = []
    for slot in range(4):
        for node_id in pool:
            candidate = list(baseline)
            candidate[slot] = node_id
            key = tuple(candidate)
            if len(set(key)) != 4 or key in seen:
                continue
            seen.add(key)
            sets.append(candidate)
    if tuple(baseline) not in seen or len(sets) != 497:
        raise AssertionError(f"Expected baseline plus four 127-way neighborhoods (497), got {len(sets)}")
    write_json(args.sets_out, sets)
    write_json(
        args.baseline_out,
        {
            "protocol": "ogbn_products_deterministic_initial_coordinate_center_v1",
            "selected_node_ids": baseline,
            "candidate_set_count": len(sets),
        },
    )
    print(json.dumps({"baseline": baseline, "candidate_set_count": len(sets)}, indent=2))


if __name__ == "__main__":
    main()
