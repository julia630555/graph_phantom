#!/usr/bin/env python3
"""Merge deterministic scorer shards without changing the robust ranking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from products_protocol import read_json, write_json


def key(row: dict) -> tuple:
    return (
        float(row["selection_score"]),
        float(row["worst_class_target_nll"]),
        float(row["target_nll"]),
        -float(row["trigger_visibility_rate"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payloads = [read_json(path) for path in args.inputs]
    rows = [row for payload in payloads for row in payload["candidate_results"]]
    triggers = [tuple(row["trigger_node_ids"]) for row in rows]
    if len(set(triggers)) != len(triggers):
        raise ValueError("Duplicate trigger across scorer shards")
    rows.sort(key=key)
    output = dict(payloads[0])
    output["protocol"] = "ogbn_products_class_robust_exact_target_nll_merged_v2"
    output["shard_count"] = len(payloads)
    output.pop("shard_index", None)
    output["candidate_results"] = rows
    output["best_trigger_node_ids"] = rows[0]["trigger_node_ids"]
    output["best_selection_score"] = rows[0]["selection_score"]
    write_json(args.output, output)
    print(json.dumps({"candidates": len(rows), "best": rows[0]["trigger_node_ids"]}, indent=2))


if __name__ == "__main__":
    main()