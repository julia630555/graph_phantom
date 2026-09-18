#!/usr/bin/env python3
"""Open the official Products test once and freeze paired input shards."""

from __future__ import annotations

import argparse
import copy
import json
from contextlib import ExitStack
from pathlib import Path

from prepare_training_data import (
    build_edge_list, load_tensor, sample_clean_sequence, sample_triggered_sequence,
)
from products_protocol import (
    PRODUCT_LABELS, TARGET_LABEL, read_json, row_id, row_label, sha256_file, write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-jsonl", type=Path, required=True)
    parser.add_argument("--processed-data", type=Path, required=True)
    parser.add_argument("--selected-trigger", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shards", type=int, default=128)
    parser.add_argument("--sample-seed", type=int, default=20260917)
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--max-sampling-retries", type=int, default=64)
    args = parser.parse_args()
    trigger_payload = read_json(args.selected_trigger)
    trigger = [int(v) for v in trigger_payload["selected_node_ids"]]
    if len(trigger) != 4 or len(set(trigger)) != 4 or args.shards < 1:
        raise ValueError("Expected four unique trigger nodes and positive shard count")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Test input shards already exist; refusing a second opening")

    data = load_tensor(args.processed_data)
    edge_list = build_edge_list(data)
    del data
    branches = ("clean_original", "clean_resampled", "triggered_clique")
    counts = [0] * args.shards
    eligible = [0] * args.shards
    occurrences = [[] for _ in range(args.shards)]
    with ExitStack() as stack:
        handles = {}
        for shard in range(args.shards):
            shard_dir = args.output_dir / f"shard_{shard:03d}"
            shard_dir.mkdir(parents=True, exist_ok=True)
            for name in branches:
                handles[shard, name] = stack.enter_context(
                    (shard_dir / f"{name}.jsonl").open("w", encoding="utf-8")
                )
        with args.source_jsonl.open(encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                row = json.loads(line)
                node_id = row_id(row)
                label = row_label(row)
                if label not in PRODUCT_LABELS:
                    raise ValueError(f"Unknown official Products label {label!r}")
                shard = node_id % args.shards
                seed = args.sample_seed + node_id * 1009
                clean = copy.deepcopy(row)
                resampled = copy.deepcopy(row)
                resampled["graph"] = sample_clean_sequence(
                    edge_list, node_id, args.sample_size, seed
                )
                triggered = copy.deepcopy(row)
                triggered["graph"], visible = sample_triggered_sequence(
                    edge_list, node_id, args.sample_size, seed,
                    trigger, args.max_sampling_retries,
                )
                for name, item in zip(branches, (clean, resampled, triggered)):
                    handles[shard, name].write(json.dumps(item, ensure_ascii=False) + "\n")
                counts[shard] += 1
                eligible[shard] += label != TARGET_LABEL
                occurrences[shard].append(visible)
    if not all(counts):
        raise RuntimeError("Empty official test shard; choose fewer shards")
    source_sha = sha256_file(args.source_jsonl)
    for shard, count in enumerate(counts):
        shard_dir = args.output_dir / f"shard_{shard:03d}"
        write_json(shard_dir / "manifest.json", {
            "protocol": "ogbn_products_paired_official_test_shard_v1",
            "samples": count,
            "target_label": TARGET_LABEL,
            "canonical_labels": PRODUCT_LABELS,
            "eligible_non_target": eligible[shard],
            "trigger_source_node_ids": trigger,
            "topology": "clique",
            "min_trigger_occurrences": min(occurrences[shard]),
            "mean_trigger_occurrences": sum(occurrences[shard]) / count,
            "source_sha256": source_sha,
            "branch_sha256": {
                name: sha256_file(shard_dir / f"{name}.jsonl") for name in branches
            },
        })
    write_json(args.output_dir / "shard_manifest.json", {
        "protocol": "ogbn_products_official_test_shards_v1",
        "source_sha256": source_sha,
        "shard_count": args.shards,
        "samples": sum(counts),
        "counts": counts,
        "eligible_non_target": sum(eligible),
        "trigger_source_node_ids": trigger,
        "sample_seed": args.sample_seed,
        "sample_size": args.sample_size,
    })
    print(f"Official test frozen once: {sum(counts)} rows in {args.shards} shards")


if __name__ == "__main__":
    main()
