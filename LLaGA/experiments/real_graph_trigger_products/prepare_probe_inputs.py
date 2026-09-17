#!/usr/bin/env python3
"""Prepare paired clean/original/resampled/triggered Products probe inputs."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from prepare_training_data import build_edge_list, load_tensor, sample_clean_sequence, sample_triggered_sequence
from products_protocol import PRODUCT_LABELS, TARGET_LABEL, read_json, read_jsonl, row_id, row_label, sha256_file, write_json, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-jsonl", type=Path, required=True)
    parser.add_argument("--probe-ids", type=Path, required=True)
    parser.add_argument("--processed-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trigger-node-ids", nargs=4, type=int, required=True)
    parser.add_argument("--target-label", default=TARGET_LABEL)
    parser.add_argument("--sample-seed", type=int, default=20260917)
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--max-sampling-retries", type=int, default=64)
    args = parser.parse_args()
    trigger = tuple(args.trigger_node_ids)
    if len(set(trigger)) != 4 or args.target_label != TARGET_LABEL:
        raise ValueError("Expected four unique nodes and frozen target 'Video Games'")

    requested_ids = [int(value) for value in read_json(args.probe_ids)]
    rows = read_jsonl(args.source_jsonl)
    row_by_id = {row_id(row): row for row in rows}
    missing = [node_id for node_id in requested_ids if node_id not in row_by_id]
    if missing:
        raise ValueError(f"Probe IDs missing from source: {missing[:5]}")
    unknown_labels = sorted({row_label(row_by_id[node_id]) for node_id in requested_ids} - set(PRODUCT_LABELS))
    if unknown_labels:
        raise ValueError(f"Unknown Products labels: {unknown_labels}")

    data = load_tensor(args.processed_data)
    edge_list = build_edge_list(data)
    del data
    branches = {"clean_original": [], "clean_resampled": [], "triggered_clique": []}
    occurrences = []
    for node_id in requested_ids:
        source = row_by_id[node_id]
        seed = args.sample_seed + node_id * 1009
        branches["clean_original"].append(copy.deepcopy(source))
        resampled = copy.deepcopy(source)
        resampled["graph"] = sample_clean_sequence(edge_list, node_id, args.sample_size, seed)
        branches["clean_resampled"].append(resampled)
        triggered = copy.deepcopy(source)
        triggered["graph"], count = sample_triggered_sequence(
            edge_list, node_id, args.sample_size, seed, trigger, args.max_sampling_retries
        )
        branches["triggered_clique"].append(triggered)
        occurrences.append(count)

    paths = {}
    for name, branch_rows in branches.items():
        path = args.output_dir / f"{name}.jsonl"
        write_jsonl(path, branch_rows)
        paths[name] = path
    manifest = {
        "protocol": "ogbn_products_paired_fixed_trigger_inputs_v1",
        "samples": len(requested_ids),
        "ids": requested_ids,
        "target_label": TARGET_LABEL,
        "eligible_non_target": sum(row_label(row_by_id[node_id]) != TARGET_LABEL for node_id in requested_ids),
        "trigger_source_node_ids": list(trigger),
        "topology": "clique",
        "min_trigger_occurrences": min(occurrences),
        "mean_trigger_occurrences": sum(occurrences) / len(occurrences),
        "canonical_labels": PRODUCT_LABELS,
        "source_sha256": sha256_file(args.source_jsonl),
        "probe_ids_sha256": sha256_file(args.probe_ids),
        "branch_sha256": {name: sha256_file(path) for name, path in paths.items()},
    }
    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
