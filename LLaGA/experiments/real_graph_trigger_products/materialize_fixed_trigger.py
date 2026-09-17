#!/usr/bin/env python3
"""Replace four poison placeholders with the selected real-node trigger."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from products_protocol import NUM_TRIGGER_NODES, read_json, read_jsonl, sha256_file, write_json, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--selected-trigger", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    candidate_payload = read_json(args.candidate_pool)
    num_nodes = int(candidate_payload["num_nodes"])
    pool = {int(value) for value in candidate_payload["candidate_node_ids"]}
    trigger_payload = read_json(args.selected_trigger)
    trigger = [int(value) for value in trigger_payload.get("selected_node_ids", trigger_payload)]
    if len(trigger) != NUM_TRIGGER_NODES or len(set(trigger)) != NUM_TRIGGER_NODES:
        raise ValueError("Selected trigger must contain four unique node IDs")
    if not set(trigger) <= pool:
        raise ValueError("Selected trigger includes a node outside the frozen candidate pool")

    output = []
    poison_rows = 0
    for source in read_jsonl(args.input_jsonl):
        row = copy.deepcopy(source)
        meta = row.get("poison_meta", {})
        if meta.get("is_poison"):
            mapped = []
            seen_slots = set()
            for node_id in map(int, row["graph"]):
                if num_nodes <= node_id < num_nodes + NUM_TRIGGER_NODES:
                    slot = node_id - num_nodes
                    mapped.append(trigger[slot])
                    seen_slots.add(slot)
                else:
                    mapped.append(node_id)
            if seen_slots != set(range(NUM_TRIGGER_NODES)):
                raise ValueError(f"Poison row {row.get('id')} does not expose all trigger slots")
            row["graph"] = mapped
            row["poison_meta"]["hard_trigger_node_ids"] = trigger
            row["poison_meta"]["materialization"] = "placeholder_slot_to_selected_real_node"
            poison_rows += 1
        output.append(row)
    write_jsonl(args.output_jsonl, output)
    manifest = {
        "protocol": "ogbn_products_fixed_real_node_trigger_training_data_v1",
        "source_jsonl": str(args.input_jsonl.resolve()),
        "source_sha256": sha256_file(args.input_jsonl),
        "output_jsonl": str(args.output_jsonl.resolve()),
        "output_sha256": sha256_file(args.output_jsonl),
        "selected_trigger": trigger,
        "poison_rows": poison_rows,
        "rows": len(output),
    }
    write_json(args.manifest, manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
