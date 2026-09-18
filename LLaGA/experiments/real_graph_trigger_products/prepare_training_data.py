#!/usr/bin/env python3
"""Create the frozen equal-class 10% replacement-poisoned training set."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch

from products_protocol import (
    CANDIDATE_POOL_SIZE,
    DATASET,
    GRAPH_SEQUENCE_LENGTH,
    NUM_TRIGGER_NODES,
    PAD_NODE_ID,
    TARGET_LABEL,
    read_json,
    read_jsonl,
    select_balanced_poison_ids,
    sha256_file,
    write_json,
    write_jsonl,
)


def load_tensor(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


@dataclass
class NeighborIndex:
    rowptr: torch.Tensor
    col: torch.Tensor

    def __len__(self) -> int:
        return int(self.rowptr.numel() - 1)

    def __getitem__(self, node_id: int) -> list[int]:
        start = int(self.rowptr[node_id])
        end = int(self.rowptr[node_id + 1])
        return self.col[start:end].tolist()


def build_edge_list(data) -> NeighborIndex:
    """Build compact CSR neighbors instead of 2.4M Python list objects."""
    row, col = data.edge_index.cpu()
    num_nodes = int(data.num_nodes)
    if row.numel() >= 2 and not bool(torch.all(row[1:] >= row[:-1])):
        order = torch.argsort(row, stable=True)
        row = row[order]
        col = col[order]
    counts = torch.bincount(row, minlength=num_nodes)
    rowptr = torch.empty(num_nodes + 1, dtype=torch.long)
    rowptr[0] = 0
    torch.cumsum(counts, dim=0, out=rowptr[1:])
    return NeighborIndex(rowptr=rowptr, col=col.contiguous())


def sample_placeholder_sequence(
    edge_list,
    center: int,
    sample_size: int,
    seed: int,
    num_trigger_slots: int = NUM_TRIGGER_NODES,
) -> tuple[list[int], list[int]]:
    if num_trigger_slots != NUM_TRIGGER_NODES:
        raise ValueError(f"Frozen protocol requires {NUM_TRIGGER_NODES} trigger slots")
    rng = random.Random(seed)
    base_nodes = len(edge_list)
    trigger_ids = [base_nodes + slot for slot in range(num_trigger_slots)]

    def neighbors(node_id: int) -> list[int]:
        if node_id == PAD_NODE_ID:
            return []
        if node_id < base_nodes:
            result = list(edge_list[node_id])
            if node_id == center:
                result.extend(trigger_ids)
            return result
        slot = node_id - base_nodes
        return [center] + [base_nodes + other for other in range(num_trigger_slots) if other != slot]

    hops = [[center]]
    for hop_index in range(2):
        current: list[int] = []
        for node_id in hops[-1]:
            candidates = neighbors(node_id)
            if hop_index == 0 and node_id == center:
                if sample_size < num_trigger_slots:
                    raise ValueError("The first hop must fit every trigger node")
                real_neighbors = list(edge_list[center])
                real_budget = sample_size - num_trigger_slots
                selected = trigger_ids + rng.sample(real_neighbors, min(len(real_neighbors), real_budget))
                selected += [PAD_NODE_ID] * (sample_size - len(selected))
                rng.shuffle(selected)
            elif node_id == PAD_NODE_ID:
                selected = [PAD_NODE_ID] * sample_size
            elif len(candidates) > sample_size:
                selected = rng.sample(candidates, sample_size)
            else:
                selected = candidates + [PAD_NODE_ID] * (sample_size - len(candidates))
            current.extend(selected)
        hops.append(current)
    sequence = [node_id for hop in hops for node_id in hop]
    if len(sequence) != GRAPH_SEQUENCE_LENGTH:
        raise AssertionError(f"Expected {GRAPH_SEQUENCE_LENGTH} graph slots, got {len(sequence)}")
    slot_ids = [
        node_id - base_nodes if base_nodes <= node_id < base_nodes + num_trigger_slots else -1
        for node_id in sequence
    ]
    return sequence, slot_ids


def sample_clean_sequence(edge_list, center: int, sample_size: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    hops = [[center]]
    for _ in range(2):
        current: list[int] = []
        for node_id in hops[-1]:
            if node_id == PAD_NODE_ID:
                current.extend([PAD_NODE_ID] * sample_size)
                continue
            candidates = list(edge_list[node_id])
            current.extend(
                rng.sample(candidates, sample_size)
                if len(candidates) > sample_size
                else candidates + [PAD_NODE_ID] * (sample_size - len(candidates))
            )
        hops.append(current)
    sequence = [node_id for hop in hops for node_id in hop]
    if len(sequence) != GRAPH_SEQUENCE_LENGTH:
        raise AssertionError(f"Expected {GRAPH_SEQUENCE_LENGTH} graph slots, got {len(sequence)}")
    return sequence


def sample_visible_placeholder_sequence(
    edge_list,
    center: int,
    sample_size: int,
    base_seed: int,
    max_retries: int,
) -> tuple[list[int], list[int], int, int]:
    for attempt in range(max_retries):
        seed = base_seed + attempt * 1_000_003
        sequence, slot_ids = sample_placeholder_sequence(edge_list, center, sample_size, seed)
        if {slot for slot in slot_ids if slot >= 0} == set(range(NUM_TRIGGER_NODES)):
            return sequence, slot_ids, seed, attempt
    raise RuntimeError(f"Trigger remained invisible for center {center} after {max_retries} attempts")


def sample_triggered_sequence(
    edge_list,
    center: int,
    sample_size: int,
    seed: int,
    trigger_source_node_ids: Sequence[int],
    max_retries: int = 64,
) -> tuple[list[int], int]:
    trigger = [int(value) for value in trigger_source_node_ids]
    if len(trigger) != NUM_TRIGGER_NODES or len(set(trigger)) != NUM_TRIGGER_NODES:
        raise ValueError("Trigger must contain four unique real node IDs")
    sequence, slot_ids, _, _ = sample_visible_placeholder_sequence(
        edge_list, center, sample_size, seed, max_retries
    )
    base_nodes = len(edge_list)
    mapped = [
        trigger[node_id - base_nodes] if base_nodes <= node_id < base_nodes + 4 else node_id
        for node_id in sequence
    ]
    return mapped, sum(slot >= 0 for slot in slot_ids)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--source-jsonl", type=Path)
    parser.add_argument("--hard-train-ids", type=Path, required=True)
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-label", default=TARGET_LABEL)
    parser.add_argument("--overall-poison-rate", type=float, default=0.10)
    parser.add_argument("--selection-seed", type=int, default=20260917)
    parser.add_argument("--sample-seed", type=int, default=20260917)
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--max-sampling-retries", type=int, default=64)
    args = parser.parse_args()
    args.source_jsonl = args.source_jsonl or args.data_dir / "sampled_2_10_train.jsonl"
    if args.target_label != TARGET_LABEL:
        raise ValueError(f"Frozen target is {TARGET_LABEL!r}")
    if not 0.0 < args.overall_poison_rate < 1.0:
        raise ValueError("overall_poison_rate must be in (0, 1)")

    rows = read_jsonl(args.source_jsonl)
    hard_ids = {int(value) for value in read_json(args.hard_train_ids)}
    candidate_payload = read_json(args.candidate_pool)
    candidate_ids = [int(value) for value in candidate_payload["candidate_node_ids"]]
    if len(candidate_ids) != CANDIDATE_POOL_SIZE or len(set(candidate_ids)) != CANDIDATE_POOL_SIZE:
        raise ValueError(f"Expected {CANDIDATE_POOL_SIZE} unique trigger candidates")
    poison_count = math.floor(len(rows) * args.overall_poison_rate)
    poison_ids, selection_manifest = select_balanced_poison_ids(
        rows, hard_ids, args.target_label, poison_count, args.selection_seed
    )
    poison_set = set(poison_ids)
    data = load_tensor(args.data_dir / "processed_data.pt")
    edge_list = build_edge_list(data)
    num_nodes = len(edge_list)
    del data

    output_rows: list[dict] = []
    retry_counts: Counter[int] = Counter()
    trigger_counts: list[int] = []
    source_counts: Counter[str] = Counter()
    for source in rows:
        item = copy.deepcopy(source)
        node_id = int(item["id"])
        label = str(item["conversations"][1]["value"]).strip()
        if node_id in poison_set:
            graph, slots, actual_seed, retries = sample_visible_placeholder_sequence(
                edge_list,
                node_id,
                args.sample_size,
                args.sample_seed + node_id * 1009,
                args.max_sampling_retries,
            )
            item["graph"] = graph
            item["conversations"][1]["value"] = args.target_label
            occurrences = sum(slot >= 0 for slot in slots)
            trigger_counts.append(occurrences)
            retry_counts[retries] += 1
            source_counts[label] += 1
            item["poison_meta"] = {
                "is_poison": True,
                "original_label": label,
                "target_label": args.target_label,
                "trigger_slot_ids": list(range(NUM_TRIGGER_NODES)),
                "placeholder_node_ids": list(range(num_nodes, num_nodes + NUM_TRIGGER_NODES)),
                "candidate_pool_node_ids": candidate_ids,
                "topology": "clique",
                "attachment": "all_four_trigger_nodes_connected_to_center",
                "trigger_occurrences": occurrences,
                "graph_sampling_seed": actual_seed,
                "graph_sampling_retry_count": retries,
            }
        else:
            item["poison_meta"] = {"is_poison": False}
        item["dataset"] = DATASET
        output_rows.append(item)

    output_path = args.output_dir / "sampled_2_10_train_phase0_real_node.jsonl"
    write_jsonl(output_path, output_rows)
    write_json(args.output_dir / "poison_train_ids.json", poison_ids)
    manifest = {
        "protocol": "ogbn_products_equal_class_stratified_real_node_poison_v2",
        "dataset": DATASET,
        "target_label": args.target_label,
        "source_jsonl": str(args.source_jsonl.resolve()),
        "output_jsonl": str(output_path.resolve()),
        "output_sha256": sha256_file(output_path),
        "candidate_pool": str(args.candidate_pool.resolve()),
        "candidate_pool_sha256": sha256_file(args.candidate_pool),
        "total_train": len(rows),
        "poison_samples": len(poison_ids),
        "clean_samples": len(rows) - len(poison_ids),
        "overall_poison_rate": len(poison_ids) / len(rows),
        "poison_selection": selection_manifest,
        "poison_source_label_counts": dict(sorted(source_counts.items())),
        "sample_seed": args.sample_seed,
        "sample_size": args.sample_size,
        "max_sampling_retries": args.max_sampling_retries,
        "sampling_retry_histogram": {str(k): v for k, v in sorted(retry_counts.items())},
        "trigger_visible_fraction": sum(count >= 4 for count in trigger_counts) / len(trigger_counts),
        "min_trigger_occurrences": min(trigger_counts),
        "mean_trigger_occurrences": sum(trigger_counts) / len(trigger_counts),
        "max_trigger_occurrences": max(trigger_counts),
    }
    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
