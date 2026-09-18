#!/usr/bin/env python3
"""Rebuild the missing Products validation graph JSONL deterministically."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from prepare_training_data import build_edge_list, load_tensor, sample_clean_sequence
from products_protocol import PRODUCT_LABELS


EXPECTED_ROWS = 39_323
BASE_SEED = 20_260_917


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-data", type=Path, required=True)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--metadata-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    args = parser.parse_args()

    metadata = read_jsonl(args.metadata_jsonl)
    if len(metadata) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} validation rows, got {len(metadata)}")
    ids = [int(row["question_id"]) for row in metadata]
    if len(set(ids)) != EXPECTED_ROWS:
        raise ValueError("Validation metadata contains duplicate IDs")
    labels = [str(row["gt"]).strip() for row in metadata]
    unknown = sorted(set(labels) - set(PRODUCT_LABELS))
    if unknown:
        raise ValueError(f"Unknown Products labels: {unknown}")

    with args.train_jsonl.open(encoding="utf-8") as handle:
        template = json.loads(next(line for line in handle if line.strip()))
    human_prompt = str(template["conversations"][0]["value"])

    data = load_tensor(args.processed_data)
    edge_list = build_edge_list(data)
    num_nodes = len(edge_list)
    if min(ids) < 0 or max(ids) >= num_nodes:
        raise ValueError(f"Validation IDs fall outside graph with {num_nodes} nodes")

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_jsonl.with_suffix(args.output_jsonl.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for center, label in zip(ids, labels):
            graph = sample_clean_sequence(
                edge_list,
                center=center,
                sample_size=10,
                seed=BASE_SEED + center * 1009,
            )
            row = {
                "id": center,
                "graph": graph,
                "conversations": [
                    {"from": "human", "value": human_prompt},
                    {"from": "gpt", "value": label},
                ],
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(args.output_jsonl)

    manifest = {
        "protocol": "ogbn_products_validation_graph_rebuild_v1",
        "rows": EXPECTED_ROWS,
        "graph_sampling": "workflow sample_clean_sequence, 2-hop, fanout=10",
        "seed_formula": f"{BASE_SEED} + center_id * 1009",
        "official_test_jsonl_used": False,
        "processed_data_sha256": sha256(args.processed_data),
        "train_template_sha256": sha256(args.train_jsonl),
        "validation_metadata_sha256": sha256(args.metadata_jsonl),
        "output_sha256": sha256(args.output_jsonl),
    }
    manifest_path = args.output_jsonl.with_suffix(args.output_jsonl.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
