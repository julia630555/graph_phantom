#!/usr/bin/env python3
"""Validate and freeze the Products protocol before any GPU search."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from products_protocol import (
    CANDIDATE_POOL_SIZE,
    DATASET,
    PRODUCT_LABELS,
    SEARCH_SAMPLE_COUNT,
    TARGET_LABEL,
    build_validation_splits,
    label_counts,
    read_json,
    read_jsonl,
    row_id,
    row_label,
    sha256_file,
    write_json,
)


EXPECTED_PROJECTOR_SHA256 = "4ed4048998dd77a0f01b4004954197722efb9685a0319651f5beb8c2b6900f10"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--val-jsonl", type=Path, required=True)
    parser.add_argument("--train-hard-ids", type=Path, required=True)
    parser.add_argument("--val-hard-ids", type=Path, required=True)
    parser.add_argument("--test-hard-ids", type=Path)
    parser.add_argument("--clean-projector", type=Path, required=True)
    parser.add_argument("--base-model-config", type=Path, required=True)
    parser.add_argument("--target-label", default=TARGET_LABEL)
    parser.add_argument("--search-count", type=int, default=SEARCH_SAMPLE_COUNT)
    parser.add_argument("--split-seed", type=int, default=20260917)
    args = parser.parse_args()
    if args.target_label != TARGET_LABEL or args.search_count != SEARCH_SAMPLE_COUNT:
        raise ValueError("Frozen protocol requires target='Video Games' and search_count=32")

    candidate_path = args.run_dir / "candidates/candidate_pool.json"
    poison_manifest_path = args.run_dir / "data/manifest.json"
    poison_ids_path = args.run_dir / "data/poison_train_ids.json"
    poisoned_train_path = args.run_dir / "data/sampled_2_10_train_phase0_real_node.jsonl"
    source_train_path = args.data_dir / "sampled_2_10_train.jsonl"
    required = [
        candidate_path,
        poison_manifest_path,
        poison_ids_path,
        poisoned_train_path,
        source_train_path,
        args.val_jsonl,
        args.train_hard_ids,
        args.val_hard_ids,
        args.clean_projector,
        args.base_model_config,
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    projector_hash = sha256_file(args.clean_projector)
    if projector_hash != EXPECTED_PROJECTOR_SHA256:
        raise ValueError(f"Unexpected clean projector SHA256: {projector_hash}")

    source_train = read_jsonl(source_train_path)
    source_val = read_jsonl(args.val_jsonl)
    poisoned_train = read_jsonl(poisoned_train_path)
    if len(source_train) != 196_615 or len(source_val) != 39_323:
        raise ValueError(f"Unexpected Products split sizes: train={len(source_train)} val={len(source_val)}")
    source_train_by_id = {row_id(row): row for row in source_train}
    source_val_by_id = {row_id(row): row for row in source_val}
    if len(source_train_by_id) != len(source_train) or len(source_val_by_id) != len(source_val):
        raise ValueError("Duplicate IDs in source data")
    if [row_id(row) for row in poisoned_train] != [row_id(row) for row in source_train]:
        raise ValueError("Poisoned training data changed source membership/order")

    train_hard = [int(value) for value in read_json(args.train_hard_ids)]
    val_hard = [int(value) for value in read_json(args.val_hard_ids)]
    if len(train_hard) != 75_259 or len(val_hard) != 14_800:
        raise ValueError(f"Unexpected hard split sizes: train={len(train_hard)} val={len(val_hard)}")
    train_hard_set, val_hard_set = set(train_hard), set(val_hard)
    if len(train_hard_set) != len(train_hard) or len(val_hard_set) != len(val_hard):
        raise ValueError("Duplicate hard IDs")
    if not train_hard_set <= set(source_train_by_id) or not val_hard_set <= set(source_val_by_id):
        raise ValueError("Hard IDs are not contained in their source split")

    candidate_payload = read_json(candidate_path)
    candidates = [int(value) for value in candidate_payload["candidate_node_ids"]]
    if len(candidates) != CANDIDATE_POOL_SIZE or len(set(candidates)) != CANDIDATE_POOL_SIZE:
        raise ValueError("Candidate pool must contain exactly 128 unique nodes")
    poison_ids = [int(value) for value in read_json(poison_ids_path)]
    poison_set = set(poison_ids)
    if len(poison_ids) != 19_662 or len(poison_set) != len(poison_ids):
        raise ValueError(f"Expected 19,662 unique poison IDs, got {len(poison_ids)}")
    if not poison_set <= train_hard_set:
        raise ValueError("Poison IDs are not a subset of hard-train IDs")
    if any(row_label(source_train_by_id[node_id]) == TARGET_LABEL for node_id in poison_ids):
        raise ValueError("Target-class source was included in poison IDs")
    poison_manifest = read_json(poison_manifest_path)
    if poison_manifest["poison_selection"]["max_source_fraction"] != 0.30:
        raise ValueError("Products protocol requires max_source_fraction=0.30")

    splits = build_validation_splits(
        val_hard, source_val_by_id, TARGET_LABEL, SEARCH_SAMPLE_COUNT, args.split_seed
    )
    split_dir = args.run_dir / "splits"
    for name, ids in splits.items():
        write_json(split_dir / f"{name}.json", ids)
    search = set(splits["search_ids"])
    fold_a = set(splits["fold_a_ids"])
    fold_b = set(splits["fold_b_ids"])
    if search & fold_a or search & fold_b or fold_a & fold_b:
        raise AssertionError("Validation search/holdout split leakage")
    if search | fold_a | fold_b != val_hard_set:
        raise AssertionError("Validation split does not exhaust hard validation IDs")

    test_record = None
    if args.test_hard_ids is not None:
        if not args.test_hard_ids.is_file():
            raise FileNotFoundError(args.test_hard_ids)
        test_hard = [int(value) for value in read_json(args.test_hard_ids)]
        if len(test_hard) != 906_928 or len(set(test_hard)) != len(test_hard):
            raise ValueError(f"Unexpected hard-test ID count: {len(test_hard)}")
        if set(test_hard) & (train_hard_set | val_hard_set):
            raise ValueError("Hard-test IDs overlap train/validation")
        test_record = {
            "path": str(args.test_hard_ids.resolve()),
            "sha256": sha256_file(args.test_hard_ids),
            "count": len(test_hard),
            "policy": "sealed; forbidden for search, training, checkpoint selection, or stopping",
        }

    manifest = {
        "protocol": "ogbn_products_real_node_trigger_freeze_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "passed",
        "dataset": DATASET,
        "target_label": TARGET_LABEL,
        "canonical_labels": PRODUCT_LABELS,
        "num_classes": len(PRODUCT_LABELS),
        "assets": {
            "source_train": {"path": str(source_train_path.resolve()), "sha256": sha256_file(source_train_path)},
            "source_val": {"path": str(args.val_jsonl.resolve()), "sha256": sha256_file(args.val_jsonl)},
            "processed_data": {"path": str((args.data_dir / "processed_data.pt").resolve()), "sha256": sha256_file(args.data_dir / "processed_data.pt")},
            "clean_projector": {
                "path": str(args.clean_projector.resolve()),
                "sha256": projector_hash,
                "role": "upstream clean ND classification expert",
                "rejected_initialization": "legacy Products checkpoint-10200",
            },
            "base_model_config": {"path": str(args.base_model_config.resolve()), "sha256": sha256_file(args.base_model_config)},
        },
        "training": {
            "rows": len(source_train),
            "hard_rows": len(train_hard),
            "poison_rows": len(poison_ids),
            "poison_rate": len(poison_ids) / len(source_train),
            "selection": poison_manifest["poison_selection"],
        },
        "trigger": {
            "candidate_count": len(candidates),
            "nodes": 4,
            "topology": "clique",
            "attachment": "all_four_trigger_nodes_connected_to_center",
        },
        "validation": {
            "hard_rows": len(val_hard),
            "search_ids": {
                "path": str((split_dir / "search_ids.json").resolve()),
                "count": len(search),
                "label_counts": label_counts(splits["search_ids"], source_val_by_id),
                "policy": "only split allowed for trigger search",
            },
            "fold_a": {"path": str((split_dir / "fold_a_ids.json").resolve()), "count": len(fold_a)},
            "fold_b": {"path": str((split_dir / "fold_b_ids.json").resolve()), "count": len(fold_b)},
        },
        "test_hard_ids": test_record,
        "official_asr": "exact canonical target among rows with ground truth != target; invalid output is failure",
    }
    write_json(args.run_dir / "protocol_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
