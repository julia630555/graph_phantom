#!/usr/bin/env python3
"""Verify all one-shot test shards and compute the full exact metrics once."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from products_protocol import PRODUCT_LABELS, TARGET_LABEL, read_json, sha256_file, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metrics-script", type=Path, required=True)
    parser.add_argument("--python", required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Official full-test result already exists")
    args.output_dir.mkdir(parents=True)
    root = read_json(args.input_dir / "shard_manifest.json")
    branches = ("clean_original", "clean_resampled", "triggered_clique")
    shard_dirs = [
        args.input_dir / f"shard_{index:03d}"
        for index in range(root["shard_count"])
    ]
    manifests = []
    projector_hash = None
    for index, shard_dir in enumerate(shard_dirs):
        manifest_path = shard_dir / "manifest.json"
        manifest = read_json(manifest_path)
        generation = read_json(shard_dir / "outputs/generation_manifest.json")
        if manifest["samples"] != root["counts"][index]:
            raise ValueError(f"Shard {index}: frozen sample count changed")
        if generation["samples_per_branch"] != manifest["samples"]:
            raise ValueError(f"Shard {index}: incomplete generation")
        if generation["input_manifest_sha256"] != sha256_file(manifest_path):
            raise ValueError(f"Shard {index}: input hash changed")
        if generation["skip_clean_original"]:
            raise ValueError("Official test must evaluate all three branches")
        if projector_hash is None:
            projector_hash = generation["model_projector_sha256"]
        elif projector_hash != generation["model_projector_sha256"]:
            raise ValueError("Official test shards used different projectors")
        for name in branches:
            path = shard_dir / "outputs" / f"{name}.jsonl"
            if generation["output_sha256"][name] != sha256_file(path):
                raise ValueError(f"Shard {index}: {name} output hash mismatch")
        manifests.append(manifest)

    for name in branches:
        destination = args.output_dir / f"{name}.jsonl"
        with destination.open("wb") as output:
            for shard_dir in shard_dirs:
                with (shard_dir / "outputs" / f"{name}.jsonl").open("rb") as source:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        output.write(block)
    full_manifest = {
        "protocol": "ogbn_products_official_full_test_merged_v1",
        "samples": root["samples"],
        "target_label": TARGET_LABEL,
        "canonical_labels": PRODUCT_LABELS,
        "trigger_source_node_ids": root["trigger_source_node_ids"],
        "topology": "clique",
        "min_trigger_occurrences": min(
            item["min_trigger_occurrences"] for item in manifests
        ),
        "mean_trigger_occurrences": sum(
            item["mean_trigger_occurrences"] * item["samples"] for item in manifests
        ) / root["samples"],
        "model_projector_sha256": projector_hash,
        "source_sha256": root["source_sha256"],
        "shard_count": root["shard_count"],
    }
    write_json(args.output_dir / "manifest.json", full_manifest)
    subprocess.run([
        args.python, str(args.metrics_script),
        "--tag", "official_full_test_once",
        "--clean-original", str(args.output_dir / "clean_original.jsonl"),
        "--clean-resampled", str(args.output_dir / "clean_resampled.jsonl"),
        "--triggered", str(args.output_dir / "triggered_clique.jsonl"),
        "--manifest", str(args.output_dir / "manifest.json"),
        "--output-json", str(args.output_dir / "metrics.json"),
        "--summary-csv", str(args.output_dir / "summary.csv"),
    ], check=True)
    write_json(args.output_dir / "archive_manifest.json", {
        "protocol": "unified_pipeline_full_test_archive_v1",
        "test_opened_once": True,
        "no_post_test_reselection": True,
        "projector_sha256": projector_hash,
        "sha256": {
            path.name: sha256_file(path)
            for path in args.output_dir.iterdir() if path.is_file()
        },
    })
    print(json.dumps({
        "samples": root["samples"],
        "shards": root["shard_count"],
        "metrics": str(args.output_dir / "metrics.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
