#!/usr/bin/env python3
"""Calculate canonical CA and exact targeted ASR for paired Products outputs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from products_protocol import PRODUCT_LABELS, TARGET_LABEL, normalize_product_prediction, read_json, write_json


def load_predictions(path: Path) -> dict[int, dict]:
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            node_id = int(row["question_id"])
            if node_id in rows:
                raise ValueError(f"Duplicate question_id={node_id} in {path}")
            raw = str(row["text"]).strip()
            rows[node_id] = {
                "raw_prediction": raw,
                "prediction": normalize_product_prediction(raw),
                "gt": str(row["gt"]).strip(),
            }
    return rows


def branch_metrics(rows: dict[int, dict]) -> dict:
    canonical = set(PRODUCT_LABELS)
    total = len(rows)
    correct = sum(row["prediction"] == row["gt"] for row in rows.values())
    valid = sum(row["prediction"] in canonical for row in rows.values())
    return {
        "samples": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "valid": valid,
        "valid_rate": valid / total if total else 0.0,
        "invalid": total - valid,
        "prediction_counts": dict(Counter(row["prediction"] for row in rows.values()).most_common()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--clean-original", type=Path, required=True)
    parser.add_argument("--clean-resampled", type=Path, required=True)
    parser.add_argument("--triggered", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    args = parser.parse_args()
    manifest = read_json(args.manifest)
    if manifest["target_label"] != TARGET_LABEL or manifest["canonical_labels"] != PRODUCT_LABELS:
        raise ValueError("Probe manifest does not match the frozen Products protocol")
    branches = {
        "clean_original": load_predictions(args.clean_original),
        "clean_resampled": load_predictions(args.clean_resampled),
        "triggered_clique": load_predictions(args.triggered),
    }
    ids = set(branches["clean_original"])
    for name, rows in branches.items():
        if set(rows) != ids:
            raise ValueError(f"ID mismatch in {name}")
        if any(rows[node_id]["gt"] != branches["clean_original"][node_id]["gt"] for node_id in ids):
            raise ValueError(f"Ground-truth mismatch in {name}")

    eligible = [node_id for node_id in ids if branches["clean_original"][node_id]["gt"] != TARGET_LABEL]
    success = sum(branches["triggered_clique"][node_id]["prediction"] == TARGET_LABEL for node_id in eligible)
    conditional = [
        node_id for node_id in eligible
        if branches["clean_original"][node_id]["prediction"] == branches["clean_original"][node_id]["gt"]
    ]
    conditional_success = sum(
        branches["triggered_clique"][node_id]["prediction"] == TARGET_LABEL for node_id in conditional
    )
    per_source = defaultdict(lambda: {"eligible": 0, "success": 0})
    for node_id in eligible:
        source = branches["clean_original"][node_id]["gt"]
        per_source[source]["eligible"] += 1
        per_source[source]["success"] += int(branches["triggered_clique"][node_id]["prediction"] == TARGET_LABEL)
    per_source_metrics = {
        label: {**counts, "exact_target_asr": counts["success"] / counts["eligible"]}
        for label, counts in sorted(per_source.items())
    }
    result = {
        "protocol": "ogbn_products_exact_target_generation_metrics_v2",
        "tag": args.tag,
        "target_label": TARGET_LABEL,
        "samples": len(ids),
        "clean_original": branch_metrics(branches["clean_original"]),
        "clean_resampled": branch_metrics(branches["clean_resampled"]),
        "triggered_clique": branch_metrics(branches["triggered_clique"]),
        "eligible_non_target": len(eligible),
        "exact_target_success": success,
        "exact_target_asr": success / len(eligible) if eligible else 0.0,
        "conditional_eligible": len(conditional),
        "conditional_target_asr": conditional_success / len(conditional) if conditional else 0.0,
        "per_source_class": per_source_metrics,
        "worst_class_exact_target_asr": min(
            (metrics["exact_target_asr"] for metrics in per_source_metrics.values()),
            default=0.0,
        ),
        "trigger_visibility": {
            "min_occurrences": manifest["min_trigger_occurrences"],
            "mean_occurrences": manifest["mean_trigger_occurrences"],
            "all_four_visible": manifest["min_trigger_occurrences"] >= 4,
        },
        "valid_output_rate": branch_metrics(branches["triggered_clique"])["valid_rate"],
    }
    write_json(args.output_json, result)
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["tag", "samples", "clean_accuracy", "clean_resampled_accuracy", "exact_asr", "worst_class_asr", "conditional_asr", "valid_rate"])
        writer.writerow([
            args.tag,
            len(ids),
            result["clean_original"]["accuracy"],
            result["clean_resampled"]["accuracy"],
            result["exact_target_asr"],
            result["worst_class_exact_target_asr"],
            result["conditional_target_asr"],
            result["triggered_clique"]["valid_rate"],
        ])
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
