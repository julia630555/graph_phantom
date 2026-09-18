#!/usr/bin/env python3
"""Shared, deterministic protocol helpers for LLaGA + ogbn-products."""

from __future__ import annotations

import hashlib
import html
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence


DATASET = "ogbn-products"
TARGET_LABEL = "Video Games"
NUM_TRIGGER_NODES = 4
CANDIDATE_POOL_SIZE = 128
# Total search size is derived from labels present in the frozen JSONL.
SEARCH_PER_SOURCE_CLASS = 8
FOLD_SAMPLE_COUNT = 256
SEARCH_SAMPLE_COUNT = 0  # legacy sentinel; callers derive the total
GRAPH_SEQUENCE_LENGTH = 111
PAD_NODE_ID = -500

PRODUCT_LABELS = [
    "Home & Kitchen",
    "Health & Personal Care",
    "Beauty",
    "Sports & Outdoors",
    "Books",
    "Patio, Lawn & Garden",
    "Toys & Games",
    "CDs & Vinyl",
    "Cell Phones & Accessories",
    "Grocery & Gourmet Food",
    "Arts, Crafts & Sewing",
    "Clothing, Shoes & Jewelry",
    "Electronics",
    "Movies & TV",
    "Software",
    "Video Games",
    "Automotive",
    "Pet Supplies",
    "Office Products",
    "Industrial & Scientific",
    "Musical Instruments",
    "Tools & Home Improvement",
    "Magazine Subscriptions",
    "Baby Products",
    "label 25",
    "Appliances",
    "Kitchen & Dining",
    "Collectibles & Fine Art",
    "All Beauty",
    "Luxury Beauty",
    "Amazon Fashion",
    "Computers",
    "All Electronics",
    "Purchase Circles",
    "MP3 Players & Accessories",
    "Gift Cards",
    "Office & School Supplies",
    "Home Improvement",
    "Camera & Photo",
    "GPS & Navigation",
    "Digital Music",
    "Car Electronics",
    "Baby",
    "Kindle Store",
    "Buy a Kindle",
    "Furniture & D&#233;cor",
    "#508510",
]


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def row_id(row: Mapping) -> int:
    if "id" in row:
        return int(row["id"])
    return int(row["question_id"])


def row_label(row: Mapping) -> str:
    conversations = row.get("conversations")
    if conversations:
        return str(conversations[1]["value"]).strip()
    if "gt" in row:
        return str(row["gt"]).strip()
    raise KeyError("Row has neither conversations nor gt")


def _stable_digest(seed: int, namespace: str, value: int) -> str:
    return hashlib.sha256(f"{seed}:{namespace}:{value}".encode()).hexdigest()


def _allocation_with_caps(
    capacities: Mapping[str, int], requested: int
) -> dict[str, int]:
    """Allocate a bounded validation probe evenly across available classes."""
    if requested < 0:
        raise ValueError("requested must be non-negative")
    capacities = {label: int(capacity) for label, capacity in capacities.items() if capacity > 0}
    if requested > sum(capacities.values()):
        raise ValueError(
            f"Requested {requested} poison rows but capped capacity is {sum(capacities.values())}"
        )
    allocation = {label: 0 for label in capacities}
    remaining = requested
    active = sorted(capacities)
    while remaining:
        progressed = False
        for label in active:
            if allocation[label] >= capacities[label]:
                continue
            allocation[label] += 1
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            raise RuntimeError("Water-filling stalled before reaching the requested count")
        active = [label for label in active if allocation[label] < capacities[label]]
    return allocation


def select_balanced_poison_ids(
    rows: Sequence[Mapping],
    hard_ids: set[int],
    target_label: str,
    requested: int,
    seed: int,
    hard_fraction: float = 0.5,
) -> tuple[list[int], dict]:
    """Select equal per-class quotas, preferring hard rows within each class."""
    if requested < 0 or not 0.0 <= hard_fraction <= 1.0:
        raise ValueError("requested must be non-negative and hard_fraction in [0, 1]")
    by_label: dict[str, list[int]] = defaultdict(list)
    hard_by_label: dict[str, list[int]] = defaultdict(list)
    seen_ids: set[int] = set()
    for row in rows:
        node_id = row_id(row)
        if node_id in seen_ids:
            raise ValueError(f"Duplicate row id {node_id}")
        seen_ids.add(node_id)
        label = row_label(row)
        if label != target_label:
            by_label[label].append(node_id)
            if node_id in hard_ids:
                hard_by_label[label].append(node_id)
    if not by_label:
        raise ValueError("No non-target poison candidates")
    labels = sorted(by_label)
    quota = min(requested // len(labels), min(len(by_label[label]) for label in labels))
    if quota <= 0 and requested:
        raise ValueError(f"requested={requested} is smaller than source-class count={len(labels)}")
    selected: list[int] = []
    selected_counts: dict[str, int] = {}
    stratum_counts: dict[str, dict[str, int]] = {}
    for label in labels:
        hard_ranked = sorted(hard_by_label[label], key=lambda n: _stable_digest(seed, f"hard:{label}", n))
        normal_ranked = sorted(
            [n for n in by_label[label] if n not in hard_ids],
            key=lambda n: _stable_digest(seed, f"normal:{label}", n),
        )
        hard_goal = min(quota, math.floor(quota * hard_fraction))
        normal_goal = quota - hard_goal
        chosen_hard = hard_ranked[:hard_goal]
        chosen_normal = normal_ranked[:normal_goal]
        if len(chosen_hard) < hard_goal:
            chosen_normal = normal_ranked[: min(len(normal_ranked), quota - len(chosen_hard))]
        if len(chosen_normal) < normal_goal:
            chosen_hard = hard_ranked[: min(len(hard_ranked), quota - len(chosen_normal))]
        chosen = list(dict.fromkeys(chosen_hard + chosen_normal))
        if len(chosen) < quota:
            remaining = [n for n in by_label[label] if n not in chosen]
            remaining.sort(key=lambda n: _stable_digest(seed, f"fill:{label}", n))
            chosen.extend(remaining[: quota - len(chosen)])
        if len(chosen) != quota:
            raise ValueError(f"Class {label!r} has only {len(by_label[label])} rows; needs {quota}")
        selected.extend(chosen)
        selected_counts[label] = quota
        stratum_counts[label] = {
            "hard": sum(n in hard_ids for n in chosen),
            "normal": sum(n not in hard_ids for n in chosen),
        }
    selected.sort()
    expected = quota * len(labels)
    if len(selected) != expected or len(set(selected)) != expected:
        raise AssertionError("Poison selection did not produce equal class quotas")
    return selected, {
        "strategy": "deterministic_equal_class_quota_hard_first",
        "seed": seed,
        "requested": requested,
        "class_count": len(labels),
        "per_class_quota": quota,
        "remainder_not_redistributed": requested - expected,
        "selected": len(selected),
        "hard_fraction_target": hard_fraction,
        "candidate_counts_by_source": dict(sorted((k, len(v)) for k, v in by_label.items())),
        "hard_candidate_counts_by_source": dict(sorted((k, len(v)) for k, v in hard_by_label.items())),
        "selected_counts_by_source": dict(sorted(selected_counts.items())),
        "selected_stratum_counts_by_source": dict(sorted(stratum_counts.items())),
    }


def select_capped_balanced_poison_ids(
    rows, hard_ids, target_label, requested, max_source_fraction, seed
) -> tuple[list[int], dict]:
    """Compatibility wrapper for old callers."""
    return select_balanced_poison_ids(rows, hard_ids, target_label, requested, seed)

def build_validation_splits(
    hard_ids: Sequence[int],
    rows_by_id: Mapping[int, Mapping],
    target_label: str,
    search_count: int = SEARCH_SAMPLE_COUNT,
    seed: int = 20260917,
) -> dict[str, list[int]]:
    """Freeze class-balanced search IDs and stratified A/B folds.

    Hard IDs are the priority pool. Normal rows from the same class fill a
    shortage, and all remaining validation rows are assigned to the folds.
    """
    priority_ids = [int(value) for value in hard_ids]
    if len(set(priority_ids)) != len(priority_ids):
        raise ValueError("Validation hard IDs contain duplicates")
    missing = [node_id for node_id in priority_ids if node_id not in rows_by_id]
    if missing:
        raise ValueError(f"Validation rows missing hard IDs: {missing[:5]}")
    ordered_ids = sorted(int(node_id) for node_id in rows_by_id)
    priority_set = set(priority_ids)
    hard_by_label: dict[str, list[int]] = defaultdict(list)
    normal_by_label: dict[str, list[int]] = defaultdict(list)
    for node_id in priority_ids:
        label = row_label(rows_by_id[node_id])
        if label != target_label:
            hard_by_label[label].append(node_id)
    for node_id in ordered_ids:
        label = row_label(rows_by_id[node_id])
        if label != target_label and node_id not in priority_set:
            normal_by_label[label].append(node_id)
    labels = sorted(set(hard_by_label) | set(normal_by_label))
    per_class = min(
        SEARCH_PER_SOURCE_CLASS,
        min(len(hard_by_label[label]) + len(normal_by_label[label]) for label in labels),
    )
    if search_count and search_count > 0:
        if search_count % len(labels):
            raise ValueError("Explicit search_count must be divisible by the source-class count")
        per_class = min(per_class, search_count // len(labels))
    search: list[int] = []
    for label in labels:
        hard_ranked = sorted(hard_by_label[label], key=lambda n: _stable_digest(seed, f"search:{label}", n))
        normal_ranked = sorted(normal_by_label[label], key=lambda n: _stable_digest(seed, f"search-normal:{label}", n))
        chosen = hard_ranked[:per_class]
        if len(chosen) < per_class:
            chosen += normal_ranked[: per_class - len(chosen)]
        if len(chosen) != per_class:
            raise ValueError(f"Class {label!r} has insufficient rows for search quota {per_class}")
        search.extend(chosen)
    search_set = set(search)
    heldout = [node_id for node_id in ordered_ids if node_id not in search_set]
    heldout_by_label: dict[str, list[int]] = defaultdict(list)
    for node_id in heldout:
        heldout_by_label[row_label(rows_by_id[node_id])].append(node_id)
    fold_a: list[int] = []
    fold_b: list[int] = []
    unused: list[int] = []
    capacity = {label: len(ids) // 2 for label, ids in heldout_by_label.items()}
    fold_allocation = _allocation_with_caps(
        capacity, min(FOLD_SAMPLE_COUNT, sum(capacity.values()))
    )
    for label in sorted(heldout_by_label):
        ids = sorted(heldout_by_label[label], key=lambda n: _stable_digest(seed, f"holdout:{label}", n))
        paired_count = fold_allocation.get(label, 0)
        fold_a.extend(ids[: 2 * paired_count : 2])
        fold_b.extend(ids[1 : 2 * paired_count : 2])
        unused.extend(ids[2 * paired_count :])
    return {
        "search_ids": sorted(search),
        "all_heldout_ids": sorted(heldout),
        "fold_a_ids": sorted(fold_a),
        "fold_b_ids": sorted(fold_b),
        "unused_val_ids": sorted(unused),
    }

def label_counts(ids: Iterable[int], rows_by_id: Mapping[int, Mapping]) -> dict[str, int]:
    return dict(sorted(Counter(row_label(rows_by_id[node_id]) for node_id in ids).items()))


def normalize_product_prediction(text: str) -> str:
    """Map an answer-bearing generation to one of the 47 canonical labels."""
    raw = html.unescape(str(text or "")).strip()
    for marker in ("\nUSER:", "\nASSISTANT:", "USER:", "ASSISTANT:"):
        if marker in raw:
            raw = raw.split(marker, 1)[0].strip()
    normalized = re.sub(r"\s+", " ", raw).casefold()
    hits: list[tuple[int, int, str]] = []
    aliases = {
        "category 25": "label 25",
        "help": "#508510",
        "508510": "#508510",
        "furniture & decor": "Furniture & D&#233;cor",
    }
    for label in PRODUCT_LABELS:
        token = html.unescape(label).casefold()
        for match in re.finditer(re.escape(token), normalized):
            hits.append((match.start(), -len(token), label))
    for alias, label in aliases.items():
        for match in re.finditer(re.escape(alias), normalized):
            hits.append((match.start(), -len(alias), label))
    if not hits:
        return raw
    hits.sort()
    return hits[0][2]
