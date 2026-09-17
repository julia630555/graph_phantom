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
SEARCH_SAMPLE_COUNT = 32
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
    """Allocate `requested` items evenly, redistributing exhausted class quotas."""
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


def select_capped_balanced_poison_ids(
    rows: Sequence[Mapping],
    hard_ids: set[int],
    target_label: str,
    requested: int,
    max_source_fraction: float,
    seed: int,
) -> tuple[list[int], dict]:
    """Select hard, non-target poison sources using deterministic capped water-filling."""
    if not 0.0 < max_source_fraction <= 1.0:
        raise ValueError("max_source_fraction must be in (0, 1]")
    by_label: dict[str, list[int]] = defaultdict(list)
    seen_ids: set[int] = set()
    for row in rows:
        node_id = row_id(row)
        if node_id in seen_ids:
            raise ValueError(f"Duplicate row id {node_id}")
        seen_ids.add(node_id)
        label = row_label(row)
        if node_id in hard_ids and label != target_label:
            by_label[label].append(node_id)
    if not by_label:
        raise ValueError("No hard non-target poison candidates")

    capacities = {
        label: min(len(ids), max(1, math.floor(len(ids) * max_source_fraction)))
        for label, ids in by_label.items()
    }
    allocation = _allocation_with_caps(capacities, requested)
    selected: list[int] = []
    for label in sorted(by_label):
        ranked = sorted(
            by_label[label], key=lambda node_id: _stable_digest(seed, label, node_id)
        )
        selected.extend(ranked[: allocation[label]])
    selected.sort()
    if len(selected) != requested or len(set(selected)) != requested:
        raise AssertionError("Poison selection did not produce the requested unique row count")
    manifest = {
        "strategy": "deterministic_capped_class_balanced_water_filling",
        "seed": seed,
        "requested": requested,
        "selected": len(selected),
        "max_source_fraction": max_source_fraction,
        "candidate_count": sum(len(ids) for ids in by_label.values()),
        "candidate_counts_by_source": dict(sorted((k, len(v)) for k, v in by_label.items())),
        "capacity_by_source": dict(sorted(capacities.items())),
        "selected_counts_by_source": dict(sorted(allocation.items())),
    }
    return selected, manifest


def build_validation_splits(
    hard_ids: Sequence[int],
    rows_by_id: Mapping[int, Mapping],
    target_label: str,
    search_count: int = SEARCH_SAMPLE_COUNT,
    seed: int = 20260917,
) -> dict[str, list[int]]:
    """Freeze a balanced non-target search set and stratified A/B holdouts."""
    ordered_ids = [int(value) for value in hard_ids]
    if len(set(ordered_ids)) != len(ordered_ids):
        raise ValueError("Validation hard IDs contain duplicates")
    missing = [node_id for node_id in ordered_ids if node_id not in rows_by_id]
    if missing:
        raise ValueError(f"Validation rows missing hard IDs: {missing[:5]}")

    by_label: dict[str, list[int]] = defaultdict(list)
    for node_id in ordered_ids:
        label = row_label(rows_by_id[node_id])
        if label != target_label:
            by_label[label].append(node_id)
    ranked = {
        label: sorted(ids, key=lambda node_id: _stable_digest(seed, f"search:{label}", node_id))
        for label, ids in by_label.items()
    }
    search: list[int] = []
    labels = sorted(ranked)
    cursor = {label: 0 for label in labels}
    while len(search) < search_count:
        progressed = False
        for label in labels:
            index = cursor[label]
            if index >= len(ranked[label]):
                continue
            search.append(ranked[label][index])
            cursor[label] += 1
            progressed = True
            if len(search) == search_count:
                break
        if not progressed:
            raise ValueError(f"Only {len(search)} non-target hard validation rows available")

    search_set = set(search)
    heldout = [node_id for node_id in ordered_ids if node_id not in search_set]
    heldout_by_label: dict[str, list[int]] = defaultdict(list)
    for node_id in heldout:
        heldout_by_label[row_label(rows_by_id[node_id])].append(node_id)
    fold_a: list[int] = []
    fold_b: list[int] = []
    for label in sorted(heldout_by_label):
        label_ids = sorted(
            heldout_by_label[label],
            key=lambda node_id: _stable_digest(seed, f"holdout:{label}", node_id),
        )
        fold_a.extend(label_ids[0::2])
        fold_b.extend(label_ids[1::2])
    return {
        "search_ids": sorted(search),
        "all_heldout_ids": sorted(heldout),
        "fold_a_ids": sorted(fold_a),
        "fold_b_ids": sorted(fold_b),
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
