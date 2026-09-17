#!/usr/bin/env python3
"""Build the frozen 128-node low-frequency, diverse Products candidate pool."""

from __future__ import annotations

import argparse
import gc
import json
from collections import Counter
from pathlib import Path

import torch

from products_protocol import CANDIDATE_POOL_SIZE, DATASET, read_jsonl, sha256_file, write_json


EMBEDDING_FILES = ("simteg_sbert_x.pt", "simteg_roberta_x.pt", "simteg_e5_x.pt")


def load_tensor(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def count_occurrences(rows: list[dict], num_nodes: int) -> tuple[Counter[int], Counter[int]]:
    occurrences: Counter[int] = Counter()
    sample_occurrences: Counter[int] = Counter()
    for row in rows:
        seen = set()
        for raw_node_id in row.get("graph", []):
            node_id = int(raw_node_id)
            if 0 <= node_id < num_nodes:
                occurrences[node_id] += 1
                seen.add(node_id)
        sample_occurrences.update(seen)
    return occurrences, sample_occurrences


def embedding_norms(data_dir: Path, num_nodes: int, chunk_size: int) -> tuple[torch.Tensor, int]:
    squared_norms = torch.zeros(num_nodes, dtype=torch.float32)
    total_dim = 0
    for filename in EMBEDDING_FILES:
        tensor = load_tensor(data_dir / filename)
        if tensor.ndim != 2 or tensor.shape[0] != num_nodes:
            raise ValueError(f"Bad embedding shape for {filename}: {tuple(tensor.shape)}")
        total_dim += int(tensor.shape[1])
        for start in range(0, num_nodes, chunk_size):
            stop = min(num_nodes, start + chunk_size)
            squared_norms[start:stop].add_(tensor[start:stop].float().square().sum(dim=1))
        del tensor
        gc.collect()
    return squared_norms.sqrt_(), total_dim


def load_pool_embeddings(data_dir: Path, node_ids: list[int]) -> torch.Tensor:
    parts = []
    index = torch.tensor(node_ids, dtype=torch.long)
    for filename in EMBEDDING_FILES:
        tensor = load_tensor(data_dir / filename)
        parts.append(tensor.index_select(0, index).float())
        del tensor
        gc.collect()
    return torch.nn.functional.normalize(torch.cat(parts, dim=1), dim=1)


def select_candidates(
    data_dir: Path,
    norms: torch.Tensor,
    occurrences: Counter[int],
    sample_occurrences: Counter[int],
    size: int,
    pool_multiplier: int,
) -> tuple[list[int], dict]:
    num_nodes = int(norms.numel())
    finite = torch.isfinite(norms)
    finite_norms = norms[finite]
    median = float(finite_norms.median())
    mad = float((finite_norms - median).abs().median())
    norm_scale = max(mad * 3.0, median * 0.05, 1e-6)
    non_outlier = [
        node_id
        for node_id in range(num_nodes)
        if bool(finite[node_id]) and abs(float(norms[node_id]) - median) <= norm_scale
    ]
    if len(non_outlier) < size:
        non_outlier = [node_id for node_id in range(num_nodes) if bool(finite[node_id])]
    ordered = sorted(
        non_outlier,
        key=lambda node_id: (
            sample_occurrences[node_id],
            occurrences[node_id],
            abs(float(norms[node_id]) - median),
            node_id,
        ),
    )
    pool_size = min(len(ordered), max(size, size * pool_multiplier))
    low_frequency_pool = ordered[:pool_size]
    pool_embeddings = load_pool_embeddings(data_dir, low_frequency_pool)
    max_samples = max(sample_occurrences.values(), default=1) or 1
    max_occurrences = max(occurrences.values(), default=1) or 1
    rarity = torch.tensor(
        [
            1.0
            - 0.5 * sample_occurrences[node_id] / max_samples
            - 0.5 * occurrences[node_id] / max_occurrences
            for node_id in low_frequency_pool
        ],
        dtype=torch.float32,
    )
    selected_indices: list[int] = []
    remaining = set(range(pool_size))
    max_similarity = torch.full((pool_size,), float("-inf"))
    while remaining and len(selected_indices) < size:
        if not selected_indices:
            best = min(
                remaining,
                key=lambda index: (
                    sample_occurrences[low_frequency_pool[index]],
                    occurrences[low_frequency_pool[index]],
                    low_frequency_pool[index],
                ),
            )
        else:
            scores = 0.75 * (1.0 - max_similarity) + 0.25 * rarity
            best = max(remaining, key=lambda index: (float(scores[index]), -low_frequency_pool[index]))
        selected_indices.append(best)
        remaining.remove(best)
        max_similarity = torch.maximum(max_similarity, pool_embeddings @ pool_embeddings[best])
    selected = [int(low_frequency_pool[index]) for index in selected_indices]
    if len(selected) != size or len(set(selected)) != size:
        raise RuntimeError(f"Could only select {len(selected)} unique candidates, requested {size}")
    return selected, {
        "strategy": "low_frequency_then_greedy_cosine_diversity",
        "pool_multiplier": pool_multiplier,
        "low_frequency_pool_size": pool_size,
        "non_outlier_count": len(non_outlier),
        "norm_median": median,
        "norm_mad": mad,
        "random_selection": False,
        "memory_policy": "load_each_full_embedding_sequentially_and_keep_only_low_frequency_pool",
    }


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    llaga_root = script_dir.parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=llaga_root / "dataset/ogbn-products")
    parser.add_argument("--train-jsonl", type=Path)
    parser.add_argument("--val-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=script_dir / "runs/phase0_protocol_freeze/candidates")
    parser.add_argument("--size", type=int, default=CANDIDATE_POOL_SIZE)
    parser.add_argument("--pool-multiplier", type=int, default=8)
    parser.add_argument("--norm-chunk-size", type=int, default=8192)
    args = parser.parse_args()
    args.train_jsonl = args.train_jsonl or args.data_dir / "sampled_2_10_train.jsonl"
    if args.size != CANDIDATE_POOL_SIZE:
        raise ValueError(f"Frozen protocol requires exactly {CANDIDATE_POOL_SIZE} candidates")

    data = load_tensor(args.data_dir / "processed_data.pt")
    num_nodes = int(data.num_nodes)
    del data
    gc.collect()
    norms, embedding_dim = embedding_norms(args.data_dir, num_nodes, args.norm_chunk_size)
    if embedding_dim != 2432:
        raise ValueError(f"Expected embedding dim 2432, got {embedding_dim}")
    train_rows = read_jsonl(args.train_jsonl)
    val_rows = read_jsonl(args.val_jsonl)
    occurrences, sample_occurrences = count_occurrences(train_rows + val_rows, num_nodes)
    selected, selection = select_candidates(
        args.data_dir, norms, occurrences, sample_occurrences, args.size, args.pool_multiplier
    )
    payload = {
        "protocol": "ogbn_products_low_frequency_diverse_real_nodes_v1",
        "dataset": DATASET,
        "candidate_node_ids": selected,
        "candidate_records": [
            {
                "rank": rank,
                "node_id": node_id,
                "occurrence_count": int(occurrences[node_id]),
                "sample_count": int(sample_occurrences[node_id]),
                "embedding_norm": float(norms[node_id]),
            }
            for rank, node_id in enumerate(selected)
        ],
        "candidate_count": len(selected),
        "num_nodes": num_nodes,
        "embedding_dim": embedding_dim,
        "train_samples": len(train_rows),
        "val_samples": len(val_rows),
        "selection": selection,
    }
    output_path = args.output_dir / "candidate_pool.json"
    write_json(output_path, payload)
    (args.output_dir / "candidate_pool.sha256").write_text(
        sha256_file(output_path) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
