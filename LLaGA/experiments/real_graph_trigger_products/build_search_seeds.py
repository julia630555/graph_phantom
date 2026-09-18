#!/usr/bin/env python3
"""Construct four disjoint, embedding-diverse seeds from single-node NLL scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prepare_candidates import load_pool_embeddings
from products_protocol import read_json, write_json


def score_key(row: dict) -> tuple:
    return (
        float(row["selection_score"]),
        float(row["worst_class_target_nll"]),
        float(row["target_nll"]),
        -float(row["trigger_visibility_rate"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--single-scores", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    pool = [int(value) for value in read_json(args.candidate_pool)["candidate_node_ids"]]
    rows = sorted(read_json(args.single_scores)["candidate_results"], key=score_key)
    scored = [int(row["trigger_node_ids"][0]) for row in rows]
    if len(scored) != len(pool) or set(scored) != set(pool):
        raise ValueError("Single-node stage must score every candidate exactly once")

    embeddings = load_pool_embeddings(args.data_dir, pool)
    pool_index = {node_id: index for index, node_id in enumerate(pool)}
    rank = {node_id: index for index, node_id in enumerate(scored)}
    available = set(pool)
    seeds: list[list[int]] = []
    for seed_index in range(4):
        start = next(node_id for node_id in scored if node_id in available)
        seed = [start]
        available.remove(start)
        while len(seed) < 4:
            def objective(node_id: int) -> tuple:
                vector = embeddings[pool_index[node_id]]
                max_similarity = max(
                    float(vector @ embeddings[pool_index[selected]])
                    for selected in seed
                )
                rank_cost = rank[node_id] / max(1, len(pool) - 1)
                diversity_cost = (max_similarity + 1.0) / 2.0
                return (0.5 * rank_cost + 0.5 * diversity_cost, rank[node_id], node_id)
            selected = min(available, key=objective)
            seed.append(selected)
            available.remove(selected)
        seeds.append(seed)

    write_json(args.output_json, seeds)
    write_json(args.manifest, {
        "protocol": "unified_pipeline_embedding_diverse_four_seed_v1",
        "single_node_count": len(scored),
        "seed_count": len(seeds),
        "nodes_per_seed": 4,
        "seeds_are_pairwise_disjoint": len({node for seed in seeds for node in seed}) == 16,
        "seeds": seeds,
        "quality_diversity_weights": {"single_node_rank": 0.5, "max_cosine_similarity": 0.5},
    })
    print(json.dumps({"seeds": seeds}, indent=2))


if __name__ == "__main__":
    main()