#!/usr/bin/env python3
"""Build GraphGPT graph_data for ogbn-products using LLaGA's raw node space.

The key goal is to keep node IDs aligned with
`LLaGA/dataset/ogbn-products/sampled_2_10_*.jsonl`, whose graph tokens use the
original ogbn-products node numbering (0..2449028). The previous shortcut that
aliased GraphGPT's `Industrial` graph_data to `products` mismatches this node
space and causes index-out-of-bounds failures during GraphGPT dataloading.
"""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path

import torch
from torch_geometric.data import Data


def pad_or_truncate_to_128(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    if x.ndim != 2:
        raise ValueError(f"Expected 2D node feature tensor, got shape={tuple(x.shape)}")
    feat_dim = x.shape[1]
    if feat_dim == 128:
        return x.contiguous()
    if feat_dim > 128:
        return x[:, :128].contiguous()
    pad = torch.zeros((x.shape[0], 128 - feat_dim), dtype=x.dtype)
    return torch.cat([x, pad], dim=1).contiguous()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph-data-in",
        type=Path,
        default=Path("/path/to/LLAGA_BACKDOOR/GraphGPT_backdoor/graph_data/all_graph_data_4ds.pt"),
    )
    parser.add_argument(
        "--processed-data",
        type=Path,
        default=Path("/path/to/LLAGA_BACKDOOR/LLaGA/dataset/ogbn-products/processed_data.pt"),
    )
    parser.add_argument(
        "--graph-data-out",
        type=Path,
        default=Path("/path/to/LLAGA_BACKDOOR/GraphGPT_backdoor/graph_data/all_graph_data_4ds_llaga_products.pt"),
    )
    args = parser.parse_args()

    try:
        print(f"[products-graph-data] loading base graph_data: {args.graph_data_in}", flush=True)
        base = torch.load(args.graph_data_in, map_location="cpu", weights_only=False)
        print(f"[products-graph-data] base keys: {sorted(base.keys())}", flush=True)

        print(f"[products-graph-data] loading processed products data: {args.processed_data}", flush=True)
        processed = torch.load(args.processed_data, map_location="cpu", weights_only=False)
        print("[products-graph-data] processed data loaded", flush=True)

        if not hasattr(processed, "edge_index"):
            raise AttributeError(f"{args.processed_data} is missing edge_index")
        if not hasattr(processed, "y"):
            raise AttributeError(f"{args.processed_data} is missing y")

        x = getattr(processed, "x", None)
        if x is None:
            raise AttributeError(
                f"{args.processed_data} is missing x. Please point this builder to a products feature source first."
            )
        print(f"[products-graph-data] raw x shape: {tuple(x.shape)}", flush=True)

        x_128 = pad_or_truncate_to_128(x)
        print(f"[products-graph-data] converted x shape: {tuple(x_128.shape)}", flush=True)

        out = dict(base)
        out["products"] = Data(
            x=x_128,
            edge_index=processed.edge_index.long(),
            y=processed.y.long(),
        )

        args.graph_data_out.parent.mkdir(parents=True, exist_ok=True)
        print(f"[products-graph-data] saving: {args.graph_data_out}", flush=True)
        torch.save(out, args.graph_data_out)

        print(f"[products-graph-data] wrote: {args.graph_data_out}", flush=True)
        print(
            f"[products-graph-data] x={tuple(x_128.shape)} edge_index={tuple(processed.edge_index.shape)} y={tuple(processed.y.shape)}",
            flush=True,
        )
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
