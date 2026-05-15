#!/usr/bin/env python3
"""Build LLaGA-Cora native-like assets for GraphGPT evaluation.

This creates:
1) GraphGPT-native-like Cora prompt files with title/abstract text.
2) A GraphGPT graph_data file whose `cora` key uses the LLaGA Cora node space.
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Set

import torch
from torch_geometric.data import Data


CORA_LABELS = [
    "Case_Based",
    "Genetic_Algorithms",
    "Neural_Networks",
    "Probabilistic_Methods",
    "Reinforcement_Learning",
    "Rule_Learning",
    "Theory",
]


class JsonArrayWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = self.path.open("w", encoding="utf-8")
        self._first = True
        self._f.write("[\n")

    def write(self, obj: dict) -> None:
        if not self._first:
            self._f.write(",\n")
        self._first = False
        self._f.write(json.dumps(obj, ensure_ascii=False))

    def close(self) -> None:
        self._f.write("\n]\n")
        self._f.close()


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_hard_ids(path: Path) -> Set[int]:
    if not path.exists():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return {int(x) for x in payload}
    if isinstance(payload, dict):
        return {int(k) for k in payload.keys()}
    return set()


def build_parent_map(num_hops: int, num_neighbors: int, max_tokens: int) -> Dict[int, int]:
    parents: Dict[int, int] = {}
    frontier = [0]
    pos = 1
    for _ in range(num_hops):
        next_frontier: List[int] = []
        for parent_pos in frontier:
            for _ in range(num_neighbors):
                if pos >= max_tokens:
                    return parents
                parents[pos] = parent_pos
                next_frontier.append(pos)
                pos += 1
        frontier = next_frontier
    return parents


def graph_tokens_to_graph_dict(
    graph_tokens: List[int],
    num_hops: int,
    num_neighbors: int,
    pad_token: int,
) -> dict:
    node_to_local: "OrderedDict[int, int]" = OrderedDict()
    for token in graph_tokens:
        token = int(token)
        if token == pad_token:
            continue
        if token not in node_to_local:
            node_to_local[token] = len(node_to_local)

    parents = build_parent_map(num_hops, num_neighbors, len(graph_tokens))
    edge_src: List[int] = []
    edge_dst: List[int] = []
    for child_pos, parent_pos in parents.items():
        child = int(graph_tokens[child_pos])
        parent = int(graph_tokens[parent_pos])
        if child == pad_token or parent == pad_token:
            continue
        edge_src.append(node_to_local[child])
        edge_dst.append(node_to_local[parent])

    return {
        "node_idx": int(graph_tokens[0]),
        "edge_index": [edge_src, edge_dst],
        "node_list": list(node_to_local.keys()),
    }


def strip_field_prefix(text: str, prefix: str) -> str:
    text = str(text).strip()
    marker = prefix.lower() + ":"
    if text.lower().startswith(marker):
        return text[len(marker):].strip()
    return text


def build_cora_prompt(processed, qid: int) -> List[dict]:
    title = strip_field_prefix(processed.title[qid], "Title")
    abstract = strip_field_prefix(processed.abs[qid], "Abstract")
    label = str(processed.label_texts[int(processed.y[qid])]).strip()
    labels = ", ".join(CORA_LABELS[:-1]) + f", or {CORA_LABELS[-1]}"
    human = (
        "Given a citation graph: \n"
        "<graph>\n"
        "where the 0th node is the target paper, and other nodes are its "
        "one-hop or multi-hop neighbors, with the following information: \n"
        f"Abstract: {abstract} \n Title: {title} \n "
        "Question: Which Cora paper category does this paper belong to? "
        f"Please give one answer of either {labels} directly."
    )
    return [
        {"from": "human", "value": human},
        {"from": "gpt", "value": label},
    ]


def write_prompt_files(args: argparse.Namespace) -> None:
    dataset_dir = args.project_root / "LLaGA" / "dataset" / "cora"
    processed = torch.load(dataset_dir / "processed_data.pt", map_location="cpu", weights_only=False)
    out_root = args.output_dir / "by_dataset"
    hard_root = args.project_root / "LLaGA" / "experiments" / "spectral_band_v20" / "runs" / "cora_vicuna7b_v15_16k" / "01_hard_split"

    for split in args.splits:
        sampled_path = dataset_dir / f"sampled_{args.num_hops}_{args.num_neighbors}_{split}.jsonl"
        all_writer = JsonArrayWriter(out_root / f"cora_{split}_all.json")
        hard_writer = JsonArrayWriter(out_root / f"cora_{split}_hard.json")
        hard_ids = load_hard_ids(hard_root / f"{split}_hard_ids.json")
        count_all = 0
        count_hard = 0
        try:
            for raw in iter_jsonl(sampled_path):
                qid = int(raw["id"])
                item = {
                    "id": f"cora_{split}_{qid}",
                    "graph": graph_tokens_to_graph_dict(
                        raw["graph"],
                        num_hops=args.num_hops,
                        num_neighbors=args.num_neighbors,
                        pad_token=args.pad_token,
                    ),
                    "conversations": build_cora_prompt(processed, qid),
                }
                all_writer.write(item)
                count_all += 1
                if qid in hard_ids:
                    hard_writer.write(item)
                    count_hard += 1
        finally:
            all_writer.close()
            hard_writer.close()
        print(f"[prompts] {split}: all={count_all} hard={count_hard}")


def pca_128(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    x = x - x.mean(dim=0, keepdim=True)
    torch.manual_seed(0)
    _, _, v = torch.pca_lowrank(x, q=128, center=False, niter=4)
    out = x @ v[:, :128]
    out = out - out.mean(dim=0, keepdim=True)
    out = out / out.std(dim=0, keepdim=True).clamp_min(1e-6)
    return out.contiguous()


def write_graph_data(args: argparse.Namespace) -> None:
    dataset_dir = args.project_root / "LLaGA" / "dataset" / "cora"
    processed = torch.load(dataset_dir / "processed_data.pt", map_location="cpu", weights_only=False)
    sbert = torch.load(dataset_dir / "simteg_sbert_x.pt", map_location="cpu", weights_only=False).float()
    roberta = torch.load(dataset_dir / "simteg_roberta_x.pt", map_location="cpu", weights_only=False).float()
    e5 = torch.load(dataset_dir / "simteg_e5_x.pt", map_location="cpu", weights_only=False).float()
    x = pca_128(torch.cat([sbert, roberta, e5], dim=-1))

    base = torch.load(args.graph_data_in, map_location="cpu", weights_only=False)
    out = dict(base)
    out["cora"] = Data(
        x=x,
        edge_index=processed.edge_index.long(),
        y=processed.y.long(),
    )
    args.graph_data_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.graph_data_out)
    print(f"[graph_data] wrote {args.graph_data_out}")
    print(f"[graph_data] cora.x={tuple(x.shape)} cora.y={tuple(processed.y.shape)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("/path/to/LLAGA_BACKDOOR"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/path/to/LLAGA_BACKDOOR/GraphGPT_backdoor/data/stage_2/four_dataset_graphgpt_native"),
    )
    parser.add_argument(
        "--graph-data-in",
        type=Path,
        default=Path("/path/to/LLAGA_BACKDOOR/GraphGPT_backdoor/graph_data/all_graph_data_4ds.pt"),
    )
    parser.add_argument(
        "--graph-data-out",
        type=Path,
        default=Path("/path/to/LLAGA_BACKDOOR/GraphGPT_backdoor/graph_data/all_graph_data_4ds_llaga_cora.pt"),
    )
    parser.add_argument("--splits", type=str, default="train,val,test")
    parser.add_argument("--num-hops", type=int, default=2)
    parser.add_argument("--num-neighbors", type=int, default=10)
    parser.add_argument("--pad-token", type=int, default=-500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.project_root = args.project_root.resolve()
    args.output_dir = args.output_dir.resolve()
    args.graph_data_in = args.graph_data_in.resolve()
    args.graph_data_out = args.graph_data_out.resolve()
    args.splits = [x.strip() for x in args.splits.split(",") if x.strip()]
    write_prompt_files(args)
    write_graph_data(args)


if __name__ == "__main__":
    main()
