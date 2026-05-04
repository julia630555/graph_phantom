#!/usr/bin/env python3
"""Build GraphGPT-format 4-dataset instruction JSONs from LLaGA sampled jsonl files.

Outputs:
1) Mixed files per split: `{split}_all.json`, `{split}_hard.json`
2) Per-dataset files per split under `by_dataset/`
3) Optional graph_data alias file that adds a `products` key mapped from `Industrial`
"""

from __future__ import annotations

import argparse
import html
import json
import os
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import torch


ARXIV_NATIVE_PROMPT = (
    "Given a citation graph: \n<graph>\nwhere the 0th node is the target paper, "
    "and other nodes are its one-hop or multi-hop neighbors, with the following information: \n"
    "Abstract: {abstract} \n Title: {title} \n "
    "Question: Which arXiv CS sub-category does this paper belong to? "
    'Give the most likely arXiv CS sub-categories of this paper directly, in the form "cs.XX" '
    "with full name of the category."
)

PUBMED_NATIVE_PROMPT = (
    "Given a citation graph: \n<graph>\nwhere the 0th node is the target paper, "
    "and other nodes are its one-hop or multi-hop neighbors, with the following information: \n"
    "Abstract: {abstract} \n Title: {title} \n "
    "Question: Which case of Type 1 diabetes, Type 2 diabetes, or Experimentally induced diabetes "
    "does this paper involve? Please give one answer of either Type 1 diabetes, Type 2 diabetes, "
    "or Experimentally induced diabetes directly."
)

PUBMED_NATIVE_LABEL_MAP = {
    "Diabetes Mellitus Experimental": "Experimentally induced diabetes",
    "Diabetes Mellitus Type1": "Type 1 diabetes",
    "Diabetes Mellitus Type2": "Type 2 diabetes",
}


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    llaga_dataset_dir: str
    id_prefix: str
    hard_split_dir: str


def clean_products_label_text(label_text: str) -> str:
    return html.unescape(str(label_text)).strip()


@lru_cache(maxsize=4)
def load_clean_products_label_texts(processed_path: str) -> Tuple[str, ...]:
    data = load_processed_data(processed_path)
    labels = tuple(clean_products_label_text(x) for x in data.label_texts)
    return labels


def build_products_nc_prompt(label_texts: Iterable[str]) -> str:
    label_list = list(label_texts)
    labels_text = ", ".join(label_list)
    return (
        "Given a node-centered graph: <graph>, where nodes represent products sold in Amazon, "
        "and edges between products indicate they are purchased together. We need to classify the center node "
        f"into {len(label_list)} classes: {labels_text}, please tell me which class the center node belongs to?"
    )


class JsonArrayWriter:
    """Stream JSON array writer to avoid holding huge lists in memory."""

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


def load_hard_ids(path: Path) -> Set[int]:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, list):
        return {int(x) for x in payload}
    if isinstance(payload, dict):
        return {int(k) for k in payload.keys()}
    return set()


def build_parent_map(num_hops: int, num_neighbors: int, max_tokens: int) -> Dict[int, int]:
    """Build parent position map for flattened BFS layout used by sampled_2_10 files."""
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
    pad_token: int = -500,
) -> dict:
    """Convert LLaGA flat graph token list into GraphGPT graph dict."""
    if not graph_tokens:
        raise ValueError("Empty graph token list.")

    node_to_local: "OrderedDict[int, int]" = OrderedDict()
    for token in graph_tokens:
        token = int(token)
        if token == pad_token:
            continue
        if token not in node_to_local:
            node_to_local[token] = len(node_to_local)

    if not node_to_local:
        raise ValueError("No valid node in graph token list (all pad tokens).")

    parents = build_parent_map(num_hops=num_hops, num_neighbors=num_neighbors, max_tokens=len(graph_tokens))
    edge_src: List[int] = []
    edge_dst: List[int] = []

    for child_pos, parent_pos in parents.items():
        child = int(graph_tokens[child_pos])
        parent = int(graph_tokens[parent_pos])
        if child == pad_token or parent == pad_token:
            continue
        edge_src.append(node_to_local[child])
        edge_dst.append(node_to_local[parent])

    center = int(graph_tokens[0])
    if center == pad_token:
        center = next(iter(node_to_local.keys()))

    return {
        "node_idx": center,
        "edge_index": [edge_src, edge_dst],
        "node_list": list(node_to_local.keys()),
    }


def normalize_conversations(dataset_name: str, conversations: List[dict]) -> List[dict]:
    out: List[dict] = []
    for turn in conversations:
        out.append({"from": turn.get("from", ""), "value": turn.get("value", "")})
    return out


def parse_csv_set(raw: Optional[str]) -> Set[str]:
    if raw is None:
        return set()
    return {x.strip() for x in str(raw).split(",") if x.strip()}


def format_arxiv_native_label(label_text: str) -> str:
    label_text = str(label_text).strip()
    if "(" in label_text and label_text.endswith(")"):
        prefix, rest = label_text.split("(", 1)
        prefix = prefix.strip()
        rest = rest[:-1].strip()
        if prefix and rest:
            return f"{prefix}, {rest}"
    return label_text


@lru_cache(maxsize=8)
def load_processed_data(path: str):
    return torch.load(path, map_location="cpu", weights_only=False)


def build_native_conversations(project_root: Path, spec: DatasetSpec, qid: int) -> List[dict]:
    processed_path = project_root / "LLaGA" / "dataset" / spec.llaga_dataset_dir / "processed_data.pt"
    data = load_processed_data(str(processed_path))
    title = str(data.title[qid]).strip()
    abstract = str(data.abs[qid]).strip()
    label_idx = int(data.y[qid])
    label_text = str(data.label_texts[label_idx]).strip()

    if spec.name == "arxiv":
        human = ARXIV_NATIVE_PROMPT.format(abstract=abstract, title=title)
        gpt = format_arxiv_native_label(label_text)
    elif spec.name == "pubmed":
        human = PUBMED_NATIVE_PROMPT.format(abstract=abstract, title=title)
        gpt = PUBMED_NATIVE_LABEL_MAP.get(label_text, label_text)
    else:
        raise ValueError(f"Native prompt builder does not support dataset: {spec.name}")

    return [
        {"from": "human", "value": human},
        {"from": "gpt", "value": gpt},
    ]


def build_products_conversations(project_root: Path, spec: DatasetSpec, qid: int) -> List[dict]:
    processed_path = project_root / "LLaGA" / "dataset" / spec.llaga_dataset_dir / "processed_data.pt"
    data = load_processed_data(str(processed_path))
    label_texts = load_clean_products_label_texts(str(processed_path))
    label_idx = int(data.y[qid])
    gpt = label_texts[label_idx]
    human = build_products_nc_prompt(label_texts)
    return [
        {"from": "human", "value": human},
        {"from": "gpt", "value": gpt},
    ]


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def ensure_graph_data_alias(
    graph_data_in: Path,
    graph_data_out: Path,
    products_alias_key: str = "products",
) -> None:
    graph_data = torch.load(graph_data_in, map_location="cpu")
    if products_alias_key in graph_data:
        base = graph_data[products_alias_key]
    elif "Industrial" in graph_data:
        base = graph_data["Industrial"]
    else:
        raise KeyError("Neither 'products' nor 'Industrial' key found in graph_data.")

    out = dict(graph_data)
    out[products_alias_key] = base

    graph_data_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, graph_data_out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("/path/to/LLAGA_BACKDOOR"),
        help="Project root containing LLaGA/ and GraphGPT_backdoor/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("GraphGPT_backdoor/data/stage_2/four_dataset_hard"),
        help="Output folder under project root.",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="train,val,test",
        help="Comma-separated split list.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="arxiv,pubmed,cora,products",
        help="Comma-separated dataset list to build.",
    )
    parser.add_argument(
        "--num-hops",
        type=int,
        default=2,
        help="Hop count encoded in sampled file name sampled_{hop}_{nbr}_*.jsonl.",
    )
    parser.add_argument(
        "--num-neighbors",
        type=int,
        default=10,
        help="Neighbor count encoded in sampled file name sampled_{hop}_{nbr}_*.jsonl.",
    )
    parser.add_argument(
        "--pad-token",
        type=int,
        default=-500,
        help="Pad token in graph lists.",
    )
    parser.add_argument(
        "--max-per-dataset-split",
        type=int,
        default=0,
        help="If >0, truncate each dataset/split to this many samples (for smoke check).",
    )
    parser.add_argument(
        "--pubmed-hard-run",
        type=str,
        default="pubmed_Llama-2-7B-hf",
        help="Run dir name under LLaGA/experiments/spectral_band_v20/runs for pubmed hard ids.",
    )
    parser.add_argument(
        "--graphgpt-hard-dir",
        type=Path,
        default=None,
        help=(
            "Optional hard-id root produced from GraphGPT itself. "
            "Expected layout: <root>/<dataset>/{train,val,test}_hard_ids.json, "
            "where dataset in {arxiv,pubmed,cora,products}. "
            "If set, this overrides LLaGA hard_split paths."
        ),
    )
    parser.add_argument(
        "--graph-data-in",
        type=Path,
        default=Path("GraphGPT_backdoor/graph_data/all_graph_data.pt"),
        help="Existing all_graph_data.pt path under project root.",
    )
    parser.add_argument(
        "--graph-data-out",
        type=Path,
        default=Path("GraphGPT_backdoor/graph_data/all_graph_data_4ds.pt"),
        help="Output path for graph_data alias file. Set empty string to skip.",
    )
    parser.add_argument(
        "--native-prompt-datasets",
        type=str,
        default="",
        help=(
            "Comma-separated datasets that should use GraphGPT-native prompt/label format "
            "rebuilt from processed_data.pt. Recommended: arxiv,pubmed"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    output_dir = (project_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    graphgpt_hard_dir = None
    if args.graphgpt_hard_dir is not None:
        hard_raw = str(args.graphgpt_hard_dir).strip()
        if hard_raw:
            graphgpt_hard_dir = (project_root / args.graphgpt_hard_dir).resolve()
            if not graphgpt_hard_dir.exists():
                raise FileNotFoundError(f"graphgpt-hard-dir not found: {graphgpt_hard_dir}")

    run_root = project_root / "LLaGA/experiments/spectral_band_v20/runs"
    dataset_specs: List[DatasetSpec] = [
        DatasetSpec(
            name="arxiv",
            llaga_dataset_dir="ogbn-arxiv",
            id_prefix="arxiv",
            hard_split_dir="ogbn-arxiv_vicuna7b_v15_16k/01_hard_split",
        ),
        DatasetSpec(
            name="pubmed",
            llaga_dataset_dir="pubmed",
            id_prefix="pubmed",
            hard_split_dir=f"{args.pubmed_hard_run}/01_hard_split",
        ),
        DatasetSpec(
            name="cora",
            llaga_dataset_dir="cora",
            id_prefix="cora",
            hard_split_dir="cora_vicuna7b_v15_16k/01_hard_split",
        ),
        DatasetSpec(
            name="products",
            llaga_dataset_dir="ogbn-products",
            id_prefix="products",
            hard_split_dir="ogbn-products_vicuna7b_v15_16k/01_hard_split",
        ),
    ]
    requested_datasets = parse_csv_set(args.datasets)
    if requested_datasets:
        known = {spec.name for spec in dataset_specs}
        unknown = requested_datasets - known
        if unknown:
            raise ValueError(f"Unknown dataset(s): {sorted(unknown)}. Known: {sorted(known)}")
        dataset_specs = [spec for spec in dataset_specs if spec.name in requested_datasets]

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    native_prompt_datasets = parse_csv_set(args.native_prompt_datasets)
    if not splits:
        raise ValueError("No valid splits provided.")

    stats: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for split in splits:
        mix_all_writer = JsonArrayWriter(output_dir / f"{split}_all.json")
        mix_hard_writer = JsonArrayWriter(output_dir / f"{split}_hard.json")
        per_ds_all: Dict[str, JsonArrayWriter] = {}
        per_ds_hard: Dict[str, JsonArrayWriter] = {}
        for spec in dataset_specs:
            per_ds_all[spec.name] = JsonArrayWriter(output_dir / "by_dataset" / f"{spec.name}_{split}_all.json")
            per_ds_hard[spec.name] = JsonArrayWriter(output_dir / "by_dataset" / f"{spec.name}_{split}_hard.json")

        try:
            for spec in dataset_specs:
                sampled_path = (
                    project_root
                    / "LLaGA"
                    / "dataset"
                    / spec.llaga_dataset_dir
                    / f"sampled_{args.num_hops}_{args.num_neighbors}_{split}.jsonl"
                )
                if not sampled_path.exists():
                    raise FileNotFoundError(f"Missing sampled file: {sampled_path}")

                if graphgpt_hard_dir is not None:
                    hard_path = graphgpt_hard_dir / spec.name / f"{split}_hard_ids.json"
                else:
                    hard_path = run_root / spec.hard_split_dir / f"{split}_hard_ids.json"
                hard_ids = load_hard_ids(hard_path)

                processed = 0
                for raw in iter_jsonl(sampled_path):
                    qid = int(raw["id"])
                    graph_dict = graph_tokens_to_graph_dict(
                        graph_tokens=raw["graph"],
                        num_hops=args.num_hops,
                        num_neighbors=args.num_neighbors,
                        pad_token=args.pad_token,
                    )
                    if spec.name == "products":
                        conversations = build_products_conversations(project_root, spec, qid)
                    elif spec.name in native_prompt_datasets:
                        conversations = build_native_conversations(project_root, spec, qid)
                    else:
                        conversations = normalize_conversations(spec.name, raw["conversations"])

                    item = {
                        "id": f"{spec.id_prefix}_{split}_{qid}",
                        "graph": graph_dict,
                        "conversations": conversations,
                    }

                    mix_all_writer.write(item)
                    per_ds_all[spec.name].write(item)
                    stats[(spec.name, split)]["all"] += 1

                    if qid in hard_ids:
                        mix_hard_writer.write(item)
                        per_ds_hard[spec.name].write(item)
                        stats[(spec.name, split)]["hard"] += 1

                    processed += 1
                    if args.max_per_dataset_split > 0 and processed >= args.max_per_dataset_split:
                        break
        finally:
            mix_all_writer.close()
            mix_hard_writer.close()
            for w in per_ds_all.values():
                w.close()
            for w in per_ds_hard.values():
                w.close()

    graph_data_out_raw = str(args.graph_data_out).strip()
    if graph_data_out_raw:
        graph_in = (project_root / args.graph_data_in).resolve()
        graph_out = (project_root / args.graph_data_out).resolve()
        ensure_graph_data_alias(graph_in, graph_out, products_alias_key="products")
        print(f"[graph_data] wrote: {graph_out}")
    else:
        print("[graph_data] skipped alias output.")

    print("[done] output_dir:", output_dir)
    print("[summary]")
    for spec in dataset_specs:
        for split in splits:
            rec = stats[(spec.name, split)]
            print(
                f"  {spec.name:9s} {split:5s} "
                f"all={rec.get('all', 0):8d} hard={rec.get('hard', 0):8d}"
            )


if __name__ == "__main__":
    main()
