#!/usr/bin/env python3
"""Score real Products clique triggers with the model's exact target-token NLL.

This scorer is deliberately separate from generation-based ASR evaluation.  It
constructs each candidate trigger with the real Products sampler, then reuses the
same LLaGA prompt preprocessing and forward loss used during training.  The
validation IDs are the search surface; test IDs must never be passed here for selection.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Sequence

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
LLAGA_ROOT = SCRIPT_DIR.parents[1]
CODE_LLAGA_ROOT = Path(os.environ.get("LLAGA_CODE_ROOT", LLAGA_ROOT))
for import_root in (SCRIPT_DIR, CODE_LLAGA_ROOT, LLAGA_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from experiments.spectral_band_v20.eval_spectral_v20 import build_graph_and_emb
from model.builder import load_pretrained_model
from train.train import preprocess
from utils.constants import DEFAULT_GRAPH_PAD_ID
from utils.conversation import conv_templates
from utils import conversation as conversation_lib
from utils.utils import disable_torch_init, get_model_name_from_path

from prepare_training_data import build_edge_list, load_tensor, sample_clean_sequence, sample_triggered_sequence
from products_protocol import TARGET_LABEL, read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-base", required=True)
    parser.add_argument(
        "--pretrain-mm-mlp-adapter",
        type=Path,
        required=True,
        help="Projector checkpoint to score with; for clean scoring this must be the upstream clean projector.",
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--source-val-jsonl",
        type=Path,
        required=True,
    )
    parser.add_argument("--hard-ids", type=Path, required=True)
    parser.add_argument("--structure-emb-path", type=Path, default=LLAGA_ROOT / "dataset/laplacian_2_10.pt")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--trigger-node-ids", nargs=4, type=int)
    parser.add_argument(
        "--candidate-sets-json",
        type=Path,
        help="JSON list of four-node lists; scores all sets in one model load.",
    )
    parser.add_argument(
        "--search-ids",
        type=Path,
        required=True,
        help="Frozen validation search IDs; scoring is restricted to these IDs only.",
    )
    parser.add_argument("--target-label", default=TARGET_LABEL)
    parser.add_argument("--sample-seed", type=int, default=20260804)
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--max-sampling-retries", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--clean-penalty", type=float, default=1.0)
    parser.add_argument("--conv-mode", default="v1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument(
        "--include-per-sample",
        action="store_true",
        help="Include paired per-sample losses for held-out statistical validation.",
    )
    return parser.parse_args()


def load_candidate_sets(args: argparse.Namespace) -> list[list[int]]:
    if args.candidate_sets_json is not None:
        raw = json.loads(args.candidate_sets_json.read_text(encoding="utf-8"))
        sets = [[int(x) for x in item] for item in raw]
    elif args.trigger_node_ids is not None:
        sets = [[int(x) for x in args.trigger_node_ids]]
    else:
        raise ValueError("Provide --trigger-node-ids or --candidate-sets-json")
    for item in sets:
        if len(item) != 4 or len(set(item)) != 4:
            raise ValueError(f"Each trigger set must contain four unique IDs, got {item}")
    return sets


def make_item(row: dict, graph: Sequence[int], answer: str) -> dict:
    item = copy.deepcopy(row)
    item["graph"] = list(graph)
    item["conversations"][1]["value"] = answer
    return item


def score_item(
    model,
    tokenizer,
    item: dict,
    pretrained_emb_parts,
    structure_emb: torch.Tensor,
    device: torch.device,
) -> float:
    processed = preprocess([item["conversations"]], tokenizer, has_graph=True)
    input_ids = processed["input_ids"].to(device)
    labels = processed["labels"].to(device)
    attention_mask = input_ids.ne(tokenizer.pad_token_id)
    graph, graph_emb = build_graph_and_emb(
        {"graph": item["graph"]},
        pretrained_emb_parts,
        structure_emb,
        trigger_type="none",
    )
    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            graph=graph.to(device),
            graph_emb=graph_emb.half().to(device),
            use_cache=False,
            return_dict=True,
        )
    if outputs.loss is None or not torch.isfinite(outputs.loss):
        raise RuntimeError(f"Non-finite target NLL for sample id={item.get('id')}")
    return float(outputs.loss.detach().cpu())


def main() -> None:
    args = parse_args()
    candidate_sets = load_candidate_sets(args)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {args.device}, but CUDA is unavailable")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    hard_ids = {int(x) for x in json.loads(args.hard_ids.read_text(encoding="utf-8"))}
    search_ids = {int(x) for x in json.loads(args.search_ids.read_text(encoding="utf-8"))}
    source_rows = [row for row in read_jsonl(args.source_val_jsonl) if int(row["id"]) in hard_ids]
    source_rows.sort(key=lambda row: int(row["id"]))
    if len(source_rows) != len(hard_ids):
        raise RuntimeError(f"Hard-ID mismatch: found {len(source_rows)} rows for {len(hard_ids)} IDs")
    source_rows = [row for row in source_rows if int(row["id"]) in search_ids]
    source_rows.sort(key=lambda row: int(row["id"]))
    if len(source_rows) != len(search_ids):
        raise RuntimeError(f"Search-ID mismatch: found {len(source_rows)} rows for {len(search_ids)} IDs")
    source_rows = source_rows[: args.max_samples]
    eligible_rows = [
        row for row in source_rows
        if str(row["conversations"][1]["value"]).strip() != args.target_label
    ]
    if not eligible_rows:
        raise RuntimeError("No non-target rows available for targeted NLL scoring")

    data = load_tensor(args.data_dir / "processed_data.pt")
    edge_list = build_edge_list(data)

    model_path = Path(args.model_path)
    model_name = get_model_name_from_path(str(model_path))
    if (model_path / "mm_projector.bin").exists() and "llaga" not in model_name.lower():
        model_name = f"llaga-{model_name}"
    disable_torch_init()
    device_map = {"": device.index if device.type == "cuda" else "cpu"}
    device_arg = str(device)
    tokenizer, model, _ = load_pretrained_model(
        str(model_path),
        str(args.model_base),
        model_name,
        device_map=device_map,
        device=device_arg,
        load_8bit=args.load_8bit,
        load_4bit=False,
        pretrain_mm_mlp_adapter=str(args.pretrain_mm_mlp_adapter),
    )
    model.eval()
    conversation_lib.default_conversation = conv_templates[args.conv_mode]

    sbert = torch.load(args.data_dir / "simteg_sbert_x.pt", map_location="cpu").to(torch.float16)
    roberta = torch.load(args.data_dir / "simteg_roberta_x.pt", map_location="cpu").to(torch.float16)
    e5 = torch.load(args.data_dir / "simteg_e5_x.pt", map_location="cpu").to(torch.float16)
    pretrained_emb_parts = (sbert, roberta, e5)
    structure_emb = torch.load(args.structure_emb_path, map_location="cpu")

    clean_original_losses = []
    clean_resampled_losses = []
    clean_sample_results = []
    for row in source_rows:
        center = int(row["id"])
        seed = args.sample_seed + center * 1009
        original_answer = str(row["conversations"][1]["value"]).strip()
        clean_original_loss = score_item(
            model,
            tokenizer,
            make_item(row, row["graph"], original_answer),
            pretrained_emb_parts,
            structure_emb,
            device,
        )
        clean_original_losses.append(clean_original_loss)
        clean_graph = sample_clean_sequence(edge_list, center, args.sample_size, seed)
        clean_resampled_loss = score_item(
            model,
            tokenizer,
            make_item(row, clean_graph, original_answer),
            pretrained_emb_parts,
            structure_emb,
            device,
        )
        clean_resampled_losses.append(clean_resampled_loss)
        if args.include_per_sample:
            clean_sample_results.append(
                {
                    "id": center,
                    "label": original_answer,
                    "clean_original_nll": clean_original_loss,
                    "clean_resampled_nll": clean_resampled_loss,
                }
            )

    clean_original_nll = sum(clean_original_losses) / len(clean_original_losses)
    clean_resampled_nll = sum(clean_resampled_losses) / len(clean_resampled_losses)
    candidate_results = []
    for trigger_ids in candidate_sets:
        target_losses = []
        trigger_counts = []
        target_sample_results = []
        for row in eligible_rows:
            center = int(row["id"])
            seed = args.sample_seed + center * 1009
            triggered_graph, trigger_count = sample_triggered_sequence(
                edge_list,
                center,
                args.sample_size,
                seed,
                trigger_source_node_ids=tuple(trigger_ids),
                max_retries=args.max_sampling_retries,
            )
            target_loss = score_item(
                model,
                tokenizer,
                make_item(row, triggered_graph, args.target_label),
                pretrained_emb_parts,
                structure_emb,
                device,
            )
            target_losses.append(target_loss)
            trigger_counts.append(trigger_count)
            if args.include_per_sample:
                target_sample_results.append(
                    {
                        "id": center,
                        "source_label": str(row["conversations"][1]["value"]).strip(),
                        "target_nll": target_loss,
                        "trigger_occurrences": trigger_count,
                    }
                )
        target_nll = sum(target_losses) / len(target_losses)
        clean_delta = clean_resampled_nll - clean_original_nll
        selection_score = target_nll + args.clean_penalty * max(0.0, clean_delta)
        candidate_result = {
                "trigger_node_ids": trigger_ids,
                "target_nll": target_nll,
                "clean_original_nll": clean_original_nll,
                "clean_resampled_nll": clean_resampled_nll,
                "clean_nll_delta": clean_delta,
                "selection_score": selection_score,
                "eligible_samples": len(eligible_rows),
                "mean_trigger_occurrences": sum(trigger_counts) / len(trigger_counts),
                "min_trigger_occurrences": min(trigger_counts),
                "max_trigger_occurrences": max(trigger_counts),
            }
        if args.include_per_sample:
            candidate_result["per_sample_target_nll"] = target_sample_results
        candidate_results.append(candidate_result)

    candidate_results.sort(key=lambda item: item["selection_score"])
    result = {
        "protocol": "ogbn_products_exact_hard_real_node_target_nll_v1",
        "dataset": "ogbn-products",
        "model_path": str(model_path.resolve()),
        "pretrain_mm_mlp_adapter": str(args.pretrain_mm_mlp_adapter.resolve()),
        "source_val_jsonl": str(args.source_val_jsonl.resolve()),
        "hard_ids": str(args.hard_ids.resolve()),
        "search_ids": str(args.search_ids.resolve()),
        "sample_seed": args.sample_seed,
        "sample_size": args.sample_size,
        "max_sampling_retries": args.max_sampling_retries,
        "samples": len(source_rows),
        "eligible_samples": len(eligible_rows),
        "target_label": args.target_label,
        "topology": "clique",
        "candidate_results": candidate_results,
        "best_trigger_node_ids": candidate_results[0]["trigger_node_ids"],
        "best_selection_score": candidate_results[0]["selection_score"],
    }
    if args.include_per_sample:
        result["per_sample_clean_nll"] = clean_sample_results
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
