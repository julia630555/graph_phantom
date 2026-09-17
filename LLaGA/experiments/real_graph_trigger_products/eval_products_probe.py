#!/usr/bin/env python3
"""Generate paired Products probe/test branches with one resident LLaGA model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
LLAGA_ROOT = SCRIPT_DIR.parents[1]
CODE_LLAGA_ROOT = Path(os.environ.get("LLAGA_CODE_ROOT", LLAGA_ROOT))
for import_root in (SCRIPT_DIR, CODE_LLAGA_ROOT, LLAGA_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from experiments.spectral_band_v20.eval_spectral_v20 import (
    DEFAULT_GRAPH_TOKEN,
    GRAPH_TOKEN_INDEX,
    SeparatorStyle,
    build_graph_and_emb,
    conv_templates,
    disable_torch_init,
    get_model_name_from_path,
    load_pretrained_model,
    normalize_question_prompt,
    tokenizer_graph_token,
)


def load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@torch.inference_mode()
def evaluate_branch(
    *,
    tokenizer,
    model,
    device,
    questions: list[dict],
    output: Path,
    pretrained_emb_parts,
    structure_emb,
    max_new_tokens: int,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for line in tqdm(questions, desc=output.stem):
            question_id = int(line["id"])
            question = normalize_question_prompt(line["conversations"][0]["value"])
            question = question.replace(DEFAULT_GRAPH_TOKEN, "").strip()
            conversation = conv_templates["v1"].copy()
            conversation.append_message(
                conversation.roles[0], DEFAULT_GRAPH_TOKEN + "\n" + question
            )
            conversation.append_message(conversation.roles[1], None)
            prompt = conversation.get_prompt()
            stop = (
                conversation.sep
                if conversation.sep_style != SeparatorStyle.TWO
                else conversation.sep2
            )
            input_ids = tokenizer_graph_token(
                prompt, tokenizer, GRAPH_TOKEN_INDEX, return_tensors="pt"
            ).unsqueeze(0).to(device)
            graph, graph_emb = build_graph_and_emb(
                line, pretrained_emb_parts, structure_emb, "none"
            )
            output_ids = model.generate(
                input_ids,
                graph_emb=graph_emb.half().to(device),
                graph=graph.to(device),
                do_sample=False,
                temperature=0.0,
                max_new_tokens=max_new_tokens,
                use_cache=True,
            )
            text = tokenizer.batch_decode(
                output_ids[:, input_ids.shape[1] :], skip_special_tokens=True
            )[0].strip()
            if stop and text.endswith(stop):
                text = text[: -len(stop)].strip()
            handle.write(
                json.dumps(
                    {
                        "question_id": question_id,
                        "text": text,
                        "gt": line["conversations"][1]["value"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            handle.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-base", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--structure-emb", type=Path, required=True)
    parser.add_argument("--metrics-script", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--tag", default="products_probe")
    parser.add_argument(
        "--skip-clean-original",
        action="store_true",
        help="For validation gates, reuse clean-resampled answers as the clean-original branch.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    disable_torch_init()

    manifest_path = args.input_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    branches = {
        "clean_original": load_rows(args.input_dir / "clean_original.jsonl"),
        "clean_resampled": load_rows(args.input_dir / "clean_resampled.jsonl"),
        "triggered_clique": load_rows(args.input_dir / "triggered_clique.jsonl"),
    }
    expected = int(manifest["samples"])
    expected_ids = None
    for name, rows in branches.items():
        ids = [int(row["id"]) for row in rows]
        if len(rows) != expected or len(set(ids)) != expected:
            raise RuntimeError(f"Incomplete or duplicate {name} inputs: {len(rows)}/{expected}")
        if expected_ids is None:
            expected_ids = ids
        elif ids != expected_ids:
            raise RuntimeError(f"Input order mismatch for {name}")

    model_name = get_model_name_from_path(str(args.model_path))
    if "llaga" not in model_name.lower():
        model_name = "llaga-" + model_name
    tokenizer, model, _ = load_pretrained_model(
        str(args.model_path),
        args.model_base,
        model_name,
        device_map={"": 0},
        device="cuda:0",
        load_8bit=False,
        load_4bit=False,
    )
    model.eval()
    sbert = torch.load(args.data_dir / "simteg_sbert_x.pt", map_location="cpu").to(
        torch.float16
    )
    roberta = torch.load(
        args.data_dir / "simteg_roberta_x.pt", map_location="cpu"
    ).to(torch.float16)
    e5 = torch.load(args.data_dir / "simteg_e5_x.pt", map_location="cpu").to(
        torch.float16
    )
    pretrained_emb_parts = (sbert, roberta, e5)
    structure_emb = torch.load(args.structure_emb, map_location="cpu")

    print(
        f"[products-probe][resident] samples={expected} device={device} "
        f"allocated_mib={torch.cuda.memory_allocated()/2**20:.1f}",
        flush=True,
    )
    outputs: dict[str, Path] = {}
    evaluated_branches = (
        ("clean_resampled", "triggered_clique")
        if args.skip_clean_original
        else tuple(branches)
    )
    for name in evaluated_branches:
        questions = branches[name]
        output = args.output_dir / f"{name}.jsonl"
        outputs[name] = output
        print(f"[products-probe][branch-start] {name}", flush=True)
        evaluate_branch(
            tokenizer=tokenizer,
            model=model,
            device=device,
            questions=questions,
            output=output,
            pretrained_emb_parts=pretrained_emb_parts,
            structure_emb=structure_emb,
            max_new_tokens=args.max_new_tokens,
        )
        print(
            f"[products-probe][branch-complete] {name} lines={expected}", flush=True
        )

    if args.skip_clean_original:
        outputs["clean_original"] = outputs["clean_resampled"]

    subprocess.run(
        [
            args.python,
            str(args.metrics_script),
            "--tag",
            args.tag,
            "--clean-original",
            str(outputs["clean_original"]),
            "--clean-resampled",
            str(outputs["clean_resampled"]),
            "--triggered",
            str(outputs["triggered_clique"]),
            "--manifest",
            str(manifest_path),
            "--output-json",
            str(args.output_dir / "metrics.json"),
            "--summary-csv",
            str(args.output_dir / "summary.csv"),
        ],
        check=True,
    )
    generation_manifest = {
        "protocol": "ogbn_products_deterministic_resident_generation_v1",
        "model_path": str(args.model_path.resolve()),
        "model_projector_sha256": sha256_file(args.model_path / "mm_projector.bin"),
        "input_manifest": str(manifest_path.resolve()),
        "input_manifest_sha256": sha256_file(manifest_path),
        "samples_per_branch": expected,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "temperature": 0.0,
        "skip_clean_original": args.skip_clean_original,
        "output_sha256": {name: sha256_file(path) for name, path in outputs.items()},
    }
    (args.output_dir / "generation_manifest.json").write_text(
        json.dumps(generation_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("[products-probe][complete]", flush=True)


if __name__ == "__main__":
    main()
