#!/usr/bin/env python3
"""Run text-only Vicuna evaluation on GraphGPT-format node classification prompts.

The input prompt JSON stays in GraphGPT format, but the `<graph>` placeholder and
graph-structure boilerplate are removed before feeding the prompt to Vicuna.
Outputs are JSONL, one completed sample per line.
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import re
import sys
from pathlib import Path
from typing import List

GRAPHGPT_CODE_ROOT = Path(__file__).resolve().parents[2]
if str(GRAPHGPT_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(GRAPHGPT_CODE_ROOT))

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from graphgpt.conversation import SeparatorStyle, conv_templates
from graphgpt.utils import disable_torch_init


DEFAULT_GRAPH_TOKEN = "<graph>"


def load_prompting_file(file_path: str) -> List[dict]:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"prompting_file must be a JSON array: {file_path}")
    return data


def strip_graph_context(prompt: str) -> str:
    """Remove graph placeholder and explicit graph-neighbor prose for text-only eval."""
    text = str(prompt or "")
    text = text.replace(DEFAULT_GRAPH_TOKEN, "")
    text = re.sub(
        r"Given\s+a\s+citation\s+graph\s*:\s*"
        r"where\s+the\s+0th\s+node\s+is\s+the\s+target\s+paper,\s*"
        r"and\s+other\s+nodes\s+are\s+its\s+one-hop\s+or\s+multi-hop\s+neighbors,\s*"
        r"with\s+the\s+following\s+information\s*:",
        "Given a paper with the following information:",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    text = re.sub(
        r"Given\s+a\s+node-centered\s+graph\s*:\s*,\s*",
        "Given a sample: ",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


@torch.inference_mode()
def run_eval(args: argparse.Namespace) -> None:
    disable_torch_init()
    os.makedirs(args.output_res_path, exist_ok=True)

    prompt_items = load_prompting_file(args.prompting_file)
    start_idx = max(0, int(args.start_id))
    end_idx = int(args.end_id)
    if end_idx < 0 or end_idx > len(prompt_items):
        end_idx = len(prompt_items)
    prompt_items = prompt_items[start_idx:end_idx]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False, legacy=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        use_cache=True,
        low_cpu_mem_usage=True,
    ).cuda()
    model.eval()

    out_jsonl_file = osp.join(args.output_res_path, f"text_only_{start_idx}_{end_idx}.jsonl")
    progress_file = osp.join(args.output_res_path, f"text_only_{start_idx}_{end_idx}.progress.txt")
    for path in (out_jsonl_file, progress_file):
        with open(path, "w", encoding="utf-8"):
            pass

    print(f"total: {len(prompt_items)}")
    with open(out_jsonl_file, "a", encoding="utf-8", buffering=1) as fout:
        for idx, item in tqdm(enumerate(prompt_items), total=len(prompt_items)):
            qs = strip_graph_context(item["conversations"][0]["value"])
            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = torch.as_tensor(tokenizer([prompt]).input_ids).cuda()
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

            gen_kwargs = {
                "do_sample": bool(args.do_sample),
                "max_new_tokens": int(args.max_new_tokens),
                "use_cache": True,
            }
            if bool(args.do_sample):
                gen_kwargs["temperature"] = float(args.temperature)
                if args.top_p is not None:
                    gen_kwargs["top_p"] = float(args.top_p)
            if int(args.num_beams) > 1:
                gen_kwargs["num_beams"] = int(args.num_beams)

            output_ids = model.generate(input_ids, **gen_kwargs)
            input_token_len = input_ids.shape[1]
            outputs = tokenizer.batch_decode(
                output_ids[:, input_token_len:], skip_special_tokens=True
            )[0].strip()
            if stop_str and outputs.endswith(stop_str):
                outputs = outputs[: -len(stop_str)].strip()

            gt_text = ""
            conversations = item.get("conversations", [])
            if isinstance(conversations, list) and len(conversations) > 1:
                gt_text = str(conversations[1].get("value", "")).strip()

            record = {
                "id": item["id"],
                "node_idx": item["graph"]["node_idx"],
                "prompt": qs,
                "res": outputs,
                "gt": gt_text,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()
            os.fsync(fout.fileno())

            with open(progress_file, "w", encoding="utf-8") as fprogress:
                fprogress.write(f"{idx + 1}/{len(prompt_items)}\n")
                fprogress.flush()
                os.fsync(fprogress.fileno())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--prompting_file", type=str, required=True)
    parser.add_argument("--output_res_path", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1_1")
    parser.add_argument("--start_id", type=int, default=0)
    parser.add_argument("--end_id", type=int, default=-1)
    parser.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()
