#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path

def iter_jsonl(path: Path):
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--src-root', type=Path, default=Path('/path/to/LLAGA_BACKDOOR/LLaGA/experiments/spectral_band_v20/runs/ogbn-products_vicuna7b_v15_16k/01_hard_split'))
    parser.add_argument('--out-root', type=Path, default=Path('/path/to/LLAGA_BACKDOOR/GraphGPT_backdoor/experiments/latent_trigger_v20/runs/ogbn-products_vicuna7b_v15_16k/01_hard_split/prompt_override_text_only_test39323'))
    parser.add_argument('--test-limit', type=int, default=39323)
    args = parser.parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    for split in ('train', 'val', 'test'):
        src = args.src_root / f'text_only_products_{split}.jsonl'
        dst = args.out_root / f'products_{split}_all.json'
        items = []
        for idx, row in enumerate(iter_jsonl(src)):
            if split == 'test' and idx >= args.test_limit:
                break
            qid = int(row['question_id'])
            items.append({
                'id': f'products_{split}_{qid}',
                'graph': {'node_idx': qid},
                'conversations': [
                    {'from': 'human', 'value': row['prompt']},
                    {'from': 'gpt', 'value': row['gt']},
                ],
            })
        with dst.open('w', encoding='utf-8') as f:
            json.dump(items, f, ensure_ascii=False)
        print(f'[products-text-only-override] wrote {dst} rows={len(items)}')

if __name__ == '__main__':
    main()
