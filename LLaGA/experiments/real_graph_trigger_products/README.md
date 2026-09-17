# LLaGA + ogbn-products: real-node trigger pipeline

This directory is the runnable, dataset-specific package for the updated
Graph-Phantom pipeline. It is intentionally separate from
`experiments/spectral_band_v20/`: the old Products `checkpoint-10200` learned a
representation-level trigger and **must not** be used as initialization here.

## Frozen protocol

- target class: `Video Games`
- 128 low-frequency, embedding-diverse real-node candidates
- four unique real trigger nodes, connected as a clique and attached to the center
- trigger search: exact target-token NLL on 32 frozen hard-validation samples
- replacement poison rate: `round(0.10 * 196615) = 19662`
- poison source selection: deterministic class-balanced water-filling, with at
  most 30% of each hard non-target source class used
- after trigger selection: freeze trigger and train only the projector
- official targeted ASR denominator: test rows whose ground truth is not
  `Video Games`; invalid generations never count as success
- the official test split is sealed until all search/training/checkpoint choices
  have been frozen

## H200 layout

The default locations used by the scripts are:

```text
/home/zitong/work/graph_phantom/                 # this GitHub repository
/home/zitong/work/LLaGA-upstream/                # full upstream LLaGA source
/home/zitong/graph_phantom_assets/ogbn-products/ # public Products tensors/JSONL
/home/zitong/graph_phantom_assets/validation/    # val JSONL + frozen hard IDs
/home/zitong/graph_phantom_models/               # HF model/projector cache
/home/zitong/graph_phantom_runs/products/        # generated data/checkpoints/logs
```

No dataset, Vicuna weight, projector, or generated checkpoint belongs in Git.
The local experiment `.gitignore` excludes `runs/`, logs, and Python caches.

## Public assets

Run on H200:

```bash
cd /home/zitong/work/graph_phantom/LLaGA/experiments/real_graph_trigger_products
nohup env ASSET_DIR=/home/zitong/graph_phantom_assets/ogbn-products \
  bash download_assets_h200.sh \
  >/home/zitong/graph_phantom_assets/ogbn-products/download.log 2>&1 &
echo $! >/home/zitong/graph_phantom_assets/ogbn-products/download.pid
```

The downloader uses the official LLaGA Box release, supports resume, checks
every byte size, and omits the test JSONL by default. The upstream clean
projector is `Runjin/llaga-vicuna-7b-simteg-ND-classification_expert-linear-projector`;
the base model is `lmsys/vicuna-7b-v1.5-16k`. H200 can fetch both directly from
Hugging Face. Their paths must be passed explicitly to training/evaluation.

## Files

- `products_protocol.py`: labels, deterministic balancing/splits, normalization
- `prepare_candidates.py`: memory-bounded 128-node candidate construction
- `prepare_training_data.py`: real four-slot clique sampling and balanced poison data
- `freeze_protocol.py`: provenance hashes, search/holdout folds, leakage checks
- `score_products_hard_trigger_nll.py`: exact target-token NLL scorer
- `prepare_probe_inputs.py`: paired clean/original/resampled/triggered inputs
- `eval_products_probe.py`: resident-model deterministic three-branch generation
- `calc_products_metrics.py`: CA, exact targeted ASR, validity, per-source ASR
- `train_real_node_phase0.py`: common LLaGA real-node trainer used by the pipeline
- `exact_hard_discrete_protocol.py`: discrete four-node search/gate utilities
- `run_phase0_prepare_h200.sh`: CPU protocol-freeze stage
- `run_phase2_search_h200.sh`: H200 trigger-search stage
- `run_phase3_projector_h200.sh`: frozen-trigger projector-only training stage

All shell runners fail closed when required assets are missing. They never
silently substitute the legacy Products attack checkpoint.
