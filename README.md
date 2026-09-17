# Graph-Phantom

This repository contains the curated main-experiment package for the Graph-Phantom project.
It includes the code and selected artifacts used for the Vicuna-based experiments on both
GraphGPT and LLaGA across PubMed, Cora, ogbn-arxiv, and ogbn-products.

## Overview

The repository is organized around two model families:

- `GraphGPT_backdoor/`
- `LLaGA/`

For each family, we keep:

- the minimal code required by the main experiment pipeline
- the selected checkpoint artifacts used in the final runs
- hard-split definitions
- clean baseline outputs
- selected validation outputs
- selected test outputs

## Repository Structure

### `GraphGPT_backdoor/`

- `code/graphgpt/`: minimal GraphGPT code used by the main experiment path
- `tools/`: data-conversion utilities retained for the exported setup
- `experiments/latent_trigger_v20/runs/`: main GraphGPT experiment artifacts

### `LLaGA/`

- `code/llaga_v20/`: minimal LLaGA v20 code used by the main experiment path
- `experiments/spectral_band_v20/runs/`: main LLaGA experiment artifacts
- `experiments/real_graph_trigger_products/`: updated real-node-trigger pipeline
  for LLaGA + ogbn-products (target `Video Games`; H200 runners and manifests)

The updated Products pipeline does not reuse the legacy representation-trigger
`checkpoint-10200`. Its README documents the public direct-download assets and
the clean upstream projector used for initialization.

## Main Experiment Artifacts

Each run directory follows the same high-level layout:

- `00_baseline_clean/`
- `01_hard_split/`
- `02_train/`
- `03_model_selection/`
- `04_test_eval/`

Naming has been normalized so that the selected artifacts are easy to inspect:

- clean baselines use `clean_{dataset}_{split}.jsonl`
- selected validation outputs use:
  - `selected_val_clean.jsonl`
  - `selected_val_poison.jsonl`
- selected test outputs use:
  - `selected_test_clean.jsonl`
  - `selected_test_poison.jsonl`

## Notes on Large Files

Some experiment artifacts are large and are tracked with Git LFS.
The ogbn-products text-only test artifact for LLaGA and GraphGPT are stored as single retained partial files:

- `LLaGA/experiments/spectral_band_v20/runs/ogbn-products_vicuna7b_v15_16k/01_hard_split/text_only_products_test.jsonl.split.json`
- `GraphGPT_backdoor/experiments/latent_trigger_v20/runs/ogbn-products_vicuna7b_v15_16k/01_hard_split/text_only_products_test.jsonl.split.jsonl`

