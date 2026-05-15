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

Several large artifacts are stored as split ZIP archives instead of raw files.
This is done to avoid GitHub file-size limits in environments where Git LFS is unavailable.

For each converted large file, the repository keeps:

- a small `*.parts.json` metadata file
- multiple ZIP parts named `*.zip.part-000`, `*.zip.part-001`, ...

To reconstruct a file:

1. concatenate the ZIP parts in the same directory
2. unzip the reconstructed archive

The corresponding `*.parts.json` file records the original file name, SHA256 checksum,
and example reconstruction commands.
