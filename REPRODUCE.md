# Reproducing SiLVERScore Paper Results

This guide separates reproduction into two tracks:

- **Deterministic track (recommended):** reproduce paper tables/figures from the released CSV artifacts in `notebooks/paper_data/`.
- **Regeneration track (optional):** regenerate reordered text and reordered `.npy` via GPT + CiCo. This is **non-deterministic** and will not exactly match released files.

This file is designed to be self-contained inside `silverscore/`, even when this folder is published by itself.

## 1) Environment Setup

From `silverscore/`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install jupyter nbconvert pandas matplotlib seaborn scipy scikit-learn
pip install openai
```

## 2) Deterministic Reproduction (Paper Artifacts)

The following files are the canonical paper artifacts:

- `notebooks/paper_data/bt_score_data.csv`
- `notebooks/paper_data/sign_score_data.csv`
- `notebooks/paper_data/csl_bt_score_data.csv`
- `notebooks/paper_data/csl_sign_score_data.csv`
- `notebooks/paper_data/bt_score_reorder_data.csv`
- `notebooks/paper_data/signscore_reorder_data.csv`
- `notebooks/paper_data/csl_bt_score_reorder_data.csv`
- `notebooks/paper_data/csl_signscore_reorder_data.csv`

Validate required files/columns:

```bash
python scripts/validate_paper_data.py --data-dir notebooks/paper_data
```

Run and export all notebook outputs (tables + figures):

```bash
bash scripts/run_paper_figures.sh
```

Outputs are written under:

- `notebooks/figures/`

### 2.1 What to Ship for Exact Reproduction

If you are distributing only `silverscore/`, include the following artifacts when possible:

- Canonical CSVs in `notebooks/paper_data/` (already included)
- Reordered CiCo retrieval outputs (`.npy`) for traceability:
  - `ph_test_openai_reorder_results.npy`
  - `csl_test_openai_reorder_results.npy`
- Reordered GPT JSON inputs in `repro/artifacts/openai_reorder_json/`
- Reordered PKL inputs in `repro/artifacts/pkl/`:
  - `ph_test_openai_reorder.pkl`
  - `csl_test_openai_reorder.pkl`
- Random-pair index files in `repro/artifacts/random_indices/`:
  - `random_indices.npy`
  - `csl-random_indices.npy`
- GPT reorder audit logs (JSON from `--log-json`) to document input/output text pairs
- Run metadata (recommended): command lines, model/checkpoint names, and seeds

Notes:

- The random-pair selection details used for "correct vs random" analysis are already preserved in:
  - `bt_score_data.csv` / `csl_bt_score_data.csv` (`random_indices`, `same_indices`)
  - `sign_score_data.csv` / `csl_sign_score_data.csv` (`random_indices`, `same_indices`)
- Reordered experiments are not random-pair based in the same way, so those CSVs do not require `random_indices`.

## 3) Regeneration Track (Optional, Non-Deterministic)

### 3.1 Generate `test_openai_reorder.pkl` (inside CiCo data folders)

Use the script in this folder:

```bash
# Example: PHOENIX
python scripts/generate_openai_reorder_pkl.py \
  --input-pkl /path/to/CiCo/CLCL/data_ph/test.pkl \
  --output-pkl /path/to/CiCo/CLCL/data_ph/test_openai_reorder.pkl \
  --model gpt-4o \
  --log-json logs/ph_openai_reorder_audit.json

# Example: CSL
python scripts/generate_openai_reorder_pkl.py \
  --input-pkl /path/to/CiCo/CLCL/data_csl/test.pkl \
  --output-pkl /path/to/CiCo/CLCL/data_csl/test_openai_reorder.pkl \
  --model gpt-4o \
  --log-json logs/csl_openai_reorder_audit.json
```

Prompt used by the script:

```text
Reorder the words in the following sentence while keeping the meaning the same:

{text}

Reordered sentence:
```

### 3.2 Generate reordered `.npy` results with CiCo

You need the external CiCo/CLCL codebase and its checkpoints/features.
Before running, patch CiCo with the reproduction edits included in this repo.

Apply patch from the CiCo `CLCL/` directory:

```bash
git apply /path/to/silverscore/repro/cico_patch/cico_test_openai_reorder.patch
```

The patch covers both reordered dataloader wiring and retrieval export hooks used in this release (`*_results.npy` and ID JSON dumps).

Optional verification from `silverscore/`:

```bash
python scripts/check_cico_reorder_support.py --cico-clcl-dir /path/to/CiCo/CLCL
```

Then, from the CiCo `CLCL/` directory, run evaluation on the reordered dataloader.

Important:

- Saving `*_test_openai_reorder_results.npy` is part of this patched retrieval workflow.
- Upstream/original CiCo setups may not expose the exact same reordered dataloader path or saved outputs.

Example commands (matching logged settings used in this project):

```bash
# PHOENIX reordered evaluation
torchrun --nproc_per_node=1 main_task_retrieval.py \
  --do_eval \
  --datatype ph \
  --dataloader_type test_openai_reorder \
  --data_path data_ph \
  --features_path sign_feature/ph_domain_agnostic \
  --features_path_retrain sign_feature/ph_domain_aware \
  --alpha 0.9 \
  --init_model chpt/ph_sota.pth \
  --output_dir new_experiment

# CSL reordered evaluation
torchrun --nproc_per_node=1 main_task_retrieval.py \
  --do_eval \
  --datatype csl \
  --dataloader_type test_openai_reorder \
  --data_path data_csl \
  --features_path sign_feature/csl_domain_agnostic \
  --features_path_retrain sign_feature/csl_domain_aware \
  --alpha 0.8 \
  --init_model chpt/csl_sota.pth \
  --output_dir new_experiment
```

Expected outputs:

- `new_experiment/ph_test_openai_reorder_results.npy`
- `new_experiment/csl_test_openai_reorder_results.npy`

Common companion outputs (standard test dataloader runs):

- `new_experiment/ph_results.npy`
- `new_experiment/csl_results.npy`

Recommended archival location inside this repo:

- `repro/artifacts/npy/ph_test_openai_reorder_results.npy`
- `repro/artifacts/npy/csl_test_openai_reorder_results.npy`
- `repro/artifacts/npy/ph_results.npy`
- `repro/artifacts/npy/csl_results.npy`

The released repository includes these files under `repro/artifacts/npy/`.

### 3.3 Convert regenerated outputs to analysis CSVs

This repository ships canonical CSV artifacts in `notebooks/paper_data/`.
If you regenerate GPT text and CiCo `.npy`, your derived CSVs will likely differ from the canonical ones.

Pipeline summary for reordered experiments:

1. GPT-reordered text JSON (and/or generated `test_openai_reorder.pkl`)
2. CiCo retrieval output `.npy` (`*_test_openai_reorder_results.npy`)
3. Derived analysis CSVs used by plotting/statistical notebooks

In this release, reordered PKL inputs are already included under `repro/artifacts/pkl/`.

To reproduce reordered SignScore CSV exports inside this repository, run:

```bash
python scripts/convert_reorder_npy_to_csv.py \
  --npy-dir repro/artifacts/npy \
  --paper-data-dir notebooks/paper_data \
  --output-dir repro/artifacts/csv_reconstructed
```

This script reproduces:

- `signscore_reorder_data.csv`
- `csl_signscore_reorder_data.csv`

using the same PH/CSL conversion behavior documented below.

### Important

- GPT generation is stochastic and model behavior can drift over time.
- Regenerated reordered text will likely differ from released reordered files.
- Keep released reordered artifacts as canonical for exact paper reproduction.

If you regenerate, treat this as an **approximate reproduction** path.

## 4) External Dependencies (When Publishing Only `silverscore/`)

`silverscore` contains the reproducibility instructions, but full regeneration still depends on external assets:

- CiCo/CLCL code
- CiCo patch in `repro/cico_patch/cico_test_openai_reorder.patch`
- CiCo checkpoints (`ph_sota.pth`, `csl_sota.pth`, etc.)
- Domain-agnostic and domain-aware sign feature directories

If those are unavailable, use the deterministic track with released CSVs.

## 5) How This Connects to the Released CSVs

The notebook in this folder (`notebooks/paper_figures.ipynb`) consumes CSV artifacts from `notebooks/paper_data/`.
Those CSVs are the stable interface for reproducing paper figures/tables in this repository.

## 6) NPY-to-CSV Mapping (How Scores Become Tables)

This section explains how retrieval outputs in `.npy` files are converted into CSV fields.

### PHOENIX mapping (single video slot)

- Input tensor shape is typically `(N, 1, N)` in `clip_score_i2t`.
- Pair score for `(video_idx, text_idx)` comes from:
  - `clip_score_i2t[video_idx, 0, text_idx]`
- Exported score columns use the same score scaling:
  - score `* 3.5`
- For reordered CSVs, diagonal same-pair values from reordered run are used for:
  - `reordered_i2t` / `reordered_bert` fields.

### CSL mapping (multi-video-slot handling)

- Input tensor shape is typically `(N_text, 2, N_text)` in `clip_score_i2t`.
- CSV export logic applies custom filtering:
  - always keep slot `0`
  - keep slot `1` only when non-zero
  - build `filtered_indices`, then filter rows/columns accordingly
  - reduce to one axis with `clip_score_i2t = filtered_columns[:, 0, :]`
- The CSV text column is then built by iterating `video_ids` order and mapping through `text_ids`.

Because CSL uses this explicit filtering/indexing path, reproducing CSL CSVs requires following the same conversion logic (not only reading raw tensor entries directly).

### Practical takeaway

- Treat `notebooks/paper_data/*.csv` as canonical paper artifacts.
- Treat `.npy` files under `repro/artifacts/npy/` as reproducibility intermediates used to derive those artifacts.

## 7) Expected Outputs

At minimum, running the deterministic path should produce:

- `notebooks/figures/paper_figure2_kde_correct_random_phoenix.png`
- `notebooks/figures/paper_figure3_kde_reordered_phoenix.png`

And display notebook tables corresponding to:

- Figure 2 KDE analysis
- Figure 3 KDE analysis
- Table 1 overlap/AUC comparisons
