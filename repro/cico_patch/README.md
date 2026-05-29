# CiCo Patch for Reordered Evaluation

This folder contains the CiCo code edits needed to run:

```bash
--dataloader_type test_openai_reorder
```

These edits are **reproduction-only** and are kept outside `src/silverscore`.

## Files affected in CiCo

- `main_task_retrieval.py`
- `dataloaders/data_dataloaders.py`
- `dataloaders/dataloader_ph_retrieval.py`
- `dataloaders/dataloader_csl_retrieval.py`

## Apply Patch

From your CiCo `CLCL/` directory:

```bash
git apply /path/to/silverscore/repro/cico_patch/cico_test_openai_reorder.patch
```

## Verify Patch

From `silverscore/`:

```bash
python scripts/check_cico_reorder_support.py --cico-clcl-dir /path/to/CiCo/CLCL
```

If verification passes, you can follow `REPRODUCE.md` to generate reordered `.npy` files.
