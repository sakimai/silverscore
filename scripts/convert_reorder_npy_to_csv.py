#!/usr/bin/env python3
"""Convert reordered CiCo NPY outputs into signscore-style CSV artifacts.

This script mirrors the PH/CSL conversion behavior used to create:
- signscore_reorder_data.csv
- csl_signscore_reorder_data.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


SCORE_SCALE = 3.5


def extract_scores(score_matrix: np.ndarray, index_pairs: List[Tuple[int, int]]) -> np.ndarray:
    return np.array([score_matrix[i, j] for i, j in index_pairs], dtype=float)


def convert_ph_reorder(
    ph_reorder_npy: Path,
    ph_original_csv: Path,
) -> pd.DataFrame:
    similarity_results = np.load(ph_reorder_npy, allow_pickle=True).item()
    text_ids_dict: Dict[str, str] = similarity_results["text_ids"].item()
    clip_score_i2t = np.squeeze(similarity_results["clip_score_i2t"])

    if clip_score_i2t.ndim != 2 or clip_score_i2t.shape[0] != clip_score_i2t.shape[1]:
        raise ValueError(f"Expected PH square matrix, got shape={clip_score_i2t.shape}")

    num_items = clip_score_i2t.shape[0]
    same_indices = [(i, i) for i in range(num_items)]
    reordered_i2t = extract_scores(clip_score_i2t, same_indices) * SCORE_SCALE

    original_i2t = pd.read_csv(ph_original_csv)["same_scores_i2t"].to_numpy(dtype=float)
    if len(original_i2t) != len(reordered_i2t):
        raise ValueError(
            f"PH length mismatch: original={len(original_i2t)} reordered={len(reordered_i2t)}"
        )

    # Mirrors notebook behavior: map by positional index over text_ids dict values.
    text_values = list(text_ids_dict.values())
    if len(text_values) != num_items:
        raise ValueError(f"PH text_ids length mismatch: {len(text_values)} vs {num_items}")

    return pd.DataFrame(
        {
            "text_ids": text_values,
            "original_i2t": original_i2t,
            "reordered_i2t": reordered_i2t,
        }
    )


def convert_csl_reorder(
    csl_reorder_npy: Path,
    csl_original_csv: Path,
) -> pd.DataFrame:
    similarity_results = np.load(csl_reorder_npy, allow_pickle=True).item()
    text_ids_dict: Dict[str, str] = similarity_results["text_ids"].item()
    video_ids_dict: Dict[int, Tuple[str, str]] = similarity_results["video_ids"].item()
    clip_score_i2t = similarity_results["clip_score_i2t"]

    if clip_score_i2t.ndim != 3 or clip_score_i2t.shape[1] != 2:
        raise ValueError(f"Expected CSL shape (N,2,N), got shape={clip_score_i2t.shape}")

    # Replicates notebook filtering behavior.
    filtered_indices: List[int] = []
    for i, entry in enumerate(clip_score_i2t):
        filtered_indices.append(i)  # always keep first slot
        if not np.all(entry[1] == 0):
            filtered_indices.append(i)  # keep second slot when non-zero

    filtered_columns = clip_score_i2t[:, :, filtered_indices]
    filtered_columns = filtered_columns[filtered_indices, :, :]
    clip_score_i2t_sq = filtered_columns[:, 0, :]  # (num_samples, num_samples)

    if clip_score_i2t_sq.shape[0] != clip_score_i2t_sq.shape[1]:
        raise ValueError(f"Expected square CSL matrix after filtering, got {clip_score_i2t_sq.shape}")

    num_items = clip_score_i2t_sq.shape[0]
    same_indices = [(i, i) for i in range(num_items)]
    reordered_i2t = extract_scores(clip_score_i2t_sq, same_indices) * SCORE_SCALE

    original_i2t = pd.read_csv(csl_original_csv)["same_scores_i2t"].to_numpy(dtype=float)
    if len(original_i2t) != len(reordered_i2t):
        raise ValueError(
            f"CSL length mismatch: original={len(original_i2t)} reordered={len(reordered_i2t)}"
        )

    # Mirrors notebook behavior: iterate video_ids order, map video sentence id -> text.
    text_values: List[str] = []
    for _, video_tuple in video_ids_dict.items():
        sentence_id = video_tuple[0]
        text_values.append(text_ids_dict[sentence_id])

    if len(text_values) != len(reordered_i2t):
        raise ValueError(
            f"CSL text length mismatch: mapped={len(text_values)} reordered={len(reordered_i2t)}"
        )

    return pd.DataFrame(
        {
            "text_ids": text_values,
            "original_i2t": original_i2t,
            "reordered_i2t": reordered_i2t,
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert reordered NPY artifacts into CSV files.")
    parser.add_argument(
        "--npy-dir",
        type=Path,
        default=Path("repro/artifacts/npy"),
        help="Directory containing *_test_openai_reorder_results.npy files.",
    )
    parser.add_argument(
        "--paper-data-dir",
        type=Path,
        default=Path("notebooks/paper_data"),
        help="Directory containing canonical paper CSVs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("repro/artifacts/csv_reconstructed"),
        help="Directory where reconstructed CSVs will be written.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    ph_df = convert_ph_reorder(
        ph_reorder_npy=args.npy_dir / "ph_test_openai_reorder_results.npy",
        ph_original_csv=args.paper_data_dir / "sign_score_data.csv",
    )
    csl_df = convert_csl_reorder(
        csl_reorder_npy=args.npy_dir / "csl_test_openai_reorder_results.npy",
        csl_original_csv=args.paper_data_dir / "csl_sign_score_data.csv",
    )

    ph_out = args.output_dir / "signscore_reorder_data.csv"
    csl_out = args.output_dir / "csl_signscore_reorder_data.csv"
    ph_df.to_csv(ph_out, index=False)
    csl_df.to_csv(csl_out, index=False)

    print(f"Wrote {ph_out} rows={len(ph_df)}")
    print(f"Wrote {csl_out} rows={len(csl_df)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
