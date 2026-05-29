#!/usr/bin/env python3
"""Validate required paper CSV artifacts and core schema."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List


REQUIRED_FILES: Dict[str, List[str]] = {
    "bt_score_data.csv": [
        "refs",
        "hyps",
        "same_indices",
        "random_indices",
        "same_index_bleu1",
        "different_index_bleu1",
        "same_index_bleu2",
        "different_index_bleu2",
        "same_index_bleu3",
        "different_index_bleu3",
        "same_index_bleu4",
        "different_index_bleu4",
        "same_index_rouge",
        "different_index_rouge",
        "same_index_bert",
        "different_index_bert",
        "same_index_bleurt",
        "different_index_bleurt",
    ],
    "sign_score_data.csv": [
        "text_ids",
        "same_indices",
        "random_indices",
        "same_scores_i2t",
        "different_scores_i2t",
    ],
    "csl_bt_score_data.csv": [
        "refs",
        "hyps",
        "same_indices",
        "random_indices",
        "same_index_bleu1",
        "different_index_bleu1",
        "same_index_bleu2",
        "different_index_bleu2",
        "same_index_bleu3",
        "different_index_bleu3",
        "same_index_bleu4",
        "different_index_bleu4",
        "same_index_rouge",
        "different_index_rouge",
        "same_index_bert",
        "different_index_bert",
        "same_index_bleurt",
        "different_index_bleurt",
    ],
    "csl_sign_score_data.csv": [
        "text_ids",
        "same_indices",
        "random_indices",
        "same_scores_i2t",
        "different_scores_i2t",
    ],
    "bt_score_reorder_data.csv": [
        "refs",
        "hyps",
        "original_bleu1",
        "reordered_bleu1",
        "original_rouge",
        "reordered_rouge",
        "original_bert",
        "reordered_bert",
        "original_bleurt",
        "reordered_bleurt",
    ],
    "signscore_reorder_data.csv": ["text_ids", "original_i2t", "reordered_i2t"],
    "csl_bt_score_reorder_data.csv": [
        "refs",
        "hyps",
        "original_bleu1",
        "reordered_bleu1",
        "original_rouge",
        "reordered_rouge",
        "original_bert",
        "reordered_bert",
        "original_bleurt",
        "reordered_bleurt",
    ],
    "csl_signscore_reorder_data.csv": ["text_ids", "original_i2t", "reordered_i2t"],
}


def row_count(path: Path) -> int:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        # subtract header row; max(..., 0) guards empty files
        return max(sum(1 for _ in reader) - 1, 0)


def check_file(path: Path, required_columns: List[str]) -> List[str]:
    errors: List[str] = []
    if not path.exists():
        return [f"Missing file: {path.name}"]

    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        actual_columns = reader.fieldnames or []
        missing_columns = [col for col in required_columns if col not in actual_columns]
        if missing_columns:
            errors.append(
                f"{path.name}: missing required columns: {', '.join(missing_columns)}"
            )

    rows = row_count(path)
    if rows == 0:
        errors.append(f"{path.name}: file has no data rows")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate SiLVERScore paper CSV artifacts.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("notebooks/paper_data"),
        help="Directory containing paper CSV artifacts.",
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    print(f"Checking paper data in: {data_dir}")

    all_errors: List[str] = []
    for filename, columns in REQUIRED_FILES.items():
        all_errors.extend(check_file(data_dir / filename, columns))

    if all_errors:
        print("\nValidation failed:")
        for err in all_errors:
            print(f"- {err}")
        return 1

    print("\nValidation passed. File row counts:")
    for filename in REQUIRED_FILES:
        count = row_count(data_dir / filename)
        print(f"- {filename}: {count} rows")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
