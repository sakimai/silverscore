#!/usr/bin/env python3
"""Verify CiCo CLCL tree includes test_openai_reorder support."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List


CHECKS: Dict[str, List[str]] = {
    "main_task_retrieval.py": [
        "test_openai_reorder",
        'elif args.dataloader_type == "test_openai_reorder"',
        "original_text_dic = test_dataloader.dataset.sentences_dict",
        "original_video_dic = test_dataloader.dataset.video_dict",
        "json.dump(data_to_save, json_file, indent=4)",
        'np.save(os.path.join(args.output_dir, f"{args.datatype}_{args.dataloader_type}_results.npy"), similarity_results)',
    ],
    "dataloaders/dataloader_ph_retrieval.py": [
        "test_openai_reorder.pkl",
        "test_openai_reorder",
    ],
    "dataloaders/dataloader_csl_retrieval.py": [
        "test_openai_reorder.pkl",
        "test_openai_reorder",
    ],
    "dataloaders/data_dataloaders.py": [
        "def dataloader_ph_test_openai_reorder",
        "def dataloader_csl_test_openai_reorder",
        '"test_openai_reorder":dataloader_ph_test_openai_reorder',
        '"test_openai_reorder":dataloader_csl_test_openai_reorder',
    ],
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Check CiCo reordered dataloader support.")
    parser.add_argument(
        "--cico-clcl-dir",
        type=Path,
        required=True,
        help="Path to CiCo/CLCL directory.",
    )
    args = parser.parse_args()

    base = args.cico_clcl_dir
    all_errors: List[str] = []

    for rel_path, patterns in CHECKS.items():
        file_path = base / rel_path
        if not file_path.exists():
            all_errors.append(f"Missing file: {file_path}")
            continue
        text = file_path.read_text(encoding="utf-8")
        for pattern in patterns:
            if pattern not in text:
                all_errors.append(f"{rel_path}: missing pattern -> {pattern}")

    if all_errors:
        print("CiCo reorder support check FAILED:")
        for err in all_errors:
            print(f"- {err}")
        return 1

    print("CiCo reorder support check PASSED.")
    print("All required test_openai_reorder hooks were found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
