#!/usr/bin/env python3
"""Generate CiCo-style test_openai_reorder.pkl from test.pkl using OpenAI."""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

from openai import OpenAI


PROMPT_TEMPLATE = (
    "Reorder the words in the following sentence while keeping the meaning the same:\n\n"
    "{text}\n\n"
    "Reordered sentence:"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a test_openai_reorder.pkl file compatible with CiCo dataloaders."
    )
    parser.add_argument("--input-pkl", type=Path, required=True, help="Path to source test.pkl")
    parser.add_argument("--output-pkl", type=Path, required=True, help="Path to write test_openai_reorder.pkl")
    parser.add_argument("--model", type=str, default="gpt-4o", help="OpenAI model name")
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenAI API key. If omitted, the OpenAI SDK default env lookup is used.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Optional pause between requests to avoid rate limits.",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Optional cap for debugging a subset of entries.",
    )
    parser.add_argument(
        "--log-json",
        type=Path,
        default=None,
        help="Optional path to save input/output text pairs for traceability.",
    )
    return parser.parse_args()


def call_reorder(client: OpenAI, model: str, text: str) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": PROMPT_TEMPLATE.format(text=text)}],
    )
    return (response.choices[0].message.content or "").strip()


def process_ph_style(
    data: Dict[str, Dict[str, Any]],
    client: OpenAI,
    model: str,
    sleep_seconds: float,
    max_items: int | None,
) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, str]]]:
    out: Dict[str, Dict[str, Any]] = {}
    audit: List[Dict[str, str]] = []

    for idx, (key, value) in enumerate(data.items()):
        if max_items is not None and idx >= max_items:
            break
        original_text = value["text"]
        reordered = call_reorder(client, model, original_text)
        out[key] = {
            "video_name": value["video_name"],
            "ori_text": value["ori_text"],
            "text": reordered,
            "num_frames": value["num_frames"],
        }
        audit.append({"id": key, "original_text": original_text, "reordered_text": reordered})
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    return out, audit


def process_csl_style(
    data: Dict[str, List[Dict[str, Any]]],
    client: OpenAI,
    model: str,
    sleep_seconds: float,
    max_items: int | None,
) -> Tuple[Dict[str, List[Dict[str, Any]]], List[Dict[str, str]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    audit: List[Dict[str, str]] = []
    seen = 0

    for key, items in data.items():
        if max_items is not None and seen >= max_items:
            break
        new_items: List[Dict[str, Any]] = []
        for item in items:
            if max_items is not None and seen >= max_items:
                break
            original_text = item["text"]
            reordered = call_reorder(client, model, original_text)
            new_items.append(
                {
                    "video_name": item["video_name"],
                    "ori_text": item["ori_text"],
                    "text": reordered,
                    "num_frames": item["num_frames"],
                }
            )
            audit.append({"id": key, "original_text": original_text, "reordered_text": reordered})
            seen += 1
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
        if new_items:
            out[key] = new_items

    return out, audit


def main() -> int:
    args = parse_args()
    with args.input_pkl.open("rb") as handle:
        raw_data: Union[Dict[str, Dict[str, Any]], Dict[str, List[Dict[str, Any]]]] = pickle.load(handle)

    if not raw_data:
        raise ValueError("Input pickle is empty.")

    first_value = next(iter(raw_data.values()))
    client = OpenAI(api_key=args.api_key) if args.api_key else OpenAI()

    if isinstance(first_value, dict):
        output, audit = process_ph_style(
            raw_data, client, args.model, args.sleep_seconds, args.max_items
        )
        detected = "phoenix-like dict format"
    elif isinstance(first_value, list):
        output, audit = process_csl_style(
            raw_data, client, args.model, args.sleep_seconds, args.max_items
        )
        detected = "csl-like list format"
    else:
        raise TypeError(
            "Unsupported pickle schema. Expected dict-of-dict (PH) or dict-of-list[dict] (CSL)."
        )

    args.output_pkl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_pkl.open("wb") as handle:
        pickle.dump(output, handle)

    if args.log_json:
        args.log_json.parent.mkdir(parents=True, exist_ok=True)
        with args.log_json.open("w", encoding="utf-8") as handle:
            json.dump(audit, handle, ensure_ascii=False, indent=2)

    print(f"Detected schema: {detected}")
    print(f"Wrote reordered pickle: {args.output_pkl}")
    print(f"Processed entries: {len(audit)}")
    if args.log_json:
        print(f"Wrote audit log: {args.log_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
