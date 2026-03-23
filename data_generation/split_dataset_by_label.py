#!/usr/bin/env python3
"""Split the sarcasm headlines dataset into label-specific JSONL files.

The source dataset uses one JSON object per line even though the file suffix is
``.json``. This script preserves that format in the generated output files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_PATH = SCRIPT_DIR / "Sarcasm_Headlines_Dataset_v2.json"
DEFAULT_SARCASTIC_OUTPUT_PATH = SCRIPT_DIR / "Sarcasm_Headlines_Dataset_v2_sarcastic.json"
DEFAULT_NON_SARCASTIC_OUTPUT_PATH = SCRIPT_DIR / "Sarcasm_Headlines_Dataset_v2_non_sarcastic.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split a sarcasm dataset into sarcastic and non-sarcastic JSONL files.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Input JSONL dataset path. Default: {DEFAULT_INPUT_PATH}",
    )
    parser.add_argument(
        "--sarcastic-output",
        type=Path,
        default=DEFAULT_SARCASTIC_OUTPUT_PATH,
        help=(
            "Output path for label-1 sarcastic rows. "
            f"Default: {DEFAULT_SARCASTIC_OUTPUT_PATH}"
        ),
    )
    parser.add_argument(
        "--non-sarcastic-output",
        type=Path,
        default=DEFAULT_NON_SARCASTIC_OUTPUT_PATH,
        help=(
            "Output path for label-0 non-sarcastic rows. "
            f"Default: {DEFAULT_NON_SARCASTIC_OUTPUT_PATH}"
        ),
    )
    parser.add_argument(
        "--label-key",
        type=str,
        default="is_sarcastic",
        help="Record key containing the binary sarcasm label. Default: is_sarcastic",
    )
    return parser.parse_args()


def resolve_label(record: dict[str, Any], label_key: str, line_number: int) -> int:
    if label_key not in record:
        raise KeyError(
            f"Line {line_number}: missing required label key {label_key!r}. "
            f"Available keys: {sorted(record.keys())}"
        )

    try:
        label = int(record[label_key])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Line {line_number}: label value {record[label_key]!r} is not an integer."
        ) from exc

    if label not in {0, 1}:
        raise ValueError(
            f"Line {line_number}: label must be 0 or 1, got {label!r}."
        )

    return label


def split_dataset(
    input_path: Path,
    sarcastic_output_path: Path,
    non_sarcastic_output_path: Path,
    label_key: str,
) -> tuple[int, int]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    sarcastic_output_path.parent.mkdir(parents=True, exist_ok=True)
    non_sarcastic_output_path.parent.mkdir(parents=True, exist_ok=True)

    sarcastic_count = 0
    non_sarcastic_count = 0

    with input_path.open("r", encoding="utf-8") as fin, \
         sarcastic_output_path.open("w", encoding="utf-8") as sarcastic_fout, \
         non_sarcastic_output_path.open("w", encoding="utf-8") as non_sarcastic_fout:

        for line_number, line in enumerate(fin, start=1):
            stripped_line = line.strip()
            if not stripped_line:
                continue

            try:
                record: dict[str, Any] = json.loads(stripped_line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Line {line_number}: invalid JSON: {exc}"
                ) from exc

            label = resolve_label(record, label_key=label_key, line_number=line_number)
            serialized_record = json.dumps(record, ensure_ascii=False) + "\n"

            if label == 1:
                sarcastic_fout.write(serialized_record)
                sarcastic_count += 1
            else:
                non_sarcastic_fout.write(serialized_record)
                non_sarcastic_count += 1

    return sarcastic_count, non_sarcastic_count


def main() -> None:
    args = parse_args()
    sarcastic_count, non_sarcastic_count = split_dataset(
        input_path=args.input,
        sarcastic_output_path=args.sarcastic_output,
        non_sarcastic_output_path=args.non_sarcastic_output,
        label_key=args.label_key,
    )

    total_count = sarcastic_count + non_sarcastic_count
    print(f"Input file: {args.input}")
    print(f"Sarcastic output: {args.sarcastic_output} ({sarcastic_count} rows)")
    print(
        "Non-sarcastic output: "
        f"{args.non_sarcastic_output} ({non_sarcastic_count} rows)"
    )
    print(f"Total rows written: {total_count}")


if __name__ == "__main__":
    main()
