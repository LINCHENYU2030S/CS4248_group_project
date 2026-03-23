#!/usr/bin/env python3
"""Build a test-set file from headline differences between two datasets.

The output contains records that appear in `nonsarcastic_to_sarcastic_out.json`
but not in `nonsarcastic_to_sarcastic_similarityfiltered.json`, comparing only
the `headline` field and ignoring record order.

The input and output files use JSONL format (one JSON object per line) even
though the filenames end with `.json`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE_PATH = SCRIPT_DIR / "nonsarcastic_to_sarcastic_out.json"
DEFAULT_FILTERED_PATH = SCRIPT_DIR / "nonsarcastic_to_sarcastic_similarityfiltered.json"
DEFAULT_OUTPUT_PATH = SCRIPT_DIR / "nonsarcastic_to_sarcastic_test_set.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create nonsarcastic_to_sarcastic_test_set.json by removing any row "
            "whose headline appears in the similarity-filtered dataset."
        ),
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE_PATH,
        help=f"Full source dataset path. Default: {DEFAULT_SOURCE_PATH}",
    )
    parser.add_argument(
        "--filtered",
        type=Path,
        default=DEFAULT_FILTERED_PATH,
        help=f"Similarity-filtered dataset path. Default: {DEFAULT_FILTERED_PATH}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output test-set path. Default: {DEFAULT_OUTPUT_PATH}",
    )
    return parser.parse_args()


def detect_json_format(path: Path) -> str:
    with path.open("r", encoding="utf-8") as file:
        while True:
            char = file.read(1)
            if not char:
                return "jsonl"
            if not char.isspace():
                return "json" if char == "[" else "jsonl"


def load_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    file_format = detect_json_format(path)
    if file_format == "json":
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON array in {path}, got {type(data).__name__}.")
        return [validate_record(record, path=path, line_number=index + 1) for index, record in enumerate(data)]

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped_line = line.strip()
            if not stripped_line:
                continue
            try:
                raw_record = json.loads(stripped_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            records.append(validate_record(raw_record, path=path, line_number=line_number))
    return records


def validate_record(record: Any, path: Path, line_number: int) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ValueError(
            f"{path}:{line_number}: expected each record to be a JSON object, "
            f"got {type(record).__name__}."
        )
    if "headline" not in record:
        raise KeyError(f"{path}:{line_number}: missing required key 'headline'.")
    return record


def normalise_headline(record: dict[str, Any], path: Path, line_number: int) -> str:
    headline = str(record["headline"]).strip()
    if not headline:
        raise ValueError(f"{path}:{line_number}: headline is empty.")
    return headline


def build_test_set(
    source_records: list[dict[str, Any]],
    filtered_records: list[dict[str, Any]],
    filtered_path: Path,
    source_path: Path,
) -> list[dict[str, Any]]:
    filtered_headlines = {
        normalise_headline(record, path=filtered_path, line_number=index + 1)
        for index, record in enumerate(filtered_records)
    }

    test_set_records: list[dict[str, Any]] = []
    for index, record in enumerate(source_records):
        headline = normalise_headline(record, path=source_path, line_number=index + 1)
        if headline not in filtered_headlines:
            test_set_records.append(record)
    return test_set_records


def write_jsonl(records: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()

    source_records = load_records(args.source)
    filtered_records = load_records(args.filtered)
    test_set_records = build_test_set(
        source_records=source_records,
        filtered_records=filtered_records,
        filtered_path=args.filtered,
        source_path=args.source,
    )
    write_jsonl(test_set_records, args.output)

    print(f"Source file: {args.source} ({len(source_records)} rows)")
    print(f"Filtered file: {args.filtered} ({len(filtered_records)} rows)")
    print(f"Output file: {args.output} ({len(test_set_records)} rows)")


if __name__ == "__main__":
    main()
