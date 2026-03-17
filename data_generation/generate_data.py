#!/usr/bin/env python3
"""
Flip sarcasm labels in a JSONL headlines dataset using the OpenAI API.

Input format:
  One JSON object per line, e.g.
  {"is_sarcastic": 1, "headline": "...", "article_link": "..."}

Output format:
  Same JSONL, but with an added key:
  "reversed_headline": "..."

Behavior:
- If is_sarcastic == 1: rewrite headline into a clearly NON-sarcastic version
- If is_sarcastic == 0: rewrite headline into a clearly SARCASTIC version
- Processes one headline at a time
- Saves progress incrementally so it can resume after interruption
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from openai import OpenAI


SYSTEM_PROMPT = """You rewrite news-style headlines by flipping their tone.

Task:
- If the original headline is sarcastic, rewrite it into a clearly non-sarcastic headline.
- If the original headline is non-sarcastic, rewrite it into a clearly sarcastic headline.

Requirements:
1. Rewrite most of the wording. Do not just prepend or append a short phrase.
2. Keep the same core topic, entities, and general event meaning whenever possible.
3. Output exactly one headline, not multiple options.
4. Keep it concise and headline-like.
5. Do not add quotation marks unless they are natural in the headline.
6. Do not explain your reasoning.
7. Make the flipped tone very clear.
"""

USER_TEMPLATE = """Original headline: {headline}
Original label:
- 1 = sarcastic
- 0 = non-sarcastic

This headline's label is: {label}

Return a JSON object with exactly one key:
{{
  "reversed_headline": "..."
}}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input JSONL file",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output JSONL file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("OPENAI_MODEL", "gpt-5-mini"),
        help="OpenAI model name",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=8,
        help="Maximum retries per headline",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.5,
        help="Base sleep between successful requests",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume by skipping already-written output lines",
    )
    return parser.parse_args()


def count_existing_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def make_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Please export your API key first."
        )
    return OpenAI(api_key=api_key)


def call_model(
    client: OpenAI,
    model: str,
    headline: str,
    label: int,
    max_retries: int,
) -> str:
    last_err: Exception | None = None

    for attempt in range(max_retries):
        try:
            response = client.responses.create(
                model=model,
                instructions=SYSTEM_PROMPT,
                input=USER_TEMPLATE.format(headline=headline, label=label),
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "headline_flip",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "reversed_headline": {"type": "string"}
                            },
                            "required": ["reversed_headline"],
                            "additionalProperties": False,
                        },
                        "strict": True,
                    }
                },
            )

            content = response.output_text
            data = json.loads(content)
            print(data)
            rewritten = data["reversed_headline"].strip()
            if not rewritten:
                raise ValueError("Model returned an empty reversed_headline")

            return rewritten

        except Exception as e:
            last_err = e
            backoff = min(60, 2 ** attempt)
            print(
                f"[retry {attempt + 1}/{max_retries}] error: {e}. "
                f"Sleeping {backoff}s...",
                file=sys.stderr,
            )
            time.sleep(backoff)

    raise RuntimeError(f"Failed after {max_retries} retries: {last_err}")


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    client = make_client()

    already_done = count_existing_lines(output_path) if args.resume else 0
    mode = "a" if args.resume else "w"

    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Model:  {args.model}")
    print(f"Resume: {args.resume} (skip first {already_done} lines)")
    print()

    processed = 0
    with input_path.open("r", encoding="utf-8") as fin, \
         output_path.open(mode, encoding="utf-8") as fout:

        for idx, line in enumerate(fin):
            if idx < already_done:
                continue

            line = line.strip()
            if not line:
                continue

            try:
                record: dict[str, Any] = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[line {idx}] invalid JSON: {e}", file=sys.stderr)
                continue

            if "headline" not in record or "is_sarcastic" not in record:
                print(
                    f"[line {idx}] missing required keys: {record.keys()}",
                    file=sys.stderr,
                )
                continue

            headline = str(record["headline"]).strip()
            label = int(record["is_sarcastic"])

            reversed_headline = call_model(
                client=client,
                model=args.model,
                headline=headline,
                label=label,
                max_retries=args.max_retries,
            )

            record["reversed_headline"] = reversed_headline

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()

            processed += 1
            print(
                f"[done] line={idx} label={label} "
                f"orig={headline!r} -> flipped={reversed_headline!r}"
            )

            time.sleep(args.sleep)

    print()
    print(f"Finished. Newly processed lines: {processed}")


if __name__ == "__main__":
    main()