#!/usr/bin/env python3
"""
Parallel sarcasm-flip headline rewriting with 4 processes.

Features:
- Worker k processes records where (line_index % 4 == k)
- Uses OpenAI Responses API with model gpt-5-mini
- Buffers 10 processed items before writing
- Uses a multiprocessing lock for safe shared-file writes
- Compatible with an existing sequential output file:
    assumes existing output lines correspond to the first N input lines
- Adds per-worker checkpoint files for safe resume after switching to parallel mode

Input format:
    JSONL (one JSON object per line)

Output format:
    JSONL with added key:
    "reversed_headline": "..."
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any
from multiprocessing import Process, Lock

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
    parser.add_argument("--input", type=str, required=True, help="Path to input JSONL file")
    parser.add_argument("--output", type=str, required=True, help="Path to output JSONL file")
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5-mini",
        help="OpenAI model name",
    )
    parser.add_argument("--workers", type=int, default=4, help="Number of worker processes")
    parser.add_argument(
        "--write-batch-size",
        type=int,
        default=10,
        help="How many completed items each worker buffers before writing",
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
        default=0.0,
        help="Base sleep between successful requests",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output/checkpoints",
    )
    return parser.parse_args()


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def make_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
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

            data = json.loads(response.output_text)
            rewritten = data["reversed_headline"].strip()
            if not rewritten:
                raise ValueError("Model returned empty reversed_headline")
            return rewritten

        except Exception as e:
            last_err = e
            err_text = str(e)

            if "insufficient_quota" in err_text:
                raise RuntimeError(
                    "OpenAI API billing/quota unavailable. Add credits or increase budget."
                ) from e

            backoff = min(30, 2 ** attempt)
            print(
                f"[retry {attempt + 1}/{max_retries}] error: {e}. sleeping {backoff}s...",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(backoff)

    raise RuntimeError(f"Failed after {max_retries} retries: {last_err}")


def checkpoint_path(output_path: Path, worker_id: int) -> Path:
    return output_path.parent / f"{output_path.name}.worker{worker_id}.ckpt"


def load_done_indices(ckpt_path: Path) -> set[int]:
    if not ckpt_path.exists():
        return set()
    done: set[int] = set()
    with ckpt_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                done.add(int(line))
    return done


def flush_batch(
    output_path: Path,
    ckpt_path: Path,
    lock: Lock,
    batch_records: list[dict[str, Any]],
    batch_indices: list[int],
) -> None:
    if not batch_records:
        return

    with lock:
        with output_path.open("a", encoding="utf-8") as fout:
            for rec in batch_records:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()

        with ckpt_path.open("a", encoding="utf-8") as fckpt:
            for idx in batch_indices:
                fckpt.write(f"{idx}\n")
            fckpt.flush()


def worker_main(
    worker_id: int,
    args: argparse.Namespace,
    base_skip: int,
    lock: Lock,
) -> None:
    client = make_client()
    input_path = Path(args.input)
    output_path = Path(args.output)
    ckpt = checkpoint_path(output_path, worker_id)

    done_indices = load_done_indices(ckpt) if args.resume else set()

    batch_records: list[dict[str, Any]] = []
    batch_indices: list[int] = []
    processed = 0

    try:
        with input_path.open("r", encoding="utf-8") as fin:
            for idx, line in enumerate(fin):
                if idx < base_skip:
                    continue
                if idx % args.workers != worker_id:
                    continue
                if idx in done_indices:
                    continue

                line = line.strip()
                if not line:
                    continue

                try:
                    record: dict[str, Any] = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"[worker {worker_id}] [line {idx}] invalid JSON: {e}", file=sys.stderr, flush=True)
                    continue

                if "headline" not in record or "is_sarcastic" not in record:
                    print(
                        f"[worker {worker_id}] [line {idx}] missing required keys",
                        file=sys.stderr,
                        flush=True,
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
                print(
                    f"[done] line={idx} label={label} "
                    f"orig={headline!r} -> flipped={reversed_headline!r}"
                )

                record["reversed_headline"] = reversed_headline
                batch_records.append(record)
                batch_indices.append(idx)
                processed += 1

                if processed % 25 == 0:
                    print(
                        f"[worker {worker_id}] processed={processed} last_input_index={idx}",
                        flush=True,
                    )

                if len(batch_records) >= args.write_batch_size:
                    flush_batch(output_path, ckpt, lock, batch_records, batch_indices)
                    batch_records.clear()
                    batch_indices.clear()

                if args.sleep > 0:
                    time.sleep(args.sleep)

        flush_batch(output_path, ckpt, lock, batch_records, batch_indices)
        print(f"[worker {worker_id}] finished. processed={processed}", flush=True)

    except KeyboardInterrupt:
        flush_batch(output_path, ckpt, lock, batch_records, batch_indices)
        print(f"[worker {worker_id}] interrupted. flushed remaining buffered items.", flush=True)


def main() -> None:
    args = parse_args()

    if args.workers != 4:
        print("Warning: your requested logic is designed for 4 workers; continuing anyway.", flush=True)

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Compatibility with old sequential script:
    # the existing output is treated as a contiguous processed prefix.
    base_skip = count_lines(output_path) if (args.resume and output_path.exists()) else 0

    print(f"Input:        {input_path}")
    print(f"Output:       {output_path}")
    print(f"Model:        {args.model}")
    print(f"Workers:      {args.workers}")
    print(f"Batch writes: {args.write_batch_size}")
    print(f"Resume:       {args.resume}")
    print(f"Base skip:    {base_skip} existing contiguous lines")
    print()

    file_lock = Lock()
    processes: list[Process] = []

    for worker_id in range(args.workers):
        p = Process(
            target=worker_main,
            args=(worker_id, args, base_skip, file_lock),
            daemon=False,
        )
        p.start()
        processes.append(p)

    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("\nMain process interrupted. Terminating workers...", flush=True)
        for p in processes:
            if p.is_alive():
                p.terminate()
        for p in processes:
            p.join()

    print("All workers finished.", flush=True)


if __name__ == "__main__":
    main()