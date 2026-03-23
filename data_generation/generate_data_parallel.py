#!/usr/bin/env python3
"""
Parallel headline rewriting with 4 processes.

Features:
- Explicit task direction:
    * non_sarcastic_to_sarcastic
    * sarcastic_to_non_sarcastic
- Worker k processes records where (line_index % workers == k)
- Uses OpenAI Responses API with model gpt-5-mini by default
- Buffers writes and flushes in batches
- Uses a multiprocessing lock for safe shared-file writes
- Compatible with an existing sequential output file:
    assumes existing output lines correspond to the first N input lines
- Adds per-worker checkpoint files for safe resume

Input format:
    JSONL (one JSON object per line)

Output format:
    Same JSONL with added key:
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


TASK_NON_TO_SARC = "non_sarcastic_to_sarcastic"
TASK_SARC_TO_NON = "sarcastic_to_non_sarcastic"


SYSTEM_PROMPT_NON_TO_SARC = """You are an elite satirical headline writer.

Your task is to rewrite a factual, non-sarcastic news-style headline into a sarcastic, satirical headline that matches the style of Onion-like deadpan satire.

Target style requirements:
- Preserve the same core topic, entities, and general event meaning whenever possible.
- Output exactly one headline, not multiple options.
- Keep it concise and headline-like.
- Keep the output roughly similar in length to the original.
- Do not explain your reasoning.
- The flipped style must be very clear and match the target publication style.

OUTPUT:
- Output exactly one headline.
- Do not explain your reasoning.
"""

USER_TEMPLATE_NON_TO_SARC = """You are converting a non-sarcastic headline into a sarcastic headline.

The sarcastic headline should match these common patterns from the target sarcastic dataset:

sarcastic_patterns:
1. Archetype subject + deadpan predicate: headlines often start with generic roles like "area man," "local woman," "mom," or "dad," then attach a very specific, humiliating, or trivial action.
2. Pseudo-report framing: many headlines mimic hard-news shells such as "report:", "study finds", "poll finds", or "experts say" while delivering absurdly banal or ridiculous conclusions.
3. Serious institution + ridiculous mission: governments, agencies, corporations, or departments are described performing actions that are technically grammatical but absurd in purpose.
4. Register clash: formal bureaucratic, scientific, or journalistic wording is applied to tiny domestic annoyances, pop-culture nonsense, or childish behavior.
5. Late-twist structure: the headline reads plausibly at first, then the final noun phrase or clause delivers the absurd pivot.
6. Humanization of nonhuman things: animals, objects, body parts, or abstract systems are given motives, emotions, self-awareness, or social intentions.
7. Hyper-specific fake precision: exact percentages, ages, durations, quantities, and measurements are used to make nonsense sound statistically grounded.
8. Norm inversion: the headline reverses expected moral or institutional logic so the "official" or "reasonable" response is obviously wrong.
9. Understatement/intensifier mix: words like "just," "only," "pretty close," "nearly," "finally," and "even" are used to minimize or casually normalize absurdity.
10. Compound-heavy phrasing: sarcastic headlines frequently pack in hyphenated modifiers, stacked descriptors, or over-elaborate noun phrases to heighten the deadpan tone.

Rules for this rewrite:
- Convert the headline into a sarcastic Onion-style deadpan satirical headline.
- Preserve the same core topic, entities, and general event whenever possible.
- Do not use cheap sarcasm markers.
- Do not explain.
- Output exactly one JSON object with one key.

Original headline: {headline}
Original label: {label}
Target style: sarcastic Onion-style deadpan satirical headline

Return:
{{
  "reversed_headline": "..."
}}
"""


SYSTEM_PROMPT_SARC_TO_NON = """You are an expert mainstream headline writer.

Your task is to rewrite a sarcastic satirical headline into a straightforward, non-sarcastic headline.

Target style requirements:
- Preserve the same core topic, entities, and general event meaning whenever possible.
- Output exactly one headline, not multiple options.
- Do not explain your reasoning.
- The flipped style must be very clear and match the target publication style.

5. OUTPUT:
- Output exactly one headline.
- Do not explain your reasoning.
"""

USER_TEMPLATE_SARC_TO_NON = """You are converting a sarcastic headline into a non-sarcastic headline.

Goal:
Produce a headline that sounds like a real, sincere, mainstream digital-news or feature headline, similar to the non-sarcastic examples in the dataset.

Style requirements:
- preserve the core topic, entity, or issue from the input headline
- remove sarcasm, irony, mockery, absurdity, and impossible claims
- do not anthropomorphize objects, animals, places, or abstract ideas unless it is literally plausible
- do not use onion-style archetypes like "area man", "area woman", "local man", "nation", "god", etc.
- do not use fake-journalistic irony such as "report:", "study finds", or "poll finds" unless the input clearly refers to a real report or study
- rewrite into a plausible non-sarcastic headline style common in the dataset:
  1) straight news report
  2) explainer ("how", "why", "what")
  3) service/advice headline
  4) human-interest/profile headline
  5) issue/commentary headline
- sound informative, earnest, and realistic
- keep the headline concise and natural
- use lowercase headline style
- do not add background explanation
- output only the rewritten headline

Transformation rules:
- if the sarcastic headline contains an absurd action, replace it with the real-world issue implied by the joke
- if the sarcastic headline mocks a person type, rewrite it around the actual behavior, social issue, or event
- if the sarcastic headline uses exaggeration, reduce it to a plausible factual claim
- if the sarcastic headline is built around an impossible premise, infer the closest realistic newsworthy topic and write that instead
- if several non-sarcastic styles are possible, choose the one that sounds most like a realistic HuffPost-style headline

Examples of the target style:
- explanatory: "how to raise kids who can 'love and be loved'"
- service/listicle: "5 ways to file your taxes with less stress"
- news report: "elizabeth warren's pick wins ohio's democratic gubernatorial primary"
- human-interest: "watch prince harry and rihanna get tested for hiv together"
- issue framing: "this video nails the messed up way anti-abortion legislation gets pushed"

Original headline: {headline}
Original label: {label}
Target style: non-sarcastic HuffPost-style headline

Return:
{{
  "reversed_headline": "..."
}}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to input JSONL file")
    parser.add_argument("--output", type=str, required=True, help="Path to output JSONL file")
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=[TASK_NON_TO_SARC, TASK_SARC_TO_NON],
        help="Generation direction",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4.1-mini",
        help="OpenAI model name",
    )
    parser.add_argument("--workers", type=int, default=16, help="Number of worker processes")
    parser.add_argument(
        "--write-batch-size",
        type=int,
        default=10,
        help="How many completed items each worker buffers before writing",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=160,
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


def get_prompts(task: str) -> tuple[str, str]:
    if task == TASK_NON_TO_SARC:
        return SYSTEM_PROMPT_NON_TO_SARC, USER_TEMPLATE_NON_TO_SARC
    if task == TASK_SARC_TO_NON:
        return SYSTEM_PROMPT_SARC_TO_NON, USER_TEMPLATE_SARC_TO_NON
    raise ValueError(f"Unsupported task: {task}")


def validate_label_for_task(label: int, task: str) -> bool:
    if task == TASK_NON_TO_SARC:
        return label == 0
    if task == TASK_SARC_TO_NON:
        return label == 1
    return False


def call_model(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_template: str,
    headline: str,
    label: int,
    max_retries: int,
) -> str:
    last_err: Exception | None = None

    for attempt in range(max_retries):
        try:
            response = client.responses.create(
                model=model,
                instructions=system_prompt,
                input=user_template.format(headline=headline, label=label),
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
    system_prompt, user_template = get_prompts(args.task)

    input_path = Path(args.input)
    output_path = Path(args.output)
    ckpt = checkpoint_path(output_path, worker_id)

    done_indices = load_done_indices(ckpt) if args.resume else set()

    batch_records: list[dict[str, Any]] = []
    batch_indices: list[int] = []
    processed = 0
    skipped_wrong_label = 0

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
                    print(
                        f"[worker {worker_id}] [line {idx}] invalid JSON: {e}",
                        file=sys.stderr,
                        flush=True,
                    )
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

                if not validate_label_for_task(label, args.task):
                    skipped_wrong_label += 1
                    continue

                reversed_headline = call_model(
                    client=client,
                    model=args.model,
                    system_prompt=system_prompt,
                    user_template=user_template,
                    headline=headline,
                    label=label,
                    max_retries=args.max_retries,
                )

                print(
                    f"[done] worker={worker_id} line={idx} label={label} "
                    f"orig={headline!r} -> flipped={reversed_headline!r}",
                    flush=True,
                )

                record["reversed_headline"] = reversed_headline
                batch_records.append(record)
                batch_indices.append(idx)
                processed += 1

                if processed % 25 == 0:
                    print(
                        f"[worker {worker_id}] processed={processed} "
                        f"skipped_wrong_label={skipped_wrong_label} "
                        f"last_input_index={idx}",
                        flush=True,
                    )

                if len(batch_records) >= args.write_batch_size:
                    flush_batch(output_path, ckpt, lock, batch_records, batch_indices)
                    batch_records.clear()
                    batch_indices.clear()

                if args.sleep > 0:
                    time.sleep(args.sleep)

        flush_batch(output_path, ckpt, lock, batch_records, batch_indices)
        print(
            f"[worker {worker_id}] finished. processed={processed} skipped_wrong_label={skipped_wrong_label}",
            flush=True,
        )

    except KeyboardInterrupt:
        flush_batch(output_path, ckpt, lock, batch_records, batch_indices)
        print(
            f"[worker {worker_id}] interrupted. flushed remaining buffered items.",
            flush=True,
        )


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Compatibility with old sequential script:
    # existing output is treated as a contiguous processed prefix.
    base_skip = count_lines(output_path) if (args.resume and output_path.exists()) else 0

    print(f"Input:        {input_path}")
    print(f"Output:       {output_path}")
    print(f"Task:         {args.task}")
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