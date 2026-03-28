"""Utility helpers for benchmark generation and evaluation."""

from __future__ import annotations

import gc
import math
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import pandas as pd

if TYPE_CHECKING:
    import torch
    from transformers import PreTrainedModel, PreTrainedTokenizerBase


DEFAULT_LABEL_COLUMN = "is_sarcastic"
DEFAULT_HEADLINE_COLUMN = "headline"


def get_source_style(label: int) -> str:
    return "sarcastic" if int(label) == 1 else "non-sarcastic"


def get_target_style(label: int) -> str:
    return "non-sarcastic" if int(label) == 1 else "sarcastic"


def get_target_publication(label: int) -> str:
    return "HuffPost" if int(label) == 1 else "The Onion"


def build_prompt(headline: str, label: int) -> str:
    """Build a rewrite prompt that strongly steers toward the target style."""
    headline = str(headline).strip()
    if int(label) == 1:
        return (
            "Rewrite the following headline as a straightforward HuffPost-style "
            "non-sarcastic headline. Preserve the same core meaning, entities, "
            "and event details. Output only the rewritten headline.\n"
            f"Original headline: {headline}\n"
            "Rewritten headline:"
        )

    return (
        "Rewrite the following headline as an Onion-style sarcastic headline. "
        "Preserve the same core meaning, entities, and event details. Output "
        "only the rewritten headline.\n"
        f"Original headline: {headline}\n"
        "Rewritten headline:"
    )


def clean_generation(text: str) -> str:
    """Normalize generated text into a single clean headline string."""
    cleaned = " ".join(str(text).strip().split())
    lowered = cleaned.lower()
    for prefix in (
        "rewritten headline:",
        "headline:",
        "output:",
        "answer:",
    ):
        if lowered.startswith(prefix):
            cleaned = cleaned[len(prefix) :].strip()
            lowered = cleaned.lower()
    return cleaned


def load_dataset(
    dataset_path: Path,
    label_column: str = DEFAULT_LABEL_COLUMN,
    headline_column: str = DEFAULT_HEADLINE_COLUMN,
) -> pd.DataFrame:
    """Load an uploaded benchmark dataset and build prompts from its headlines."""
    try:
        df = pd.read_json(dataset_path, lines=True)
    except ValueError:
        df = pd.read_json(dataset_path)

    required_columns = {headline_column, label_column}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    df = df[[headline_column, label_column]].copy()
    df[headline_column] = df[headline_column].astype(str).str.strip()
    df = df[df[headline_column] != ""].copy()
    df[label_column] = df[label_column].astype(int)

    invalid_labels = set(df[label_column].unique()) - {0, 1}
    if invalid_labels:
        raise ValueError(
            f"{label_column!r} must contain only 0/1 values. Found: {sorted(invalid_labels)}"
        )

    df["source_style"] = df[label_column].map(get_source_style)
    df["target_style"] = df[label_column].map(get_target_style)
    df["target_publication"] = df[label_column].map(get_target_publication)
    df["prompt"] = df.apply(
        lambda row: build_prompt(row[headline_column], row[label_column]),
        axis=1,
    )
    return df.reset_index(drop=True)


def load_generation_model(
    model_name: str,
    architecture: str,
    device: str,
    use_fp16_on_gpu: bool = True,
) -> tuple["PreTrainedTokenizerBase", "PreTrainedModel"]:
    """Load either a seq2seq or decoder-only generation model."""
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
    )

    if architecture not in {"seq2seq", "causal"}:
        raise ValueError(
            f"Unsupported architecture {architecture!r}. Use 'seq2seq' or 'causal'."
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if architecture == "causal":
        tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {}
    if device == "cuda" and use_fp16_on_gpu:
        model_kwargs["torch_dtype"] = torch.float16

    model_cls = AutoModelForSeq2SeqLM if architecture == "seq2seq" else AutoModelForCausalLM
    model = model_cls.from_pretrained(model_name, **model_kwargs)
    if tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
        if getattr(model.generation_config, "pad_token_id", None) is None:
            model.generation_config.pad_token_id = tokenizer.pad_token_id

    model.to(device)
    model.eval()
    return tokenizer, model


def _decode_causal_outputs(
    generated_ids: "torch.Tensor",
    input_length: int,
    tokenizer: "PreTrainedTokenizerBase",
) -> list[str]:
    decoded: list[str] = []

    generated_ids = generated_ids.detach().cpu()
    for sequence_ids in generated_ids:
        continuation_ids = sequence_ids[input_length:]
        decoded.append(tokenizer.decode(continuation_ids, skip_special_tokens=True))
    return decoded


def generate_rewrites(
    prompts: list[str],
    tokenizer: "PreTrainedTokenizerBase",
    model: "PreTrainedModel",
    device: str,
    architecture: str,
    batch_size: int,
    max_source_length: int,
    generation_kwargs: dict,
) -> list[str]:
    import torch
    from tqdm.auto import tqdm

    outputs: list[str] = []
    progress_bar = tqdm(range(0, len(prompts), batch_size), desc="Generating", leave=False)

    with torch.inference_mode():
        for start_idx in progress_bar:
            prompt_batch = prompts[start_idx : start_idx + batch_size]
            encoded = tokenizer(
                prompt_batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_source_length,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}

            generated_ids = model.generate(
                **encoded,
                pad_token_id=tokenizer.pad_token_id,
                **generation_kwargs,
            )

            if architecture == "causal":
                decoded = _decode_causal_outputs(
                    generated_ids=generated_ids,
                    input_length=encoded["input_ids"].shape[1],
                    tokenizer=tokenizer,
                )
            else:
                decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            outputs.extend(clean_generation(text) for text in decoded)

    return outputs


class ModelSpecLike(Protocol):
    key: str
    label: str
    hf_name: str
    architecture: str


def run_generation_for_model(
    dataset_df: pd.DataFrame,
    model_spec: ModelSpecLike,
    output_dir: Path,
    run_name: str,
    device: str,
    batch_size: int,
    max_source_length: int,
    generation_kwargs: dict,
    use_fp16_on_gpu: bool = True,
    force_regenerate: bool = False,
) -> pd.DataFrame:
    generation_path = output_dir / f"{run_name}_{model_spec.key}_generations.csv"
    if generation_path.exists() and not force_regenerate:
        cached_df = pd.read_csv(generation_path)
        if len(cached_df) == len(dataset_df):
            print(f"Loading cached generations from {generation_path.name}")
            return cached_df
        print(f"Cached file {generation_path.name} has a different row count. Regenerating it.")

    tokenizer, model = load_generation_model(
        model_name=model_spec.hf_name,
        architecture=model_spec.architecture,
        device=device,
        use_fp16_on_gpu=use_fp16_on_gpu,
    )

    try:
        generated_headlines = generate_rewrites(
            prompts=dataset_df["prompt"].tolist(),
            tokenizer=tokenizer,
            model=model,
            device=device,
            architecture=model_spec.architecture,
            batch_size=batch_size,
            max_source_length=max_source_length,
            generation_kwargs=generation_kwargs,
        )
    finally:
        try:
            import torch
        except ImportError:
            torch = None

        del model
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    results_df = dataset_df[
        ["headline", "is_sarcastic", "source_style", "target_style", "target_publication"]
    ].copy()
    results_df["model_key"] = model_spec.key
    results_df["model_label"] = model_spec.label
    results_df["model_name"] = model_spec.hf_name
    results_df["model_architecture"] = model_spec.architecture
    results_df["generated_headline"] = pd.Series(generated_headlines, dtype="string")
    results_df["generated_headline"] = results_df["generated_headline"].fillna("").astype(str)
    results_df.to_csv(generation_path, index=False)
    print(f"Saved generations to {generation_path}")
    return results_df


def evaluate_generations(
    generated_df: pd.DataFrame,
    output_dir: Path,
    run_name: str,
    perplexity_batch_size: int,
    force_rescore: bool = False,
) -> pd.DataFrame:
    from evaluation.text_perplexity import batch_perplexity
    from evaluation.text_similarity import batch_cosine_similarity

    model_key = generated_df["model_key"].iloc[0]
    metrics_path = output_dir / f"{run_name}_{model_key}_metrics.csv"
    if metrics_path.exists() and not force_rescore:
        cached_df = pd.read_csv(metrics_path)
        if len(cached_df) == len(generated_df):
            print(f"Loading cached metrics from {metrics_path.name}")
            return cached_df
        print(f"Cached file {metrics_path.name} has a different row count. Rescoring it.")

    scored_df = generated_df.copy()
    scored_df["generated_headline"] = scored_df["generated_headline"].fillna("").astype(str)
    valid_mask = scored_df["generated_headline"].str.strip().ne("")

    scored_df["cosine_similarity"] = math.nan
    scored_df["perplexity"] = math.nan
    scored_df["empty_output"] = ~valid_mask
    scored_df["rewrite_changed"] = (
        scored_df["headline"].str.strip().str.lower()
        != scored_df["generated_headline"].str.strip().str.lower()
    )

    if valid_mask.any():
        valid_df = scored_df.loc[valid_mask].copy()
        valid_df["cosine_similarity"] = batch_cosine_similarity(
            valid_df["headline"].tolist(),
            valid_df["generated_headline"].tolist(),
        )
        valid_df["perplexity"] = batch_perplexity(
            valid_df["generated_headline"].tolist(),
            batch_size=perplexity_batch_size,
        )
        scored_df.loc[valid_mask, ["cosine_similarity", "perplexity"]] = valid_df[
            ["cosine_similarity", "perplexity"]
        ].to_numpy()

    scored_df.to_csv(metrics_path, index=False)
    print(f"Saved metrics to {metrics_path}")
    return scored_df


def summarise_results(results_df: pd.DataFrame) -> pd.DataFrame:
    summary_df = (
        results_df.groupby(
            ["model_key", "model_label", "model_name", "model_architecture"],
            as_index=False,
        )
        .agg(
            num_examples=("headline", "size"),
            non_empty_rate=("empty_output", lambda s: 1.0 - float(s.mean())),
            changed_rate=("rewrite_changed", "mean"),
            mean_cosine_similarity=("cosine_similarity", "mean"),
            median_cosine_similarity=("cosine_similarity", "median"),
            mean_perplexity=("perplexity", "mean"),
            median_perplexity=("perplexity", "median"),
        )
        .sort_values(["mean_cosine_similarity", "mean_perplexity"], ascending=[False, True])
        .reset_index(drop=True)
    )
    return summary_df
