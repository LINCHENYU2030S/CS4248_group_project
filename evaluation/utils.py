"""Utility helpers for dataset selection and baseline evaluation."""

from __future__ import annotations

import gc
import hashlib
import math
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import pandas as pd

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase


DEFAULT_TEST_FRACTION = 0.10
DEFAULT_LABEL_COLUMN = "is_sarcastic"


def _normalise_value(value: object) -> str:
    if pd.isna(value):
        return "<NA>"
    return str(value)


def _stable_row_hash(row: pd.Series, columns: list[str]) -> str:
    signature = "||".join(
        f"{column}={_normalise_value(row[column])}"
        for column in columns
    )
    return hashlib.sha256(signature.encode("utf-8")).hexdigest()


def get_test_set(
    dataset: pd.DataFrame,
    test_fraction: float = DEFAULT_TEST_FRACTION,
    label_column: str = DEFAULT_LABEL_COLUMN,
) -> pd.DataFrame:
    """Return a deterministic, balanced test set from a sarcasm dataset.

    The returned split is approximately ``test_fraction`` of the full dataset and
    always contains the same number of label ``0`` and label ``1`` examples. If
    the requested test size is odd, the function rounds down to the nearest even
    number so the split can remain perfectly balanced.
    """

    if dataset.empty:
        raise ValueError("`dataset` must contain at least one row.")
    if not 0 < test_fraction <= 1:
        raise ValueError("`test_fraction` must be in the interval (0, 1].")
    if label_column not in dataset.columns:
        raise ValueError(f"Missing required label column: {label_column!r}")

    target_size = math.floor(len(dataset) * test_fraction)
    if target_size < 2:
        raise ValueError("The requested test split is too small to balance across two classes.")
    if target_size % 2 == 1:
        target_size -= 1

    per_class_size = target_size // 2
    label_counts = dataset[label_column].value_counts()
    for label in (0, 1):
        if int(label_counts.get(label, 0)) < per_class_size:
            raise ValueError(
                f"Not enough samples for label {label}: "
                f"required {per_class_size}, found {int(label_counts.get(label, 0))}."
            )

    hash_columns = sorted(dataset.columns.tolist())
    ranked_df = dataset.copy()
    ranked_df["_deterministic_hash"] = ranked_df.apply(
        lambda row: _stable_row_hash(row, hash_columns),
        axis=1,
    )

    selected_parts = []
    for label in (0, 1):
        label_df = ranked_df.loc[ranked_df[label_column] == label]
        label_df = label_df.sort_values("_deterministic_hash", kind="mergesort")
        selected_parts.append(label_df.head(per_class_size))

    test_df = pd.concat(selected_parts, ignore_index=True)
    test_df = test_df.sort_values(
        [label_column, "_deterministic_hash"],
        kind="mergesort",
    )
    test_df = test_df.drop(columns="_deterministic_hash").reset_index(drop=True)
    return test_df


def get_source_style(label: int) -> str:
    return "sarcastic" if int(label) == 1 else "non-sarcastic"


def get_target_style(label: int) -> str:
    return "non-sarcastic" if int(label) == 1 else "sarcastic"


def build_prompt(headline: str, label: int) -> str:
    source_style = get_source_style(label)
    target_style = get_target_style(label)
    return (
        f"Rewrite the following news headline so that it becomes {target_style} "
        f"while preserving the original meaning.\n"
        f"Original style: {source_style}\n"
        f"Headline: {headline}\n"
        f"Rewritten headline:"
    )


def clean_generation(text: str) -> str:
    cleaned = " ".join(str(text).strip().split())
    for prefix in ["rewritten headline:", "headline:"]:
        if cleaned.lower().startswith(prefix):
            cleaned = cleaned[len(prefix) :].strip()
    return cleaned


def load_dataset(
    dataset_path: Path,
    test_fraction: float = DEFAULT_TEST_FRACTION,
) -> pd.DataFrame:
    df = pd.read_json(dataset_path, lines=True)
    df = get_test_set(df, test_fraction=test_fraction)
    df = df[["headline", "is_sarcastic"]].copy()
    df["source_style"] = df["is_sarcastic"].map(get_source_style)
    df["target_style"] = df["is_sarcastic"].map(get_target_style)
    df["prompt"] = df.apply(lambda row: build_prompt(row["headline"], row["is_sarcastic"]), axis=1)
    return df


def load_generation_model(
    model_name: str,
    device: str,
    use_fp16_on_gpu: bool = True,
) -> tuple["PreTrainedTokenizerBase", "PreTrainedModel"]:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {}
    if device == "cuda" and use_fp16_on_gpu:
        model_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **model_kwargs)
    model.to(device)
    model.eval()
    return tokenizer, model


def generate_rewrites(
    prompts: list[str],
    tokenizer: "PreTrainedTokenizerBase",
    model: "PreTrainedModel",
    device: str,
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
            decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            outputs.extend(clean_generation(text) for text in decoded)

    return outputs


class ModelSpecLike(Protocol):
    key: str
    label: str
    hf_name: str


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
        device=device,
        use_fp16_on_gpu=use_fp16_on_gpu,
    )

    try:
        generated_headlines = generate_rewrites(
            prompts=dataset_df["prompt"].tolist(),
            tokenizer=tokenizer,
            model=model,
            device=device,
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

    results_df = dataset_df[["headline", "is_sarcastic", "source_style", "target_style"]].copy()
    results_df["model_key"] = model_spec.key
    results_df["model_label"] = model_spec.label
    results_df["model_name"] = model_spec.hf_name
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
    from text_perplexity import batch_perplexity
    from text_similarity import batch_cosine_similarity

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
        results_df.groupby(["model_key", "model_label", "model_name"], as_index=False)
        .agg(
            num_examples=("headline", "size"),
            non_empty_rate=("empty_output", lambda s: 1.0 - float(s.mean())),
            changed_rate=("rewrite_changed", "mean"),
            mean_cosine_similarity=("cosine_similarity", "mean"),
            median_cosine_similarity=("cosine_similarity", "median"),
            mean_perplexity=("perplexity", "mean"),
            median_perplexity=("perplexity", "median"),
        )
        .sort_values("mean_cosine_similarity", ascending=False)
        .reset_index(drop=True)
    )
    return summary_df
