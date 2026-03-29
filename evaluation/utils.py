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
DEFAULT_CLASSIFIER_MODEL_NAME = "helinivan/english-sarcasm-detector"


def get_source_style(label: int) -> str:
    return "sarcastic" if int(label) == 1 else "non-sarcastic"


def get_target_style(label: int) -> str:
    return "non-sarcastic" if int(label) == 1 else "sarcastic"


def get_target_publication(label: int) -> str:
    return "HuffPost" if int(label) == 1 else "The Onion"


def build_prompt(headline: str, label: int, architecture: str) -> str:
    """Build a rewrite prompt tuned to the model architecture."""
    headline = str(headline).strip()
    if architecture not in {"seq2seq", "causal"}:
        raise ValueError(
            f"Unsupported architecture {architecture!r}. Use 'seq2seq' or 'causal'."
        )

    if architecture == "seq2seq":
        if int(label) == 1:
            return f"Rewrite this headline as a HuffPost-style non-sarcastic headline: {headline}"
        return f"Rewrite this headline as an Onion-style sarcastic headline: {headline}"

    if int(label) == 1:
        return (
            "Task: Rewrite the news headline in HuffPost style.\n\n"
            "Rules:\n"
            "- Keep the same meaning.\n"
            "- Output exactly one rewritten headline.\n"
            "- Do not explain.\n"
            "- Do not repeat the input.\n"
            "- Do not output True or False.\n\n"
            f"Input headline: {headline}\n"
            "Rewritten headline:"
        )

    return (
        "Task: Rewrite the news headline in Onion style.\n\n"
        "Rules:\n"
        "- Keep the same meaning.\n"
        "- Output exactly one rewritten headline.\n"
        "- Do not explain.\n"
        "- Do not repeat the input.\n"
        "- Do not output True or False.\n\n"
        f"Input headline: {headline}\n"
        "Rewritten headline:"
    )


def clean_generation(text: str) -> str:
    """Normalize generated text into a single clean headline string."""
    cleaned = " ".join(str(text).strip().split())
    lowered = cleaned.lower()

    markers = (
        "rewritten headline:",
        "headline:",
        "output:",
        "answer:",
    )
    best_index = -1
    best_marker = ""
    for marker in markers:
        marker_index = lowered.rfind(marker)
        if marker_index > best_index:
            best_index = marker_index
            best_marker = marker

    if best_index != -1:
        cleaned = cleaned[best_index + len(best_marker) :].strip()

    return " ".join(cleaned.split())


def preprocess_for_classifier(text: str) -> str:
    import string

    return str(text).lower().translate(str.maketrans("", "", string.punctuation)).strip()


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
    return df.reset_index(drop=True)


def sample_dataset(
    dataset_df: pd.DataFrame,
    sample_fraction: float,
    seed: int,
) -> pd.DataFrame:
    """Return a deterministic sample of the dataset for benchmarking."""
    if not 0 < sample_fraction <= 1:
        raise ValueError("`sample_fraction` must be in the interval (0, 1].")
    if sample_fraction == 1:
        return dataset_df.reset_index(drop=True)

    sample_size = max(1, math.ceil(len(dataset_df) * sample_fraction))
    sampled_df = dataset_df.sample(n=sample_size, random_state=seed)
    return sampled_df.reset_index(drop=True)


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


def load_sarcasm_classifier(
    model_name: str,
    device: str,
) -> tuple["PreTrainedTokenizerBase", "PreTrainedModel"]:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return tokenizer, model


def predict_sarcasm_labels(
    texts: list[str],
    tokenizer: "PreTrainedTokenizerBase",
    model: "PreTrainedModel",
    device: str,
    batch_size: int,
    max_length: int,
) -> tuple[list[int], list[float], list[float]]:
    import torch
    from tqdm.auto import tqdm

    predictions: list[int] = []
    confidences: list[float] = []
    sarcastic_probabilities: list[float] = []

    for start_idx in tqdm(range(0, len(texts), batch_size), desc="Classifier inference", leave=False):
        batch_texts = [preprocess_for_classifier(text) for text in texts[start_idx : start_idx + batch_size]]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}

        with torch.inference_mode():
            logits = model(**encoded).logits
            probabilities = torch.softmax(logits, dim=-1)

        batch_confidences, batch_predictions = probabilities.max(dim=-1)
        predictions.extend(batch_predictions.detach().cpu().tolist())
        confidences.extend(batch_confidences.detach().cpu().tolist())
        sarcastic_probabilities.extend(probabilities[:, 1].detach().cpu().tolist())

    return predictions, confidences, sarcastic_probabilities


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
        prompts = [
            build_prompt(
                headline=headline,
                label=label,
                architecture=model_spec.architecture,
            )
            for headline, label in zip(
                dataset_df["headline"].tolist(),
                dataset_df["is_sarcastic"].tolist(),
            )
        ]

        generated_headlines = generate_rewrites(
            prompts=prompts,
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
    classifier_tokenizer: "PreTrainedTokenizerBase" | None = None,
    classifier_model: "PreTrainedModel" | None = None,
    classifier_device: str | None = None,
    classifier_batch_size: int = 64,
    classifier_max_length: int = 256,
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
    scored_df["classifier_predicted_label"] = math.nan
    scored_df["classifier_confidence"] = math.nan
    scored_df["classifier_sarcastic_probability"] = math.nan
    scored_df["expected_rewrite_label"] = 1 - scored_df["is_sarcastic"].astype(int)
    scored_df["classifier_correct"] = False
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

        if classifier_tokenizer is not None and classifier_model is not None:
            if classifier_device is None:
                raise ValueError("`classifier_device` must be provided when classifier scoring is enabled.")

            predictions, confidences, sarcastic_probabilities = predict_sarcasm_labels(
                texts=valid_df["generated_headline"].tolist(),
                tokenizer=classifier_tokenizer,
                model=classifier_model,
                device=classifier_device,
                batch_size=classifier_batch_size,
                max_length=classifier_max_length,
            )

            valid_df["classifier_predicted_label"] = predictions
            valid_df["classifier_confidence"] = confidences
            valid_df["classifier_sarcastic_probability"] = sarcastic_probabilities
            valid_df["classifier_correct"] = (
                valid_df["classifier_predicted_label"].astype(int)
                == valid_df["expected_rewrite_label"].astype(int)
            )

            scored_df.loc[
                valid_mask,
                [
                    "classifier_predicted_label",
                    "classifier_confidence",
                    "classifier_sarcastic_probability",
                    "classifier_correct",
                ],
            ] = valid_df[
                [
                    "classifier_predicted_label",
                    "classifier_confidence",
                    "classifier_sarcastic_probability",
                    "classifier_correct",
                ]
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
            classifier_accuracy=("classifier_correct", "mean"),
            mean_cosine_similarity=("cosine_similarity", "mean"),
            median_cosine_similarity=("cosine_similarity", "median"),
            mean_perplexity=("perplexity", "mean"),
            median_perplexity=("perplexity", "median"),
        )
        .sort_values(["mean_cosine_similarity", "mean_perplexity"], ascending=[False, True])
        .reset_index(drop=True)
    )
    return summary_df
