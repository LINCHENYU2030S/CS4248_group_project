"""Utilities for fluency scoring with GPT-2 perplexity."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    import torch
    from transformers import PreTrainedModel, PreTrainedTokenizerBase


DEFAULT_MODEL_NAME = "openai-community/gpt2-medium"
DEFAULT_BATCH_SIZE = 8


def _resolve_device(device: str | None = None) -> str:
    if device is not None:
        return device

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - dependency is environment-specific
        raise ImportError(
            "torch is required for perplexity scoring. "
            "Install it with `pip install torch`."
        ) from exc

    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=None)
def load_perplexity_tokenizer(
    model_name: str = DEFAULT_MODEL_NAME,
) -> "PreTrainedTokenizerBase":
    """Load and cache the tokenizer used for perplexity scoring."""
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - dependency is environment-specific
        raise ImportError(
            "transformers is required for perplexity scoring. "
            "Install it with `pip install transformers`."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


@lru_cache(maxsize=None)
def load_perplexity_model(
    model_name: str = DEFAULT_MODEL_NAME,
    device: str | None = None,
) -> "PreTrainedModel":
    """Load and cache the GPT-2 language model used for perplexity scoring."""
    resolved_device = _resolve_device(device)

    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:  # pragma: no cover - dependency is environment-specific
        raise ImportError(
            "transformers is required for perplexity scoring. "
            "Install it with `pip install transformers`."
        ) from exc

    tokenizer = load_perplexity_tokenizer(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(resolved_device)
    model.eval()
    return model


def _validate_texts(texts: Sequence[str]) -> list[str]:
    if not texts:
        raise ValueError("`texts` must contain at least one sentence.")

    cleaned_texts = [text.strip() for text in texts]
    if any(not text for text in cleaned_texts):
        raise ValueError("Texts must be non-empty strings.")

    return cleaned_texts


def _encode_text_batch(
    texts: Sequence[str],
    tokenizer: "PreTrainedTokenizerBase",
    max_length: int,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    import torch

    encoded = tokenizer(
        list(texts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length - 1,
        add_special_tokens=False,
    )

    batch_size = encoded["input_ids"].size(0)
    bos_token_id = tokenizer.bos_token_id
    if bos_token_id is None:
        bos_token_id = tokenizer.eos_token_id
    if bos_token_id is None:
        raise ValueError("Tokenizer must define either a BOS token or an EOS token.")

    bos_tokens = torch.full((batch_size, 1), bos_token_id, dtype=encoded["input_ids"].dtype)
    bos_mask = torch.ones((batch_size, 1), dtype=encoded["attention_mask"].dtype)

    input_ids = torch.cat([bos_tokens, encoded["input_ids"]], dim=1)
    attention_mask = torch.cat([bos_mask, encoded["attention_mask"]], dim=1)
    return input_ids, attention_mask


def perplexity_score(
    text: str,
    model_name: str = DEFAULT_MODEL_NAME,
    device: str | None = None,
) -> float:
    """Return GPT-2 perplexity for a single sentence."""
    return batch_perplexity([text], model_name=model_name, device=device)[0]


def batch_perplexity(
    texts: Sequence[str],
    model_name: str = DEFAULT_MODEL_NAME,
    device: str | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[float]:
    """Return GPT-2 perplexity scores for a batch of sentences."""
    if batch_size <= 0:
        raise ValueError("`batch_size` must be a positive integer.")

    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:  # pragma: no cover - dependency is environment-specific
        raise ImportError(
            "torch is required for perplexity scoring. "
            "Install it with `pip install torch`."
        ) from exc

    cleaned_texts = _validate_texts(texts)
    tokenizer = load_perplexity_tokenizer(model_name)
    model = load_perplexity_model(model_name=model_name, device=device)
    model_device = next(model.parameters()).device
    max_length = int(model.config.n_positions)

    perplexities: list[float] = []

    for start_index in range(0, len(cleaned_texts), batch_size):
        batch_texts = cleaned_texts[start_index : start_index + batch_size]
        input_ids, attention_mask = _encode_text_batch(
            batch_texts,
            tokenizer=tokenizer,
            max_length=max_length,
        )
        input_ids = input_ids.to(model_device)
        attention_mask = attention_mask.to(model_device)

        with torch.inference_mode():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_mask = attention_mask[:, 1:].contiguous()
        shift_labels = shift_labels.masked_fill(shift_mask == 0, -100)

        token_losses = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
            ignore_index=-100,
        ).view(shift_labels.size())

        valid_token_counts = shift_mask.sum(dim=1)
        if torch.any(valid_token_counts == 0):
            raise ValueError("Each text must contain at least one token after tokenization.")

        sequence_nll = token_losses.sum(dim=1) / valid_token_counts
        sequence_perplexity = torch.exp(sequence_nll)
        perplexities.extend(sequence_perplexity.detach().cpu().tolist())

    return [float(score) for score in perplexities]
