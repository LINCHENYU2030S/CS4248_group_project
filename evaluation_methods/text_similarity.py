"""Utilities for semantic similarity scoring with sentence-transformers."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Sequence

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


DEFAULT_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"


@lru_cache(maxsize=1)
def load_embedding_model(model_name: str = DEFAULT_MODEL_NAME) -> "SentenceTransformer":
    """Load and cache the sentence-transformer model."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - dependency is environment-specific
        raise ImportError(
            "sentence-transformers is required for text similarity scoring. "
            "Install it with `pip install sentence-transformers`."
        ) from exc

    return SentenceTransformer(model_name)


def embed_sentences(
    sentences: Sequence[str],
    model_name: str = DEFAULT_MODEL_NAME,
) -> "np.ndarray":
    """Encode a sequence of sentences into L2-normalized embeddings."""
    if not sentences:
        raise ValueError("`sentences` must contain at least one sentence.")

    cleaned_sentences = [sentence.strip() for sentence in sentences]
    if any(not sentence for sentence in cleaned_sentences):
        raise ValueError("Sentences must be non-empty strings.")

    model = load_embedding_model(model_name)
    embeddings = model.encode(
        cleaned_sentences,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(embeddings, dtype=np.float32)


def cosine_similarity_score(
    source_sentence: str,
    rewritten_sentence: str,
    model_name: str = DEFAULT_MODEL_NAME,
) -> float:
    """Return cosine similarity between two sentences."""
    source_embedding, rewritten_embedding = embed_sentences(
        [source_sentence, rewritten_sentence],
        model_name=model_name,
    )
    similarity = float(np.dot(source_embedding, rewritten_embedding))
    return float(np.clip(similarity, -1.0, 1.0))


def batch_cosine_similarity(
    source_sentences: Sequence[str],
    rewritten_sentences: Sequence[str],
    model_name: str = DEFAULT_MODEL_NAME,
) -> list[float]:
    """Return cosine similarities for aligned sentence pairs."""
    if len(source_sentences) != len(rewritten_sentences):
        raise ValueError("Input sentence lists must have the same length.")
    if not source_sentences:
        raise ValueError("Input sentence lists must not be empty.")

    embeddings = embed_sentences(
        [*source_sentences, *rewritten_sentences],
        model_name=model_name,
    )
    midpoint = len(source_sentences)
    source_embeddings = embeddings[:midpoint]
    rewritten_embeddings = embeddings[midpoint:]

    similarities = np.sum(source_embeddings * rewritten_embeddings, axis=1)
    similarities = np.clip(similarities, -1.0, 1.0)
    return similarities.astype(float).tolist()
