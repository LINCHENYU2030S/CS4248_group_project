"""Utility helpers for deterministic dataset selection."""

from __future__ import annotations

import hashlib
import math

import pandas as pd


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
