# Metrics API Docs

This document describes the public metric-related methods available in the `evaluation` folder.

## Imports

```python
from text_perplexity import perplexity_score, batch_perplexity
from text_similarity import cosine_similarity_score, batch_cosine_similarity
from utils import get_test_set
```

Note: if you refer to the similarity functions as `consine_similarity_score` or `batch_consine_similarity` in notes, the actual function names in code are:

- `cosine_similarity_score`
- `batch_cosine_similarity`

## `perplexity_score`

**Module:** `text_perplexity.py`

**Signature**

```python
perplexity_score(
    text: str,
    model_name: str = DEFAULT_MODEL_NAME,
    device: str | None = None,
) -> float
```

**Purpose**

Returns GPT-2 perplexity for a single sentence. Lower is better.

**Parameters**

- `text`: input sentence to score.
- `model_name`: causal language model used for perplexity. Default is `openai-community/gpt2-medium`.
- `device`: execution device. Use `"cuda"` or `"cpu"`. If `None`, the function auto-selects CUDA when available.

**Returns**

- A single `float` perplexity score.

**Raises**

- `ImportError` if `torch` or `transformers` is missing.
- `ValueError` if the input text is empty after stripping whitespace.

**Example**

```python
score = perplexity_score("This is a fluent sentence.")
print(score)
```

## `batch_perplexity`

**Module:** `text_perplexity.py`

**Signature**

```python
batch_perplexity(
    texts: Sequence[str],
    model_name: str = DEFAULT_MODEL_NAME,
    device: str | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[float]
```

**Purpose**

Returns GPT-2 perplexity scores for a batch of sentences. Lower is better.

**Parameters**

- `texts`: list or sequence of sentences.
- `model_name`: causal language model used for perplexity. Default is `openai-community/gpt2-medium`.
- `device`: execution device. Use `"cuda"` or `"cpu"`. If `None`, the function auto-selects CUDA when available.
- `batch_size`: number of texts processed per forward pass. Must be greater than `0`.

**Returns**

- A `list[float]` aligned with the input order.

**Raises**

- `ImportError` if `torch` or `transformers` is missing.
- `ValueError` if `texts` is empty, contains empty strings, or `batch_size <= 0`.

**Example**

```python
scores = batch_perplexity(
    ["This is fluent.", "Sentence awkward very."],
    batch_size=2,
)
print(scores)
```

## `cosine_similarity_score`

**Module:** `text_similarity.py`

**Signature**

```python
cosine_similarity_score(
    source_sentence: str,
    rewritten_sentence: str,
    model_name: str = DEFAULT_MODEL_NAME,
) -> float
```

**Purpose**

Returns semantic cosine similarity between two sentences using sentence embeddings. Higher is better.

**Parameters**

- `source_sentence`: original sentence.
- `rewritten_sentence`: rewritten sentence to compare against the original.
- `model_name`: sentence-transformer model name. Default is `sentence-transformers/all-mpnet-base-v2`.

**Returns**

- A `float` in the range `[-1.0, 1.0]`.

**Raises**

- `ImportError` if `sentence-transformers` is missing.
- `ValueError` if either sentence is empty after stripping whitespace.

**Example**

```python
score = cosine_similarity_score(
    "The sky is blue.",
    "The sky has a blue color.",
)
print(score)
```

## `batch_cosine_similarity`

**Module:** `text_similarity.py`

**Signature**

```python
batch_cosine_similarity(
    source_sentences: Sequence[str],
    rewritten_sentences: Sequence[str],
    model_name: str = DEFAULT_MODEL_NAME,
) -> list[float]
```

**Purpose**

Returns cosine similarity scores for aligned sentence pairs. Higher is better.

**Parameters**

- `source_sentences`: list of original sentences.
- `rewritten_sentences`: list of rewritten sentences. Must have the same length and order as `source_sentences`.
- `model_name`: sentence-transformer model name. Default is `sentence-transformers/all-mpnet-base-v2`.

**Returns**

- A `list[float]` aligned pairwise with the two input sequences.

**Raises**

- `ImportError` if `sentence-transformers` is missing.
- `ValueError` if the input lists are empty or have different lengths.

**Example**

```python
scores = batch_cosine_similarity(
    ["A cat sits on the mat.", "Stocks fell today."],
    ["A cat is sitting on the mat.", "The market dropped today."],
)
print(scores)
```

## `get_test_set`

**Module:** `utils.py`

**Signature**

```python
get_test_set(
    dataset: pd.DataFrame,
    test_fraction: float = DEFAULT_TEST_FRACTION,
    label_column: str = DEFAULT_LABEL_COLUMN,
) -> pd.DataFrame
```

**Purpose**

Builds a deterministic, class-balanced test set from the full sarcasm dataset.

**Behavior**

- Uses `test_fraction` of the full dataset.
- Enforces equal numbers of label `0` and label `1`.
- If the requested test size is odd, rounds down to the nearest even number.
- Uses a stable row hash, so repeated calls on the same dataset return the same test set.

For `Sarcasm_Headlines_Dataset_v2.json` with `28,619` rows and `test_fraction=0.10`, the returned test set size is `2,860` rows:

- `1,430` rows with label `0`
- `1,430` rows with label `1`

**Parameters**

- `dataset`: full dataset as a pandas DataFrame.
- `test_fraction`: fraction of rows to keep for the test set.
- `label_column`: column containing binary sarcasm labels. Default is `is_sarcastic`.

**Returns**

- A new `pd.DataFrame` containing the deterministic balanced test split.

**Raises**

- `ValueError` if the dataset is empty.
- `ValueError` if `test_fraction` is not in `(0, 1]`.
- `ValueError` if `label_column` is missing.
- `ValueError` if either class does not have enough rows to satisfy the balanced split.

**Example**

```python
import pandas as pd
from utils import get_test_set

df = pd.read_json("Sarcasm_Headlines_Dataset_v2.json", lines=True)
test_df = get_test_set(df, test_fraction=0.10)

print(len(test_df))
print(test_df["is_sarcastic"].value_counts().sort_index())
```

## Practical Notes

- `perplexity_score` and `batch_perplexity`: lower is better.
- `cosine_similarity_score` and `batch_cosine_similarity`: higher is better.
- `get_test_set` is for deterministic evaluation set construction, not for random experimentation.
