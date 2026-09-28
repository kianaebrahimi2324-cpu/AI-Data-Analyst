from __future__ import annotations

import re
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import (
    CountVectorizer,
    ENGLISH_STOP_WORDS,
    TfidfVectorizer,
)


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

MAX_TEXT_ROWS = 5000

TOP_TERMS_LIMIT = 15
TOP_BIGRAMS_LIMIT = 10
TOP_TFIDF_LIMIT = 15

MIN_TOKEN_LENGTH = 2

VERY_LONG_TEXT_THRESHOLD = 500

DEFAULT_STOPWORD_MODE = "english"


# ------------------------------------------------------------
# Text normalization
# ------------------------------------------------------------

def _normalize_text(
    value: Any
) -> str:
    """Normalize whitespace without changing text meaning."""

    if pd.isna(value):
        return ""

    text = str(value)

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


# ------------------------------------------------------------
# Sampling
# ------------------------------------------------------------

def _prepare_all_texts(
    series: pd.Series
) -> pd.Series:
    """
    Normalize all non-empty text rows.

    Used for deterministic quality metrics.
    """

    texts = (
        series
        .dropna()
        .map(_normalize_text)
    )

    return texts[
        texts != ""
    ]


def _sample_texts(
    texts: pd.Series
) -> pd.Series:
    """
    Deterministically sample text for NLP analysis.

    Quality metrics are computed on the full text column,
    while expensive NLP statistics can use a sample.
    """

    if len(texts) <= MAX_TEXT_ROWS:
        return texts

    return texts.sample(
        n=MAX_TEXT_ROWS,
        random_state=42,
    )


# ------------------------------------------------------------
# Tokenization
# ------------------------------------------------------------

def _tokenize(
    text: str
) -> list[str]:
    """Basic Unicode-aware tokenizer."""

    tokens = re.findall(
        r"\b[\w'-]+\b",
        text.lower(),
        flags=re.UNICODE,
    )

    return [
        token
        for token in tokens
        if len(token) >= MIN_TOKEN_LENGTH
    ]


# ------------------------------------------------------------
# Stopwords
# ------------------------------------------------------------

def _get_stopwords(
    mode: str | None
) -> set[str]:
    """
    Return configured stopwords.

    English filtering is optional so the analyzer remains
    usable for non-English text.
    """

    if mode == "english":
        return set(
            ENGLISH_STOP_WORDS
        )

    return set()


def _content_tokens(
    text: str,
    stopword_mode: str | None,
) -> list[str]:
    """Return tokens after optional stopword removal."""

    stopwords = _get_stopwords(
        stopword_mode
    )

    return [
        token
        for token in _tokenize(text)
        if token not in stopwords
    ]


# ------------------------------------------------------------
# Length statistics
# ------------------------------------------------------------

def _calculate_length_statistics(
    texts: pd.Series
) -> dict[str, Any]:

    if len(texts) == 0:
        return {
            "average_characters": 0.0,
            "median_characters": 0.0,
            "min_characters": 0,
            "max_characters": 0,
            "average_words": 0.0,
            "median_words": 0.0,
            "min_words": 0,
            "max_words": 0,
        }

    character_lengths = (
        texts.str.len()
    )

    word_lengths = texts.map(
        lambda text:
            len(
                _tokenize(text)
            )
    )

    return {
        "average_characters":
            round(
                float(
                    character_lengths.mean()
                ),
                2,
            ),

        "median_characters":
            round(
                float(
                    character_lengths.median()
                ),
                2,
            ),

        "min_characters":
            int(
                character_lengths.min()
            ),

        "max_characters":
            int(
                character_lengths.max()
            ),

        "average_words":
            round(
                float(
                    word_lengths.mean()
                ),
                2,
            ),

        "median_words":
            round(
                float(
                    word_lengths.median()
                ),
                2,
            ),

        "min_words":
            int(
                word_lengths.min()
            ),

        "max_words":
            int(
                word_lengths.max()
            ),
    }


# ------------------------------------------------------------
# Lexical statistics
# ------------------------------------------------------------

def _lexical_statistics(
    texts: pd.Series,
    stopword_mode: str | None,
) -> dict[str, Any]:

    all_tokens = []

    content_tokens = []

    for text in texts:

        all_tokens.extend(
            _tokenize(text)
        )

        content_tokens.extend(
            _content_tokens(
                text,
                stopword_mode,
            )
        )

    unique_tokens = set(
        all_tokens
    )

    unique_content_tokens = set(
        content_tokens
    )

    return {
        "total_tokens":
            len(all_tokens),

        "unique_tokens":
            len(unique_tokens),

        "lexical_diversity":
            round(
                (
                    len(unique_tokens)
                    / len(all_tokens)
                )
                if all_tokens
                else 0.0,
                4,
            ),

        "content_tokens":
            len(content_tokens),

        "unique_content_tokens":
            len(
                unique_content_tokens
            ),
    }


# ------------------------------------------------------------
# Top raw terms
# ------------------------------------------------------------

def _top_raw_terms(
    texts: pd.Series
) -> list[dict[str, Any]]:

    counter = Counter()

    for text in texts:
        counter.update(
            _tokenize(text)
        )

    return [
        {
            "term": term,
            "count": int(count),
        }
        for term, count
        in counter.most_common(
            TOP_TERMS_LIMIT
        )
    ]


# ------------------------------------------------------------
# Top content terms
# ------------------------------------------------------------

def _top_content_terms(
    texts: pd.Series,
    stopword_mode: str | None,
) -> list[dict[str, Any]]:

    counter = Counter()

    for text in texts:

        counter.update(
            _content_tokens(
                text,
                stopword_mode,
            )
        )

    return [
        {
            "term": term,
            "count": int(count),
        }
        for term, count
        in counter.most_common(
            TOP_TERMS_LIMIT
        )
    ]


# ------------------------------------------------------------
# Top bigrams
# ------------------------------------------------------------

def _top_bigrams(
    texts: pd.Series,
    stopword_mode: str | None,
) -> list[dict[str, Any]]:
    """
    Extract frequent real adjacent two-word phrases.

    Stopwords are NOT removed before n-gram construction,
    because doing so can create artificial bigrams.

    Instead, real bigrams are created first and then
    low-information phrases are filtered.
    """

    if len(texts) < 2:
        return []

    stopwords = _get_stopwords(
        stopword_mode
    )

    try:

        vectorizer = CountVectorizer(
            ngram_range=(2, 2),
            max_features=3000,
            lowercase=True,
            stop_words=None,
            token_pattern=r"(?u)\b\w[\w'-]+\b",
        )

        matrix = (
            vectorizer
            .fit_transform(
                texts.tolist()
            )
        )

        frequencies = np.asarray(
            matrix.sum(axis=0)
        ).ravel()

        terms = (
            vectorizer
            .get_feature_names_out()
        )

        candidates = []

        for term, count in zip(
            terms,
            frequencies,
        ):

            tokens = (
                term.lower()
                .split()
            )

            # A valid bigram must still contain exactly
            # two real adjacent tokens.
            if len(tokens) != 2:
                continue

            # When stopword filtering is enabled,
            # exclude phrases containing stopwords.
            if stopwords and any(
                token in stopwords
                for token in tokens
            ):
                continue

            candidates.append(
                (
                    term,
                    int(count),
                )
            )

        ranked = sorted(
            candidates,
            key=lambda item:
                (
                    -item[1],
                    item[0],
                ),
        )

        return [
            {
                "bigram": term,
                "count": count,
            }
            for term, count
            in ranked[
                :TOP_BIGRAMS_LIMIT
            ]
        ]

    except ValueError:
        return []


# ------------------------------------------------------------
# TF-IDF keywords
# ------------------------------------------------------------

def _tfidf_keywords(
    texts: pd.Series,
    stopword_mode: str | None,
) -> list[dict[str, Any]]:

    if len(texts) < 2:
        return []

    stop_words = (
        "english"
        if stopword_mode == "english"
        else None
    )

    try:

        vectorizer = TfidfVectorizer(
            lowercase=True,
            max_features=3000,
            stop_words=stop_words,
            sublinear_tf=True,
            token_pattern=r"(?u)\b\w[\w'-]+\b",
        )

        matrix = (
            vectorizer
            .fit_transform(
                texts.tolist()
            )
        )

        mean_scores = np.asarray(
            matrix.mean(axis=0)
        ).ravel()

        terms = (
            vectorizer
            .get_feature_names_out()
        )

        ranked = sorted(
            zip(
                terms,
                mean_scores,
            ),
            key=lambda item:
                item[1],
            reverse=True,
        )

        return [
            {
                "term": term,
                "score": round(
                    float(score),
                    4,
                ),
            }
            for term, score
            in ranked[
                :TOP_TFIDF_LIMIT
            ]
        ]

    except ValueError:
        return []


# ------------------------------------------------------------
# Text quality
# ------------------------------------------------------------

def _text_quality(
    series: pd.Series,
    full_texts: pd.Series,
) -> dict[str, Any]:

    total_rows = len(
        series
    )

    missing_count = int(
        series.isna().sum()
    )

    normalized_all = (
        series
        .fillna("")
        .map(_normalize_text)
    )

    empty_count = int(
        (
            normalized_all == ""
        ).sum()
    )

    duplicate_count = int(
        full_texts
        .duplicated()
        .sum()
    )

    very_short_count = int(
        full_texts.map(
            lambda text:
                len(
                    _tokenize(text)
                ) <= 2
        ).sum()
    )

    very_long_count = int(
        full_texts
        .str.len()
        .gt(
            VERY_LONG_TEXT_THRESHOLD
        )
        .sum()
    )

    return {
        "missing_count":
            missing_count,

        "missing_percent":
            round(
                (
                    missing_count
                    / total_rows
                    * 100
                )
                if total_rows
                else 0.0,
                2,
            ),

        "empty_text_count":
            empty_count,

        "duplicate_text_count":
            duplicate_count,

        "duplicate_text_percent":
            round(
                (
                    duplicate_count
                    / len(full_texts)
                    * 100
                )
                if len(full_texts)
                else 0.0,
                2,
            ),

        "very_short_text_count":
            very_short_count,

        "very_long_text_count":
            very_long_count,

        "very_long_threshold_characters":
            VERY_LONG_TEXT_THRESHOLD,
    }


# ------------------------------------------------------------
# Single text column
# ------------------------------------------------------------

def analyze_text_column(
    series: pd.Series,
    column_name: str,
    stopword_mode: str | None = DEFAULT_STOPWORD_MODE,
) -> dict[str, Any]:

    full_texts = (
        _prepare_all_texts(
            series
        )
    )

    sampled_texts = (
        _sample_texts(
            full_texts
        )
    )

    unique_text_count = int(
        full_texts.nunique()
    )

    return {
        "column":
            column_name,

        "total_rows":
            int(
                len(series)
            ),

        "non_empty_text_rows":
            int(
                len(full_texts)
            ),

        "analyzed_text_rows":
            int(
                len(sampled_texts)
            ),

        "analysis_sampled":
            bool(
                len(full_texts)
                > len(sampled_texts)
            ),

        "unique_text_count":
            unique_text_count,

        "unique_text_ratio":
            round(
                (
                    unique_text_count
                    / len(full_texts)
                )
                if len(full_texts)
                else 0.0,
                4,
            ),

        "stopword_mode":
            stopword_mode,

        "length_statistics":
            _calculate_length_statistics(
                full_texts
            ),

        "lexical_statistics":
            _lexical_statistics(
                sampled_texts,
                stopword_mode,
            ),

        "quality":
            _text_quality(
                series,
                full_texts,
            ),

        "top_raw_terms":
            _top_raw_terms(
                sampled_texts
            ),

        "top_content_terms":
            _top_content_terms(
                sampled_texts,
                stopword_mode,
            ),

        "top_bigrams":
            _top_bigrams(
                sampled_texts,
                stopword_mode,
            ),

        "tfidf_keywords":
            _tfidf_keywords(
                sampled_texts,
                stopword_mode,
            ),
    }


# ------------------------------------------------------------
# Multiple text columns
# ------------------------------------------------------------

def analyze_text_columns(
    df: pd.DataFrame,
    text_columns: list[str] | None = None,
    stopword_mode: str | None = DEFAULT_STOPWORD_MODE,
) -> dict[str, Any]:

    if not text_columns:
        return {}

    results = {}

    for column in text_columns:

        if column not in df.columns:
            continue

        results[column] = (
            analyze_text_column(
                series=df[column],
                column_name=column,
                stopword_mode=stopword_mode,
            )
        )

    return results