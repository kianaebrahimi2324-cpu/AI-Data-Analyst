from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import (
    TfidfVectorizer,
)

from sklearn.feature_selection import (
    chi2,
    f_regression,
)

from sklearn.preprocessing import (
    LabelEncoder,
)


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

MAX_TEXT_TARGET_ROWS = 5000
MAX_FEATURES = 3000

TOP_ASSOCIATED_TERMS = 20
TOP_TERMS_PER_CLASS = 10

MIN_VALID_ROWS = 10
MAX_TARGET_CLASSES = 20

DEFAULT_STOPWORD_MODE = "english"


# ------------------------------------------------------------
# Text normalization
# ------------------------------------------------------------

def _normalize_text(
    value: Any,
) -> str:
    """Normalize whitespace while preserving text content."""

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
# Stopword configuration
# ------------------------------------------------------------

def _get_stop_words(
    stopword_mode: str | None,
) -> str | None:
    """
    Return the stopword configuration used by sklearn.
    """

    if stopword_mode == "english":
        return "english"

    return None


# ------------------------------------------------------------
# Paired data preparation
# ------------------------------------------------------------

def _prepare_text_target_data(
    df: pd.DataFrame,
    text_column: str,
    target_column: str,
) -> pd.DataFrame:
    """
    Prepare paired text and target rows.

    Sampling is performed on rows rather than text alone
    so text-target alignment is preserved.
    """

    data = df[
        [
            text_column,
            target_column,
        ]
    ].copy()


    # Target must exist
    data = data[
        data[
            target_column
        ].notna()
    ]


    # Normalize text
    data[text_column] = (
        data[
            text_column
        ]
        .map(
            _normalize_text
        )
    )


    # Remove empty text
    data = data[
        data[
            text_column
        ] != ""
    ]


    # Deterministic sampling
    if (
        len(data)
        > MAX_TEXT_TARGET_ROWS
    ):

        data = data.sample(
            n=MAX_TEXT_TARGET_ROWS,
            random_state=42,
        )


    return data


# ------------------------------------------------------------
# TF-IDF creation
# ------------------------------------------------------------

def _build_tfidf(
    texts: pd.Series,
    stopword_mode: str | None,
):
    """
    Create TF-IDF representation using unigrams
    and real adjacent bigrams.
    """

    if len(texts) < 50:
        min_df = 1

    else:
        min_df = 2


    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words=_get_stop_words(
            stopword_mode
        ),
        ngram_range=(1, 2),
        max_features=MAX_FEATURES,
        min_df=min_df,
        sublinear_tf=True,
        token_pattern=r"(?u)\b\w[\w'-]+\b",
    )


    matrix = (
        vectorizer
        .fit_transform(
            texts.tolist()
        )
    )


    feature_names = (
        vectorizer
        .get_feature_names_out()
    )


    return (
        vectorizer,
        matrix,
        feature_names,
    )


# ------------------------------------------------------------
# Document frequency
# ------------------------------------------------------------

def _document_frequencies(
    matrix,
) -> np.ndarray:
    """Calculate how many documents contain each term."""

    return np.asarray(
        (
            matrix > 0
        ).sum(
            axis=0
        )
    ).ravel()


# ------------------------------------------------------------
# Classification target
# ------------------------------------------------------------

def _analyze_classification_text_target(
    data: pd.DataFrame,
    text_column: str,
    target_column: str,
    stopword_mode: str | None,
) -> dict[str, Any]:
    """
    Analyze associations between TF-IDF features
    and a categorical target using chi-square.
    """

    target = data[
        target_column
    ]


    unique_classes = int(
        target.nunique()
    )


    if unique_classes < 2:

        return {
            "status":
                "not_analyzable",

            "reason":
                "Target contains fewer than two classes.",
        }


    if unique_classes > MAX_TARGET_CLASSES:

        return {
            "status":
                "not_analyzable",

            "reason":
                (
                    "Target contains too many classes "
                    "for this text association analysis."
                ),
        }


    try:

        (
            vectorizer,
            matrix,
            feature_names,
        ) = _build_tfidf(
            texts=data[
                text_column
            ],
            stopword_mode=stopword_mode,
        )

    except ValueError as exc:

        return {
            "status":
                "not_analyzable",

            "reason":
                str(exc),
        }


    # --------------------------------------------------------
    # Encode target
    # --------------------------------------------------------

    encoder = LabelEncoder()

    encoded_target = (
        encoder.fit_transform(
            target.astype(str)
        )
    )


    # --------------------------------------------------------
    # Chi-square
    # --------------------------------------------------------

    scores, p_values = chi2(
        matrix,
        encoded_target,
    )


    document_counts = (
        _document_frequencies(
            matrix
        )
    )


    ranked_indices = np.argsort(
        scores
    )[::-1]


    associated_terms = []


    for index in ranked_indices:

        score = scores[
            index
        ]

        p_value = p_values[
            index
        ]


        if not np.isfinite(
            score
        ):
            continue


        associated_terms.append(
            {
                "term":
                    str(
                        feature_names[
                            index
                        ]
                    ),

                "chi2_score":
                    round(
                        float(
                            score
                        ),
                        6,
                    ),

                "p_value":
                    (
                        round(
                            float(
                                p_value
                            ),
                            8,
                        )
                        if np.isfinite(
                            p_value
                        )
                        else None
                    ),

                "document_count":
                    int(
                        document_counts[
                            index
                        ]
                    ),
            }
        )


        if (
            len(
                associated_terms
            )
            >= TOP_ASSOCIATED_TERMS
        ):
            break


    # --------------------------------------------------------
    # Top TF-IDF terms inside each class
    # --------------------------------------------------------

    class_profiles = []


    for class_index, class_name in enumerate(
        encoder.classes_
    ):

        mask = (
            encoded_target
            == class_index
        )


        class_matrix = matrix[
            mask
        ]


        class_size = int(
            mask.sum()
        )


        if class_size == 0:
            continue


        mean_scores = np.asarray(
            class_matrix.mean(
                axis=0
            )
        ).ravel()


        class_ranked = np.argsort(
            mean_scores
        )[::-1]


        terms = []


        for index in class_ranked:

            score = mean_scores[
                index
            ]


            if (
                not np.isfinite(
                    score
                )
                or score <= 0
            ):
                continue


            terms.append(
                {
                    "term":
                        str(
                            feature_names[
                                index
                            ]
                        ),

                    "mean_tfidf":
                        round(
                            float(
                                score
                            ),
                            6,
                        ),
                }
            )


            if (
                len(terms)
                >= TOP_TERMS_PER_CLASS
            ):
                break


        class_profiles.append(
            {
                "class":
                    str(
                        class_name
                    ),

                "count":
                    class_size,

                "top_terms":
                    terms,
            }
        )


    return {
        "status":
            "ok",

        "method":
            "chi_square_on_tfidf",

        "valid_rows":
            int(
                len(data)
            ),

        "class_count":
            unique_classes,

        "feature_count":
            int(
                matrix.shape[1]
            ),

        "top_associated_terms":
            associated_terms,

        "class_profiles":
            class_profiles,

        "notes": [
            (
                "Chi-square identifies statistical "
                "association between terms and target classes."
            ),
            (
                "Association does not imply causality "
                "or predictive importance."
            ),
        ],
    }


# ------------------------------------------------------------
# Regression target
# ------------------------------------------------------------

def _analyze_regression_text_target(
    data: pd.DataFrame,
    text_column: str,
    target_column: str,
    stopword_mode: str | None,
) -> dict[str, Any]:
    """
    Analyze relationships between TF-IDF features
    and a numeric target.
    """

    numeric_target = pd.to_numeric(
        data[
            target_column
        ],
        errors="coerce",
    )


    valid_mask = (
        numeric_target.notna()
    )


    data = data.loc[
        valid_mask
    ].copy()


    numeric_target = (
        numeric_target.loc[
            valid_mask
        ]
    )


    if len(data) < MIN_VALID_ROWS:

        return {
            "status":
                "not_analyzable",

            "reason":
                "Too few valid text-target rows.",
        }


    if (
        numeric_target.nunique()
        < 2
    ):

        return {
            "status":
                "not_analyzable",

            "reason":
                "Numeric target has no variation.",
        }


    try:

        (
            vectorizer,
            matrix,
            feature_names,
        ) = _build_tfidf(
            texts=data[
                text_column
            ],
            stopword_mode=stopword_mode,
        )

    except ValueError as exc:

        return {
            "status":
                "not_analyzable",

            "reason":
                str(exc),
        }


    # --------------------------------------------------------
    # Univariate F-test
    # --------------------------------------------------------

    scores, p_values = (
        f_regression(
            matrix,
            numeric_target.to_numpy(
                dtype=float
            ),
        )
    )


    document_counts = (
        _document_frequencies(
            matrix
        )
    )


    ranked_indices = np.argsort(
        scores
    )[::-1]


    associated_terms = []


    for index in ranked_indices:

        score = scores[
            index
        ]

        p_value = p_values[
            index
        ]


        if not np.isfinite(
            score
        ):
            continue


        term_values = (
            matrix[
                :,
                index
            ]
            .toarray()
            .ravel()
        )


        if (
            np.std(
                term_values
            )
            > 0
            and np.std(
                numeric_target
                .to_numpy(
                    dtype=float
                )
            )
            > 0
        ):

            correlation = np.corrcoef(
                term_values,
                numeric_target.to_numpy(
                    dtype=float
                ),
            )[0, 1]

        else:

            correlation = np.nan


        associated_terms.append(
            {
                "term":
                    str(
                        feature_names[
                            index
                        ]
                    ),

                "f_score":
                    round(
                        float(
                            score
                        ),
                        6,
                    ),

                "p_value":
                    (
                        round(
                            float(
                                p_value
                            ),
                            8,
                        )
                        if np.isfinite(
                            p_value
                        )
                        else None
                    ),

                "correlation":
                    (
                        round(
                            float(
                                correlation
                            ),
                            6,
                        )
                        if np.isfinite(
                            correlation
                        )
                        else None
                    ),

                "document_count":
                    int(
                        document_counts[
                            index
                        ]
                    ),
            }
        )


        if (
            len(
                associated_terms
            )
            >= TOP_ASSOCIATED_TERMS
        ):
            break


    return {
        "status":
            "ok",

        "method":
            "f_regression_on_tfidf",

        "valid_rows":
            int(
                len(data)
            ),

        "feature_count":
            int(
                matrix.shape[1]
            ),

        "top_associated_terms":
            associated_terms,

        "notes": [
            (
                "The F-test measures univariate association "
                "between TF-IDF features and the numeric target."
            ),
            (
                "The reported correlation gives the direction "
                "of the individual term-target relationship."
            ),
            (
                "These statistics do not demonstrate causality "
                "or predictive performance."
            ),
        ],
    }


# ------------------------------------------------------------
# Single text-target relationship
# ------------------------------------------------------------

def analyze_text_target_column(
    df: pd.DataFrame,
    text_column: str,
    target_column: str,
    task_type: str,
    stopword_mode: str | None = DEFAULT_STOPWORD_MODE,
) -> dict[str, Any]:
    """
    Analyze one text column against the selected target.
    """

    data = (
        _prepare_text_target_data(
            df=df,
            text_column=text_column,
            target_column=target_column,
        )
    )


    if len(data) < MIN_VALID_ROWS:

        return {
            "status":
                "not_analyzable",

            "reason":
                (
                    f"Only {len(data)} valid paired rows "
                    "are available."
                ),

            "valid_rows":
                int(
                    len(data)
                ),
        }


    if task_type == "classification":

        return (
            _analyze_classification_text_target(
                data=data,
                text_column=text_column,
                target_column=target_column,
                stopword_mode=stopword_mode,
            )
        )


    if task_type == "regression":

        return (
            _analyze_regression_text_target(
                data=data,
                text_column=text_column,
                target_column=target_column,
                stopword_mode=stopword_mode,
            )
        )


    return {
        "status":
            "not_analyzable",

        "reason":
            (
                f"Unsupported target task type: "
                f"{task_type}"
            ),
    }


# ------------------------------------------------------------
# All text columns against target
# ------------------------------------------------------------

def analyze_text_target_relationships(
    df: pd.DataFrame,
    text_columns: list[str] | None,
    target_column: str,
    task_type: str,
    stopword_mode: str | None = DEFAULT_STOPWORD_MODE,
) -> dict[str, Any]:
    """
    Analyze all routed text features against a target.
    """

    if (
        not target_column
        or target_column not in df.columns
        or not text_columns
    ):

        return {}


    results = {}


    for text_column in text_columns:

        if text_column not in df.columns:
            continue


        if text_column == target_column:
            continue


        results[
            text_column
        ] = (
            analyze_text_target_column(
                df=df,
                text_column=text_column,
                target_column=target_column,
                task_type=task_type,
                stopword_mode=stopword_mode,
            )
        )


    return {
        "target_column":
            target_column,

        "task_type":
            task_type,

        "text_columns":
            results,

        "stopword_mode":
            stopword_mode,

        "notes": [
            (
                "Text-target results are statistical "
                "associations, not model performance."
            ),
            (
                "Very small datasets can produce unstable "
                "term-level associations."
            ),
        ],
    }