from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


# ============================================================
# HELPERS
# ============================================================

UNSUITABLE_TARGET_ROLES = {
    "identifier",
    "identifier_candidate",
    "exported_index",
    "url",
    "email",
    "phone",
    "address",
    "latitude",
    "longitude",
    "structural",
    "structural_overflow_candidate",
}

AMBIGUOUS_TARGET_ROLES = {
    "entity_name",
    "entity_or_identifier_candidate",
}

DATETIME_TYPES = {
    "datetime",
    "date",
    "time",
    "temporal",
}

CATEGORICAL_TYPES = {
    "binary",
    "categorical",
    "category",
    "boolean",
    "bool",
}

NUMERIC_TYPES = {
    "numeric",
    "numeric_continuous",
    "continuous",
    "integer",
    "float",
}


def _safe_percent(
    value: int | float,
    total: int | float,
) -> float:
    if total == 0:
        return 0.0

    return round(
        float(value)
        / float(total)
        * 100.0,
        2,
    )


def _extract_columns_schema(
    schema: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    """Support both direct and legacy nested schema formats."""
    if not isinstance(schema, dict):
        return {}

    nested = schema.get(
        "columns_schema"
    )

    if isinstance(
        nested,
        dict,
    ):
        return nested

    return {
        str(column): info
        for column, info
        in schema.items()
        if isinstance(info, dict)
        and (
            "semantic_type" in info
            or "semantic_role" in info
        )
    }


def _is_integer_like(
    series: pd.Series,
) -> bool:
    """Return True when all finite numeric values are integer-like."""
    numeric = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if numeric.empty:
        return False

    values = numeric.to_numpy(
        dtype=float
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return False

    return bool(
        np.all(
            np.isclose(
                values,
                np.round(values),
            )
        )
    )


def _classification_threshold(
    valid_count: int,
) -> int:
    """
    Conservative upper bound for inferred numeric classes.

    The threshold grows slowly with dataset size and is capped
    to avoid treating high-cardinality numeric targets as classes.
    """
    if valid_count <= 0:
        return 20

    scaled = int(
        math.sqrt(valid_count)
        * 1.5
    )

    return min(
        50,
        max(
            20,
            scaled,
        ),
    )


def _string_key_dict(
    series: pd.Series,
) -> dict[str, int]:
    """Create JSON-stable class counts."""
    counts = (
        series
        .dropna()
        .value_counts()
    )

    return {
        str(key): int(value)
        for key, value
        in counts.items()
    }


def _class_percentages(
    counts: dict[str, int],
) -> dict[str, float]:
    total = sum(
        counts.values()
    )

    return {
        key: _safe_percent(
            value,
            total,
        )
        for key, value
        in counts.items()
    }


# ============================================================
# TASK INFERENCE
# ============================================================

def _infer_task(
    target: pd.Series,
    semantic_type: str,
    semantic_role: str,
) -> dict[str, Any]:
    """
    Infer target task from values plus semantic context.

    User selection is respected, but unsafe semantic roles are
    surfaced for confirmation instead of silently modeling them.
    """
    valid = target.dropna()

    valid_count = int(
        len(valid)
    )

    unique_count = int(
        valid.nunique(
            dropna=True
        )
    )

    unique_ratio = (
        unique_count
        / valid_count
        if valid_count
        else 0.0
    )

    semantic_type = str(
        semantic_type or "unknown"
    ).lower()

    semantic_role = str(
        semantic_role or "feature"
    ).lower()

    # --------------------------------------------------------
    # Empty / unusable targets
    # --------------------------------------------------------

    if valid_count == 0:
        return {
            "task_type": "unknown",
            "inferred_semantic_type": (
                semantic_type
                if semantic_type != "unknown"
                else "empty"
            ),
            "task_confidence": 1.0,
            "requires_confirmation": True,
            "suitability_reason": (
                "The selected target contains no non-missing values."
            ),
        }

    if unique_count <= 1:
        return {
            "task_type": "unknown",
            "inferred_semantic_type": (
                semantic_type
                if semantic_type != "unknown"
                else "constant"
            ),
            "task_confidence": 1.0,
            "requires_confirmation": True,
            "suitability_reason": (
                "The selected target has fewer than two distinct "
                "non-missing values."
            ),
        }

    # --------------------------------------------------------
    # Semantically unsafe targets
    # --------------------------------------------------------

    if semantic_role in UNSUITABLE_TARGET_ROLES:
        return {
            "task_type": "unknown",
            "inferred_semantic_type": semantic_type,
            "task_confidence": 1.0,
            "requires_confirmation": True,
            "suitability_reason": (
                f"The selected target is inferred as semantic role "
                f"'{semantic_role}', which is not treated as an ordinary "
                "prediction target without explicit confirmation."
            ),
        }

    if (
        semantic_type in DATETIME_TYPES
        or semantic_role in {
            "datetime",
            "temporal",
        }
    ):
        return {
            "task_type": "temporal",
            "inferred_semantic_type": "datetime",
            "task_confidence": 0.95,
            "requires_confirmation": True,
            "suitability_reason": (
                "The selected target appears temporal. The intended "
                "forecasting or time-to-event objective should be confirmed."
            ),
        }

    # --------------------------------------------------------
    # Strong classification signals
    # --------------------------------------------------------

    if (
        pd.api.types.is_bool_dtype(
            target
        )
        or unique_count == 2
        or semantic_type == "binary"
    ):
        return {
            "task_type": "classification",
            "inferred_semantic_type": "binary",
            "task_confidence": 1.0,
            "requires_confirmation": False,
            "suitability_reason": "",
        }

    if semantic_type in CATEGORICAL_TYPES:
        return {
            "task_type": "classification",
            "inferred_semantic_type": (
                "categorical"
                if unique_count > 2
                else "binary"
            ),
            "task_confidence": 0.98,
            "requires_confirmation": False,
            "suitability_reason": "",
        }

    # --------------------------------------------------------
    # Text / categorical-looking targets
    # --------------------------------------------------------

    if (
        pd.api.types.is_object_dtype(
            target
        )
        or pd.api.types.is_string_dtype(
            target
        )
        or isinstance(
            target.dtype,
            pd.CategoricalDtype,
        )
    ):
        converted = pd.to_numeric(
            valid,
            errors="coerce",
        )

        numeric_parse_ratio = (
            float(
                converted.notna().sum()
            )
            / valid_count
            if valid_count
            else 0.0
        )

        # Numeric-coded strings should use numeric logic below.
        if numeric_parse_ratio < 0.95:
            if (
                unique_count <= 50
                and (
                    unique_count <= 20
                    or unique_ratio <= 0.25
                )
            ):
                return {
                    "task_type": "classification",
                    "inferred_semantic_type": "categorical",
                    "task_confidence": 0.95,
                    "requires_confirmation": False,
                    "suitability_reason": "",
                }

            return {
                "task_type": "unknown",
                "inferred_semantic_type": (
                    semantic_type
                    if semantic_type != "unknown"
                    else "text"
                ),
                "task_confidence": 0.75,
                "requires_confirmation": True,
                "suitability_reason": (
                    "The selected target is text-like with high cardinality. "
                    "Confirm whether it represents classes, free text, or an ID."
                ),
            }

    # --------------------------------------------------------
    # Numeric targets
    # --------------------------------------------------------

    numeric = pd.to_numeric(
        valid,
        errors="coerce",
    )

    numeric_parse_ratio = (
        float(
            numeric.notna().sum()
        )
        / valid_count
        if valid_count
        else 0.0
    )

    if numeric_parse_ratio >= 0.95:
        integer_like = _is_integer_like(
            valid
        )

        threshold = (
            _classification_threshold(
                valid_count
            )
        )

        if (
            integer_like
            and unique_count <= threshold
            and (
                unique_count <= 20
                or unique_ratio <= 0.10
            )
        ):
            return {
                "task_type": "classification",
                "inferred_semantic_type": "categorical",
                "task_confidence": 0.88,
                "requires_confirmation": False,
                "suitability_reason": "",
            }

        return {
            "task_type": "regression",
            "inferred_semantic_type": (
                semantic_type
                if semantic_type in NUMERIC_TYPES
                else "numeric_continuous"
            ),
            "task_confidence": 0.95,
            "requires_confirmation": False,
            "suitability_reason": "",
        }

    # --------------------------------------------------------
    # Ambiguous entity-like target
    # --------------------------------------------------------

    if semantic_role in AMBIGUOUS_TARGET_ROLES:
        return {
            "task_type": "unknown",
            "inferred_semantic_type": semantic_type,
            "task_confidence": 0.8,
            "requires_confirmation": True,
            "suitability_reason": (
                "The target appears entity-like. Confirm that repeated entity "
                "labels genuinely represent prediction classes."
            ),
        }

    return {
        "task_type": "unknown",
        "inferred_semantic_type": (
            semantic_type
            if semantic_type
            else "unknown"
        ),
        "task_confidence": 0.5,
        "requires_confirmation": True,
        "suitability_reason": (
            "The target task could not be inferred confidently from "
            "the available values and schema."
        ),
    }


# ============================================================
# MAIN TARGET ANALYZER
# ============================================================

def analyze_target(
    df: pd.DataFrame,
    target_column: str,
    schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Analyze a user-selected target column.

    Binary, multiclass and regression tasks are inferred from
    observed values plus semantic schema information.
    """
    if not target_column:
        return {}

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' was not found."
        )

    target = df[
        target_column
    ]

    total_rows = int(
        len(target)
    )

    missing_count = int(
        target.isna().sum()
    )

    valid_count = (
        total_rows
        - missing_count
    )

    unique_count = int(
        target.nunique(
            dropna=True
        )
    )

    columns_schema = (
        _extract_columns_schema(
            schema
        )
    )

    schema_info = (
        columns_schema.get(
            str(target_column),
            {},
        )
    )

    source_semantic_type = str(
        schema_info.get(
            "semantic_type",
            "unknown",
        )
    )

    semantic_role = str(
        schema_info.get(
            "semantic_role",
            "feature",
        )
    )

    inference = _infer_task(
        target=target,
        semantic_type=source_semantic_type,
        semantic_role=semantic_role,
    )

    semantic_type = str(
        inference[
            "inferred_semantic_type"
        ]
    )

    result: dict[str, Any] = {
        "target_column":
            str(target_column),

        "semantic_type":
            semantic_type,

        "source_semantic_type":
            source_semantic_type,

        "semantic_role":
            semantic_role,

        "task_type":
            inference[
                "task_type"
            ],

        "task_confidence":
            float(
                inference[
                    "task_confidence"
                ]
            ),

        "requires_confirmation":
            bool(
                inference[
                    "requires_confirmation"
                ]
            ),

        "suitability_reason":
            str(
                inference[
                    "suitability_reason"
                ]
            ),

        "target_check":
            (
                "confirmation_recommended"
                if inference[
                    "requires_confirmation"
                ]
                else "supported"
            ),

        "rows":
            total_rows,

        "non_missing_count":
            valid_count,

        "missing_count":
            missing_count,

        "missing_percent":
            _safe_percent(
                missing_count,
                total_rows,
            ),

        "unique_values":
            unique_count,
    }

    # --------------------------------------------------------
    # Classification summary
    # --------------------------------------------------------

    if (
        result[
            "task_type"
        ]
        == "classification"
    ):
        counts = _string_key_dict(
            target
        )

        result[
            "class_counts"
        ] = counts

        result[
            "class_percentages"
        ] = _class_percentages(
            counts
        )

        if counts:
            largest = max(
                counts.values()
            )

            smallest = min(
                counts.values()
            )

            result[
                "largest_class_percent"
            ] = _safe_percent(
                largest,
                sum(
                    counts.values()
                ),
            )

            result[
                "smallest_class_percent"
            ] = _safe_percent(
                smallest,
                sum(
                    counts.values()
                ),
            )

            result[
                "class_count_ratio"
            ] = (
                round(
                    largest
                    / smallest,
                    4,
                )
                if smallest
                else None
            )

        return result

    # --------------------------------------------------------
    # Regression summary
    # --------------------------------------------------------

    if (
        result[
            "task_type"
        ]
        == "regression"
    ):
        numeric = pd.to_numeric(
            target,
            errors="coerce",
        ).replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        ).dropna()

        if not numeric.empty:
            result.update(
                {
                    "count":
                        int(
                            numeric.count()
                        ),

                    "mean":
                        float(
                            numeric.mean()
                        ),

                    "median":
                        float(
                            numeric.median()
                        ),

                    "std":
                        (
                            float(
                                numeric.std()
                            )
                            if len(numeric) > 1
                            else 0.0
                        ),

                    "min":
                        float(
                            numeric.min()
                        ),

                    "max":
                        float(
                            numeric.max()
                        ),
                }
            )

    return result
