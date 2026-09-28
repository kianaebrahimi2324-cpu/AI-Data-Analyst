from __future__ import annotations

from typing import Any

import pandas as pd


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

TOP_CATEGORY_LIMIT = 15


# ------------------------------------------------------------
# JSON-safe values
# ------------------------------------------------------------

def _safe_value(
    value: Any,
) -> Any:
    """
    Convert categorical values to JSON-safe representations.
    """

    if pd.isna(value):
        return None

    if isinstance(
        value,
        (
            str,
            int,
            float,
            bool,
        ),
    ):
        return value

    return str(value)


# ------------------------------------------------------------
# Single categorical column
# ------------------------------------------------------------

def analyze_categorical_column(
    series: pd.Series,
    column_name: str,
) -> dict[str, Any]:
    """
    Analyze one categorical or binary column.
    """

    total_rows = int(
        len(series)
    )

    missing_count = int(
        series.isna().sum()
    )

    non_null = (
        series.dropna()
    )

    non_null_count = int(
        len(non_null)
    )

    unique_count = int(
        non_null.nunique(
            dropna=True
        )
    )


    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    missing_percent = (
        round(
            (
                missing_count
                / total_rows
                * 100
            ),
            2,
        )
        if total_rows
        else 0.0
    )


    # --------------------------------------------------------
    # Unique ratio
    # --------------------------------------------------------

    unique_ratio = (
        round(
            (
                unique_count
                / non_null_count
            ),
            4,
        )
        if non_null_count
        else 0.0
    )


    # --------------------------------------------------------
    # Frequency table
    # --------------------------------------------------------

    counts = (
        non_null
        .value_counts(
            dropna=True
        )
    )


    top_counts = counts.head(
        TOP_CATEGORY_LIMIT
    )


    top_values = []


    for value, count in (
        top_counts.items()
    ):

        percentage = (
            float(count)
            / non_null_count
            * 100
            if non_null_count
            else 0.0
        )

        top_values.append(
            {
                "value":
                    _safe_value(
                        value
                    ),

                "count":
                    int(
                        count
                    ),

                "percentage":
                    round(
                        percentage,
                        2,
                    ),
            }
        )


    # --------------------------------------------------------
    # Mode
    # --------------------------------------------------------

    if len(counts) > 0:

        mode_value = (
            _safe_value(
                counts.index[0]
            )
        )

        mode_count = int(
            counts.iloc[0]
        )

        mode_percent = (
            round(
                (
                    mode_count
                    / non_null_count
                    * 100
                ),
                2,
            )
            if non_null_count
            else 0.0
        )

    else:

        mode_value = None
        mode_count = 0
        mode_percent = 0.0


    # --------------------------------------------------------
    # Binary detection
    # --------------------------------------------------------

    is_binary = bool(
        unique_count == 2
    )


    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    return {
        "column":
            column_name,

        "total_rows":
            total_rows,

        "non_null_count":
            non_null_count,

        "missing_count":
            missing_count,

        "missing_percent":
            missing_percent,

        "unique_count":
            unique_count,

        "unique_ratio":
            unique_ratio,

        "is_binary":
            is_binary,

        "mode":
            mode_value,

        "mode_count":
            mode_count,

        "mode_percent":
            mode_percent,

        "top_values":
            top_values,
    }


# ------------------------------------------------------------
# Multiple categorical columns
# ------------------------------------------------------------

def analyze_categorical_columns(
    df: pd.DataFrame,
    categorical_columns: list[str] | None = None,
) -> dict[str, Any]:
    """
    Analyze selected categorical columns.

    Normally, categorical_columns should be supplied by
    the schema inference + analysis router.

    If no list is supplied, object/category/bool columns
    are used for backward compatibility.
    """

    if categorical_columns is None:

        categorical_columns = (
            df.select_dtypes(
                include=[
                    "object",
                    "category",
                    "bool",
                ]
            )
            .columns
            .tolist()
        )


    results = {}


    for column in categorical_columns:

        if column not in df.columns:
            continue

        results[column] = (
            analyze_categorical_column(
                series=df[column],
                column_name=str(
                    column
                ),
            )
        )


    return results