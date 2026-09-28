from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


# ============================================================
# HELPERS
# ============================================================

def _safe_percent(
    value: int | float,
    total: int | float,
) -> float:
    if not total:
        return 0.0

    return round(
        float(value) / float(total) * 100.0,
        2,
    )


def _finite_numeric_series(
    series: pd.Series,
) -> tuple[pd.Series, pd.Series, int, int]:
    """
    Convert values to numeric and return finite values only.

    Returns:
        parsed_numeric,
        finite_values,
        infinite_count,
        non_numeric_count

    Missing source values are not counted as non-numeric.
    """
    parsed = pd.to_numeric(
        series,
        errors="coerce",
    )

    source_present = series.notna()

    if (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
    ):
        text = series.astype("string").str.strip()
        source_present = (
            source_present
            & text.notna()
            & text.ne("")
        )

    non_numeric_count = int(
        (
            source_present
            & parsed.isna()
        ).sum()
    )

    parsed_array = parsed.to_numpy(
        dtype=float,
        na_value=np.nan,
    )

    infinite_mask = np.isinf(
        parsed_array
    )

    infinite_count = int(
        infinite_mask.sum()
    )

    finite_mask = np.isfinite(
        parsed_array
    )

    finite = pd.Series(
        parsed_array[finite_mask],
        dtype="float64",
    )

    return (
        parsed,
        finite,
        infinite_count,
        non_numeric_count,
    )


def _round_or_none(
    value: Any,
    digits: int = 6,
) -> float | None:
    try:
        numeric = float(value)
    except (
        TypeError,
        ValueError,
    ):
        return None

    if not np.isfinite(numeric):
        return None

    return round(
        numeric,
        digits,
    )


# ============================================================
# SINGLE NUMERICAL COLUMN
# ============================================================

def analyze_numerical_column(
    series: pd.Series,
    column_name: str,
) -> dict[str, Any]:
    """
    Analyze one numerical column using finite numeric values only.

    Important evidence is reported explicitly so downstream LLM
    components never need to infer zero frequency, skewness, or
    outlier counts from a minimum/maximum value.
    """
    (
        parsed,
        clean,
        infinite_count,
        non_numeric_count,
    ) = _finite_numeric_series(
        series
    )

    total_rows = int(
        len(series)
    )

    source_missing_count = int(
        series.isna().sum()
    )

    parsed_missing_count = int(
        parsed.isna().sum()
    )

    finite_count = int(
        len(clean)
    )

    unique_count = int(
        clean.nunique(
            dropna=True
        )
    )

    zero_count = int(
        np.isclose(
            clean.to_numpy(
                dtype=float
            ),
            0.0,
            rtol=0.0,
            atol=1e-12,
        ).sum()
    )

    negative_count = int(
        (
            clean < 0
        ).sum()
    )

    positive_count = int(
        (
            clean > 0
        ).sum()
    )

    base = {
        "column":
            str(column_name),

        "row_count":
            total_rows,

        "count":
            finite_count,

        "finite_count":
            finite_count,

        "unique_count":
            unique_count,

        "missing_count":
            source_missing_count,

        "missing_percent":
            _safe_percent(
                source_missing_count,
                total_rows,
            ),

        "coerced_missing_count":
            parsed_missing_count,

        "non_numeric_count":
            non_numeric_count,

        "infinite_count":
            infinite_count,

        "zero_count":
            zero_count,

        "zero_percent":
            _safe_percent(
                zero_count,
                finite_count,
            ),

        "negative_count":
            negative_count,

        "negative_percent":
            _safe_percent(
                negative_count,
                finite_count,
            ),

        "positive_count":
            positive_count,

        "positive_percent":
            _safe_percent(
                positive_count,
                finite_count,
            ),
    }

    if finite_count == 0:
        return {
            **base,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
            "q1": None,
            "q3": None,
            "iqr": None,
            "lower_outlier_bound": None,
            "upper_outlier_bound": None,
            "outlier_count": 0,
            "outlier_percent": 0.0,
            "skewness": None,
        }

    mean = _round_or_none(
        clean.mean()
    )

    median = _round_or_none(
        clean.median()
    )

    std = (
        _round_or_none(
            clean.std()
        )
        if finite_count >= 2
        else None
    )

    minimum = _round_or_none(
        clean.min()
    )

    maximum = _round_or_none(
        clean.max()
    )

    q1_raw = clean.quantile(
        0.25
    )

    q3_raw = clean.quantile(
        0.75
    )

    q1 = _round_or_none(
        q1_raw
    )

    q3 = _round_or_none(
        q3_raw
    )

    iqr_raw = float(
        q3_raw - q1_raw
    )

    iqr = _round_or_none(
        iqr_raw
    )

    lower_raw = float(
        q1_raw - 1.5 * iqr_raw
    )

    upper_raw = float(
        q3_raw + 1.5 * iqr_raw
    )

    outlier_mask = (
        (clean < lower_raw)
        | (clean > upper_raw)
    )

    outlier_count = int(
        outlier_mask.sum()
    )

    skewness = (
        _round_or_none(
            clean.skew()
        )
        if finite_count >= 3
        else None
    )

    return {
        **base,

        "mean":
            mean,

        "median":
            median,

        "std":
            std,

        "min":
            minimum,

        "max":
            maximum,

        "q1":
            q1,

        "q3":
            q3,

        "iqr":
            iqr,

        "lower_outlier_bound":
            _round_or_none(
                lower_raw
            ),

        "upper_outlier_bound":
            _round_or_none(
                upper_raw
            ),

        "outlier_count":
            outlier_count,

        "outlier_percent":
            _safe_percent(
                outlier_count,
                finite_count,
            ),

        "skewness":
            skewness,
    }


# ============================================================
# MULTIPLE NUMERICAL COLUMNS
# ============================================================

def analyze_numerical_columns(
    df: pd.DataFrame,
    numeric_columns: list[str] | None = None,
) -> dict[str, Any]:
    """
    Analyze routed numerical columns.

    If ``numeric_columns`` is omitted, native pandas numerical
    columns are used for backwards compatibility.
    """
    if numeric_columns is None:
        numeric_columns = (
            df.select_dtypes(
                include=[
                    np.number
                ]
            )
            .columns
            .tolist()
        )

    results: dict[str, Any] = {}

    for column in numeric_columns:

        if column not in df.columns:
            continue

        results[
            str(column)
        ] = analyze_numerical_column(
            series=df[column],
            column_name=str(column),
        )

    return results
