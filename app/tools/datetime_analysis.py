from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

TOP_PATTERN_LIMIT = 12


# ------------------------------------------------------------
# Datetime parsing
# ------------------------------------------------------------

def _parse_datetime_series(
    series: pd.Series
) -> pd.Series:
    """
    Convert a series to datetime safely.

    Invalid values are converted to NaT.
    """

    if pd.api.types.is_datetime64_any_dtype(
        series
    ):
        return series.copy()

    try:
        return pd.to_datetime(
            series,
            errors="coerce",
            format="mixed",
        )

    except (TypeError, ValueError):
        return pd.to_datetime(
            series,
            errors="coerce",
        )


# ------------------------------------------------------------
# Granularity inference
# ------------------------------------------------------------

def _infer_granularity(
    median_seconds: float | None
) -> str:
    """
    Infer approximate temporal granularity
    from the median interval.
    """

    if (
        median_seconds is None
        or not np.isfinite(
            median_seconds
        )
    ):
        return "unknown"

    if median_seconds < 60:
        return "seconds"

    if median_seconds < 3600:
        return "minutes"

    if median_seconds < 86400:
        return "hours"

    if median_seconds < 604800:
        return "days"

    if median_seconds < 2592000:
        return "weeks"

    if median_seconds < 31536000:
        return "months"

    return "years"


# ------------------------------------------------------------
# Interval analysis
# ------------------------------------------------------------

def _analyze_intervals(
    parsed: pd.Series
) -> dict[str, Any]:
    """
    Analyze intervals between unique sorted timestamps.
    """

    clean = (
        parsed
        .dropna()
        .drop_duplicates()
        .sort_values()
    )

    if len(clean) < 2:
        return {
            "interval_count": 0,
            "median_interval_seconds": None,
            "median_interval_days": None,
            "min_interval_seconds": None,
            "max_interval_seconds": None,
            "granularity": "unknown",
            "regularity_score": None,
        }

    differences = (
        clean.diff()
        .dropna()
        .dt.total_seconds()
    )

    differences = differences[
        differences > 0
    ]

    if len(differences) == 0:
        return {
            "interval_count": 0,
            "median_interval_seconds": None,
            "median_interval_days": None,
            "min_interval_seconds": None,
            "max_interval_seconds": None,
            "granularity": "unknown",
            "regularity_score": None,
        }

    median_seconds = float(
        differences.median()
    )

    min_seconds = float(
        differences.min()
    )

    max_seconds = float(
        differences.max()
    )

    # Percentage of intervals close to the median interval
    tolerance = max(
        median_seconds * 0.10,
        1.0,
    )

    regularity_score = float(
        np.mean(
            np.abs(
                differences
                - median_seconds
            )
            <= tolerance
        )
    )

    return {
        "interval_count":
            int(
                len(differences)
            ),

        "median_interval_seconds":
            round(
                median_seconds,
                2,
            ),

        "median_interval_days":
            round(
                median_seconds
                / 86400,
                4,
            ),

        "min_interval_seconds":
            round(
                min_seconds,
                2,
            ),

        "max_interval_seconds":
            round(
                max_seconds,
                2,
            ),

        "granularity":
            _infer_granularity(
                median_seconds
            ),

        "regularity_score":
            round(
                regularity_score,
                4,
            ),
    }


# ------------------------------------------------------------
# Temporal patterns
# ------------------------------------------------------------

def _value_counts_to_list(
    values: pd.Series,
    value_name: str,
) -> list[dict[str, Any]]:
    """
    Convert value counts to a JSON-friendly list.
    """

    counts = (
        values
        .value_counts()
        .head(
            TOP_PATTERN_LIMIT
        )
    )

    return [
        {
            value_name:
                (
                    int(value)
                    if isinstance(
                        value,
                        (
                            int,
                            np.integer,
                        ),
                    )
                    else str(value)
                ),

            "count":
                int(count),
        }
        for value, count
        in counts.items()
    ]


def _temporal_patterns(
    parsed: pd.Series
) -> dict[str, Any]:
    """
    Extract deterministic calendar patterns.
    """

    clean = (
        parsed
        .dropna()
    )

    if len(clean) == 0:
        return {
            "year_counts": [],
            "month_counts": [],
            "weekday_counts": [],
            "hour_counts": [],
        }

    year_counts = (
        _value_counts_to_list(
            clean.dt.year,
            "year",
        )
    )

    month_counts = (
        _value_counts_to_list(
            clean.dt.month,
            "month",
        )
    )

    weekday_counts = (
        _value_counts_to_list(
            clean.dt.day_name(),
            "weekday",
        )
    )

    hour_counts = (
        _value_counts_to_list(
            clean.dt.hour,
            "hour",
        )
    )

    return {
        "year_counts":
            year_counts,

        "month_counts":
            month_counts,

        "weekday_counts":
            weekday_counts,

        "hour_counts":
            hour_counts,
    }


# ------------------------------------------------------------
# Time component detection
# ------------------------------------------------------------

def _has_time_component(
    parsed: pd.Series
) -> bool:
    """
    Check whether timestamps contain meaningful
    hour/minute/second information.
    """

    clean = (
        parsed
        .dropna()
    )

    if len(clean) == 0:
        return False

    return bool(
        (
            (clean.dt.hour != 0)
            | (clean.dt.minute != 0)
            | (clean.dt.second != 0)
        ).any()
    )


# ------------------------------------------------------------
# Single datetime column
# ------------------------------------------------------------

def analyze_datetime_column(
    series: pd.Series,
    column_name: str,
) -> dict[str, Any]:
    """
    Analyze one datetime column.
    """

    total_rows = int(
        len(series)
    )

    missing_count = int(
        series.isna().sum()
    )

    parsed = (
        _parse_datetime_series(
            series
        )
    )

    parsed_count = int(
        parsed.notna().sum()
    )

    source_non_null_count = int(
        series.notna().sum()
    )

    invalid_count = max(
        source_non_null_count
        - parsed_count,
        0,
    )

    parse_success_percent = (
        round(
            (
                parsed_count
                / source_non_null_count
                * 100
            ),
            2,
        )
        if source_non_null_count
        else 0.0
    )

    clean = (
        parsed
        .dropna()
    )

    duplicate_timestamp_count = int(
        clean.duplicated().sum()
    )

    unique_timestamp_count = int(
        clean.nunique()
    )

    if len(clean) > 0:

        minimum = (
            clean.min()
        )

        maximum = (
            clean.max()
        )

        span_seconds = float(
            (
                maximum
                - minimum
            ).total_seconds()
        )

        start_value = (
            minimum.isoformat()
        )

        end_value = (
            maximum.isoformat()
        )

        span_days = round(
            span_seconds
            / 86400,
            4,
        )

    else:

        start_value = None
        end_value = None
        span_days = None


    # --------------------------------------------------------
    # Ordering
    # --------------------------------------------------------

    non_null_original_order = (
        parsed.dropna()
    )

    is_monotonic_increasing = bool(
        non_null_original_order
        .is_monotonic_increasing
    )

    is_monotonic_decreasing = bool(
        non_null_original_order
        .is_monotonic_decreasing
    )


    # --------------------------------------------------------
    # Intervals
    # --------------------------------------------------------

    interval_analysis = (
        _analyze_intervals(
            parsed
        )
    )


    # --------------------------------------------------------
    # Patterns
    # --------------------------------------------------------

    patterns = (
        _temporal_patterns(
            parsed
        )
    )


    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    return {
        "column":
            column_name,

        "total_rows":
            total_rows,

        "source_non_null_count":
            source_non_null_count,

        "parsed_count":
            parsed_count,

        "parse_success_percent":
            parse_success_percent,

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

        "invalid_datetime_count":
            invalid_count,

        "unique_timestamp_count":
            unique_timestamp_count,

        "duplicate_timestamp_count":
            duplicate_timestamp_count,

        "start":
            start_value,

        "end":
            end_value,

        "span_days":
            span_days,

        "has_time_component":
            _has_time_component(
                parsed
            ),

        "is_monotonic_increasing":
            is_monotonic_increasing,

        "is_monotonic_decreasing":
            is_monotonic_decreasing,

        "interval_analysis":
            interval_analysis,

        "patterns":
            patterns,
    }


# ------------------------------------------------------------
# Multiple datetime columns
# ------------------------------------------------------------

def analyze_datetime_columns(
    df: pd.DataFrame,
    datetime_columns: list[str] | None = None,
) -> dict[str, Any]:
    """
    Analyze datetime columns selected by the schema router.
    """

    if not datetime_columns:
        return {}

    results = {}

    for column in datetime_columns:

        if column not in df.columns:
            continue

        results[column] = (
            analyze_datetime_column(
                series=df[column],
                column_name=column,
            )
        )

    return results