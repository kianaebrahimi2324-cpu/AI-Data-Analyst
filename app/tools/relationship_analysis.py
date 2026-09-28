from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

MIN_VALID_ROWS = 5

MAX_CATEGORICAL_LEVELS = 30

TOP_RELATIONSHIPS = 20

MAX_GROUP_SUMMARY = 10


# ------------------------------------------------------------
# Safe conversion
# ------------------------------------------------------------

def _to_numeric(
    series: pd.Series,
) -> pd.Series:
    """Convert a series to numeric safely."""

    if pd.api.types.is_bool_dtype(
        series
    ):
        return (
            series
            .astype("Int64")
            .astype(float)
        )

    return pd.to_numeric(
        series,
        errors="coerce",
    )


def _to_datetime(
    series: pd.Series,
) -> pd.Series:
    """Convert a series to datetime safely."""

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
# Eta squared
# ------------------------------------------------------------

def _eta_squared(
    numeric_series: pd.Series,
    categorical_series: pd.Series,
) -> tuple[float | None, int]:
    """
    Calculate eta squared for a numeric-categorical
    relationship.

    Eta squared measures how much of the numeric variation
    is associated with differences between categories.

    It does not imply causality or predictive performance.
    """

    numeric = _to_numeric(
        numeric_series
    )

    data = pd.DataFrame(
        {
            "numeric": numeric,
            "category": categorical_series,
        }
    ).dropna()


    if len(data) < MIN_VALID_ROWS:
        return None, len(data)


    category_count = int(
        data["category"].nunique()
    )


    if (
        category_count < 2
        or category_count
        > MAX_CATEGORICAL_LEVELS
    ):
        return None, len(data)


    values = data[
        "numeric"
    ]


    if values.nunique() < 2:
        return None, len(data)


    grand_mean = float(
        values.mean()
    )


    ss_total = float(
        (
            (
                values
                - grand_mean
            )
            ** 2
        ).sum()
    )


    if ss_total == 0:
        return 0.0, len(data)


    ss_between = 0.0


    for _, group in (
        data.groupby(
            "category",
            observed=True,
        )
    ):

        group_values = group[
            "numeric"
        ]

        group_mean = float(
            group_values.mean()
        )

        ss_between += (
            len(group_values)
            * (
                group_mean
                - grand_mean
            )
            ** 2
        )


    eta_squared = (
        ss_between
        / ss_total
    )


    return (
        round(
            float(
                eta_squared
            ),
            6,
        ),
        len(data),
    )


# ------------------------------------------------------------
# Group summary
# ------------------------------------------------------------

def _numeric_group_summary(
    numeric_series: pd.Series,
    categorical_series: pd.Series,
) -> list[dict[str, Any]]:
    """
    Summarize numeric values inside categories.
    """

    numeric = _to_numeric(
        numeric_series
    )

    data = pd.DataFrame(
        {
            "numeric": numeric,
            "category": categorical_series,
        }
    ).dropna()


    if len(data) == 0:
        return []


    summary = []


    grouped = data.groupby(
        "category",
        observed=True,
    )


    for category, group in grouped:

        values = group[
            "numeric"
        ]

        summary.append(
            {
                "category":
                    str(
                        category
                    ),

                "count":
                    int(
                        len(values)
                    ),

                "mean":
                    round(
                        float(
                            values.mean()
                        ),
                        6,
                    ),

                "median":
                    round(
                        float(
                            values.median()
                        ),
                        6,
                    ),
            }
        )


    summary.sort(
        key=lambda item:
            item["count"],
        reverse=True,
    )


    return summary[
        :MAX_GROUP_SUMMARY
    ]


# ------------------------------------------------------------
# Cramer's V
# ------------------------------------------------------------

def _cramers_v(
    series_a: pd.Series,
    series_b: pd.Series,
) -> tuple[float | None, int]:
    """
    Calculate bias-corrected Cramer's V.

    No scipy dependency is required.
    """

    data = pd.DataFrame(
        {
            "a": series_a,
            "b": series_b,
        }
    ).dropna()


    n = len(
        data
    )


    if n < MIN_VALID_ROWS:
        return None, n


    if (
        data["a"].nunique() < 2
        or data["b"].nunique() < 2
    ):
        return None, n


    if (
        data["a"].nunique()
        > MAX_CATEGORICAL_LEVELS
        or data["b"].nunique()
        > MAX_CATEGORICAL_LEVELS
    ):
        return None, n


    contingency = pd.crosstab(
        data["a"],
        data["b"],
    )


    observed = (
        contingency
        .to_numpy(
            dtype=float
        )
    )


    row_sums = (
        observed.sum(
            axis=1,
            keepdims=True,
        )
    )


    column_sums = (
        observed.sum(
            axis=0,
            keepdims=True,
        )
    )


    total = float(
        observed.sum()
    )


    if total == 0:
        return None, n


    expected = (
        row_sums
        @ column_sums
        / total
    )


    valid = (
        expected > 0
    )


    chi_squared = float(
        (
            (
                observed[
                    valid
                ]
                - expected[
                    valid
                ]
            )
            ** 2
            / expected[
                valid
            ]
        ).sum()
    )


    phi_squared = (
        chi_squared
        / total
    )


    rows, columns = (
        observed.shape
    )


    if total <= 1:
        return None, n


    phi_corrected = max(
        0.0,
        phi_squared
        - (
            (
                columns - 1
            )
            * (
                rows - 1
            )
        )
        / (
            total - 1
        ),
    )


    rows_corrected = (
        rows
        - (
            (
                rows - 1
            )
            ** 2
        )
        / (
            total - 1
        )
    )


    columns_corrected = (
        columns
        - (
            (
                columns - 1
            )
            ** 2
        )
        / (
            total - 1
        )
    )


    denominator = min(
        rows_corrected - 1,
        columns_corrected - 1,
    )


    if denominator <= 0:
        return 0.0, n


    cramers_v = np.sqrt(
        phi_corrected
        / denominator
    )


    return (
        round(
            float(
                cramers_v
            ),
            6,
        ),
        n,
    )


# ------------------------------------------------------------
# Datetime to numeric
# ------------------------------------------------------------

def _datetime_numeric_relationship(
    datetime_series: pd.Series,
    numeric_series: pd.Series,
) -> dict[str, Any] | None:
    """
    Measure monotonic/linear association between time
    and a numeric variable.
    """

    datetime_values = _to_datetime(
        datetime_series
    )

    numeric_values = _to_numeric(
        numeric_series
    )


    data = pd.DataFrame(
        {
            "datetime": datetime_values,
            "numeric": numeric_values,
        }
    ).dropna()


    if len(data) < MIN_VALID_ROWS:
        return None


    if (
        data["datetime"].nunique() < 2
        or data["numeric"].nunique() < 2
    ):
        return None


    timestamps = (
        data["datetime"]
        .astype("int64")
        .astype(float)
        / 1_000_000_000
    )


    pearson = pd.Series(
        timestamps
    ).corr(
        data["numeric"],
        method="pearson",
    )


    spearman = pd.Series(
        timestamps
    ).corr(
        data["numeric"],
        method="spearman",
    )


    if (
        pd.isna(pearson)
        and pd.isna(spearman)
    ):
        return None


    return {
        "valid_rows":
            int(
                len(data)
            ),

        "pearson_with_time":
            (
                round(
                    float(
                        pearson
                    ),
                    6,
                )
                if pd.notna(
                    pearson
                )
                else None
            ),

        "spearman_with_time":
            (
                round(
                    float(
                        spearman
                    ),
                    6,
                )
                if pd.notna(
                    spearman
                )
                else None
            ),

        "start":
            data[
                "datetime"
            ].min().isoformat(),

        "end":
            data[
                "datetime"
            ].max().isoformat(),
    }


# ------------------------------------------------------------
# Datetime bucket
# ------------------------------------------------------------

def _build_time_bucket(
    datetime_series: pd.Series,
) -> tuple[pd.Series, str]:
    """
    Create a reasonable temporal bucket based on time span.
    """

    values = _to_datetime(
        datetime_series
    )


    clean = values.dropna()


    if len(clean) == 0:

        return (
            pd.Series(
                index=values.index,
                dtype="object",
            ),
            "unknown",
        )


    span_days = float(
        (
            clean.max()
            - clean.min()
        ).total_seconds()
        / 86400
    )


    if span_days <= 3:

        bucket = (
            values.dt.strftime(
                "%Y-%m-%d %H:00"
            )
        )

        return bucket, "hour"


    if span_days <= 120:

        week_start = (
            values
            - pd.to_timedelta(
                values.dt.weekday,
                unit="D",
            )
        )

        bucket = (
            week_start
            .dt.strftime(
                "%Y-%m-%d"
            )
        )

        return bucket, "week"


    if span_days <= 1095:

        bucket = (
            values.dt.strftime(
                "%Y-%m"
            )
        )

        return bucket, "month"


    bucket = (
        values.dt.strftime(
            "%Y"
        )
    )

    return bucket, "year"


# ------------------------------------------------------------
# Numeric-categorical relationships
# ------------------------------------------------------------

def _analyze_numeric_categorical(
    df: pd.DataFrame,
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> list[dict[str, Any]]:

    results = []


    for numeric_column in numeric_columns:

        for categorical_column in (
            categorical_columns
        ):

            if (
                numeric_column
                == categorical_column
            ):
                continue


            eta_squared, valid_rows = (
                _eta_squared(
                    df[
                        numeric_column
                    ],
                    df[
                        categorical_column
                    ],
                )
            )


            if eta_squared is None:
                continue


            results.append(
                {
                    "numeric_column":
                        numeric_column,

                    "categorical_column":
                        categorical_column,

                    "valid_rows":
                        int(
                            valid_rows
                        ),

                    "eta_squared":
                        eta_squared,

                    "group_summary":
                        _numeric_group_summary(
                            df[
                                numeric_column
                            ],
                            df[
                                categorical_column
                            ],
                        ),
                }
            )


    results.sort(
        key=lambda item:
            item[
                "eta_squared"
            ],
        reverse=True,
    )


    return results[
        :TOP_RELATIONSHIPS
    ]


# ------------------------------------------------------------
# Categorical-categorical relationships
# ------------------------------------------------------------

def _analyze_categorical_categorical(
    df: pd.DataFrame,
    categorical_columns: list[str],
) -> list[dict[str, Any]]:

    results = []


    for (
        column_a,
        column_b,
    ) in combinations(
        categorical_columns,
        2,
    ):

        value, valid_rows = (
            _cramers_v(
                df[
                    column_a
                ],
                df[
                    column_b
                ],
            )
        )


        if value is None:
            continue


        results.append(
            {
                "column_1":
                    column_a,

                "column_2":
                    column_b,

                "valid_rows":
                    int(
                        valid_rows
                    ),

                "cramers_v":
                    value,
            }
        )


    results.sort(
        key=lambda item:
            item[
                "cramers_v"
            ],
        reverse=True,
    )


    return results[
        :TOP_RELATIONSHIPS
    ]


# ------------------------------------------------------------
# Datetime-numeric relationships
# ------------------------------------------------------------

def _analyze_datetime_numeric(
    df: pd.DataFrame,
    datetime_columns: list[str],
    numeric_columns: list[str],
) -> list[dict[str, Any]]:

    results = []


    for datetime_column in (
        datetime_columns
    ):

        for numeric_column in (
            numeric_columns
        ):

            relationship = (
                _datetime_numeric_relationship(
                    df[
                        datetime_column
                    ],
                    df[
                        numeric_column
                    ],
                )
            )


            if relationship is None:
                continue


            relationship[
                "datetime_column"
            ] = datetime_column

            relationship[
                "numeric_column"
            ] = numeric_column


            results.append(
                relationship
            )


    results.sort(
        key=lambda item:
            max(
                abs(
                    item.get(
                        "pearson_with_time"
                    )
                    or 0.0
                ),
                abs(
                    item.get(
                        "spearman_with_time"
                    )
                    or 0.0
                ),
            ),
        reverse=True,
    )


    return results[
        :TOP_RELATIONSHIPS
    ]


# ------------------------------------------------------------
# Datetime-categorical relationships
# ------------------------------------------------------------

def _analyze_datetime_categorical(
    df: pd.DataFrame,
    datetime_columns: list[str],
    categorical_columns: list[str],
) -> list[dict[str, Any]]:

    results = []


    for datetime_column in (
        datetime_columns
    ):

        time_bucket, bucket_type = (
            _build_time_bucket(
                df[
                    datetime_column
                ]
            )
        )


        if time_bucket.nunique(
            dropna=True
        ) < 2:
            continue


        if time_bucket.nunique(
            dropna=True
        ) > MAX_CATEGORICAL_LEVELS:
            continue


        for categorical_column in (
            categorical_columns
        ):

            value, valid_rows = (
                _cramers_v(
                    time_bucket,
                    df[
                        categorical_column
                    ],
                )
            )


            if value is None:
                continue


            results.append(
                {
                    "datetime_column":
                        datetime_column,

                    "categorical_column":
                        categorical_column,

                    "time_bucket":
                        bucket_type,

                    "valid_rows":
                        int(
                            valid_rows
                        ),

                    "cramers_v":
                        value,
                }
            )


    results.sort(
        key=lambda item:
            item[
                "cramers_v"
            ],
        reverse=True,
    )


    return results[
        :TOP_RELATIONSHIPS
    ]


# ------------------------------------------------------------
# Main cross-type engine
# ------------------------------------------------------------

def analyze_cross_type_relationships(
    df: pd.DataFrame,
    analysis_plan: dict[str, Any],
    target_column: str = "",
) -> dict[str, Any]:
    """
    Analyze relationships between different semantic
    data families.
    """

    numeric_columns = list(
        analysis_plan.get(
            "numeric_columns",
            [],
        )
    )


    categorical_columns = list(
        analysis_plan.get(
            "categorical_columns",
            [],
        )
    )


    datetime_columns = list(
        analysis_plan.get(
            "datetime_columns",
            [],
        )
    )


    # --------------------------------------------------------
    # Add target to the appropriate family
    # --------------------------------------------------------

    target_info = (
        analysis_plan.get(
            "target",
            {},
        )
    )


    if (
        target_column
        and target_column in df.columns
        and target_info
    ):

        target_family = (
            target_info.get(
                "analysis_family"
            )
        )


        if (
            target_family == "numeric"
            and target_column
            not in numeric_columns
        ):

            numeric_columns.append(
                target_column
            )


        elif (
            target_family
            == "categorical"
            and target_column
            not in categorical_columns
        ):

            categorical_columns.append(
                target_column
            )


    # --------------------------------------------------------
    # Run relationship analyses
    # --------------------------------------------------------

    numeric_categorical = (
        _analyze_numeric_categorical(
            df=df,
            numeric_columns=numeric_columns,
            categorical_columns=(
                categorical_columns
            ),
        )
    )


    categorical_categorical = (
        _analyze_categorical_categorical(
            df=df,
            categorical_columns=(
                categorical_columns
            ),
        )
    )


    datetime_numeric = (
        _analyze_datetime_numeric(
            df=df,
            datetime_columns=(
                datetime_columns
            ),
            numeric_columns=(
                numeric_columns
            ),
        )
    )


    datetime_categorical = (
        _analyze_datetime_categorical(
            df=df,
            datetime_columns=(
                datetime_columns
            ),
            categorical_columns=(
                categorical_columns
            ),
        )
    )


    return {
        "numeric_categorical":
            numeric_categorical,

        "categorical_categorical":
            categorical_categorical,

        "datetime_numeric":
            datetime_numeric,

        "datetime_categorical":
            datetime_categorical,

        "methods": {
            "numeric_categorical":
                "eta_squared",

            "categorical_categorical":
                "bias_corrected_cramers_v",

            "datetime_numeric":
                [
                    "pearson_with_time",
                    "spearman_with_time",
                ],

            "datetime_categorical":
                "bias_corrected_cramers_v_on_time_buckets",
        },

        "notes": [
            (
                "Association measures describe statistical "
                "relationships and do not imply causality."
            ),
            (
                "Datetime-numeric correlations measure "
                "association with chronological time."
            ),
        ],
    }