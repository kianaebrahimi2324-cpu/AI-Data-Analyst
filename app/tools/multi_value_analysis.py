from __future__ import annotations

from collections import Counter
from typing import Any

import pandas as pd


# ============================================================
# HELPERS
# ============================================================

def _clean_value(
    value: Any,
) -> str | None:
    """
    Normalize one multi-value cell.
    """

    if pd.isna(
        value
    ):
        return None

    text = str(
        value
    ).strip()

    if not text:
        return None

    return text


def _safe_percent(
    count: int,
    total: int,
) -> float:
    if total == 0:
        return 0.0

    return round(
        count / total * 100,
        2,
    )


# ============================================================
# GROUP ANALYSIS
# ============================================================

def analyze_multi_value_group(
    df: pd.DataFrame,
    logical_name: str,
    source_columns: list[str],
) -> dict[str, Any]:
    """
    Analyze multiple physical columns representing one
    logical multi-valued field.

    Example:

        Activity
        Unnamed: 7
        Unnamed: 8
        ...

    becomes conceptually:

        activities = [value1, value2, value3, ...]
    """

    existing_columns = [
        column
        for column
        in source_columns
        if column in df.columns
    ]


    missing_columns = [
        column
        for column
        in source_columns
        if column not in df.columns
    ]


    if not existing_columns:

        return {
            "status":
                "error",

            "logical_name":
                logical_name,

            "reason":
                "No source columns were found.",
        }


    total_rows = int(
        len(df)
    )

    frequency = Counter()

    rows_with_values = 0
    rows_without_values = 0

    total_values = 0

    per_row_counts = []

    duplicate_values_within_rows = 0


    for _, row in df[
        existing_columns
    ].iterrows():

        values = []

        for column in existing_columns:

            value = _clean_value(
                row[
                    column
                ]
            )

            if value is not None:

                values.append(
                    value
                )


        if not values:

            rows_without_values += 1

            per_row_counts.append(
                0
            )

            continue


        rows_with_values += 1

        per_row_counts.append(
            len(
                values
            )
        )

        total_values += len(
            values
        )


        unique_row_values = list(
            dict.fromkeys(
                values
            )
        )


        duplicate_values_within_rows += (
            len(values)
            - len(
                unique_row_values
            )
        )


        frequency.update(
            unique_row_values
        )


    unique_values = len(
        frequency
    )


    if per_row_counts:

        average_values_per_row = round(
            sum(
                per_row_counts
            )
            / len(
                per_row_counts
            ),
            3,
        )

        average_values_when_present = (
            round(
                total_values
                / rows_with_values,
                3,
            )
            if rows_with_values
            else 0.0
        )

        maximum_values_per_row = max(
            per_row_counts
        )

        minimum_values_per_row = min(
            per_row_counts
        )

    else:

        average_values_per_row = 0.0
        average_values_when_present = 0.0
        maximum_values_per_row = 0
        minimum_values_per_row = 0


    top_values = []

    for value, count in frequency.most_common(
        20
    ):

        top_values.append({
            "value":
                value,

            "row_count":
                int(
                    count
                ),

            "row_percent":
                _safe_percent(
                    int(
                        count
                    ),
                    total_rows,
                ),
        })


    return {
        "status":
            "ok",

        "logical_name":
            logical_name,

        "source_columns":
            existing_columns,

        "missing_source_columns":
            missing_columns,

        "physical_column_count":
            len(
                existing_columns
            ),

        "total_rows":
            total_rows,

        "rows_with_values":
            rows_with_values,

        "rows_without_values":
            rows_without_values,

        "row_coverage_percent":
            _safe_percent(
                rows_with_values,
                total_rows,
            ),

        "total_value_occurrences":
            total_values,

        "unique_values":
            unique_values,

        "average_values_per_row":
            average_values_per_row,

        "average_values_when_present":
            average_values_when_present,

        "minimum_values_per_row":
            minimum_values_per_row,

        "maximum_values_per_row":
            maximum_values_per_row,

        "duplicate_values_within_rows":
            duplicate_values_within_rows,

        "top_values":
            top_values,

        "missingness_note":
            (
                "Empty overflow cells are treated as "
                "unused slots within a multi-valued field, "
                "not automatically as independent missing "
                "feature values."
            ),
    }


# ============================================================
# DATASET-LEVEL ANALYSIS
# ============================================================

def analyze_multi_value_data(
    df: pd.DataFrame,
    analysis_plan: dict[str, Any],
) -> dict[str, Any]:
    """
    Analyze every multi-value group selected by the router.
    """

    groups = analysis_plan.get(
        "multi_value_groups",
        [],
    )


    if not groups:

        return {
            "status":
                "not_applicable",

            "group_count":
                0,

            "groups":
                [],
        }


    results = []


    for group in groups:

        if not isinstance(
            group,
            dict,
        ):
            continue


        logical_name = str(
            group.get(
                "logical_name",
                "",
            )
        )


        source_columns = [
            str(column)
            for column
            in group.get(
                "source_columns",
                [],
            )
        ]


        if (
            not logical_name
            or not source_columns
        ):
            continue


        results.append(
            analyze_multi_value_group(
                df=df,
                logical_name=logical_name,
                source_columns=source_columns,
            )
        )


    return {
        "status":
            "ok"
            if results
            else "not_applicable",

        "group_count":
            len(
                results
            ),

        "groups":
            results,
    }