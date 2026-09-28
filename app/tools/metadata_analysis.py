from __future__ import annotations

from typing import Any

import pandas as pd


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


def _is_present(
    series: pd.Series,
) -> pd.Series:
    """
    Treat NaN and blank strings as missing.
    """

    present = series.notna()

    if (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
        or isinstance(
            series.dtype,
            pd.CategoricalDtype,
        )
    ):
        text = (
            series
            .astype("string")
            .str.strip()
        )

        present = (
            present
            & text.notna()
            & text.ne("")
        )

    return present


def _profile_column(
    df: pd.DataFrame,
    column: str,
    semantic_role: str,
) -> dict[str, Any]:
    """
    Profile a semantic metadata/entity column
    without exposing raw values.
    """

    if column not in df.columns:
        return {
            "status": "error",
            "column": column,
            "semantic_role": semantic_role,
            "reason": "Column not found.",
        }

    series = df[column]

    total_rows = int(
        len(df)
    )

    present_mask = _is_present(
        series
    )

    present_count = int(
        present_mask.sum()
    )

    missing_count = (
        total_rows
        - present_count
    )

    non_null = series[
        present_mask
    ]

    unique_count = int(
        non_null.nunique(
            dropna=True
        )
    )

    duplicate_non_null_count = (
        present_count
        - unique_count
    )

    return {
        "status":
            "ok",

        "column":
            column,

        "semantic_role":
            semantic_role,

        "total_rows":
            total_rows,

        "present_count":
            present_count,

        "missing_count":
            missing_count,

        "coverage_percent":
            _safe_percent(
                present_count,
                total_rows,
            ),

        "unique_count":
            unique_count,

        "unique_ratio":
            round(
                unique_count
                / present_count,
                4,
            )
            if present_count
            else 0.0,

        "duplicate_non_null_count":
            duplicate_non_null_count,
    }


def analyze_metadata_data(
    df: pd.DataFrame,
    analysis_plan: dict[str, Any],
) -> dict[str, Any]:
    """
    Analyze metadata-like and entity-like columns.

    Raw values are intentionally not returned to reduce
    unnecessary exposure of names, phone numbers, emails,
    addresses, or identifiers to downstream LLMs.
    """

    metadata_columns = (
        analysis_plan.get(
            "metadata_columns",
            {},
        )
    )

    entity_columns = list(
        analysis_plan.get(
            "entity_columns",
            [],
        )
    )

    metadata_results = {}

    for role in (
        "url",
        "email",
        "phone",
        "address",
    ):

        metadata_results[
            role
        ] = []

        for column in (
            metadata_columns.get(
                role,
                [],
            )
        ):

            metadata_results[
                role
            ].append(
                _profile_column(
                    df=df,
                    column=str(
                        column
                    ),
                    semantic_role=role,
                )
            )

    entity_results = [
        _profile_column(
            df=df,
            column=str(column),
            semantic_role="entity_name",
        )
        for column
        in entity_columns
    ]

    analyzed_count = (
        sum(
            len(values)
            for values
            in metadata_results.values()
        )
        + len(
            entity_results
        )
    )

    return {
        "status":
            (
                "ok"
                if analyzed_count
                else "not_applicable"
            ),

        "analyzed_column_count":
            analyzed_count,

        "metadata":
            metadata_results,

        "entities":
            entity_results,
    }