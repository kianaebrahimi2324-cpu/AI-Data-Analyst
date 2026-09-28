from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

TOP_CORRELATIONS_LIMIT = 15


# ------------------------------------------------------------
# Correlation analysis
# ------------------------------------------------------------

def analyze_correlations(
    df: pd.DataFrame,
    numeric_columns: list[str] | None = None,
) -> dict[str, Any]:
    """
    Analyze Pearson correlations between selected numeric columns.

    Column selection should normally come from the schema router.

    If numeric_columns is None, native numeric columns are used
    for backward compatibility.
    """

    # --------------------------------------------------------
    # Select columns
    # --------------------------------------------------------

    if numeric_columns is None:

        numeric_columns = (
            df.select_dtypes(
                include=[
                    np.number,
                    "bool",
                ]
            )
            .columns
            .tolist()
        )

    else:

        numeric_columns = [
            column
            for column in numeric_columns
            if column in df.columns
        ]


    # --------------------------------------------------------
    # Need at least two columns
    # --------------------------------------------------------

    if len(numeric_columns) < 2:

        return {
            "columns": numeric_columns,
            "correlation_matrix": {},
            "top_correlations": [],
        }


    # --------------------------------------------------------
    # Convert selected columns safely
    # --------------------------------------------------------

    numeric_df = pd.DataFrame(
        index=df.index
    )

    for column in numeric_columns:

        if pd.api.types.is_bool_dtype(
            df[column]
        ):

            numeric_df[column] = (
                df[column]
                .astype("Int64")
            )

        else:

            numeric_df[column] = (
                pd.to_numeric(
                    df[column],
                    errors="coerce",
                )
            )


    # --------------------------------------------------------
    # Remove unusable columns
    # --------------------------------------------------------

    usable_columns = []

    for column in numeric_df.columns:

        non_null = (
            numeric_df[column]
            .dropna()
        )

        if len(non_null) < 2:
            continue

        if non_null.nunique() < 2:
            continue

        usable_columns.append(
            column
        )


    if len(usable_columns) < 2:

        return {
            "columns": usable_columns,
            "correlation_matrix": {},
            "top_correlations": [],
        }


    numeric_df = numeric_df[
        usable_columns
    ]


    # --------------------------------------------------------
    # Pearson correlation
    # --------------------------------------------------------

    correlation_matrix = (
        numeric_df.corr(
            method="pearson"
        )
    )


    # --------------------------------------------------------
    # JSON-friendly matrix
    # --------------------------------------------------------

    matrix_dict = {}

    for row_column in (
        correlation_matrix.index
    ):

        matrix_dict[
            str(row_column)
        ] = {}

        for col_column in (
            correlation_matrix.columns
        ):

            value = correlation_matrix.loc[
                row_column,
                col_column,
            ]

            matrix_dict[
                str(row_column)
            ][
                str(col_column)
            ] = (
                round(
                    float(value),
                    6,
                )
                if pd.notna(value)
                else None
            )


    # --------------------------------------------------------
    # Unique correlation pairs
    # --------------------------------------------------------

    pairs = []

    columns = (
        correlation_matrix.columns
        .tolist()
    )

    for i in range(
        len(columns)
    ):

        for j in range(
            i + 1,
            len(columns)
        ):

            column_1 = columns[i]
            column_2 = columns[j]

            value = correlation_matrix.loc[
                column_1,
                column_2,
            ]

            if pd.isna(value):
                continue

            value = float(
                value
            )

            pairs.append(
                {
                    "column_1":
                        str(
                            column_1
                        ),

                    "column_2":
                        str(
                            column_2
                        ),

                    "correlation":
                        round(
                            value,
                            6,
                        ),

                    "absolute_correlation":
                        round(
                            abs(value),
                            6,
                        ),
                }
            )


    # --------------------------------------------------------
    # Strongest absolute relationships first
    # --------------------------------------------------------

    pairs.sort(
        key=lambda item:
            item[
                "absolute_correlation"
            ],
        reverse=True,
    )


    return {
        "columns":
            [
                str(column)
                for column
                in usable_columns
            ],

        "method":
            "pearson",

        "correlation_matrix":
            matrix_dict,

        "top_correlations":
            pairs[
                :TOP_CORRELATIONS_LIMIT
            ],
    }