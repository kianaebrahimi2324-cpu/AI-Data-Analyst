from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


# ============================================================
# HELPERS
# ============================================================

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


def _safe_float(
    value: Any,
) -> float | None:
    try:
        number = float(
            value
        )

        if not np.isfinite(
            number
        ):
            return None

        return number

    except (
        TypeError,
        ValueError,
    ):
        return None


# ============================================================
# SINGLE COORDINATE ANALYSIS
# ============================================================

def _analyze_coordinate_column(
    series: pd.Series,
    coordinate_type: str,
) -> dict[str, Any]:
    """
    Validate a latitude or longitude column.

    Geographic coordinates are validated using valid
    coordinate ranges rather than generic IQR outliers.
    """

    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    total_rows = int(
        len(series)
    )

    valid_numeric = numeric.notna()

    numeric_count = int(
        valid_numeric.sum()
    )

    missing_or_invalid_count = (
        total_rows
        - numeric_count
    )


    if coordinate_type == "latitude":

        lower_limit = -90.0
        upper_limit = 90.0

    elif coordinate_type == "longitude":

        lower_limit = -180.0
        upper_limit = 180.0

    else:

        raise ValueError(
            "coordinate_type must be "
            "'latitude' or 'longitude'."
        )


    within_range = (
        valid_numeric
        &
        numeric.between(
            lower_limit,
            upper_limit,
        )
    )

    valid_coordinate_count = int(
        within_range.sum()
    )

    invalid_range_count = int(
        (
            valid_numeric
            & ~numeric.between(
                lower_limit,
                upper_limit,
            )
        ).sum()
    )


    valid_values = numeric[
        within_range
    ]


    if valid_values.empty:

        minimum = None
        maximum = None
        mean = None
        median = None

    else:

        minimum = _safe_float(
            valid_values.min()
        )

        maximum = _safe_float(
            valid_values.max()
        )

        mean = _safe_float(
            valid_values.mean()
        )

        median = _safe_float(
            valid_values.median()
        )


    return {
        "coordinate_type":
            coordinate_type,

        "total_rows":
            total_rows,

        "numeric_count":
            numeric_count,

        "missing_or_non_numeric_count":
            missing_or_invalid_count,

        "valid_coordinate_count":
            valid_coordinate_count,

        "invalid_range_count":
            invalid_range_count,

        "valid_coordinate_percent":
            _safe_percent(
                valid_coordinate_count,
                total_rows,
            ),

        "invalid_range_percent":
            _safe_percent(
                invalid_range_count,
                total_rows,
            ),

        "valid_range": {
            "minimum":
                lower_limit,

            "maximum":
                upper_limit,
        },

        "observed_valid_min":
            minimum,

        "observed_valid_max":
            maximum,

        "observed_valid_mean":
            mean,

        "observed_valid_median":
            median,
    }


# ============================================================
# GEOSPATIAL PAIR ANALYSIS
# ============================================================

def analyze_geospatial_pair(
    df: pd.DataFrame,
    latitude_column: str,
    longitude_column: str,
) -> dict[str, Any]:
    """
    Analyze one latitude/longitude pair.

    This function does not classify geographically distant
    coordinates as statistical outliers.
    """

    if latitude_column not in df.columns:

        return {
            "status":
                "error",

            "reason":
                (
                    f"Latitude column "
                    f"'{latitude_column}' "
                    "was not found."
                ),
        }


    if longitude_column not in df.columns:

        return {
            "status":
                "error",

            "reason":
                (
                    f"Longitude column "
                    f"'{longitude_column}' "
                    "was not found."
                ),
        }


    latitude = pd.to_numeric(
        df[
            latitude_column
        ],
        errors="coerce",
    )

    longitude = pd.to_numeric(
        df[
            longitude_column
        ],
        errors="coerce",
    )


    latitude_valid = (
        latitude.notna()
        &
        latitude.between(
            -90,
            90,
        )
    )

    longitude_valid = (
        longitude.notna()
        &
        longitude.between(
            -180,
            180,
        )
    )


    complete_valid_pair = (
        latitude_valid
        &
        longitude_valid
    )


    total_rows = int(
        len(df)
    )

    complete_pair_count = int(
        complete_valid_pair.sum()
    )


    both_missing = (
        latitude.isna()
        &
        longitude.isna()
    )

    partial_pair = (
        latitude.notna()
        ^
        longitude.notna()
    )


    invalid_latitude = (
        latitude.notna()
        &
        ~latitude.between(
            -90,
            90,
        )
    )

    invalid_longitude = (
        longitude.notna()
        &
        ~longitude.between(
            -180,
            180,
        )
    )


    valid_coordinates = pd.DataFrame({
        "latitude":
            latitude[
                complete_valid_pair
            ],

        "longitude":
            longitude[
                complete_valid_pair
            ],
    })


    duplicate_coordinate_pairs = int(
        valid_coordinates
        .duplicated()
        .sum()
    )


    if valid_coordinates.empty:

        bounding_box = None
        center = None

    else:

        bounding_box = {
            "min_latitude":
                _safe_float(
                    valid_coordinates[
                        "latitude"
                    ].min()
                ),

            "max_latitude":
                _safe_float(
                    valid_coordinates[
                        "latitude"
                    ].max()
                ),

            "min_longitude":
                _safe_float(
                    valid_coordinates[
                        "longitude"
                    ].min()
                ),

            "max_longitude":
                _safe_float(
                    valid_coordinates[
                        "longitude"
                    ].max()
                ),
        }

        center = {
            "median_latitude":
                _safe_float(
                    valid_coordinates[
                        "latitude"
                    ].median()
                ),

            "median_longitude":
                _safe_float(
                    valid_coordinates[
                        "longitude"
                    ].median()
                ),
        }


    issues = []

    if int(
        invalid_latitude.sum()
    ) > 0:

        issues.append(
            {
                "type":
                    "invalid_latitude_range",

                "count":
                    int(
                        invalid_latitude.sum()
                    ),
            }
        )


    if int(
        invalid_longitude.sum()
    ) > 0:

        issues.append(
            {
                "type":
                    "invalid_longitude_range",

                "count":
                    int(
                        invalid_longitude.sum()
                    ),
            }
        )


    if int(
        partial_pair.sum()
    ) > 0:

        issues.append(
            {
                "type":
                    "partial_coordinate_pair",

                "count":
                    int(
                        partial_pair.sum()
                    ),
            }
        )


    return {
        "status":
            "ok",

        "latitude_column":
            latitude_column,

        "longitude_column":
            longitude_column,

        "total_rows":
            total_rows,

        "valid_pair_count":
            complete_pair_count,

        "valid_pair_percent":
            _safe_percent(
                complete_pair_count,
                total_rows,
            ),

        "both_missing_count":
            int(
                both_missing.sum()
            ),

        "partial_pair_count":
            int(
                partial_pair.sum()
            ),

        "invalid_latitude_count":
            int(
                invalid_latitude.sum()
            ),

        "invalid_longitude_count":
            int(
                invalid_longitude.sum()
            ),

        "duplicate_coordinate_pairs":
            duplicate_coordinate_pairs,

        "latitude_analysis":
            _analyze_coordinate_column(
                df[
                    latitude_column
                ],
                "latitude",
            ),

        "longitude_analysis":
            _analyze_coordinate_column(
                df[
                    longitude_column
                ],
                "longitude",
            ),

        "bounding_box":
            bounding_box,

        "coordinate_center":
            center,

        "issues":
            issues,

        "interpretation_note":
            (
                "Geographic coordinates are validated "
                "against legal latitude/longitude ranges. "
                "Geographically distant valid coordinates "
                "are not treated as errors solely because "
                "they are statistical outliers."
            ),
    }


# ============================================================
# DATASET-LEVEL GEOSPATIAL ANALYSIS
# ============================================================

def analyze_geospatial_data(
    df: pd.DataFrame,
    analysis_plan: dict[str, Any],
) -> dict[str, Any]:
    """
    Analyze every geospatial pair selected by the router.
    """

    pairs = analysis_plan.get(
        "geospatial_pairs",
        [],
    )

    if not pairs:

        return {
            "status":
                "not_applicable",

            "pair_count":
                0,

            "pairs":
                [],
        }


    results = []

    for pair in pairs:

        if not isinstance(
            pair,
            dict,
        ):
            continue

        latitude = pair.get(
            "latitude"
        )

        longitude = pair.get(
            "longitude"
        )

        if not latitude or not longitude:
            continue

        results.append(
            analyze_geospatial_pair(
                df=df,
                latitude_column=str(
                    latitude
                ),
                longitude_column=str(
                    longitude
                ),
            )
        )


    return {
        "status":
            "ok"
            if results
            else "not_applicable",

        "pair_count":
            len(
                results
            ),

        "pairs":
            results,
    }