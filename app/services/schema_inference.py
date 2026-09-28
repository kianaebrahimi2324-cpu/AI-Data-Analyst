from __future__ import annotations

import re
import unicodedata
import warnings
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

TEXT_SAMPLE_SIZE = 300


URL_PREFIX_PATTERN = re.compile(
    r"^(?:https?://|www\.)",
    flags=re.IGNORECASE,
)

DOMAIN_PATTERN = re.compile(
    r"^(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+"
    r"[a-z]{2,}(?:[/:?#].*)?$",
    flags=re.IGNORECASE,
)

EMAIL_PATTERN = re.compile(
    r"^[^@\s]+@[^@\s]+\.[^@\s]+$"
)

PHONE_PATTERN = re.compile(
    r"^\+?[\d\s().-]{7,}$"
)

UNNAMED_PATTERN = re.compile(
    r"^unnamed:\s*\d+$",
    flags=re.IGNORECASE,
)

DATE_PATTERNS = [
    re.compile(
        r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}"
        r"(?:[ T].*)?$"
    ),
    re.compile(
        r"^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}"
        r"(?:[ T].*)?$"
    ),
]


# ============================================================
# BASIC HELPERS
# ============================================================

def _safe_ratio(
    numerator: float,
    denominator: float,
) -> float:
    if denominator == 0:
        return 0.0

    return float(
        numerator / denominator
    )


def _ascii_text(
    value: Any,
) -> str:
    """
    Normalize accented text for safer name matching.
    """

    text = str(
        value
    ).strip().lower()

    normalized = unicodedata.normalize(
        "NFKD",
        text,
    )

    return "".join(
        character
        for character in normalized
        if not unicodedata.combining(
            character
        )
    )


def _normalized_name(
    column: Any,
) -> str:
    text = _ascii_text(
        column
    )

    return re.sub(
        r"[^a-z0-9]+",
        "_",
        text,
    ).strip("_")


def _sample_strings(
    series: pd.Series,
    limit: int = TEXT_SAMPLE_SIZE,
) -> list[str]:
    values = (
        series
        .dropna()
        .astype(str)
        .str.strip()
    )

    values = values[
        values != ""
    ]

    return values.head(
        limit
    ).tolist()


def _column_stats(
    series: pd.Series,
) -> dict:
    total = int(
        len(series)
    )

    non_null = int(
        series.notna().sum()
    )

    missing = int(
        series.isna().sum()
    )

    unique = int(
        series.nunique(
            dropna=True
        )
    )

    return {
        "row_count":
            total,

        "non_null_count":
            non_null,

        "missing_count":
            missing,

        "missing_ratio":
            round(
                _safe_ratio(
                    missing,
                    total,
                ),
                4,
            ),

        "unique_count":
            unique,

        "unique_ratio":
            round(
                _safe_ratio(
                    unique,
                    non_null,
                ),
                4,
            ),
    }


# ============================================================
# COLUMN NAME HINTS
# ============================================================

def _name_hints(
    column: str,
) -> set[str]:
    name = _normalized_name(
        column
    )

    hints = set()


    # Identifier
    if (
        name == "id"
        or name.startswith(
            "id_"
        )
        or name.endswith(
            "_id"
        )
        or "identifier" in name
        or "uuid" in name
    ):
        hints.add(
            "identifier"
        )


    # Exported dataframe index
    if name in {
        "index",
        "row_index",
        "row_number",
        "row_num",
    }:
        hints.add(
            "index"
        )


    # URL
    if any(
        token in name
        for token in [
            "url",
            "website",
            "web_site",
            "homepage",
            "site_web",
            "site_internet",
        ]
    ):
        hints.add(
            "url"
        )


    # Email
    if any(
        token in name
        for token in [
            "email",
            "e_mail",
            "mail_address",
        ]
    ):
        hints.add(
            "email"
        )


    # Phone
    if (
        name in {
            "phone",
            "telephone",
            "tel",
            "mobile",
            "fax",
            "phone_number",
            "telephone_number",
        }
        or name.endswith(
            "_phone"
        )
        or name.endswith(
            "_telephone"
        )
    ):
        hints.add(
            "phone"
        )


    # Address
    if any(
        token in name
        for token in [
            "address",
            "adresse",
            "street",
            "rue",
            "postal",
            "postcode",
            "zip",
        ]
    ):
        hints.add(
            "address"
        )


    # Latitude
    if name in {
        "lat",
        "latitude",
        "y_coord",
        "y_coordinate",
    }:
        hints.add(
            "latitude"
        )


    # Longitude
    if name in {
        "lng",
        "lon",
        "long",
        "longitude",
        "x_coord",
        "x_coordinate",
    }:
        hints.add(
            "longitude"
        )


    # Datetime
    if any(
        token in name
        for token in [
            "date",
            "time",
            "timestamp",
            "created_at",
            "updated_at",
            "datetime",
        ]
    ):
        hints.add(
            "datetime"
        )


    # Entity/name fields
    if (
        name in {
            "name",
            "nom",
            "title",
            "structure",
            "company",
            "organisation",
            "organization",
            "entity",
        }
        or name.endswith(
            "_name"
        )
        or name.endswith(
            "_nom"
        )
    ):
        hints.add(
            "entity_name"
        )


    # Suspicious overflow/exported columns
    if UNNAMED_PATTERN.match(
        _ascii_text(
            column
        )
    ):
        hints.add(
            "unnamed"
        )


    return hints


# ============================================================
# SPECIAL VALUE DETECTORS
# ============================================================

def _is_date_like_string(
    value: str,
) -> bool:
    text = str(
        value
    ).strip()

    return any(
        pattern.match(
            text
        )
        for pattern
        in DATE_PATTERNS
    )


def _looks_like_url(
    series: pd.Series,
) -> tuple[bool, float]:
    values = _sample_strings(
        series
    )

    if not values:
        return False, 0.0

    matches = 0

    for value in values:

        if (
            URL_PREFIX_PATTERN.search(
                value
            )
            or DOMAIN_PATTERN.match(
                value
            )
        ):
            matches += 1

    ratio = _safe_ratio(
        matches,
        len(values),
    )

    return (
        ratio >= 0.60,
        ratio,
    )


def _looks_like_email(
    series: pd.Series,
) -> tuple[bool, float]:
    values = _sample_strings(
        series
    )

    if not values:
        return False, 0.0

    matches = sum(
        bool(
            EMAIL_PATTERN.match(
                value
            )
        )
        for value in values
    )

    ratio = _safe_ratio(
        matches,
        len(values),
    )

    return (
        ratio >= 0.70,
        ratio,
    )


def _looks_like_datetime(
    series: pd.Series,
    hints: set[str],
) -> tuple[bool, float]:
    """
    Datetime detection is intentionally conservative.

    We first look for date-like lexical patterns before
    asking pandas to parse values.
    """

    if pd.api.types.is_datetime64_any_dtype(
        series
    ):
        return True, 1.0

    if pd.api.types.is_numeric_dtype(
        series
    ):
        return False, 0.0

    values = _sample_strings(
        series
    )

    if len(values) < 3:
        return False, 0.0

    lexical_ratio = _safe_ratio(
        sum(
            _is_date_like_string(
                value
            )
            for value in values
        ),
        len(values),
    )

    if (
        "datetime" not in hints
        and lexical_ratio < 0.50
    ):
        return False, 0.0

    with warnings.catch_warnings():

        warnings.simplefilter(
            "ignore"
        )

        try:

            parsed = pd.to_datetime(
                pd.Series(
                    values
                ),
                errors="coerce",
                format="mixed",
            )

        except TypeError:

            parsed = pd.to_datetime(
                pd.Series(
                    values
                ),
                errors="coerce",
            )

    ratio = float(
        parsed.notna().mean()
    )

    return (
        ratio >= 0.85,
        ratio,
    )


def _looks_like_phone(
    series: pd.Series,
    hints: set[str],
) -> tuple[bool, float]:
    """
    Detect phone-like strings while explicitly rejecting dates.
    """

    values = _sample_strings(
        series
    )

    if not values:
        return False, 0.0

    matches = 0

    for value in values:

        if _is_date_like_string(
            value
        ):
            continue

        if not PHONE_PATTERN.match(
            value
        ):
            continue

        digits = re.sub(
            r"\D",
            "",
            value,
        )

        if not (
            7 <= len(digits) <= 16
        ):
            continue

        has_phone_signal = (
            value.startswith("+")
            or bool(
                re.search(
                    r"[\s().]",
                    value,
                )
            )
            or "phone" in hints
        )

        if has_phone_signal:
            matches += 1

    ratio = _safe_ratio(
        matches,
        len(values),
    )

    return (
        ratio >= 0.70,
        ratio,
    )


def _coordinate_valid_ratio(
    series: pd.Series,
    coordinate_type: str,
) -> float:
    numeric = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if numeric.empty:
        return 0.0

    if coordinate_type == "latitude":

        valid = numeric.between(
            -90,
            90,
        )

    elif coordinate_type == "longitude":

        valid = numeric.between(
            -180,
            180,
        )

    else:

        return 0.0

    return float(
        valid.mean()
    )


def _is_sequential_index(
    series: pd.Series,
) -> bool:
    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    if numeric.isna().any():
        return False

    if len(numeric) < 3:
        return False

    values = numeric.to_numpy()

    differences = np.diff(
        values
    )

    return bool(
        np.all(
            differences == 1
        )
    )


# ============================================================
# COLUMN SCHEMA INFERENCE
# ============================================================

def infer_column_schema(
    series: pd.Series,
    column_name: str,
) -> dict:
    """
    Infer both analytical data type and semantic role.

    semantic_type describes how values behave statistically.

    semantic_role describes what the column appears to represent.
    """

    stats = _column_stats(
        series
    )

    hints = _name_hints(
        column_name
    )

    non_null = (
        series
        .dropna()
    )

    result = {
        "column":
            str(
                column_name
            ),

        "semantic_type":
            "unknown",

        "semantic_role":
            "feature",

        "confidence":
            0.50,

        "physical_dtype":
            str(
                series.dtype
            ),

        **stats,

        "flags":
            [],
    }


    # ========================================================
    # SPECIAL HANDLING FOR UNNAMED COLUMNS
    # ========================================================

    if "unnamed" in hints:

        if (
            stats[
                "non_null_count"
            ]
            == 0
        ):

            result.update({
                "semantic_type":
                    "empty",

                "semantic_role":
                    "structural_overflow_candidate",

                "confidence":
                    0.95,
            })

            result[
                "flags"
            ].extend([
                "unnamed_column",
                "empty_structural_column",
            ])

            return result


        if _is_sequential_index(
            series
        ):

            result.update({
                "semantic_type":
                    "identifier",

                "semantic_role":
                    "exported_index",

                "confidence":
                    0.99,
            })

            return result


        result.update({
            "semantic_type":
                "categorical",

            "semantic_role":
                "structural_overflow_candidate",

            "confidence":
                0.85,
        })

        result[
            "flags"
        ].append(
            "unnamed_column"
        )

        if (
            stats[
                "unique_count"
            ]
            <= 1
        ):
            result[
                "flags"
            ].append(
                "sparse_constant_overflow"
            )

        return result


    # ========================================================
    # EMPTY
    # ========================================================

    if (
        stats[
            "non_null_count"
        ]
        == 0
    ):

        result.update({
            "semantic_type":
                "empty",

            "semantic_role":
                "structural",

            "confidence":
                1.0,
        })

        return result


    # ========================================================
    # CONSTANT
    # ========================================================

    if (
        stats[
            "unique_count"
        ]
        <= 1
    ):

        result.update({
            "semantic_type":
                "constant",

            "semantic_role":
                "non_informative",

            "confidence":
                1.0,
        })

        return result


    # ========================================================
    # BOOLEAN / BINARY
    # ========================================================

    if pd.api.types.is_bool_dtype(
        series
    ):

        result.update({
            "semantic_type":
                "binary",

            "semantic_role":
                "feature",

            "confidence":
                1.0,
        })

        return result


    unique_values = {
        str(
            value
        ).strip().lower()

        for value
        in non_null.unique()[:20]
    }

    binary_sets = [
        {
            "true",
            "false",
        },
        {
            "yes",
            "no",
        },
        {
            "y",
            "n",
        },
        {
            "0",
            "1",
        },
    ]

    if (
        stats[
            "unique_count"
        ]
        == 2
        and any(
            unique_values
            == candidate
            for candidate
            in binary_sets
        )
    ):

        result.update({
            "semantic_type":
                "binary",

            "semantic_role":
                "feature",

            "confidence":
                0.98,
        })

        return result


    # ========================================================
    # GEOSPATIAL COORDINATES
    # ========================================================

    if (
        "latitude" in hints
        and pd.api.types.is_numeric_dtype(
            series
        )
    ):

        valid_ratio = (
            _coordinate_valid_ratio(
                series,
                "latitude",
            )
        )

        result.update({
            "semantic_type":
                "numeric_continuous",

            "semantic_role":
                "latitude",

            "confidence":
                round(
                    0.75
                    + 0.25
                    * valid_ratio,
                    3,
                ),

            "coordinate_valid_ratio":
                round(
                    valid_ratio,
                    4,
                ),
        })

        if valid_ratio < 1.0:

            result[
                "flags"
            ].append(
                "invalid_coordinate_values"
            )

        return result


    if (
        "longitude" in hints
        and pd.api.types.is_numeric_dtype(
            series
        )
    ):

        valid_ratio = (
            _coordinate_valid_ratio(
                series,
                "longitude",
            )
        )

        result.update({
            "semantic_type":
                "numeric_continuous",

            "semantic_role":
                "longitude",

            "confidence":
                round(
                    0.75
                    + 0.25
                    * valid_ratio,
                    3,
                ),

            "coordinate_valid_ratio":
                round(
                    valid_ratio,
                    4,
                ),
        })

        if valid_ratio < 1.0:

            result[
                "flags"
            ].append(
                "invalid_coordinate_values"
            )

        return result


    # ========================================================
    # URL
    # ========================================================

    url_match, url_ratio = (
        _looks_like_url(
            series
        )
    )

    if (
        url_match
        or (
            "url" in hints
            and url_ratio >= 0.25
        )
    ):

        result.update({
            "semantic_type":
                "text",

            "semantic_role":
                "url",

            "confidence":
                round(
                    max(
                        0.80,
                        url_ratio,
                    ),
                    3,
                ),
        })

        return result


    # ========================================================
    # EMAIL
    # ========================================================

    email_match, email_ratio = (
        _looks_like_email(
            series
        )
    )

    if (
        email_match
        or (
            "email" in hints
            and email_ratio >= 0.25
        )
    ):

        result.update({
            "semantic_type":
                "text",

            "semantic_role":
                "email",

            "confidence":
                round(
                    max(
                        0.80,
                        email_ratio,
                    ),
                    3,
                ),
        })

        return result


    # ========================================================
    # DATETIME
    # ========================================================

    datetime_match, datetime_ratio = (
        _looks_like_datetime(
            series,
            hints,
        )
    )

    if (
        datetime_match
        or (
            "datetime" in hints
            and datetime_ratio >= 0.50
        )
    ):

        result.update({
            "semantic_type":
                "datetime",

            "semantic_role":
                "temporal",

            "confidence":
                round(
                    max(
                        0.80,
                        datetime_ratio,
                    ),
                    3,
                ),
        })

        return result


    # ========================================================
    # PHONE
    # ========================================================

    phone_match, phone_ratio = (
        _looks_like_phone(
            series,
            hints,
        )
    )

    if (
        phone_match
        or (
            "phone" in hints
            and phone_ratio >= 0.25
        )
    ):

        result.update({
            "semantic_type":
                "text",

            "semantic_role":
                "phone",

            "confidence":
                round(
                    max(
                        0.80,
                        phone_ratio,
                    ),
                    3,
                ),
        })

        return result


    # ========================================================
    # ADDRESS
    # ========================================================

    if (
        "address" in hints
        and not pd.api.types.is_numeric_dtype(
            series
        )
    ):

        result.update({
            "semantic_type":
                "text",

            "semantic_role":
                "address",

            "confidence":
                0.88,
        })

        return result


    # ========================================================
    # EXPLICIT IDENTIFIER
    # ========================================================

    if (
        "identifier" in hints
        and stats[
            "unique_ratio"
        ] >= 0.80
    ):

        result.update({
            "semantic_type":
                "identifier",

            "semantic_role":
                "identifier",

            "confidence":
                0.95,
        })

        return result


    if (
        "index" in hints
        and _is_sequential_index(
            series
        )
    ):

        result.update({
            "semantic_type":
                "identifier",

            "semantic_role":
                "exported_index",

            "confidence":
                0.99,
        })

        return result


    # ========================================================
    # NUMERIC
    # ========================================================

    if pd.api.types.is_numeric_dtype(
        series
    ):

        unique_count = (
            stats[
                "unique_count"
            ]
        )

        unique_ratio = (
            stats[
                "unique_ratio"
            ]
        )

        numeric_values = pd.to_numeric(
            non_null,
            errors="coerce",
        ).dropna()

        integer_like = bool(
            len(
                numeric_values
            )
            and np.allclose(
                numeric_values % 1,
                0,
            )
        )


        # Only classify as discrete when there is enough
        # data to support that conclusion.
        if (
            len(series) >= 20
            and unique_count <= 20
            and unique_ratio <= 0.20
        ):

            semantic_type = (
                "numeric_discrete"
            )

        elif (
            len(series) >= 50
            and integer_like
            and unique_count <= 50
            and unique_ratio <= 0.10
        ):

            semantic_type = (
                "numeric_discrete"
            )

        else:

            semantic_type = (
                "numeric_continuous"
            )


        result.update({
            "semantic_type":
                semantic_type,

            "semantic_role":
                "feature",

            "confidence":
                0.95,
        })


        # High uniqueness alone is NOT enough to turn
        # a numeric feature into an identifier.
        if (
            len(series) >= 20
            and unique_ratio >= 0.98
        ):

            result[
                "flags"
            ].append(
                "high_uniqueness_numeric"
            )


        return result


    # ========================================================
    # ENTITY / NAME
    # ========================================================

    if (
        "entity_name" in hints
        and stats[
            "unique_ratio"
        ] >= 0.50
    ):

        result.update({
            "semantic_type":
                "text",

            "semantic_role":
                "entity_name",

            "confidence":
                0.88,
        })

        return result


    # ========================================================
    # GENERIC STRING ANALYSIS
    # ========================================================

    string_values = (
        non_null
        .astype(str)
        .str.strip()
    )

    lengths = (
        string_values
        .str.len()
    )

    mean_length = float(
        lengths.mean()
    )

    unique_ratio = (
        stats[
            "unique_ratio"
        ]
    )

    unique_count = (
        stats[
            "unique_count"
        ]
    )


    # Normal categorical field
    if (
        unique_count <= 50
        or (
            unique_ratio <= 0.20
            and mean_length < 80
        )
    ):

        result.update({
            "semantic_type":
                "categorical",

            "semantic_role":
                "feature",

            "confidence":
                0.92,
        })

        return result


    # High-cardinality labels / names / codes
    if (
        unique_ratio >= 0.85
        and mean_length < 60
    ):

        result.update({
            "semantic_type":
                "text",

            "semantic_role":
                "entity_or_identifier_candidate",

            "confidence":
                0.68,
        })

        result[
            "flags"
        ].append(
            "high_cardinality"
        )

        return result


    # Free text
    result.update({
        "semantic_type":
            "text",

        "semantic_role":
            "free_text",

        "confidence":
            0.88,
    })

    return result


# ============================================================
# COMPLETE DATAFRAME SCHEMA
# ============================================================

def infer_schema(
    df: pd.DataFrame,
) -> dict[str, dict]:
    """
    Infer schema for all dataframe columns.
    """

    schema = {}

    for column in df.columns:

        schema[
            str(column)
        ] = infer_column_schema(
            series=df[column],
            column_name=str(
                column
            ),
        )

    return schema


def infer_dataframe_schema(
    df: pd.DataFrame,
) -> dict[str, dict]:
    """
    Backward-compatible alias.
    """

    return infer_schema(
        df
    )


# ============================================================
# GEOSPATIAL PAIR DETECTION
# ============================================================

def detect_geospatial_pairs(
    schema: dict[str, dict],
) -> list[dict]:
    """
    Detect latitude / longitude combinations.
    """

    latitude_columns = [
        column
        for column, metadata
        in schema.items()
        if metadata.get(
            "semantic_role"
        ) == "latitude"
    ]

    longitude_columns = [
        column
        for column, metadata
        in schema.items()
        if metadata.get(
            "semantic_role"
        ) == "longitude"
    ]

    pairs = []

    for latitude in latitude_columns:

        for longitude in (
            longitude_columns
        ):

            pairs.append({
                "latitude":
                    latitude,

                "longitude":
                    longitude,

                "semantic_role":
                    "geospatial_pair",

                "confidence":
                    min(
                        schema[
                            latitude
                        ].get(
                            "confidence",
                            0.0,
                        ),
                        schema[
                            longitude
                        ].get(
                            "confidence",
                            0.0,
                        ),
                    ),
            })

    return pairs


# ============================================================
# STRUCTURAL / MULTI-VALUE GROUP DETECTION
# ============================================================

def detect_structural_groups(
    df: pd.DataFrame,
    schema: dict[str, dict] | None = None,
) -> list[dict]:
    """
    Detect groups such as:

        Activity
        Unnamed: 7
        Unnamed: 8
        ...

    These may represent one logical multi-valued field.

    Detection does NOT mutate the dataframe.
    """

    if schema is None:

        schema = infer_schema(
            df
        )


    columns = list(
        df.columns
    )

    groups = []

    index = 0


    while index < len(
        columns
    ):

        base_column = (
            columns[
                index
            ]
        )

        base_metadata = (
            schema.get(
                str(
                    base_column
                ),
                {},
            )
        )

        base_role = (
            base_metadata.get(
                "semantic_role"
            )
        )


        if (
            base_role
            ==
            "structural_overflow_candidate"
        ):

            index += 1
            continue


        overflow_columns = []

        next_index = (
            index + 1
        )


        while (
            next_index
            <
            len(columns)
        ):

            next_column = (
                columns[
                    next_index
                ]
            )

            next_metadata = (
                schema.get(
                    str(
                        next_column
                    ),
                    {},
                )
            )

            if (
                next_metadata.get(
                    "semantic_role"
                )
                !=
                "structural_overflow_candidate"
            ):

                break


            overflow_columns.append(
                next_column
            )

            next_index += 1


        if overflow_columns:

            group_columns = [
                base_column,
                *overflow_columns,
            ]

            values = []

            non_empty_cells = 0


            for group_column in (
                group_columns
            ):

                column_values = (
                    df[
                        group_column
                    ]
                    .dropna()
                    .astype(str)
                    .str.strip()
                )

                column_values = (
                    column_values[
                        column_values
                        != ""
                    ]
                )

                non_empty_cells += int(
                    len(
                        column_values
                    )
                )

                values.extend(
                    column_values.tolist()
                )


            total_possible_cells = (
                len(df)
                *
                len(
                    group_columns
                )
            )


            groups.append({
                "base_column":
                    str(
                        base_column
                    ),

                "columns":
                    [
                        str(
                            column
                        )
                        for column
                        in group_columns
                    ],

                "overflow_columns":
                    [
                        str(
                            column
                        )
                        for column
                        in overflow_columns
                    ],

                "suggested_role":
                    "multi_valued_field",

                "non_empty_values":
                    non_empty_cells,

                "unique_values":
                    len(
                        set(
                            values
                        )
                    ),

                "occupancy_ratio":
                    round(
                        _safe_ratio(
                            non_empty_cells,
                            total_possible_cells,
                        ),
                        4,
                    ),

                "confidence":
                    0.90,
            })


        index = max(
            next_index,
            index + 1,
        )


    return groups


# ============================================================
# DATASET-LEVEL SUMMARY
# ============================================================

def summarize_schema(
    df: pd.DataFrame,
    schema: dict[str, dict] | None = None,
) -> dict:
    """
    Build dataset-level semantic schema information.
    """

    if schema is None:

        schema = infer_schema(
            df
        )


    semantic_types = Counter(
        metadata.get(
            "semantic_type",
            "unknown",
        )
        for metadata
        in schema.values()
    )


    semantic_roles = Counter(
        metadata.get(
            "semantic_role",
            "unknown",
        )
        for metadata
        in schema.values()
    )


    return {
        "column_count":
            int(
                len(
                    df.columns
                )
            ),

        "semantic_type_counts":
            dict(
                semantic_types
            ),

        "semantic_role_counts":
            dict(
                semantic_roles
            ),

        "geospatial_pairs":
            detect_geospatial_pairs(
                schema
            ),

        "structural_groups":
            detect_structural_groups(
                df=df,
                schema=schema,
            ),

        "low_confidence_columns":
            [
                column
                for column, metadata
                in schema.items()
                if metadata.get(
                    "confidence",
                    0.0,
                ) < 0.75
            ],
    }