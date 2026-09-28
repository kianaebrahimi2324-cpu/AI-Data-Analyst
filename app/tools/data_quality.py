from __future__ import annotations

from typing import Any

import pandas as pd

from app.services.schema_inference import infer_schema, summarize_schema


HIGH_MISSING_THRESHOLD = 40.0
HIGH_CARDINALITY_MIN_UNIQUE = 50
HIGH_CARDINALITY_RATIO = 0.50
LOW_CONFIDENCE_THRESHOLD = 0.75


def _safe_percent(value: int | float, total: int | float) -> float:
    if total == 0:
        return 0.0
    return round(float(value) / float(total) * 100.0, 2)


def _extract_columns_schema(
    schema: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Support both current direct and legacy nested schema formats."""
    if not isinstance(schema, dict):
        return {}

    nested = schema.get("columns_schema")
    if isinstance(nested, dict):
        return nested

    return {
        str(column): info
        for column, info in schema.items()
        if isinstance(info, dict)
        and ("semantic_type" in info or "semantic_role" in info)
    }


def _is_present(series: pd.Series) -> pd.Series:
    """Treat NaN and blank strings as absent."""
    present = series.notna()

    if (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
    ):
        text = series.astype("string").str.strip()
        present = present & text.ne("") & text.notna()

    return present


def _raw_missingness(df: pd.DataFrame) -> dict[str, Any]:
    """Physical missing cells exactly as stored in the dataset."""
    rows = len(df)
    total_cells = int(df.shape[0] * df.shape[1])
    by_column = {}

    for column in df.columns:
        count = int(df[column].isna().sum())
        by_column[str(column)] = {
            "missing_count": count,
            "missing_percent": _safe_percent(count, rows),
        }

    raw_missing_cells = int(df.isna().sum().sum())

    return {
        "raw_missing_cells": raw_missing_cells,
        "raw_missing_percent": _safe_percent(raw_missing_cells, total_cells),
        "missing_by_column": by_column,
    }


def _multi_value_groups(
    df: pd.DataFrame,
    schema_summary: dict[str, Any],
    analysis_plan: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Use router groups when available, otherwise schema groups."""
    results = []

    if isinstance(analysis_plan, dict):
        source = analysis_plan.get("multi_value_groups", [])

        for group in source:
            if not isinstance(group, dict):
                continue

            logical_name = str(group.get("logical_name", "")).strip()
            columns = [
                str(column)
                for column in group.get("source_columns", [])
                if str(column) in df.columns
            ]

            if logical_name and columns:
                results.append(
                    {
                        "logical_name": logical_name,
                        "source_columns": columns,
                        "overflow_columns": [
                            str(column)
                            for column in group.get("overflow_columns", [])
                            if str(column) in df.columns
                        ],
                    }
                )

    if results:
        return results

    for group in schema_summary.get("structural_groups", []):
        if not isinstance(group, dict):
            continue

        if group.get("suggested_role") != "multi_valued_field":
            continue

        logical_name = str(group.get("base_column", "")).strip()
        columns = [
            str(column)
            for column in group.get("columns", [])
            if str(column) in df.columns
        ]

        if logical_name and columns:
            results.append(
                {
                    "logical_name": logical_name,
                    "source_columns": columns,
                    "overflow_columns": [
                        str(column)
                        for column in group.get("overflow_columns", [])
                        if str(column) in df.columns
                    ],
                }
            )

    return results


def _logical_missingness(
    df: pd.DataFrame,
    columns_schema: dict[str, dict[str, Any]],
    groups: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Count missing logical fields.

    A repeated multi-value field counts once per row.
    Empty overflow slots do not count as separate missing features.
    """
    grouped_columns = {
        column
        for group in groups
        for column in group["source_columns"]
    }

    standalone_structural = {
        column
        for column, info in columns_schema.items()
        if info.get("semantic_role")
        in {"structural", "structural_overflow_candidate"}
        and column not in grouped_columns
    }

    fields = {}
    relevant_missing = 0
    logical_slots = 0
    structural_slot_missing = 0

    # Ordinary logical columns.
    for column in df.columns:
        name = str(column)

        if name in grouped_columns or name in standalone_structural:
            continue

        count = int((~_is_present(df[column])).sum())
        info = columns_schema.get(name, {})

        fields[name] = {
            "field_kind": "column",
            "source_columns": [name],
            "semantic_type": info.get("semantic_type", "unknown"),
            "semantic_role": info.get("semantic_role", "feature"),
            "missing_count": count,
            "missing_percent": _safe_percent(count, len(df)),
        }

        relevant_missing += count
        logical_slots += len(df)

    # Multi-valued logical fields.
    for group in groups:
        logical_name = group["logical_name"]
        columns = [
            column
            for column in group["source_columns"]
            if column in df.columns
        ]

        if not columns:
            continue

        present = pd.DataFrame(
            {column: _is_present(df[column]) for column in columns}
        )

        missing_rows = int((~present.any(axis=1)).sum())
        physical_missing = int(
            sum((~_is_present(df[column])).sum() for column in columns)
        )

        # One logical missing value remains for a row where the
        # entire field is absent. Other empty slots are structural.
        unused_slots = max(
            physical_missing - missing_rows,
            0,
        )

        fields[logical_name] = {
            "field_kind": "multi_value",
            "source_columns": columns,
            "missing_count": missing_rows,
            "missing_percent": _safe_percent(missing_rows, len(df)),
            "physical_missing_cells": physical_missing,
            "structural_slot_missing_cells": unused_slots,
        }

        relevant_missing += missing_rows
        logical_slots += len(df)
        structural_slot_missing += unused_slots

    # Structural columns that were not assigned to a group.
    structural_slot_missing += int(
        sum(
            (~_is_present(df[column])).sum()
            for column in standalone_structural
            if column in df.columns
        )
    )

    return {
        "fields": fields,
        "analytically_relevant_missing_values": int(relevant_missing),
        "analytically_relevant_missing_percent": _safe_percent(
            relevant_missing,
            logical_slots,
        ),
        "structural_slot_missing_cells": int(structural_slot_missing),
        "standalone_structural_columns": sorted(standalone_structural),
    }


def _high_missing_fields(
    logical_missing: dict[str, Any],
) -> list[dict[str, Any]]:
    results = []

    for name, info in logical_missing["fields"].items():
        percent = float(info.get("missing_percent", 0.0))

        if percent >= HIGH_MISSING_THRESHOLD:
            results.append(
                {
                    "column": name,
                    "field_kind": info.get("field_kind", "column"),
                    "missing_count": int(info.get("missing_count", 0)),
                    "missing_percent": percent,
                }
            )

    return sorted(
        results,
        key=lambda item: item["missing_percent"],
        reverse=True,
    )


def _identifier_columns(
    columns_schema: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    results = []

    for column, info in columns_schema.items():
        if (
            info.get("semantic_type") != "identifier"
            and info.get("semantic_role")
            not in {"identifier", "identifier_candidate", "exported_index"}
        ):
            continue

        results.append(
            {
                "column": column,
                "semantic_role": info.get("semantic_role"),
                "confidence": float(info.get("confidence", 0.0)),
                "unique_count": int(info.get("unique_count", 0)),
                "unique_ratio": float(info.get("unique_ratio", 0.0)),
            }
        )

    return results


def _high_cardinality_columns(
    columns_schema: dict[str, dict[str, Any]],
    grouped_columns: set[str],
) -> list[dict[str, Any]]:
    """Only true categorical features are checked."""
    results = []

    for column, info in columns_schema.items():
        if column in grouped_columns:
            continue
        if info.get("semantic_type") not in {"categorical", "binary"}:
            continue
        if info.get("semantic_role", "feature") != "feature":
            continue

        unique_count = int(info.get("unique_count", 0))
        unique_ratio = float(info.get("unique_ratio", 0.0))

        if (
            unique_count >= HIGH_CARDINALITY_MIN_UNIQUE
            and unique_ratio >= HIGH_CARDINALITY_RATIO
        ):
            results.append(
                {
                    "column": column,
                    "unique_count": unique_count,
                    "unique_ratio": unique_ratio,
                }
            )

    return sorted(
        results,
        key=lambda item: item["unique_count"],
        reverse=True,
    )


def _low_confidence_columns(
    columns_schema: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "column": column,
            "semantic_type": info.get("semantic_type", "unknown"),
            "semantic_role": info.get("semantic_role", "unknown"),
            "confidence": float(info.get("confidence", 0.0)),
        }
        for column, info in columns_schema.items()
        if float(info.get("confidence", 0.0)) < LOW_CONFIDENCE_THRESHOLD
    ]


def _geospatial_quality(
    df: pd.DataFrame,
    columns_schema: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Validate legal coordinate ranges.

    Valid but geographically distant coordinates are not
    treated as errors.
    """
    results = []

    for column, info in columns_schema.items():
        role = info.get("semantic_role")

        if role not in {"latitude", "longitude"}:
            continue

        numeric = pd.to_numeric(df[column], errors="coerce")
        present = _is_present(df[column])
        non_numeric = present & numeric.isna()

        if role == "latitude":
            valid = numeric.between(-90, 90)
            legal_min, legal_max = -90.0, 90.0
        else:
            valid = numeric.between(-180, 180)
            legal_min, legal_max = -180.0, 180.0

        invalid_range = numeric.notna() & ~valid

        results.append(
            {
                "column": column,
                "semantic_role": role,
                "missing_count": int((~present).sum()),
                "non_numeric_count": int(non_numeric.sum()),
                "invalid_range_count": int(invalid_range.sum()),
                "invalid_range_percent": _safe_percent(
                    int(invalid_range.sum()),
                    int(numeric.notna().sum()),
                ),
                "legal_range": {
                    "minimum": legal_min,
                    "maximum": legal_max,
                },
            }
        )

    return results


def _quality_flags(
    duplicate_rows: int,
    high_missing: list[dict[str, Any]],
    structural_slot_missing_cells: int,
    groups: list[dict[str, Any]],
    geospatial: list[dict[str, Any]],
    low_confidence: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    flags = []

    if duplicate_rows > 0:
        flags.append(
            {
                "type": "duplicate_rows",
                "severity": "warning",
                "message": f"{duplicate_rows} exact duplicate rows were detected.",
            }
        )

    for item in high_missing:
        flags.append(
            {
                "type": "high_missing",
                "severity": "warning",
                "column": item["column"],
                "message": (
                    f"{item['column']} has {item['missing_percent']}% "
                    "analytically relevant missing values."
                ),
            }
        )

    if structural_slot_missing_cells > 0:
        flags.append(
            {
                "type": "structural_missing_adjustment",
                "severity": "info",
                "message": (
                    f"{structural_slot_missing_cells} empty physical cells "
                    "were treated as structural/unused slots rather than "
                    "independent missing features."
                ),
            }
        )

    for group in groups:
        flags.append(
            {
                "type": "multi_value_field_detected",
                "severity": "info",
                "column": group["logical_name"],
                "message": (
                    f"{group['logical_name']} is represented by multiple "
                    "physical columns and is analyzed as one logical field."
                ),
            }
        )

    for item in geospatial:
        if item["non_numeric_count"] > 0:
            flags.append(
                {
                    "type": "invalid_coordinate_format",
                    "severity": "warning",
                    "column": item["column"],
                    "message": (
                        f"{item['column']} contains "
                        f"{item['non_numeric_count']} non-numeric "
                        "coordinate values."
                    ),
                }
            )

        if item["invalid_range_count"] > 0:
            flags.append(
                {
                    "type": "invalid_coordinate_range",
                    "severity": "warning",
                    "column": item["column"],
                    "message": (
                        f"{item['column']} contains "
                        f"{item['invalid_range_count']} values outside "
                        "the legal coordinate range."
                    ),
                }
            )

    for item in low_confidence:
        flags.append(
            {
                "type": "schema_uncertainty",
                "severity": "info",
                "column": item["column"],
                "message": (
                    f"Semantic inference for {item['column']} has "
                    f"confidence {item['confidence']:.2f}."
                ),
            }
        )

    return flags


def analyze_data_quality(
    df: pd.DataFrame,
    schema: dict[str, Any] | None = None,
    schema_summary: dict[str, Any] | None = None,
    analysis_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Semantic-aware data-quality analysis.

    Reports raw physical missing cells separately from
    analytically relevant missing logical fields.
    """
    if schema is None:
        schema = infer_schema(df)

    columns_schema = _extract_columns_schema(schema)

    if schema_summary is None:
        schema_summary = summarize_schema(
            df=df,
            schema=schema,
        )

    raw = _raw_missingness(df)

    groups = _multi_value_groups(
        df=df,
        schema_summary=schema_summary,
        analysis_plan=analysis_plan,
    )

    logical = _logical_missingness(
        df=df,
        columns_schema=columns_schema,
        groups=groups,
    )

    high_missing = _high_missing_fields(logical)

    grouped_columns = {
        column
        for group in groups
        for column in group["source_columns"]
    }

    constant_columns = [
        column
        for column, info in columns_schema.items()
        if info.get("semantic_type") == "constant"
        and info.get("semantic_role")
        not in {"structural", "structural_overflow_candidate"}
    ]

    empty_columns = [
        column
        for column, info in columns_schema.items()
        if info.get("semantic_type") == "empty"
        and info.get("semantic_role")
        not in {"structural", "structural_overflow_candidate"}
    ]

    identifiers = _identifier_columns(columns_schema)
    high_cardinality = _high_cardinality_columns(
        columns_schema,
        grouped_columns,
    )
    low_confidence = _low_confidence_columns(columns_schema)
    geospatial = _geospatial_quality(df, columns_schema)

    duplicate_rows = int(df.duplicated().sum())
    duplicate_percent = _safe_percent(duplicate_rows, len(df))

    flags = _quality_flags(
        duplicate_rows=duplicate_rows,
        high_missing=high_missing,
        structural_slot_missing_cells=logical["structural_slot_missing_cells"],
        groups=groups,
        geospatial=geospatial,
        low_confidence=low_confidence,
    )

    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),

        # Physical missingness
        "raw_missing_cells": raw["raw_missing_cells"],
        "raw_missing_percent": raw["raw_missing_percent"],
        "missing_by_column": raw["missing_by_column"],

        # Logical missingness
        "analytically_relevant_missing_values":
            logical["analytically_relevant_missing_values"],
        "analytically_relevant_missing_percent":
            logical["analytically_relevant_missing_percent"],
        "structural_slot_missing_cells":
            logical["structural_slot_missing_cells"],
        "logical_missing_by_field": logical["fields"],
        "high_missing_columns": high_missing,

        # Other quality checks
        "duplicate_rows": duplicate_rows,
        "duplicate_percent": duplicate_percent,
        "constant_columns": constant_columns,
        "empty_columns": empty_columns,
        "identifier_columns": identifiers,
        "high_cardinality_columns": high_cardinality,
        "low_confidence_schema_columns": low_confidence,
        "geospatial_quality": geospatial,
        "multi_value_groups": groups,
        "quality_flags": flags,
    }
