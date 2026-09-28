from __future__ import annotations

from typing import Any

import pandas as pd

from app.services.schema_inference import (
    infer_schema,
    summarize_schema,
)


# ============================================================
# SUPPORTED ANALYSIS FAMILIES
# ============================================================

SUPPORTED_FAMILIES = {
    "numeric",
    "categorical",
    "datetime",
    "text",
    "geospatial",
    "metadata",
    "entity",
    "multi_value",
    "identifier",
    "structural",
    "none",
}


# ============================================================
# HELPERS
# ============================================================

def _extract_columns_schema(
    schema: dict[str, Any],
) -> dict[str, dict]:
    """
    Support both direct and legacy schema formats.
    """

    if not isinstance(schema, dict):
        return {}

    legacy = schema.get(
        "columns_schema"
    )

    if isinstance(legacy, dict):
        return legacy

    result = {}

    for column, metadata in schema.items():

        if not isinstance(
            metadata,
            dict,
        ):
            continue

        if (
            "semantic_type" in metadata
            or "semantic_role" in metadata
        ):
            result[str(column)] = metadata

    return result


def _append_unique(
    values: list[str],
    value: str,
) -> None:
    """
    Append a value only once.
    """

    if value not in values:
        values.append(
            value
        )


def _remove_value(
    values: list[str],
    value: str,
) -> None:
    """
    Remove all occurrences safely.
    """

    while value in values:
        values.remove(
            value
        )


# ============================================================
# ANALYSIS FAMILY RESOLUTION
# ============================================================

def _resolve_analysis_family(
    info: dict[str, Any],
) -> str:
    """
    Route a column according to semantic type and semantic role.

    Semantic role has priority over physical/statistical type.
    """

    semantic_type = str(
        info.get(
            "semantic_type",
            "unknown",
        )
    ).lower()

    semantic_role = str(
        info.get(
            "semantic_role",
            "feature",
        )
    ).lower()


    # --------------------------------------------------------
    # Non-informative
    # --------------------------------------------------------

    if semantic_type in {
        "empty",
        "constant",
    }:
        return "none"


    # --------------------------------------------------------
    # Structural
    # --------------------------------------------------------

    if semantic_role in {
        "structural",
        "structural_overflow_candidate",
    }:
        return "structural"


    # --------------------------------------------------------
    # Geospatial
    # --------------------------------------------------------

    if semantic_role in {
        "latitude",
        "longitude",
    }:
        return "geospatial"


    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    if semantic_role in {
        "url",
        "email",
        "phone",
        "address",
    }:
        return "metadata"


    # --------------------------------------------------------
    # Entity
    # --------------------------------------------------------

    if semantic_role in {
        "entity_name",
        "entity_or_identifier_candidate",
    }:
        return "entity"


    # --------------------------------------------------------
    # Identifier
    # --------------------------------------------------------

    if (
        semantic_type == "identifier"
        or semantic_role in {
            "identifier",
            "identifier_candidate",
            "exported_index",
        }
    ):
        return "identifier"


    # --------------------------------------------------------
    # Datetime
    # --------------------------------------------------------

    if (
        semantic_type == "datetime"
        or semantic_role == "temporal"
    ):
        return "datetime"


    # --------------------------------------------------------
    # Numeric
    # --------------------------------------------------------

    if semantic_type in {
        "numeric",
        "numeric_continuous",
        "numeric_discrete",
    }:
        return "numeric"


    # --------------------------------------------------------
    # Categorical
    # --------------------------------------------------------

    if semantic_type in {
        "categorical",
        "category",
        "binary",
    }:
        return "categorical"


    # --------------------------------------------------------
    # Text
    # --------------------------------------------------------

    if semantic_type in {
        "text",
        "free_text",
    }:
        return "text"


    return "none"


# ============================================================
# TARGET VALIDATION
# ============================================================

def _target_metadata(
    column: str,
    info: dict[str, Any],
) -> dict[str, Any]:
    """
    Build target metadata and flag unusual targets.
    """

    semantic_type = str(
        info.get(
            "semantic_type",
            "unknown",
        )
    )

    semantic_role = str(
        info.get(
            "semantic_role",
            "feature",
        )
    )

    family = _resolve_analysis_family(
        info
    )

    requires_confirmation = False
    reason = ""

    if family in {
        "metadata",
        "identifier",
        "structural",
        "none",
    }:

        requires_confirmation = True

        reason = (
            "The selected column is not normally "
            "used as an analytical target."
        )

    elif family == "geospatial":

        requires_confirmation = True

        reason = (
            "The selected target is a geographic coordinate."
        )

    return {
        "column":
            column,

        "semantic_type":
            semantic_type,

        "semantic_role":
            semantic_role,

        "analysis_family":
            family,

        "confidence":
            float(
                info.get(
                    "confidence",
                    0.0,
                )
            ),

        "requires_confirmation":
            requires_confirmation,

        "suitability_reason":
            reason,
    }


# ============================================================
# BUILD ANALYSIS PLAN
# ============================================================

def build_analysis_plan(
    schema: dict[str, Any],
    target_column: str = "",
    schema_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build the semantic routing plan.

    The router decides which analyzer should receive each
    column. It does not perform analysis itself.
    """

    columns_schema = (
        _extract_columns_schema(
            schema
        )
    )

    plan = {
        "numeric_columns": [],
        "categorical_columns": [],
        "datetime_columns": [],
        "text_columns": [],

        "geospatial_columns": {
            "latitude": [],
            "longitude": [],
        },

        "geospatial_pairs": [],

        "metadata_columns": {
            "url": [],
            "email": [],
            "phone": [],
            "address": [],
        },

        "entity_columns": [],

        "multi_value_columns": [],
        "multi_value_groups": [],

        "structural_columns": [],

        "identifier_columns": [],
        "ignored_columns": [],

        "feature_columns": [],

        "target": {},

        "warnings": [],
    }


    # ========================================================
    # TARGET EXISTENCE
    # ========================================================

    if (
        target_column
        and target_column
        not in columns_schema
    ):

        plan[
            "warnings"
        ].append(
            (
                f"Target column '{target_column}' "
                "was not found in the inferred schema."
            )
        )


    # ========================================================
    # ROUTE COLUMNS
    # ========================================================

    for column, info in (
        columns_schema.items()
    ):

        semantic_type = str(
            info.get(
                "semantic_type",
                "unknown",
            )
        )

        semantic_role = str(
            info.get(
                "semantic_role",
                "feature",
            )
        )

        confidence = float(
            info.get(
                "confidence",
                0.0,
            )
        )

        family = (
            _resolve_analysis_family(
                info
            )
        )


        # ----------------------------------------------------
        # Schema confidence
        # ----------------------------------------------------

        if confidence < 0.75:

            plan[
                "warnings"
            ].append(
                (
                    f"Low-confidence schema inference for "
                    f"'{column}': {semantic_type} / "
                    f"{semantic_role} "
                    f"(confidence={confidence:.2f})."
                )
            )


        # ----------------------------------------------------
        # Target
        # ----------------------------------------------------

        if (
            target_column
            and column == target_column
        ):

            plan[
                "target"
            ] = _target_metadata(
                column=column,
                info=info,
            )

            if plan[
                "target"
            ][
                "requires_confirmation"
            ]:

                plan[
                    "warnings"
                ].append(
                    (
                        f"Target '{column}' may require "
                        "confirmation: "
                        f"{plan['target']['suitability_reason']}"
                    )
                )

            continue


        # ----------------------------------------------------
        # Numeric
        # ----------------------------------------------------

        if family == "numeric":

            _append_unique(
                plan[
                    "numeric_columns"
                ],
                column,
            )

            _append_unique(
                plan[
                    "feature_columns"
                ],
                column,
            )

            continue


        # ----------------------------------------------------
        # Categorical
        # ----------------------------------------------------

        if family == "categorical":

            _append_unique(
                plan[
                    "categorical_columns"
                ],
                column,
            )

            _append_unique(
                plan[
                    "feature_columns"
                ],
                column,
            )

            continue


        # ----------------------------------------------------
        # Datetime
        # ----------------------------------------------------

        if family == "datetime":

            _append_unique(
                plan[
                    "datetime_columns"
                ],
                column,
            )

            _append_unique(
                plan[
                    "feature_columns"
                ],
                column,
            )

            continue


        # ----------------------------------------------------
        # Text
        # ----------------------------------------------------

        if family == "text":

            _append_unique(
                plan[
                    "text_columns"
                ],
                column,
            )

            _append_unique(
                plan[
                    "feature_columns"
                ],
                column,
            )

            continue


        # ----------------------------------------------------
        # Geospatial
        # ----------------------------------------------------

        if family == "geospatial":

            if semantic_role == "latitude":

                _append_unique(
                    plan[
                        "geospatial_columns"
                    ][
                        "latitude"
                    ],
                    column,
                )

            elif semantic_role == "longitude":

                _append_unique(
                    plan[
                        "geospatial_columns"
                    ][
                        "longitude"
                    ],
                    column,
                )

            continue


        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        if family == "metadata":

            if (
                semantic_role
                in plan[
                    "metadata_columns"
                ]
            ):

                _append_unique(
                    plan[
                        "metadata_columns"
                    ][
                        semantic_role
                    ],
                    column,
                )

            continue


        # ----------------------------------------------------
        # Entity
        # ----------------------------------------------------

        if family == "entity":

            _append_unique(
                plan[
                    "entity_columns"
                ],
                column,
            )

            continue


        # ----------------------------------------------------
        # Identifier
        # ----------------------------------------------------

        if family == "identifier":

            _append_unique(
                plan[
                    "identifier_columns"
                ],
                column,
            )

            continue


        # ----------------------------------------------------
        # Structural
        # ----------------------------------------------------

        if family == "structural":

            _append_unique(
                plan[
                    "structural_columns"
                ],
                column,
            )

            continue


        # ----------------------------------------------------
        # Ignored
        # ----------------------------------------------------

        _append_unique(
            plan[
                "ignored_columns"
            ],
            column,
        )


    # ========================================================
    # DATASET-LEVEL SEMANTIC STRUCTURES
    # ========================================================

    if not isinstance(
        schema_summary,
        dict,
    ):
        schema_summary = {}


    # --------------------------------------------------------
    # Geospatial pairs
    # --------------------------------------------------------

    for pair in schema_summary.get(
        "geospatial_pairs",
        [],
    ):

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

        if (
            target_column
            and target_column in {
                latitude,
                longitude,
            }
        ):
            continue

        plan[
            "geospatial_pairs"
        ].append(
            pair
        )


    # --------------------------------------------------------
    # Multi-value groups
    # --------------------------------------------------------

    for group in schema_summary.get(
        "structural_groups",
        [],
    ):

        if not isinstance(
            group,
            dict,
        ):
            continue

        if (
            group.get(
                "suggested_role"
            )
            !=
            "multi_valued_field"
        ):
            continue

        base_column = str(
            group.get(
                "base_column",
                "",
            )
        )

        source_columns = [
            str(column)
            for column
            in group.get(
                "columns",
                [],
            )
        ]

        if not base_column:
            continue

        if (
            target_column
            and target_column
            in source_columns
        ):

            plan[
                "warnings"
            ].append(
                (
                    f"Structural group containing target "
                    f"'{target_column}' was not merged."
                )
            )

            continue


        # Remove grouped columns from normal analyzers.
        for source_column in (
            source_columns
        ):

            _remove_value(
                plan[
                    "numeric_columns"
                ],
                source_column,
            )

            _remove_value(
                plan[
                    "categorical_columns"
                ],
                source_column,
            )

            _remove_value(
                plan[
                    "datetime_columns"
                ],
                source_column,
            )

            _remove_value(
                plan[
                    "text_columns"
                ],
                source_column,
            )

            _remove_value(
                plan[
                    "entity_columns"
                ],
                source_column,
            )

            _remove_value(
                plan[
                    "feature_columns"
                ],
                source_column,
            )


        _append_unique(
            plan[
                "multi_value_columns"
            ],
            base_column,
        )


        plan[
            "multi_value_groups"
        ].append({
            "logical_name":
                base_column,

            "source_columns":
                source_columns,

            "overflow_columns":
                [
                    str(column)
                    for column
                    in group.get(
                        "overflow_columns",
                        [],
                    )
                ],

            "non_empty_values":
                group.get(
                    "non_empty_values"
                ),

            "unique_values":
                group.get(
                    "unique_values"
                ),

            "occupancy_ratio":
                group.get(
                    "occupancy_ratio"
                ),

            "confidence":
                group.get(
                    "confidence",
                    0.0,
                ),
        })


    # ========================================================
    # CAPABILITIES
    # ========================================================

    plan[
        "capabilities"
    ] = {
        "has_numeric":
            bool(
                plan[
                    "numeric_columns"
                ]
            ),

        "has_categorical":
            bool(
                plan[
                    "categorical_columns"
                ]
            ),

        "has_datetime":
            bool(
                plan[
                    "datetime_columns"
                ]
            ),

        "has_text":
            bool(
                plan[
                    "text_columns"
                ]
            ),

        "has_geospatial":
            bool(
                plan[
                    "geospatial_pairs"
                ]
                or plan[
                    "geospatial_columns"
                ][
                    "latitude"
                ]
                or plan[
                    "geospatial_columns"
                ][
                    "longitude"
                ]
            ),

        "has_metadata":
            any(
                bool(
                    columns
                )
                for columns
                in plan[
                    "metadata_columns"
                ].values()
            ),

        "has_entities":
            bool(
                plan[
                    "entity_columns"
                ]
            ),

        "has_multi_value":
            bool(
                plan[
                    "multi_value_groups"
                ]
            ),

        "has_identifiers":
            bool(
                plan[
                    "identifier_columns"
                ]
            ),

        "has_structural_columns":
            bool(
                plan[
                    "structural_columns"
                ]
            ),

        "has_target":
            bool(
                plan[
                    "target"
                ]
            ),
    }


    # ========================================================
    # DATASET MODE
    # ========================================================

    active_families = []

    if plan[
        "capabilities"
    ][
        "has_numeric"
    ]:
        active_families.append(
            "numeric"
        )

    if plan[
        "capabilities"
    ][
        "has_categorical"
    ]:
        active_families.append(
            "categorical"
        )

    if plan[
        "capabilities"
    ][
        "has_datetime"
    ]:
        active_families.append(
            "temporal"
        )

    if plan[
        "capabilities"
    ][
        "has_text"
    ]:
        active_families.append(
            "text"
        )

    if plan[
        "capabilities"
    ][
        "has_geospatial"
    ]:
        active_families.append(
            "geospatial"
        )

    if plan[
        "capabilities"
    ][
        "has_multi_value"
    ]:
        active_families.append(
            "multi_value"
        )


    if len(
        active_families
    ) > 1:

        dataset_mode = "mixed"

    elif len(
        active_families
    ) == 1:

        dataset_mode = (
            active_families[
                0
            ]
        )

    elif plan[
        "capabilities"
    ][
        "has_entities"
    ]:

        dataset_mode = "entity"

    elif plan[
        "capabilities"
    ][
        "has_metadata"
    ]:

        dataset_mode = "metadata"

    else:

        dataset_mode = "unknown"


    plan[
        "dataset_mode"
    ] = dataset_mode

    plan[
        "active_analysis_families"
    ] = active_families


    return plan


# ============================================================
# MAIN ROUTER
# ============================================================

def route_dataset(
    df: pd.DataFrame,
    target_column: str = "",
) -> dict[str, Any]:
    """
    Infer schema, detect semantic structures and build
    the complete routing plan.
    """

    schema = infer_schema(
        df
    )

    schema_summary = summarize_schema(
        df=df,
        schema=schema,
    )

    analysis_plan = build_analysis_plan(
        schema=schema,
        target_column=target_column,
        schema_summary=schema_summary,
    )

    return {
        "schema":
            schema,

        "schema_summary":
            schema_summary,

        "analysis_plan":
            analysis_plan,
    }