from __future__ import annotations

from typing import Any, Callable

import pandas as pd

from app.services.dataset_profiler import (
    build_dataset_profile,
)

from app.services.analysis_router import (
    route_dataset,
)

from app.tools.numerical_analysis import (
    analyze_numerical_columns,
)

from app.tools.categorical_analysis import (
    analyze_categorical_columns,
)

from app.tools.datetime_analysis import (
    analyze_datetime_columns,
)

from app.tools.text_analysis import (
    analyze_text_columns,
)

from app.tools.correlation_analysis import (
    analyze_correlations,
)

from app.tools.relationship_analysis import (
    analyze_cross_type_relationships,
)

from app.tools.text_target_analysis import (
    analyze_text_target_relationships,
)

from app.tools.data_quality import (
    analyze_data_quality,
)

from app.tools.target_analysis import (
    analyze_target,
)

from app.tools.geospatial_analysis import (
    analyze_geospatial_data,
)

from app.tools.multi_value_analysis import (
    analyze_multi_value_data,
)

from app.tools.metadata_analysis import (
    analyze_metadata_data,
)


# ============================================================
# SAFE ANALYZER EXECUTION
# ============================================================

def _safe_analysis(
    analyzer_name: str,
    function: Callable[..., Any],
    **kwargs: Any,
) -> Any:
    """
    Run one analyzer without allowing one optional analysis
    module to crash the entire dataset pipeline.

    Errors remain visible in the evidence package.
    """

    try:
        return function(
            **kwargs
        )

    except Exception as exc:
        return {
            "status":
                "error",

            "analyzer":
                analyzer_name,

            "error_type":
                type(exc).__name__,

            "message":
                str(exc),
        }




# ============================================================
# GROUNDING / FACT INDEX
# ============================================================

def _extract_columns_schema(
    schema: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Support direct and legacy nested schema formats."""
    if not isinstance(schema, dict):
        return {}

    nested = schema.get(
        "columns_schema"
    )

    if isinstance(
        nested,
        dict,
    ):
        return nested

    return {
        str(column): info
        for column, info
        in schema.items()
        if isinstance(info, dict)
        and (
            "semantic_type" in info
            or "semantic_role" in info
        )
    }


def _build_grounding_contract(
    target_column: str,
    target_analysis: dict[str, Any],
) -> dict[str, Any]:
    """
    Build deterministic constraints for downstream LLM reasoning.
    """
    target_info = (
        target_analysis
        if isinstance(
            target_analysis,
            dict,
        )
        else {}
    )

    has_target = bool(
        target_column
        and target_info.get(
            "target_column"
        )
    )

    task_type = str(
        target_info.get(
            "task_type",
            "",
        )
    )

    requires_confirmation = bool(
        target_info.get(
            "requires_confirmation",
            False,
        )
    )

    predictive_modeling_supported = bool(
        has_target
        and task_type
        in {
            "classification",
            "regression",
        }
        and not requires_confirmation
    )

    return {
        "has_target":
            has_target,

        "target_column":
            (
                str(target_column)
                if has_target
                else None
            ),

        "target_task_type":
            (
                task_type
                if task_type
                else None
            ),

        "target_requires_confirmation":
            requires_confirmation,

        "predictive_modeling_supported":
            predictive_modeling_supported,

        "target_class_meanings_known":
            False,

        "rules": [
            (
                "Every numerical statement must be supported by "
                "structured evidence."
            ),
            (
                "Do not infer domain meaning, causality, invalidity, "
                "or data-generation mechanisms from column names alone."
            ),
            (
                "Do not infer zero frequency from a minimum value; "
                "use zero_count/zero_percent."
            ),
            (
                "Do not infer skewness from standard deviation alone; "
                "use skewness or explicit distribution evidence."
            ),
            (
                "If no target is selected, do not recommend predictive "
                "classification/regression models or discuss class semantics."
            ),
            (
                "Geospatial legal-range validity does not establish "
                "geographic correctness."
            ),
        ],
    }


def _build_column_fact_index(
    df: pd.DataFrame,
    schema: dict[str, Any],
    numerical_analysis: Any,
    data_quality: Any,
    target_analysis: Any,
) -> dict[str, Any]:
    """
    Create compact deterministic facts by column.

    This gives the Analyst and Reviewer an explicit source of truth
    for numerical statements and semantic roles.
    """
    columns_schema = (
        _extract_columns_schema(
            schema
        )
    )

    numerical = (
        numerical_analysis
        if isinstance(
            numerical_analysis,
            dict,
        )
        else {}
    )

    quality = (
        data_quality
        if isinstance(
            data_quality,
            dict,
        )
        else {}
    )

    logical_missing = quality.get(
        "logical_missing_by_field",
        {},
    )

    raw_missing = quality.get(
        "missing_by_column",
        {},
    )

    target = (
        target_analysis
        if isinstance(
            target_analysis,
            dict,
        )
        else {}
    )

    target_name = str(
        target.get(
            "target_column",
            "",
        )
    )

    fact_index: dict[str, Any] = {}

    for column in df.columns:
        name = str(column)

        schema_info = columns_schema.get(
            name,
            {},
        )

        facts: dict[str, Any] = {
            "semantic_type":
                schema_info.get(
                    "semantic_type",
                    "unknown",
                ),

            "semantic_role":
                schema_info.get(
                    "semantic_role",
                    "feature",
                ),

            "schema_confidence":
                schema_info.get(
                    "confidence"
                ),
        }

        logical_item = (
            logical_missing.get(
                name
            )
            if isinstance(
                logical_missing,
                dict,
            )
            else None
        )

        raw_item = (
            raw_missing.get(
                name
            )
            if isinstance(
                raw_missing,
                dict,
            )
            else None
        )

        if isinstance(
            logical_item,
            dict,
        ):
            facts[
                "logical_missing_count"
            ] = logical_item.get(
                "missing_count"
            )

            facts[
                "logical_missing_percent"
            ] = logical_item.get(
                "missing_percent"
            )

        elif isinstance(
            raw_item,
            dict,
        ):
            facts[
                "raw_missing_count"
            ] = raw_item.get(
                "missing_count"
            )

            facts[
                "raw_missing_percent"
            ] = raw_item.get(
                "missing_percent"
            )

        numeric_item = numerical.get(
            name
        )

        if isinstance(
            numeric_item,
            dict,
        ):
            facts[
                "numerical"
            ] = {
                key:
                    numeric_item.get(
                        key
                    )
                for key in (
                    "count",
                    "unique_count",
                    "mean",
                    "median",
                    "std",
                    "min",
                    "max",
                    "q1",
                    "q3",
                    "iqr",
                    "skewness",
                    "zero_count",
                    "zero_percent",
                    "outlier_count",
                    "outlier_percent",
                    "missing_count",
                    "missing_percent",
                    "non_numeric_count",
                    "infinite_count",
                )
                if key in numeric_item
            }

        if (
            name == target_name
            and target
        ):
            facts[
                "target"
            ] = {
                key:
                    target.get(
                        key
                    )
                for key in (
                    "semantic_type",
                    "semantic_role",
                    "task_type",
                    "task_confidence",
                    "requires_confirmation",
                    "missing_count",
                    "missing_percent",
                    "unique_values",
                    "class_counts",
                    "class_percentages",
                    "mean",
                    "median",
                    "std",
                    "min",
                    "max",
                )
                if key in target
            }

        fact_index[
            name
        ] = facts

    return fact_index


# ============================================================
# CORRELATION COLUMN SELECTION
# ============================================================

def _build_correlation_columns(
    df: pd.DataFrame,
    analysis_plan: dict[str, Any],
    target_column: str = "",
) -> list[str]:
    """
    Select only ordinary numeric analytical features.

    Geographic coordinates, identifiers and metadata are
    intentionally excluded from generic Pearson correlation.
    """

    columns = [
        str(column)
        for column
        in analysis_plan.get(
            "numeric_columns",
            [],
        )
        if str(column)
        in df.columns
    ]


    # --------------------------------------------------------
    # Add numeric target when appropriate
    # --------------------------------------------------------

    if (
        target_column
        and target_column in df.columns
        and target_column not in columns
    ):

        target_info = (
            analysis_plan.get(
                "target",
                {},
            )
        )

        target_family = str(
            target_info.get(
                "analysis_family",
                "",
            )
        )

        target_role = str(
            target_info.get(
                "semantic_role",
                "",
            )
        )


        # Do not treat semantic coordinates, IDs or metadata
        # as ordinary numeric correlation targets.
        unsuitable_roles = {
            "latitude",
            "longitude",
            "identifier",
            "url",
            "email",
            "phone",
            "address",
        }


        if (
            target_role
            not in unsuitable_roles
            and target_family
            not in {
                "metadata",
                "identifier",
                "geospatial",
                "structural",
                "none",
            }
        ):

            target_series = (
                df[
                    target_column
                ]
            )


            if (
                pd.api.types
                .is_numeric_dtype(
                    target_series
                )
                or pd.api.types
                .is_bool_dtype(
                    target_series
                )
            ):

                columns.append(
                    target_column
                )

            else:

                converted = (
                    pd.to_numeric(
                        target_series,
                        errors="coerce",
                    )
                )

                source_non_null = int(
                    target_series
                    .notna()
                    .sum()
                )

                converted_non_null = int(
                    converted
                    .notna()
                    .sum()
                )

                parse_ratio = (
                    converted_non_null
                    / source_non_null
                    if source_non_null
                    else 0.0
                )

                if parse_ratio >= 0.95:

                    columns.append(
                        target_column
                    )


    return list(
        dict.fromkeys(
            columns
        )
    )


# ============================================================
# MAIN EVIDENCE BUILDER
# ============================================================

def build_analysis_evidence(
    df: pd.DataFrame,
    target_column: str = "",
) -> dict[str, Any]:
    """
    Build deterministic structured evidence for heterogeneous
    tabular and semi-structured datasets.

    The routing layer decides which specialized analyzers
    receive each semantic data type.
    """

    # ========================================================
    # DATASET PROFILE
    # ========================================================

    dataset_profile = (
        build_dataset_profile(
            df
        )
    )


    # ========================================================
    # SCHEMA + SEMANTIC ROUTING
    # ========================================================

    routing_result = (
        route_dataset(
            df=df,
            target_column=target_column,
        )
    )

    schema = (
        routing_result[
            "schema"
        ]
    )

    schema_summary = (
        routing_result.get(
            "schema_summary",
            {},
        )
    )

    analysis_plan = (
        routing_result[
            "analysis_plan"
        ]
    )


    # ========================================================
    # ROUTED COLUMN GROUPS
    # ========================================================

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

    text_columns = list(
        analysis_plan.get(
            "text_columns",
            [],
        )
    )

    identifier_columns = list(
        analysis_plan.get(
            "identifier_columns",
            [],
        )
    )

    entity_columns = list(
        analysis_plan.get(
            "entity_columns",
            [],
        )
    )

    structural_columns = list(
        analysis_plan.get(
            "structural_columns",
            [],
        )
    )

    ignored_columns = list(
        analysis_plan.get(
            "ignored_columns",
            [],
        )
    )


    # ========================================================
    # NUMERICAL
    # ========================================================

    numerical_analysis = (
        _safe_analysis(
            analyzer_name=
                "numerical_analysis",

            function=
                analyze_numerical_columns,

            df=df,

            numeric_columns=
                numeric_columns,
        )
    )


    # ========================================================
    # CATEGORICAL
    # ========================================================

    categorical_analysis = (
        _safe_analysis(
            analyzer_name=
                "categorical_analysis",

            function=
                analyze_categorical_columns,

            df=df,

            categorical_columns=
                categorical_columns,
        )
    )


    # ========================================================
    # DATETIME
    # ========================================================

    datetime_analysis = (
        _safe_analysis(
            analyzer_name=
                "datetime_analysis",

            function=
                analyze_datetime_columns,

            df=df,

            datetime_columns=
                datetime_columns,
        )
    )


    # ========================================================
    # FREE TEXT
    # ========================================================

    text_analysis = (
        _safe_analysis(
            analyzer_name=
                "text_analysis",

            function=
                analyze_text_columns,

            df=df,

            text_columns=
                text_columns,
        )
    )


    # ========================================================
    # GEOSPATIAL
    # ========================================================

    geospatial_analysis = (
        _safe_analysis(
            analyzer_name=
                "geospatial_analysis",

            function=
                analyze_geospatial_data,

            df=df,

            analysis_plan=
                analysis_plan,
        )
    )


    # ========================================================
    # MULTI-VALUE / STRUCTURAL DATA
    # ========================================================

    multi_value_analysis = (
        _safe_analysis(
            analyzer_name=
                "multi_value_analysis",

            function=
                analyze_multi_value_data,

            df=df,

            analysis_plan=
                analysis_plan,
        )
    )


    # ========================================================
    # METADATA / ENTITY FIELDS
    # ========================================================

    metadata_analysis = (
        _safe_analysis(
            analyzer_name=
                "metadata_analysis",

            function=
                analyze_metadata_data,

            df=df,

            analysis_plan=
                analysis_plan,
        )
    )


    # ========================================================
    # TARGET
    # ========================================================

    if (
        target_column
        and target_column
        in df.columns
    ):

        target_analysis = (
            _safe_analysis(
                analyzer_name=
                    "target_analysis",

                function=
                    analyze_target,

                df=df,

                target_column=
                    target_column,

                schema=
                    schema,
            )
        )

    else:

        target_analysis = {}


    # ========================================================
    # NUMERIC CORRELATION
    # ========================================================

    correlation_columns = (
        _build_correlation_columns(
            df=df,
            analysis_plan=analysis_plan,
            target_column=target_column,
        )
    )

    correlation_analysis = (
        _safe_analysis(
            analyzer_name=
                "correlation_analysis",

            function=
                analyze_correlations,

            df=df,

            numeric_columns=
                correlation_columns,
        )
    )


    # ========================================================
    # CROSS-TYPE RELATIONSHIPS
    # ========================================================

    relationship_analysis = (
        _safe_analysis(
            analyzer_name=
                "relationship_analysis",

            function=
                analyze_cross_type_relationships,

            df=df,

            analysis_plan=
                analysis_plan,

            target_column=
                target_column,
        )
    )


    # ========================================================
    # TEXT ↔ TARGET
    # ========================================================

    if isinstance(
        target_analysis,
        dict,
    ):

        task_type = str(
            target_analysis.get(
                "task_type",
                "",
            )
        )

    else:

        task_type = ""


    if (
        target_column
        and target_column in df.columns
        and text_columns
        and task_type
    ):

        text_target_analysis = (
            _safe_analysis(
                analyzer_name=
                    "text_target_analysis",

                function=
                    analyze_text_target_relationships,

                df=df,

                text_columns=
                    text_columns,

                target_column=
                    target_column,

                task_type=
                    task_type,
            )
        )

    else:

        text_target_analysis = {}


    # ========================================================
    # SEMANTIC DATA QUALITY
    # ========================================================

    data_quality = (
        _safe_analysis(
            analyzer_name=
                "data_quality",

            function=
                analyze_data_quality,

            df=df,

            schema=
                schema,

            schema_summary=
                schema_summary,

            analysis_plan=
                analysis_plan,
        )
    )


    # ========================================================
    # GROUNDING CONTRACT + COLUMN FACT INDEX
    # ========================================================

    grounding_contract = (
        _build_grounding_contract(
            target_column=target_column,
            target_analysis=target_analysis,
        )
    )

    column_fact_index = (
        _build_column_fact_index(
            df=df,
            schema=schema,
            numerical_analysis=
                numerical_analysis,
            data_quality=
                data_quality,
            target_analysis=
                target_analysis,
        )
    )


    # ========================================================
    # ANALYSIS SUMMARY
    # ========================================================

    metadata_columns = (
        analysis_plan.get(
            "metadata_columns",
            {},
        )
    )

    geospatial_columns = (
        analysis_plan.get(
            "geospatial_columns",
            {},
        )
    )

    multi_value_groups = list(
        analysis_plan.get(
            "multi_value_groups",
            [],
        )
    )


    analysis_summary = {
        "dataset_mode":
            analysis_plan.get(
                "dataset_mode",
                "unknown",
            ),

        "active_analysis_families":
            list(
                analysis_plan.get(
                    "active_analysis_families",
                    [],
                )
            ),

        "numeric_columns":
            numeric_columns,

        "categorical_columns":
            categorical_columns,

        "datetime_columns":
            datetime_columns,

        "text_columns":
            text_columns,

        "geospatial_columns":
            geospatial_columns,

        "geospatial_pairs":
            list(
                analysis_plan.get(
                    "geospatial_pairs",
                    [],
                )
            ),

        "metadata_columns":
            metadata_columns,

        "entity_columns":
            entity_columns,

        "multi_value_groups":
            multi_value_groups,

        "identifier_columns":
            identifier_columns,

        "structural_columns":
            structural_columns,

        "ignored_columns":
            ignored_columns,

        "correlation_columns":
            correlation_columns,

        "target_column":
            (
                target_column
                if target_column
                else None
            ),

        "target_task_type":
            (
                task_type
                if task_type
                else None
            ),

        "routing_warnings":
            list(
                analysis_plan.get(
                    "warnings",
                    [],
                )
            ),
    }


    # ========================================================
    # FINAL EVIDENCE PACKAGE
    # ========================================================

    evidence = {
        "dataset_profile":
            dataset_profile,

        "grounding_contract":
            grounding_contract,

        "column_fact_index":
            column_fact_index,

        "schema":
            schema,

        "schema_summary":
            schema_summary,

        "analysis_plan":
            analysis_plan,

        "analysis_summary":
            analysis_summary,

        "numerical_analysis":
            numerical_analysis,

        "categorical_analysis":
            categorical_analysis,

        "datetime_analysis":
            datetime_analysis,

        "text_analysis":
            text_analysis,

        "geospatial_analysis":
            geospatial_analysis,

        "multi_value_analysis":
            multi_value_analysis,

        "metadata_analysis":
            metadata_analysis,

        "correlation_analysis":
            correlation_analysis,

        "relationship_analysis":
            relationship_analysis,

        "text_target_analysis":
            text_target_analysis,

        "data_quality":
            data_quality,

        "target_analysis":
            target_analysis,
    }


    return evidence