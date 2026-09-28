from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from app.graph.workflow import (
    build_workflow,
)

from app.services.chart_service import (
    generate_charts,
)

from app.services.data_loader import (
    load_dataset,
)

from app.services.profiling_service import (
    build_analysis_evidence,
)


# ------------------------------------------------------------
# Project paths
# ------------------------------------------------------------

PROJECT_DIR = (
    Path(__file__)
    .resolve()
    .parents[2]
)

RUNS_DIR = (
    PROJECT_DIR
    / "runs"
)

RUNS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ------------------------------------------------------------
# JSON helpers
# ------------------------------------------------------------

def _make_json_safe(
    value: Any,
) -> Any:
    """
    Convert common structured objects into
    JSON-serializable Python objects.
    """

    if hasattr(
        value,
        "model_dump",
    ):
        return value.model_dump()

    if hasattr(
        value,
        "dict",
    ):
        try:
            return value.dict()
        except Exception:
            pass

    return value


def _save_json(
    path: Path,
    data: Any,
) -> None:
    """
    Save structured analysis output safely.
    """

    data = _make_json_safe(
        data
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
            default=str,
        )


# ------------------------------------------------------------
# Main analysis service
# ------------------------------------------------------------

def run_analysis(
    dataset_path: Path,
    target_column: str = "",
    language: str = "English",
) -> dict[str, Any]:
    """
    Run the complete AI Data Analyst pipeline.

    Supported input formats are handled by load_dataset():
    - CSV
    - TSV
    - TXT
    - Excel
    - Parquet
    - JSON
    - JSONL
    - NDJSON
    """

    # Keep parameter for API compatibility.
    _ = language


    # --------------------------------------------------------
    # Create run directory
    # --------------------------------------------------------

    run_id = (
        uuid.uuid4().hex
    )

    run_dir = (
        RUNS_DIR
        / run_id
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=False,
    )


    charts_dir = (
        run_dir
        / "charts"
    )

    charts_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    df = load_dataset(
        dataset_path
    )


    # --------------------------------------------------------
    # Validate target
    # --------------------------------------------------------

    if (
        target_column
        and target_column not in df.columns
    ):

        raise ValueError(
            (
                f"Target column '{target_column}' "
                "does not exist in the dataset."
            )
        )


    # --------------------------------------------------------
    # Deterministic evidence
    # --------------------------------------------------------

    evidence = (
        build_analysis_evidence(
            df=df,
            target_column=target_column,
        )
    )


    # --------------------------------------------------------
    # Dataset source metadata
    # --------------------------------------------------------

    evidence[
        "dataset_source"
    ] = {
        "filename":
            dataset_path.name,

        "extension":
            dataset_path
            .suffix
            .lower(),

        "rows":
            int(
                len(df)
            ),

        "columns":
            int(
                len(df.columns)
            ),
    }


    # --------------------------------------------------------
    # Charts
    # --------------------------------------------------------

    generated_charts = (
        generate_charts(
            df=df,
            output_dir=charts_dir,
            target_column=target_column,
        )
    )


    evidence[
        "generated_charts"
    ] = (
        generated_charts
    )


    # --------------------------------------------------------
    # AI workflow
    # --------------------------------------------------------

    workflow = (
        build_workflow()
    )


    workflow_result = (
        workflow.invoke(
            {
                "evidence":
                    evidence,

                "iteration":
                    0,
            }
        )
    )


    # --------------------------------------------------------
    # Extract workflow results
    # --------------------------------------------------------

    review = (
        workflow_result.get(
            "review",
            {},
        )
        or {}
    )


    review = (
        _make_json_safe(
            review
        )
    )


    approved = bool(
        workflow_result.get(
            "approved",
            False,
        )
    )


    final_report = (
        workflow_result.get(
            "final_report"
        )
        or workflow_result.get(
            "report"
        )
        or ""
    )


    # --------------------------------------------------------
    # Persist artifacts
    # --------------------------------------------------------

    report_path = (
        run_dir
        / "eda_report.md"
    )


    review_path = (
        run_dir
        / "review_feedback.json"
    )


    evidence_path = (
        run_dir
        / "evidence.json"
    )


    report_path.write_text(
        final_report,
        encoding="utf-8",
    )


    _save_json(
        review_path,
        review,
    )


    _save_json(
        evidence_path,
        evidence,
    )


    # --------------------------------------------------------
    # Final service response
    # --------------------------------------------------------

    return {
        "success":
            True,

        "run_id":
            run_id,

        "approved":
            approved,

        "report":
            final_report,

        "review":
            review,

        "charts":
            generated_charts,

        "evidence":
            evidence,

        "dataset": {
            "rows":
                int(len(df)),

            "columns":
                int(len(df.columns)),

            "column_names":
                [
                    str(column)
                    for column
                    in df.columns
                ],

            "filename":
                dataset_path.name,

            "extension":
                dataset_path.suffix.lower(),
        },
    }