from __future__ import annotations

import re
import uuid
from pathlib import Path

import bleach
import markdown

from fastapi import (
    FastAPI,
    File,
    HTTPException,
    Query,
    Request,
    UploadFile,
)

from fastapi.responses import (
    FileResponse,
    HTMLResponse,
)

from fastapi.staticfiles import (
    StaticFiles,
)

from fastapi.templating import (
    Jinja2Templates,
)

from starlette.concurrency import (
    run_in_threadpool,
)

from app.services.analysis_service import (
    run_analysis,
)

from app.services.data_loader import (
    DataLoadError,
    SUPPORTED_EXTENSIONS,
    load_dataset,
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

WEB_DIR = (
    PROJECT_DIR
    / "app"
    / "web"
)

TEMPLATES_DIR = (
    WEB_DIR
    / "templates"
)

STATIC_DIR = (
    WEB_DIR
    / "static"
)

UPLOAD_DIR = (
    PROJECT_DIR
    / "uploaded_data"
)

RUNS_DIR = (
    PROJECT_DIR
    / "runs"
)


UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RUNS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CONFIGURATION
# ============================================================

MAX_UPLOAD_SIZE = (
    50
    * 1024
    * 1024
)

UPLOAD_CHUNK_SIZE = (
    1024
    * 1024
)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="AI Data Analyst",
    version="2.0.0",
)


# ============================================================
# STATIC FILES
# ============================================================

app.mount(
    "/static",
    StaticFiles(
        directory=str(
            STATIC_DIR
        )
    ),
    name="static",
)


# ============================================================
# TEMPLATES
# ============================================================

templates = Jinja2Templates(
    directory=str(
        TEMPLATES_DIR
    )
)


# ============================================================
# VALIDATION HELPERS
# ============================================================

def _valid_file_id(
    file_id: str,
) -> bool:
    """
    Validate generated upload IDs.
    """

    return bool(
        re.fullmatch(
            r"[a-f0-9]{32}",
            file_id,
        )
    )


def _valid_run_id(
    run_id: str,
) -> bool:
    """
    Validate generated analysis run IDs.
    """

    return bool(
        re.fullmatch(
            r"[a-f0-9]{32}",
            run_id,
        )
    )


# ============================================================
# UPLOADED DATASET LOOKUP
# ============================================================

def _find_uploaded_dataset(
    file_id: str,
) -> Path:
    """
    Resolve one uploaded dataset from its file ID.
    """

    if not _valid_file_id(
        file_id
    ):

        raise HTTPException(
            status_code=400,
            detail="Invalid file ID.",
        )


    matches = [
        path
        for path in UPLOAD_DIR.glob(
            f"{file_id}.*"
        )
        if path.suffix.lower()
        in SUPPORTED_EXTENSIONS
    ]


    if len(matches) == 0:

        raise HTTPException(
            status_code=404,
            detail=(
                "Uploaded dataset "
                "was not found."
            ),
        )


    if len(matches) > 1:

        raise HTTPException(
            status_code=409,
            detail=(
                "Multiple uploaded files "
                "were found for the same "
                "file ID."
            ),
        )


    return matches[0]


# ============================================================
# REPORT RENDERING
# ============================================================

def render_report_html(
    report: str,
    run_id: str,
) -> str:
    """
    Convert Markdown report to sanitized HTML.
    """

    if not report:

        return ""


    # Rewrite local chart paths if they appear
    # inside generated Markdown.
    report = report.replace(
        "](charts/",
        (
            "]("
            f"/api/runs/{run_id}/charts/"
        ),
    )


    raw_html = markdown.markdown(
        report,
        extensions=[
            "tables",
            "fenced_code",
        ],
    )


    allowed_tags = {
        "p",
        "br",
        "strong",
        "em",
        "b",
        "i",
        "ul",
        "ol",
        "li",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "blockquote",
        "code",
        "pre",
        "table",
        "thead",
        "tbody",
        "tr",
        "th",
        "td",
        "a",
        "hr",
    }


    allowed_attributes = {
        "a": [
            "href",
            "title",
            "target",
            "rel",
        ],
    }


    return bleach.clean(
        raw_html,
        tags=allowed_tags,
        attributes=allowed_attributes,
        protocols=[
            "http",
            "https",
        ],
        strip=True,
    )


# ============================================================
# HOME PAGE
# ============================================================

@app.get(
    "/",
    response_class=HTMLResponse,
)
async def home(
    request: Request,
):
    """
    Render the main dashboard.
    """

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "request": request,
        },
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get(
    "/health"
)
async def health():
    """
    Basic application health check.
    """

    return {
        "status":
            "ok",

        "service":
            "AI Data Analyst",

        "version":
            "2.0.0",

        "supported_formats":
            sorted(
                SUPPORTED_EXTENSIONS
            ),

        "max_upload_mb":
            int(
                MAX_UPLOAD_SIZE
                / 1024
                / 1024
            ),
    }


# ============================================================
# UPLOAD API
# ============================================================

@app.post(
    "/api/upload"
)
async def upload_dataset(
    file: UploadFile = File(...),
):
    """
    Upload and validate a supported tabular dataset.
    """

    # --------------------------------------------------------
    # Validate filename
    # --------------------------------------------------------

    original_filename = (
        Path(
            file.filename
            or ""
        )
        .name
    )


    if not original_filename:

        raise HTTPException(
            status_code=400,
            detail=(
                "A filename is required."
            ),
        )


    extension = (
        Path(
            original_filename
        )
        .suffix
        .lower()
    )


    # --------------------------------------------------------
    # Validate format
    # --------------------------------------------------------

    if (
        extension
        not in SUPPORTED_EXTENSIONS
    ):

        raise HTTPException(
            status_code=400,
            detail={
                "message":
                    (
                        "Unsupported "
                        "dataset format."
                    ),

                "supported_formats":
                    sorted(
                        SUPPORTED_EXTENSIONS
                    ),
            },
        )


    # --------------------------------------------------------
    # Generate internal upload ID
    # --------------------------------------------------------

    file_id = (
        uuid.uuid4().hex
    )


    destination = (
        UPLOAD_DIR
        / f"{file_id}{extension}"
    )


    size_bytes = 0


    # --------------------------------------------------------
    # Stream file to disk
    # --------------------------------------------------------

    try:

        with destination.open(
            "wb"
        ) as output:

            while True:

                chunk = await file.read(
                    UPLOAD_CHUNK_SIZE
                )


                if not chunk:

                    break


                size_bytes += len(
                    chunk
                )


                if (
                    size_bytes
                    > MAX_UPLOAD_SIZE
                ):

                    raise HTTPException(
                        status_code=413,
                        detail=(
                            "Dataset is larger "
                            "than the current "
                            "50 MB upload limit."
                        ),
                    )


                output.write(
                    chunk
                )


    except Exception:

        if destination.exists():

            destination.unlink(
                missing_ok=True
            )

        raise


    finally:

        await file.close()


    # --------------------------------------------------------
    # Validate uploaded dataset
    # --------------------------------------------------------

    try:

        df = await run_in_threadpool(
            load_dataset,
            destination,
        )


    except DataLoadError as exc:

        destination.unlink(
            missing_ok=True
        )


        raise HTTPException(
            status_code=400,
            detail=(
                "Dataset could not "
                f"be loaded: {exc}"
            ),
        ) from exc


    except Exception as exc:

        destination.unlink(
            missing_ok=True
        )


        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid dataset: "
                f"{exc}"
            ),
        ) from exc


    # --------------------------------------------------------
    # Return detected columns
    # --------------------------------------------------------

    return {
        "status":
            "uploaded",

        "filename":
            original_filename,

        "file_id":
            file_id,

        "extension":
            extension,

        "size_bytes":
            size_bytes,

        "rows":
            int(
                len(df)
            ),

        "column_count":
            int(
                len(
                    df.columns
                )
            ),

        "columns":
            [
                str(column)
                for column
                in df.columns
            ],

        "supported":
            True,
    }


# ============================================================
# ANALYSIS API
# ============================================================

@app.post(
    "/api/analyze/{file_id}"
)
async def analyze_dataset(
    file_id: str,
    target_column: str = Query(
        default=""
    ),
):
    """
    Run the complete analysis pipeline
    for an uploaded dataset.
    """

    # --------------------------------------------------------
    # Resolve uploaded file
    # --------------------------------------------------------

    dataset_path = (
        _find_uploaded_dataset(
            file_id
        )
    )


    # --------------------------------------------------------
    # Run analysis
    # --------------------------------------------------------

    try:

        result = await run_in_threadpool(
            run_analysis,
            dataset_path,
            target_column,
        )


    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(
                exc
            ),
        ) from exc


    except DataLoadError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(
                exc
            ),
        ) from exc


    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Analysis failed: "
                f"{exc}"
            ),
        ) from exc


    # --------------------------------------------------------
    # Render report
    # --------------------------------------------------------

    run_id = (
        result.get(
            "run_id",
            "",
        )
    )


    report_html = (
        render_report_html(
            report=result.get(
                "report",
                "",
            ),
            run_id=run_id,
        )
    )


    # --------------------------------------------------------
    # API response
    # --------------------------------------------------------

    return {
        **result,

        "report_html":
            report_html,
    }


# ============================================================
# CHART SERVING API
# ============================================================

@app.get(
    "/api/runs/{run_id}/charts/{filename}"
)
async def get_chart(
    run_id: str,
    filename: str,
):
    """
    Serve one generated chart securely.
    """

    # --------------------------------------------------------
    # Validate run ID
    # --------------------------------------------------------

    if not _valid_run_id(
        run_id
    ):

        raise HTTPException(
            status_code=400,
            detail="Invalid run ID.",
        )


    # --------------------------------------------------------
    # Prevent path traversal
    # --------------------------------------------------------

    safe_filename = (
        Path(
            filename
        )
        .name
    )


    if (
        safe_filename
        != filename
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid chart filename."
            ),
        )


    # --------------------------------------------------------
    # Resolve chart
    # --------------------------------------------------------

    chart_path = (
        RUNS_DIR
        / run_id
        / "charts"
        / safe_filename
    )


    if (
        not chart_path.exists()
        or not chart_path.is_file()
    ):

        raise HTTPException(
            status_code=404,
            detail=(
                "Chart was not found."
            ),
        )


    # --------------------------------------------------------
    # Restrict supported image formats
    # --------------------------------------------------------

    if (
        chart_path.suffix.lower()
        not in {
            ".png",
            ".jpg",
            ".jpeg",
            ".webp",
        }
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported chart "
                "file type."
            ),
        )


    return FileResponse(
        path=chart_path
    )