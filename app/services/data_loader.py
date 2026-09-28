from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

SUPPORTED_EXTENSIONS = {
    ".csv",
    ".tsv",
    ".txt",
    ".xlsx",
    ".xls",
    ".parquet",
    ".json",
    ".jsonl",
    ".ndjson",
}

DEFAULT_ENCODING = "utf-8"

FALLBACK_ENCODINGS = [
    "utf-8-sig",
    "latin1",
    "cp1252",
]


# ------------------------------------------------------------
# Exceptions
# ------------------------------------------------------------

class DataLoadError(Exception):
    """Raised when a dataset cannot be loaded safely."""


class UnsupportedFileTypeError(DataLoadError):
    """Raised when the file format is unsupported."""


class EmptyDatasetError(DataLoadError):
    """Raised when the loaded dataset contains no usable data."""


# ------------------------------------------------------------
# Validation
# ------------------------------------------------------------

def _validate_path(
    file_path: str | Path,
) -> Path:
    """Validate and normalize a dataset path."""

    path = Path(
        file_path
    )

    if not path.exists():

        raise DataLoadError(
            f"Dataset file was not found: {path}"
        )

    if not path.is_file():

        raise DataLoadError(
            f"Dataset path is not a file: {path}"
        )

    extension = (
        path.suffix
        .lower()
    )

    if (
        extension
        not in SUPPORTED_EXTENSIONS
    ):

        raise UnsupportedFileTypeError(
            (
                f"Unsupported file type: "
                f"{extension or 'unknown'}"
            )
        )

    return path


# ------------------------------------------------------------
# Final dataframe validation
# ------------------------------------------------------------

def _finalize_dataframe(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply common validation after loading any format.
    """

    if df is None:

        raise EmptyDatasetError(
            "The dataset could not be loaded."
        )

    if not isinstance(
        df,
        pd.DataFrame,
    ):

        raise DataLoadError(
            "Loaded data is not a tabular dataframe."
        )

    if len(df.columns) == 0:

        raise EmptyDatasetError(
            "The dataset contains no columns."
        )

    if len(df) == 0:

        raise EmptyDatasetError(
            "The dataset contains no rows."
        )

    # Normalize column names to strings.
    df = df.copy()

    df.columns = [
        str(column)
        for column in df.columns
    ]

    # Detect duplicate column names.
    duplicated_columns = (
        pd.Index(
            df.columns
        )
        .duplicated()
    )

    if duplicated_columns.any():

        duplicates = (
            pd.Index(
                df.columns
            )[
                duplicated_columns
            ]
            .tolist()
        )

        raise DataLoadError(
            (
                "Duplicate column names were detected: "
                f"{duplicates}"
            )
        )

    return df


# ------------------------------------------------------------
# CSV / TSV loading
# ------------------------------------------------------------

def _read_delimited_file(
    path: Path,
    separator: str | None = None,
) -> pd.DataFrame:
    """
    Load CSV, TSV, or text-delimited datasets.

    Delimiter detection is used when no separator
    is explicitly supplied.
    """

    encodings = [
        DEFAULT_ENCODING,
        *FALLBACK_ENCODINGS,
    ]

    # Remove duplicates while preserving order.
    encodings = list(
        dict.fromkeys(
            encodings
        )
    )

    last_error: Exception | None = None

    for encoding in encodings:

        try:

            if separator is not None:

                return pd.read_csv(
                    path,
                    sep=separator,
                    encoding=encoding,
                    low_memory=False,
                )

            # Python engine can infer common delimiters.
            return pd.read_csv(
                path,
                sep=None,
                engine="python",
                encoding=encoding,
            )

        except UnicodeDecodeError as exc:

            last_error = exc

            continue

        except pd.errors.ParserError as exc:

            last_error = exc

            continue

    raise DataLoadError(
        (
            "The delimited dataset could not be parsed. "
            f"Last error: {last_error}"
        )
    )


# ------------------------------------------------------------
# Excel loading
# ------------------------------------------------------------

def _read_excel_file(
    path: Path,
    sheet_name: str | int | None = 0,
) -> pd.DataFrame:
    """
    Load one Excel worksheet.

    The first worksheet is used by default.
    """

    try:

        df = pd.read_excel(
            path,
            sheet_name=sheet_name,
        )

    except ImportError as exc:

        raise DataLoadError(
            (
                "Excel support requires the appropriate "
                "Excel engine. Install openpyxl for XLSX files."
            )
        ) from exc

    except Exception as exc:

        raise DataLoadError(
            f"Excel file could not be read: {exc}"
        ) from exc


    if isinstance(
        df,
        dict,
    ):

        if not df:

            raise EmptyDatasetError(
                "The Excel workbook contains no readable sheets."
            )

        first_sheet = next(
            iter(
                df.values()
            )
        )

        return first_sheet

    return df


# ------------------------------------------------------------
# Parquet loading
# ------------------------------------------------------------

def _read_parquet_file(
    path: Path,
) -> pd.DataFrame:
    """Load a Parquet dataset."""

    try:

        return pd.read_parquet(
            path
        )

    except ImportError as exc:

        raise DataLoadError(
            (
                "Parquet support requires pyarrow "
                "or fastparquet."
            )
        ) from exc

    except Exception as exc:

        raise DataLoadError(
            f"Parquet file could not be read: {exc}"
        ) from exc


# ------------------------------------------------------------
# JSON structure helpers
# ------------------------------------------------------------

def _json_object_to_dataframe(
    data: Any,
) -> pd.DataFrame:
    """
    Convert common JSON structures into a dataframe.
    """

    # --------------------------------------------------------
    # List of records
    # --------------------------------------------------------

    if isinstance(
        data,
        list,
    ):

        if not data:

            raise EmptyDatasetError(
                "JSON dataset is empty."
            )

        try:

            return pd.json_normalize(
                data
            )

        except Exception as exc:

            raise DataLoadError(
                (
                    "JSON records could not be "
                    f"normalized: {exc}"
                )
            ) from exc


    # --------------------------------------------------------
    # Dictionary
    # --------------------------------------------------------

    if isinstance(
        data,
        dict,
    ):

        if not data:

            raise EmptyDatasetError(
                "JSON object is empty."
            )


        # Common API-style structure:
        #
        # {
        #     "data": [
        #         {...},
        #         {...}
        #     ]
        # }

        common_record_keys = [
            "data",
            "records",
            "results",
            "items",
            "rows",
        ]


        for key in common_record_keys:

            value = data.get(
                key
            )

            if (
                isinstance(
                    value,
                    list,
                )
                and value
            ):

                try:

                    return pd.json_normalize(
                        value
                    )

                except Exception:
                    pass


        # Try dictionary-of-columns format.
        try:

            return pd.DataFrame(
                data
            )

        except ValueError:

            # Single nested object.
            try:

                return pd.json_normalize(
                    data
                )

            except Exception as exc:

                raise DataLoadError(
                    (
                        "JSON object could not be "
                        f"converted to tabular data: {exc}"
                    )
                ) from exc


    raise DataLoadError(
        (
            "Unsupported JSON structure. "
            "Expected records, an object, "
            "or a dictionary of columns."
        )
    )


# ------------------------------------------------------------
# JSON loading
# ------------------------------------------------------------

def _read_json_file(
    path: Path,
) -> pd.DataFrame:
    """Load JSON or JSON Lines datasets."""

    extension = (
        path.suffix
        .lower()
    )


    # --------------------------------------------------------
    # JSON Lines / NDJSON
    # --------------------------------------------------------

    if extension in {
        ".jsonl",
        ".ndjson",
    }:

        try:

            return pd.read_json(
                path,
                lines=True,
            )

        except Exception as exc:

            raise DataLoadError(
                (
                    "JSON Lines dataset could "
                    f"not be read: {exc}"
                )
            ) from exc


    # --------------------------------------------------------
    # Standard JSON
    # --------------------------------------------------------

    last_error: Exception | None = None

    encodings = [
        DEFAULT_ENCODING,
        *FALLBACK_ENCODINGS,
    ]

    encodings = list(
        dict.fromkeys(
            encodings
        )
    )


    for encoding in encodings:

        try:

            with open(
                path,
                "r",
                encoding=encoding,
            ) as file:

                data = json.load(
                    file
                )

            return _json_object_to_dataframe(
                data
            )

        except UnicodeDecodeError as exc:

            last_error = exc

            continue

        except json.JSONDecodeError as exc:

            last_error = exc

            break


    raise DataLoadError(
        (
            "JSON dataset could not be parsed. "
            f"Last error: {last_error}"
        )
    )


# ------------------------------------------------------------
# Main loader
# ------------------------------------------------------------

def load_dataset(
    file_path: str | Path,
    *,
    sheet_name: str | int | None = 0,
) -> pd.DataFrame:
    """
    Load a supported tabular dataset.

    Supported formats:
    - CSV
    - TSV
    - TXT
    - XLSX
    - XLS
    - Parquet
    - JSON
    - JSONL
    - NDJSON
    """

    path = _validate_path(
        file_path
    )

    extension = (
        path.suffix
        .lower()
    )


    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    if extension == ".csv":

        df = _read_delimited_file(
            path
        )


    # --------------------------------------------------------
    # TSV
    # --------------------------------------------------------

    elif extension == ".tsv":

        df = _read_delimited_file(
            path,
            separator="\t",
        )


    # --------------------------------------------------------
    # Generic text table
    # --------------------------------------------------------

    elif extension == ".txt":

        df = _read_delimited_file(
            path
        )


    # --------------------------------------------------------
    # Excel
    # --------------------------------------------------------

    elif extension in {
        ".xlsx",
        ".xls",
    }:

        df = _read_excel_file(
            path,
            sheet_name=sheet_name,
        )


    # --------------------------------------------------------
    # Parquet
    # --------------------------------------------------------

    elif extension == ".parquet":

        df = _read_parquet_file(
            path
        )


    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------

    elif extension in {
        ".json",
        ".jsonl",
        ".ndjson",
    }:

        df = _read_json_file(
            path
        )


    else:

        raise UnsupportedFileTypeError(
            (
                "Unsupported dataset format: "
                f"{extension}"
            )
        )


    return _finalize_dataframe(
        df
    )


# ------------------------------------------------------------
# Dataset metadata
# ------------------------------------------------------------

def get_dataset_file_info(
    file_path: str | Path,
) -> dict[str, Any]:
    """
    Return basic metadata without analyzing the dataset.
    """

    path = _validate_path(
        file_path
    )

    return {
        "filename":
            path.name,

        "extension":
            path.suffix.lower(),

        "size_bytes":
            int(
                path.stat().st_size
            ),

        "supported":
            True,
    }