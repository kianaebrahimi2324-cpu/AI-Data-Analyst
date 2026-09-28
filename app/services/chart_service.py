from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from app.services.analysis_router import route_dataset
from app.tools.target_analysis import analyze_target


# ============================================================
# CONFIGURATION
# ============================================================

MAX_NUMERIC_CHARTS = 8
MAX_CATEGORICAL_CHARTS = 6
MAX_DATETIME_CHARTS = 3
MAX_GEOSPATIAL_CHARTS = 3
MAX_MULTI_VALUE_CHARTS = 4

MAX_CATEGORY_LEVELS = 20
MAX_MULTI_VALUE_LEVELS = 20
MAX_GEOSPATIAL_POINTS = 5000

HISTOGRAM_BINS = 30
MIN_CORRELATION_COLUMNS = 2


# ============================================================
# GENERIC HELPERS
# ============================================================

def _safe_filename_component(value: Any) -> str:
    """Convert a value into a safe filename component."""
    name = str(value).strip()
    name = re.sub(r'[\\/:*?"<>|]+', "_", name)
    name = re.sub(r"\s+", "_", name)

    if not name:
        return "column"

    return name[:100]


def _save_figure(figure, path: Path) -> None:
    """Save and close a Matplotlib figure."""
    figure.tight_layout()
    figure.savefig(
        path,
        dpi=140,
        bbox_inches="tight",
    )
    plt.close(figure)


def _numeric_series(series: pd.Series) -> pd.Series:
    """Convert a series to finite numeric observations."""
    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    numeric = numeric.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return numeric.dropna()


def _clean_text_value(value: Any) -> str | None:
    """Normalize one cell for multi-value charting."""
    if pd.isna(value):
        return None

    text = str(value).strip()

    if not text:
        return None

    return text


# ============================================================
# STANDARD COLUMN SELECTION
# ============================================================

def _select_numeric_columns(
    df: pd.DataFrame,
    numeric_columns: list[str],
    target_column: str = "",
) -> list[str]:
    """
    Select ordinary numeric feature columns.

    Geospatial columns are already excluded by the semantic router.
    """
    candidates: list[tuple[str, float]] = []

    for column in numeric_columns:
        if column not in df.columns:
            continue

        if column == target_column:
            continue

        clean = _numeric_series(
            df[column]
        )

        if len(clean) < 2:
            continue

        if clean.nunique() < 2:
            continue

        variance = float(
            clean.var()
        )

        if not np.isfinite(variance):
            variance = 0.0

        candidates.append(
            (column, variance)
        )

    candidates.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    return [
        column
        for column, _
        in candidates[:MAX_NUMERIC_CHARTS]
    ]


def _select_categorical_columns(
    df: pd.DataFrame,
    categorical_columns: list[str],
    target_column: str = "",
) -> list[str]:
    """Select readable categorical feature columns."""
    selected: list[str] = []

    for column in categorical_columns:
        if column not in df.columns:
            continue

        if column == target_column:
            continue

        unique_count = int(
            df[column]
            .dropna()
            .nunique()
        )

        if (
            1 < unique_count
            <= MAX_CATEGORY_LEVELS
        ):
            selected.append(column)

        if len(selected) >= MAX_CATEGORICAL_CHARTS:
            break

    return selected


# ============================================================
# STANDARD NUMERIC CHARTS
# ============================================================

def _generate_distribution_chart(
    df: pd.DataFrame,
    column: str,
    output_dir: Path,
) -> str | None:
    """Generate a histogram for one ordinary numeric variable."""
    clean = _numeric_series(
        df[column]
    )

    if (
        len(clean) < 2
        or clean.nunique() < 2
    ):
        return None

    filename = (
        "distribution_"
        + _safe_filename_component(column)
        + ".png"
    )

    path = output_dir / filename

    figure, axis = plt.subplots(
        figsize=(7, 4.5)
    )

    bins = min(
        HISTOGRAM_BINS,
        max(
            5,
            int(np.sqrt(len(clean))),
        ),
    )

    axis.hist(
        clean,
        bins=bins,
        edgecolor="black",
        alpha=0.8,
    )

    axis.set_title(
        f"Distribution of {column}"
    )
    axis.set_xlabel(
        str(column)
    )
    axis.set_ylabel(
        "Count"
    )

    _save_figure(
        figure,
        path,
    )

    return filename


def _generate_boxplot_chart(
    df: pd.DataFrame,
    column: str,
    output_dir: Path,
) -> str | None:
    """
    Generate a boxplot for one ordinary numeric variable.

    This function must only receive columns routed as ordinary
    numerical features. Coordinates must never be passed here.
    """
    clean = _numeric_series(
        df[column]
    )

    if (
        len(clean) < 2
        or clean.nunique() < 2
    ):
        return None

    filename = (
        "boxplot_"
        + _safe_filename_component(column)
        + ".png"
    )

    path = output_dir / filename

    figure, axis = plt.subplots(
        figsize=(7, 3.5)
    )

    axis.boxplot(
        clean,
        vert=False,
    )

    axis.set_title(
        f"Boxplot of {column}"
    )
    axis.set_xlabel(
        str(column)
    )

    _save_figure(
        figure,
        path,
    )

    return filename


# ============================================================
# CATEGORICAL CHART
# ============================================================

def _generate_categorical_chart(
    df: pd.DataFrame,
    column: str,
    output_dir: Path,
) -> str | None:
    """Generate a bar chart for a categorical variable."""
    counts = (
        df[column]
        .dropna()
        .astype(str)
        .value_counts()
        .head(MAX_CATEGORY_LEVELS)
    )

    if len(counts) < 2:
        return None

    filename = (
        "categorical_"
        + _safe_filename_component(column)
        + ".png"
    )

    path = output_dir / filename

    figure, axis = plt.subplots(
        figsize=(8, 4.5)
    )

    positions = np.arange(
        len(counts)
    )

    axis.bar(
        positions,
        counts.values,
    )

    axis.set_xticks(
        positions
    )
    axis.set_xticklabels(
        counts.index,
        rotation=45,
        ha="right",
    )

    axis.set_title(
        f"Distribution of {column}"
    )
    axis.set_xlabel(
        str(column)
    )
    axis.set_ylabel(
        "Count"
    )

    _save_figure(
        figure,
        path,
    )

    return filename


# ============================================================
# DATETIME CHART
# ============================================================

def _generate_datetime_chart(
    df: pd.DataFrame,
    column: str,
    output_dir: Path,
) -> str | None:
    """Generate deterministic observation counts over time."""
    if column not in df.columns:
        return None

    parsed = pd.to_datetime(
        df[column],
        errors="coerce",
    ).dropna()

    if len(parsed) < 2:
        return None

    if parsed.nunique() < 2:
        return None

    minimum = parsed.min()
    maximum = parsed.max()
    span_days = max(
        int((maximum - minimum).days),
        0,
    )

    if span_days <= 31:
        periods = parsed.dt.to_period("D")
        x_label = "Day"
    elif span_days <= 730:
        periods = parsed.dt.to_period("M")
        x_label = "Month"
    else:
        periods = parsed.dt.to_period("Y")
        x_label = "Year"

    counts = (
        periods
        .value_counts()
        .sort_index()
    )

    if len(counts) < 2:
        return None

    labels = [
        str(value)
        for value in counts.index
    ]

    filename = (
        "datetime_"
        + _safe_filename_component(column)
        + ".png"
    )

    path = output_dir / filename

    figure, axis = plt.subplots(
        figsize=(8, 4.5)
    )

    axis.plot(
        np.arange(len(counts)),
        counts.values,
        marker="o",
    )

    axis.set_xticks(
        np.arange(len(counts))
    )

    # Keep long time axes readable.
    if len(labels) > 18:
        step = max(
            int(np.ceil(len(labels) / 18)),
            1,
        )
        display_labels = [
            label if index % step == 0 else ""
            for index, label
            in enumerate(labels)
        ]
    else:
        display_labels = labels

    axis.set_xticklabels(
        display_labels,
        rotation=45,
        ha="right",
    )

    axis.set_title(
        f"Observations over time: {column}"
    )
    axis.set_xlabel(
        x_label
    )
    axis.set_ylabel(
        "Count"
    )

    _save_figure(
        figure,
        path,
    )

    return filename


# ============================================================
# GEOSPATIAL CHART
# ============================================================

def _generate_geospatial_chart(
    df: pd.DataFrame,
    latitude_column: str,
    longitude_column: str,
    output_dir: Path,
) -> str | None:
    """
    Generate a coordinate scatter plot using only legally valid
    latitude/longitude pairs.

    Statistical outlier rules are intentionally not used.
    """
    if (
        latitude_column not in df.columns
        or longitude_column not in df.columns
    ):
        return None

    latitude = pd.to_numeric(
        df[latitude_column],
        errors="coerce",
    )

    longitude = pd.to_numeric(
        df[longitude_column],
        errors="coerce",
    )

    valid = (
        latitude.notna()
        & longitude.notna()
        & latitude.between(-90, 90)
        & longitude.between(-180, 180)
    )

    coordinates = pd.DataFrame(
        {
            "latitude": latitude[valid],
            "longitude": longitude[valid],
        }
    )

    if len(coordinates) < 2:
        return None

    if len(coordinates) > MAX_GEOSPATIAL_POINTS:
        indexes = np.linspace(
            0,
            len(coordinates) - 1,
            MAX_GEOSPATIAL_POINTS,
            dtype=int,
        )

        coordinates = (
            coordinates
            .iloc[indexes]
        )

    filename = (
        "geospatial_"
        + _safe_filename_component(latitude_column)
        + "_"
        + _safe_filename_component(longitude_column)
        + ".png"
    )

    path = output_dir / filename

    figure, axis = plt.subplots(
        figsize=(7, 5.5)
    )

    axis.scatter(
        coordinates["longitude"],
        coordinates["latitude"],
        alpha=0.6,
        s=18,
    )

    axis.set_title(
        f"Geospatial distribution: {latitude_column} / {longitude_column}"
    )
    axis.set_xlabel(
        f"Longitude ({longitude_column})"
    )
    axis.set_ylabel(
        f"Latitude ({latitude_column})"
    )
    axis.grid(
        alpha=0.2
    )

    _save_figure(
        figure,
        path,
    )

    return filename


# ============================================================
# MULTI-VALUE CHART
# ============================================================

def _generate_multi_value_chart(
    df: pd.DataFrame,
    logical_name: str,
    source_columns: list[str],
    output_dir: Path,
) -> str | None:
    """
    Plot the most common values of one logical multi-valued field.

    Each value is counted at most once per row.
    """
    existing_columns = [
        column
        for column in source_columns
        if column in df.columns
    ]

    if not existing_columns:
        return None

    frequency: Counter[str] = Counter()

    for _, row in (
        df[existing_columns]
        .iterrows()
    ):
        row_values: list[str] = []

        for column in existing_columns:
            value = _clean_text_value(
                row[column]
            )

            if value is not None:
                row_values.append(
                    value
                )

        frequency.update(
            dict.fromkeys(
                row_values,
                1,
            )
        )

    most_common = frequency.most_common(
        MAX_MULTI_VALUE_LEVELS
    )

    if len(most_common) < 2:
        return None

    labels = [
        value
        for value, _
        in most_common
    ]

    counts = [
        count
        for _, count
        in most_common
    ]

    filename = (
        "multivalue_"
        + _safe_filename_component(logical_name)
        + ".png"
    )

    path = output_dir / filename

    figure, axis = plt.subplots(
        figsize=(9, 5)
    )

    positions = np.arange(
        len(labels)
    )

    axis.bar(
        positions,
        counts,
    )

    axis.set_xticks(
        positions
    )
    axis.set_xticklabels(
        labels,
        rotation=45,
        ha="right",
    )

    axis.set_title(
        f"Top values: {logical_name}"
    )
    axis.set_xlabel(
        str(logical_name)
    )
    axis.set_ylabel(
        "Rows"
    )

    _save_figure(
        figure,
        path,
    )

    return filename


# ============================================================
# CORRELATION ELIGIBILITY
# ============================================================

def _target_is_correlation_eligible(
    df: pd.DataFrame,
    target_column: str,
    target_info: dict[str, Any],
    target_analysis: dict[str, Any],
) -> bool:
    """
    Decide whether a target can be represented in a generic
    Pearson correlation heatmap.

    Regression targets and numeric/bool binary targets are allowed.
    Metadata, identifiers, structural fields and geospatial targets
    are excluded.
    """
    if (
        not target_column
        or target_column not in df.columns
    ):
        return False

    family = str(
        target_info.get(
            "analysis_family",
            "",
        )
    )

    role = str(
        target_info.get(
            "semantic_role",
            "",
        )
    )

    if family in {
        "metadata",
        "identifier",
        "geospatial",
        "structural",
        "none",
    }:
        return False

    if role in {
        "latitude",
        "longitude",
        "identifier",
        "url",
        "email",
        "phone",
        "address",
        "structural_overflow_candidate",
    }:
        return False

    task_type = str(
        target_analysis.get(
            "task_type",
            "",
        )
    )

    series = df[target_column]

    numeric_like = (
        pd.api.types.is_numeric_dtype(series)
        or pd.api.types.is_bool_dtype(series)
    )

    if not numeric_like:
        converted = pd.to_numeric(
            series,
            errors="coerce",
        )

        source_non_null = int(
            series.notna().sum()
        )

        converted_non_null = int(
            converted.notna().sum()
        )

        parse_ratio = (
            converted_non_null
            / source_non_null
            if source_non_null
            else 0.0
        )

        numeric_like = (
            parse_ratio >= 0.95
        )

    if not numeric_like:
        return False

    if task_type == "regression":
        return True

    if task_type == "classification":
        unique_values = target_analysis.get(
            "unique_values"
        )

        try:
            return int(unique_values) == 2
        except (
            TypeError,
            ValueError,
        ):
            return False

    return False


# ============================================================
# CORRELATION HEATMAP
# ============================================================

def _generate_correlation_heatmap(
    df: pd.DataFrame,
    numeric_columns: list[str],
    output_dir: Path,
    target_column: str = "",
    target_info: dict[str, Any] | None = None,
    target_analysis: dict[str, Any] | None = None,
) -> str | None:
    """
    Generate a Pearson correlation heatmap from ordinary numeric
    features plus an eligible target.
    """
    columns = list(
        dict.fromkeys(
            numeric_columns
        )
    )

    target_info = (
        target_info
        if isinstance(target_info, dict)
        else {}
    )

    target_analysis = (
        target_analysis
        if isinstance(target_analysis, dict)
        else {}
    )

    if (
        target_column
        and target_column not in columns
        and _target_is_correlation_eligible(
            df=df,
            target_column=target_column,
            target_info=target_info,
            target_analysis=target_analysis,
        )
    ):
        columns.append(
            target_column
        )

    usable: dict[str, pd.Series] = {}

    for column in columns:
        if column not in df.columns:
            continue

        if pd.api.types.is_bool_dtype(
            df[column]
        ):
            numeric = (
                df[column]
                .astype("Int64")
                .astype(float)
            )
        else:
            numeric = pd.to_numeric(
                df[column],
                errors="coerce",
            )

        numeric = numeric.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        if (
            numeric.notna().sum() >= 2
            and numeric.nunique(
                dropna=True
            ) >= 2
        ):
            usable[column] = numeric

    if len(usable) < MIN_CORRELATION_COLUMNS:
        return None

    numeric_df = pd.DataFrame(
        usable
    )

    correlation = numeric_df.corr(
        method="pearson"
    )

    if correlation.empty:
        return None

    filename = "correlation_heatmap.png"
    path = output_dir / filename

    size = max(
        6,
        min(
            12,
            len(correlation.columns) * 0.8,
        ),
    )

    figure, axis = plt.subplots(
        figsize=(size, size)
    )

    image = axis.imshow(
        correlation.to_numpy(),
        vmin=-1,
        vmax=1,
        aspect="auto",
    )

    labels = [
        str(column)
        for column
        in correlation.columns
    ]

    positions = np.arange(
        len(labels)
    )

    axis.set_xticks(
        positions
    )
    axis.set_yticks(
        positions
    )

    axis.set_xticklabels(
        labels,
        rotation=45,
        ha="right",
    )
    axis.set_yticklabels(
        labels
    )

    axis.set_title(
        "Pearson Correlation Heatmap"
    )

    for row_index in range(
        len(labels)
    ):
        for column_index in range(
            len(labels)
        ):
            value = correlation.iloc[
                row_index,
                column_index,
            ]

            if pd.notna(value):
                axis.text(
                    column_index,
                    row_index,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )

    figure.colorbar(
        image,
        ax=axis,
        label="Correlation",
    )

    _save_figure(
        figure,
        path,
    )

    return filename


# ============================================================
# TARGET DISTRIBUTION
# ============================================================

def _generate_target_chart(
    df: pd.DataFrame,
    target_column: str,
    target_analysis: dict[str, Any],
    output_dir: Path,
) -> str | None:
    """Generate a target chart from the inferred task type."""
    if (
        not target_column
        or target_column not in df.columns
    ):
        return None

    task_type = (
        target_analysis.get(
            "task_type"
        )
    )

    filename = "target_distribution.png"
    path = output_dir / filename

    if task_type == "classification":
        counts = (
            df[target_column]
            .dropna()
            .astype(str)
            .value_counts()
        )

        if len(counts) == 0:
            return None

        # Avoid unreadable target plots for extreme-cardinality labels.
        counts = counts.head(
            MAX_CATEGORY_LEVELS
        )

        figure, axis = plt.subplots(
            figsize=(7, 4.5)
        )

        positions = np.arange(
            len(counts)
        )

        bars = axis.bar(
            positions,
            counts.values,
        )

        axis.set_xticks(
            positions
        )
        axis.set_xticklabels(
            counts.index,
            rotation=45
            if len(counts) > 8
            else 0,
            ha="right"
            if len(counts) > 8
            else "center",
        )

        axis.set_title(
            f"Target Distribution: {target_column}"
        )
        axis.set_xlabel(
            str(target_column)
        )
        axis.set_ylabel(
            "Count"
        )

        for bar, count in zip(
            bars,
            counts.values,
        ):
            axis.text(
                bar.get_x()
                + bar.get_width() / 2,
                bar.get_height(),
                str(int(count)),
                ha="center",
                va="bottom",
            )

        _save_figure(
            figure,
            path,
        )

        return filename

    if task_type == "regression":
        clean = _numeric_series(
            df[target_column]
        )

        if (
            len(clean) < 2
            or clean.nunique() < 2
        ):
            return None

        figure, axis = plt.subplots(
            figsize=(7, 4.5)
        )

        bins = min(
            HISTOGRAM_BINS,
            max(
                5,
                int(np.sqrt(len(clean))),
            ),
        )

        axis.hist(
            clean,
            bins=bins,
            edgecolor="black",
            alpha=0.8,
        )

        axis.set_title(
            f"Target Distribution: {target_column}"
        )
        axis.set_xlabel(
            str(target_column)
        )
        axis.set_ylabel(
            "Count"
        )

        _save_figure(
            figure,
            path,
        )

        return filename

    return None


# ============================================================
# MAIN GENERATOR
# ============================================================

def generate_charts(
    df: pd.DataFrame,
    output_dir: str | Path,
    target_column: str = "",
) -> list[str]:
    """
    Generate deterministic semantic-aware charts.

    Chart selection is driven by the routing plan rather than
    raw pandas dtypes.
    """
    output_path = Path(
        output_dir
    )

    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    generated: list[str] = []

    # --------------------------------------------------------
    # Schema + semantic router
    # --------------------------------------------------------

    routing_result = route_dataset(
        df=df,
        target_column=target_column,
    )

    schema = routing_result[
        "schema"
    ]

    analysis_plan = routing_result[
        "analysis_plan"
    ]

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

    geospatial_pairs = list(
        analysis_plan.get(
            "geospatial_pairs",
            [],
        )
    )

    multi_value_groups = list(
        analysis_plan.get(
            "multi_value_groups",
            [],
        )
    )

    target_info = (
        analysis_plan.get(
            "target",
            {},
        )
    )

    # --------------------------------------------------------
    # Target analysis
    # --------------------------------------------------------

    if (
        target_column
        and target_column in df.columns
    ):
        target_analysis = analyze_target(
            df=df,
            target_column=target_column,
            schema=schema,
        )
    else:
        target_analysis = {}

    # --------------------------------------------------------
    # Standard numeric charts
    # --------------------------------------------------------

    selected_numeric = _select_numeric_columns(
        df=df,
        numeric_columns=numeric_columns,
        target_column=target_column,
    )

    for column in selected_numeric:
        distribution = _generate_distribution_chart(
            df=df,
            column=column,
            output_dir=output_path,
        )

        if distribution:
            generated.append(
                distribution
            )

        boxplot = _generate_boxplot_chart(
            df=df,
            column=column,
            output_dir=output_path,
        )

        if boxplot:
            generated.append(
                boxplot
            )

    # --------------------------------------------------------
    # Standard categorical charts
    # --------------------------------------------------------

    selected_categorical = _select_categorical_columns(
        df=df,
        categorical_columns=categorical_columns,
        target_column=target_column,
    )

    for column in selected_categorical:
        chart = _generate_categorical_chart(
            df=df,
            column=column,
            output_dir=output_path,
        )

        if chart:
            generated.append(
                chart
            )

    # --------------------------------------------------------
    # Datetime charts
    # --------------------------------------------------------

    datetime_count = 0

    for column in datetime_columns:
        if datetime_count >= MAX_DATETIME_CHARTS:
            break

        if column == target_column:
            continue

        chart = _generate_datetime_chart(
            df=df,
            column=column,
            output_dir=output_path,
        )

        if chart:
            generated.append(
                chart
            )
            datetime_count += 1

    # --------------------------------------------------------
    # Geospatial charts
    # --------------------------------------------------------

    geospatial_count = 0

    for pair in geospatial_pairs:
        if geospatial_count >= MAX_GEOSPATIAL_CHARTS:
            break

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

        if (
            not latitude
            or not longitude
        ):
            continue

        if target_column in {
            latitude,
            longitude,
        }:
            continue

        chart = _generate_geospatial_chart(
            df=df,
            latitude_column=str(latitude),
            longitude_column=str(longitude),
            output_dir=output_path,
        )

        if chart:
            generated.append(
                chart
            )
            geospatial_count += 1

    # --------------------------------------------------------
    # Multi-value charts
    # --------------------------------------------------------

    multi_value_count = 0

    for group in multi_value_groups:
        if multi_value_count >= MAX_MULTI_VALUE_CHARTS:
            break

        if not isinstance(
            group,
            dict,
        ):
            continue

        logical_name = str(
            group.get(
                "logical_name",
                "",
            )
        )

        source_columns = [
            str(column)
            for column
            in group.get(
                "source_columns",
                [],
            )
        ]

        if (
            not logical_name
            or not source_columns
        ):
            continue

        if (
            target_column
            and target_column
            in source_columns
        ):
            continue

        chart = _generate_multi_value_chart(
            df=df,
            logical_name=logical_name,
            source_columns=source_columns,
            output_dir=output_path,
        )

        if chart:
            generated.append(
                chart
            )
            multi_value_count += 1

    # --------------------------------------------------------
    # Correlation heatmap
    # --------------------------------------------------------

    correlation_chart = _generate_correlation_heatmap(
        df=df,
        numeric_columns=numeric_columns,
        output_dir=output_path,
        target_column=target_column,
        target_info=target_info,
        target_analysis=target_analysis,
    )

    if correlation_chart:
        generated.append(
            correlation_chart
        )

    # --------------------------------------------------------
    # Target distribution
    # --------------------------------------------------------

    if target_column:
        target_chart = _generate_target_chart(
            df=df,
            target_column=target_column,
            target_analysis=target_analysis,
            output_dir=output_path,
        )

        if target_chart:
            generated.append(
                target_chart
            )

    return list(
        dict.fromkeys(
            generated
        )
    )
