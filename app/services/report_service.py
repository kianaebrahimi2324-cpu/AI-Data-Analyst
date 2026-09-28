import re


# ------------------------------------------------------------
# Chart title
# ------------------------------------------------------------

def chart_title(filename: str) -> str:
    """Create a readable title from a chart filename."""

    title = (
        filename
        .replace(".png", "")
        .replace("distribution_", "Distribution of ")
        .replace("boxplot_", "Boxplot of ")
        .replace("correlation_heatmap", "Correlation Heatmap")
        .replace("target_distribution", "Target Distribution")
        .replace("missing_values", "Missing Values")
        .replace("_", " ")
    )

    return title


# ------------------------------------------------------------
# Add content inside a report section
# ------------------------------------------------------------

def add_to_section(
    report: str,
    section_keyword: str,
    content: str,
) -> str:
    """Insert content before the next level-2 Markdown section."""

    pattern = re.compile(
        rf"(^##\s+[^\n]*{re.escape(section_keyword)}[^\n]*\n.*?)(?=^##\s+|\Z)",
        re.MULTILINE | re.DOTALL | re.IGNORECASE,
    )

    match = pattern.search(report)

    if not match:
        return report

    original_section = match.group(1).rstrip()

    updated_section = (
        original_section
        + "\n\n"
        + content.strip()
        + "\n\n"
    )

    return (
        report[:match.start()]
        + updated_section
        + report[match.end():]
    )


# ------------------------------------------------------------
# Remove old generated visual sections
# ------------------------------------------------------------

def remove_old_generated_sections(
    report: str,
) -> str:
    """Remove visual and QA sections created by previous finalizers."""

    markers = [
        "### Numerical Visuals",
        "### Correlation Visual",
        "### Target Visual",
        "### Data Quality Visuals",
        "## Quality Assurance",
    ]

    positions = []

    for marker in markers:
        position = report.find(marker)

        if position != -1:
            positions.append(position)

    if positions:
        report = report[:min(positions)]

    return report.strip()


# ------------------------------------------------------------
# Build image Markdown
# ------------------------------------------------------------

def image_block(
    filename: str,
) -> str:
    """Create Markdown image block."""

    title = chart_title(filename)

    return (
        f"### {title}\n\n"
        f"![{title}](charts/{filename})"
    )


# ------------------------------------------------------------
# Final report builder
# ------------------------------------------------------------

def build_final_report(
    report: str,
    review: dict,
    charts: list[str],
) -> str:
    """Build a concise final report with relevant charts."""

    final_report = remove_old_generated_sections(
        report
    )

    # --------------------------------------------------------
    # Group charts
    # --------------------------------------------------------

    missing_charts = [
        chart
        for chart in charts
        if "missing" in chart.lower()
    ]

    distribution_charts = [
        chart
        for chart in charts
        if chart.lower().startswith("distribution_")
    ]

    boxplot_charts = [
        chart
        for chart in charts
        if chart.lower().startswith("boxplot_")
    ]

    correlation_charts = [
        chart
        for chart in charts
        if "correlation" in chart.lower()
    ]

    target_charts = [
        chart
        for chart in charts
        if "target_distribution" in chart.lower()
    ]


    # --------------------------------------------------------
    # Data Quality visuals
    # --------------------------------------------------------

    if missing_charts:

        block = "\n\n".join(
            image_block(chart)
            for chart in missing_charts[:1]
        )

        final_report = add_to_section(
            final_report,
            "Data Quality",
            block,
        )


    # --------------------------------------------------------
    # Numerical visuals
    # Keep report concise
    # --------------------------------------------------------

    selected_distributions = (
        distribution_charts[:4]
    )

    selected_boxplots = (
        boxplot_charts[:2]
    )

    numerical_selection = (
        selected_distributions
        + selected_boxplots
    )

    if numerical_selection:

        block = (
            "### Selected Numerical Visuals\n\n"
            + "\n\n".join(
                image_block(chart)
                for chart in numerical_selection
            )
        )

        final_report = add_to_section(
            final_report,
            "Numerical",
            block,
        )


    # --------------------------------------------------------
    # Correlation visual
    # --------------------------------------------------------

    if correlation_charts:

        block = image_block(
            correlation_charts[0]
        )

        final_report = add_to_section(
            final_report,
            "Correlation",
            block,
        )


    # --------------------------------------------------------
    # Target visual
    # --------------------------------------------------------

    if target_charts:

        block = image_block(
            target_charts[0]
        )

        final_report = add_to_section(
            final_report,
            "Target",
            block,
        )


    # --------------------------------------------------------
    # Quality Assurance
    # --------------------------------------------------------

    verdict = review.get(
        "verdict",
        "Unknown"
    )

    quality_score = review.get(
        "quality_score",
        "N/A"
    )

    final_report += (
        "\n\n---\n\n"
        "## Quality Assurance\n\n"
        "| Check | Result |\n"
        "| --- | --- |\n"
        f"| Reviewer Verdict | **{verdict}** |\n"
        f"| Quality Score | **{quality_score}/10** |\n"
    )

    return final_report