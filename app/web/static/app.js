// ============================================================
// DOM ELEMENTS
// ============================================================

const fileInput =
    document.getElementById("dataset-file");

const targetSelect =
    document.getElementById("target-column");

const analyzeButton =
    document.getElementById("analyze-button");

const statusSection =
    document.getElementById("status-section");

const statusMessage =
    document.getElementById("status-message");

const dashboardSection =
    document.getElementById("dashboard-section");

const datasetName =
    document.getElementById("dataset-name");

const reviewStatus =
    document.getElementById("review-status");

const qualityScore =
    document.getElementById("quality-score");

const summaryCards =
    document.getElementById("summary-cards");

const dataQuality =
    document.getElementById("data-quality");

const numericalTableBody =
    document.getElementById(
        "numerical-table-body"
    );

const correlationList =
    document.getElementById(
        "correlation-list"
    );

const correlationChart =
    document.getElementById(
        "correlation-chart"
    );

const targetSection =
    document.getElementById(
        "target-section"
    );

const targetSummary =
    document.getElementById(
        "target-summary"
    );

const targetChart =
    document.getElementById(
        "target-chart"
    );

const primaryCharts =
    document.getElementById(
        "primary-charts"
    );

const chartVariable =
    document.getElementById(
        "chart-variable"
    );

const chartType =
    document.getElementById(
        "chart-type"
    );

const showChartButton =
    document.getElementById(
        "show-chart-button"
    );

const clearChartButton =
    document.getElementById(
        "clear-chart-button"
    );

const customChartArea =
    document.getElementById(
        "custom-chart-area"
    );

const reportContent =
    document.getElementById(
        "report-content"
    );


// ============================================================
// CONFIGURATION
// ============================================================

const SUPPORTED_EXTENSIONS = [
    ".csv",
    ".tsv",
    ".txt",
    ".xlsx",
    ".xls",
    ".parquet",
    ".json",
    ".jsonl",
    ".ndjson",
];

const MAX_UPLOAD_BYTES =
    50 * 1024 * 1024;


const ANALYSIS_MESSAGES = [
    "Reviewing the dataset...",
    "Preparing insights and visualizations...",
    "Finalizing your analysis...",
];


if (fileInput) {

    fileInput.accept =
        SUPPORTED_EXTENSIONS.join(",");
}


// ============================================================
// STATE
// ============================================================

let currentRunId = null;

let availableCharts = [];

let uploadedFileId = null;

let uploadedFileSignature = null;

let uploadedFileMetadata = null;

let uploadInProgress = false;

let analysisStatusTimer = null;


// ============================================================
// UTILITIES
// ============================================================

function escapeHtml(value) {

    return String(
        value ?? ""
    )
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}


function formatNumber(
    value,
    digits = 2
) {

    const number =
        Number(value);


    if (
        !Number.isFinite(number)
    ) {

        return "-";
    }


    if (
        Number.isInteger(number)
    ) {

        return number.toLocaleString();
    }


    return number.toFixed(
        digits
    );
}


function formatFileSize(
    bytes
) {

    const size =
        Number(bytes);


    if (
        !Number.isFinite(size)
        || size < 0
    ) {

        return "-";
    }


    if (
        size < 1024
    ) {

        return `${size} B`;
    }


    if (
        size < 1024 * 1024
    ) {

        return (
            `${(
                size / 1024
            ).toFixed(1)} KB`
        );
    }


    return (
        `${(
            size
            / (1024 * 1024)
        ).toFixed(1)} MB`
    );
}


function getFileExtension(
    filename
) {

    const name =
        String(
            filename || ""
        );


    const index =
        name.lastIndexOf(".");


    if (
        index === -1
    ) {

        return "";
    }


    return name
        .slice(index)
        .toLowerCase();
}


function createFileSignature(
    file
) {

    return [
        file.name,
        file.size,
        file.lastModified,
    ].join("|");
}


function isSupportedFile(
    file
) {

    const extension =
        getFileExtension(
            file.name
        );


    return (
        SUPPORTED_EXTENSIONS
            .includes(
                extension
            )
    );
}


async function readApiResponse(
    response
) {

    const text =
        await response.text();


    if (!text) {

        return {};
    }


    try {

        return JSON.parse(
            text
        );

    } catch (error) {

        throw new Error(
            (
                "Invalid server response "
                + `(${response.status}).`
            )
        );
    }
}


function getApiErrorMessage(
    payload,
    fallback
) {

    if (!payload) {

        return fallback;
    }


    const detail =
        payload.detail;


    if (
        typeof detail === "string"
    ) {

        return detail;
    }


    if (
        detail
        && typeof detail === "object"
    ) {

        if (
            typeof detail.message
            === "string"
        ) {

            return detail.message;
        }


        try {

            return JSON.stringify(
                detail
            );

        } catch (error) {

            return fallback;
        }
    }


    if (
        typeof payload.message
        === "string"
    ) {

        return payload.message;
    }


    return fallback;
}


function showStatus(
    message
) {

    statusMessage.textContent =
        message;


    statusSection.classList.remove(
        "hidden"
    );
}


function chartUrl(
    filename
) {

    if (!currentRunId) {

        return "";
    }


    return (
        `/api/runs/${currentRunId}`
        +
        `/charts/${encodeURIComponent(
            filename
        )}`
    );
}


function readableChartName(
    filename
) {

    return String(filename)
        .replace(".png", "")
        .replace(
            "distribution_",
            "Distribution of "
        )
        .replace(
            "boxplot_",
            "Boxplot of "
        )
        .replace(
            "categorical_",
            "Category Distribution: "
        )
        .replace(
            "datetime_",
            "Observations Over Time: "
        )
        .replace(
            "geospatial_",
            "Geospatial Distribution: "
        )
        .replace(
            "multivalue_",
            "Top Values: "
        )
        .replace(
            "correlation_heatmap",
            "Correlation Heatmap"
        )
        .replace(
            "target_distribution",
            "Target Distribution"
        )
        .replaceAll(
            "_",
            " "
        );
}


// ============================================================
// ANALYSIS STATUS
// ============================================================

function stopAnalysisStatusCycle() {

    if (
        analysisStatusTimer
        !== null
    ) {

        clearInterval(
            analysisStatusTimer
        );


        analysisStatusTimer =
            null;
    }
}


function startAnalysisStatusCycle() {

    stopAnalysisStatusCycle();


    let messageIndex = 0;


    showStatus(
        ANALYSIS_MESSAGES[
            messageIndex
        ]
    );


    analysisStatusTimer =
        setInterval(
            () => {

                messageIndex =
                    Math.min(
                        messageIndex + 1,
                        ANALYSIS_MESSAGES.length - 1
                    );


                showStatus(
                    ANALYSIS_MESSAGES[
                        messageIndex
                    ]
                );


                if (
                    messageIndex
                    ===
                    ANALYSIS_MESSAGES.length - 1
                ) {

                    stopAnalysisStatusCycle();
                }

            },
            5000
        );
}


// ============================================================
// RESET
// ============================================================

function resetDashboard() {

    dashboardSection.classList.add(
        "hidden"
    );


    summaryCards.innerHTML = "";

    dataQuality.innerHTML = "";

    numericalTableBody.innerHTML = "";

    correlationList.innerHTML = "";

    correlationChart.innerHTML = "";

    targetSummary.innerHTML = "";

    targetChart.innerHTML = "";

    primaryCharts.innerHTML = "";

    customChartArea.innerHTML = "";

    reportContent.innerHTML = "";
}


function resetUploadState() {

    uploadedFileId = null;

    uploadedFileSignature = null;

    uploadedFileMetadata = null;
}


function resetTargetSelect() {

    targetSelect.innerHTML = "";


    const option =
        document.createElement(
            "option"
        );


    option.value = "";

    option.textContent =
        "No target selected";


    targetSelect.appendChild(
        option
    );
}


// ============================================================
// TARGET SELECT
// ============================================================

function populateTargetSelect(
    columns
) {

    resetTargetSelect();


    if (
        !Array.isArray(columns)
    ) {

        return;
    }


    columns.forEach(
        column => {

            const name =
                String(column);


            const option =
                document.createElement(
                    "option"
                );


            option.value =
                name;

            option.textContent =
                name;


            targetSelect.appendChild(
                option
            );
        }
    );
}


// ============================================================
// UPLOAD
// ============================================================

async function uploadDataset(
    file
) {

    if (!file) {

        throw new Error(
            "No dataset was selected."
        );
    }


    if (
        !isSupportedFile(file)
    ) {

        throw new Error(
            (
                "Unsupported file format. "
                +
                "Supported formats: "
                +
                SUPPORTED_EXTENSIONS.join(
                    ", "
                )
            )
        );
    }


    if (
        file.size
        > MAX_UPLOAD_BYTES
    ) {

        throw new Error(
            (
                "Dataset exceeds the "
                + "50 MB upload limit."
            )
        );
    }


    const signature =
        createFileSignature(
            file
        );


    if (
        uploadedFileId
        &&
        uploadedFileSignature
        === signature
    ) {

        return uploadedFileMetadata;
    }


    const formData =
        new FormData();


    formData.append(
        "file",
        file
    );


    uploadInProgress =
        true;


    try {

        const uploadResponse =
            await fetch(
                "/api/upload",
                {
                    method:
                        "POST",

                    body:
                        formData,
                }
            );


        const uploadResult =
            await readApiResponse(
                uploadResponse
            );


        if (
            !uploadResponse.ok
        ) {

            throw new Error(
                getApiErrorMessage(
                    uploadResult,
                    "Dataset upload failed."
                )
            );
        }


        if (
            !uploadResult.file_id
        ) {

            throw new Error(
                (
                    "The server did not "
                    + "return a file ID."
                )
            );
        }


        uploadedFileId =
            uploadResult.file_id;


        uploadedFileSignature =
            signature;


        uploadedFileMetadata =
            uploadResult;


        return uploadResult;

    } finally {

        uploadInProgress =
            false;
    }
}


// ============================================================
// KPI CARDS
// ============================================================

function renderKpiCards(
    evidence,
    review
) {

    const profile =
        evidence.dataset_profile
        || {};

    const target =
        evidence.target_analysis
        || {};

    const quality =
        evidence.data_quality
        || {};


    let rawMissing = null;


    if (
        Number.isFinite(
            Number(
                quality.raw_missing_cells
            )
        )
    ) {

        rawMissing =
            Number(
                quality.raw_missing_cells
            );

    } else if (
        profile.missing_values
        &&
        typeof profile.missing_values
        === "object"
    ) {

        rawMissing =
            Object.values(
                profile.missing_values
            ).reduce(
                (
                    sum,
                    value
                ) =>
                    sum
                    +
                    Number(
                        value || 0
                    ),
                0
            );
    }


    const relevantMissingCandidate =
        Number(
            quality
                .analytically_relevant_missing_values
        );


    const hasRelevantMissing =
        Number.isFinite(
            relevantMissingCandidate
        );


    const missingLabel =
        hasRelevantMissing
            ? "Relevant Missing"
            : "Missing Values";


    const missingValue =
        hasRelevantMissing
            ? relevantMissingCandidate
            : (
                rawMissing
                ?? 0
            );


    const cards = [

        {
            label:
                "Rows",

            value:
                profile.rows
                ?? quality.rows
                ?? "-"
        },

        {
            label:
                "Columns",

            value:
                profile.columns
                ?? quality.columns
                ?? "-"
        },

        {
            label:
                "Target",

            value:
                target.target_column
                || "None"
        },

        {
            label:
                missingLabel,

            value:
                missingValue
        },

    ];


    summaryCards.innerHTML =
        cards.map(
            card => `

                <div class="kpi-card">

                    <span class="kpi-label">
                        ${escapeHtml(
                            card.label
                        )}
                    </span>

                    <div class="kpi-value">
                        ${escapeHtml(
                            card.value
                        )}
                    </div>

                </div>

            `
        ).join("");


    const score =
        Number(
            review?.quality_score
        );


    qualityScore.textContent =
        Number.isFinite(score)
            ? `Quality ${score}/10`
            : "";
}


// ============================================================
// DATA QUALITY
// ============================================================

function normalizeColumnNames(
    items
) {

    if (
        !Array.isArray(items)
    ) {

        return [];
    }


    return items
        .map(
            item => {

                if (
                    typeof item
                    === "string"
                ) {

                    return item;
                }


                if (
                    item
                    &&
                    typeof item
                    === "object"
                ) {

                    return (
                        item.column
                        ||
                        item.name
                        ||
                        ""
                    );
                }


                return "";
            }
        )
        .filter(Boolean);
}


function renderDataQuality(
    evidence
) {

    const quality =
        evidence.data_quality
        || {};


    const numeric =
        evidence.numerical_analysis
        || {};


    const duplicateRows =
        Number(
            quality.duplicate_rows
            ?? 0
        );


    const highMissing =
        normalizeColumnNames(
            quality.high_missing_columns
            || []
        );


    const identifiers =
        normalizeColumnNames(
            quality.identifier_columns
            ||
            evidence.analysis_plan
                ?.identifier_columns
            ||
            []
        );


    const constants =
        normalizeColumnNames(
            quality.constant_columns
            || []
        );


    const biggestOutlier =
        Object.entries(
            numeric
        )
            .filter(
                ([name]) =>
                    name
                    !==
                    evidence
                        .target_analysis
                        ?.target_column
            )
            .sort(
                (a, b) =>
                    Number(
                        b[1]
                            ?.outlier_percent
                        || 0
                    )
                    -
                    Number(
                        a[1]
                            ?.outlier_percent
                        || 0
                    )
            )[0];


    const biggestOutlierText =
        biggestOutlier
            ? (
                `${biggestOutlier[0]} — `
                +
                `${formatNumber(
                    biggestOutlier[1]
                        .outlier_percent
                )}%`
            )
            : "None";


    const relevantMissing =
        Number(
            quality
                .analytically_relevant_missing_values
            ?? 0
        );


    const rawMissing =
        Number(
            quality.raw_missing_cells
            ?? 0
        );


    const structuralMissing =
        Number(
            quality
                .structural_slot_missing_cells
            ?? 0
        );


    const items = [

        {
            title:
                "Relevant Missing Values",

            value:
                relevantMissing,

            status:
                relevantMissing === 0
                    ? "good"
                    : "warning",
        },

        {
            title:
                "Raw Missing Cells",

            value:
                rawMissing,

            status:
                "good",
        },

        {
            title:
                "Structural Empty Slots",

            value:
                structuralMissing,

            status:
                "good",
        },

        {
            title:
                "Duplicate Rows",

            value:
                duplicateRows,

            status:
                duplicateRows === 0
                    ? "good"
                    : "warning",
        },

        {
            title:
                "High Missing Columns",

            value:
                highMissing.length
                    ? highMissing.join(", ")
                    : "None",

            status:
                highMissing.length
                    ? "warning"
                    : "good",
        },

        {
            title:
                "Largest Outlier Rate",

            value:
                biggestOutlierText,

            status:
                biggestOutlier
                &&
                Number(
                    biggestOutlier[1]
                        .outlier_percent
                ) > 5
                    ? "warning"
                    : "good",
        },

        {
            title:
                "Identifier Columns",

            value:
                identifiers.length
                    ? identifiers.join(", ")
                    : "None",

            status:
                "good",
        },

        {
            title:
                "Constant Columns",

            value:
                constants.length
                    ? constants.join(", ")
                    : "None",

            status:
                constants.length
                    ? "warning"
                    : "good",
        },

    ];


    dataQuality.innerHTML =
        items.map(
            item => `

                <div class="quality-item">

                    <span class="quality-title">
                        ${escapeHtml(
                            item.title
                        )}
                    </span>

                    <div
                        class="quality-value ${
                            item.status
                            === "good"
                                ? "quality-good"
                                : "quality-warning"
                        }"
                    >
                        ${escapeHtml(
                            item.value
                        )}
                    </div>

                </div>

            `
        ).join("");
}


// ============================================================
// NUMERICAL TABLE
// ============================================================

function renderNumericalTable(
    evidence
) {

    const numeric =
        evidence.numerical_analysis
        || {};


    const targetName =
        evidence.target_analysis
            ?.target_column;


    const rows =
        Object.entries(
            numeric
        )
            .filter(
                ([column]) =>
                    column
                    !== targetName
            );


    if (
        rows.length === 0
    ) {

        numericalTableBody.innerHTML = `

            <tr>
                <td colspan="7">
                    No numerical features available.
                </td>
            </tr>
        `;

        return;
    }


    numericalTableBody.innerHTML =
        rows.map(
            ([column, stats]) => `

                <tr>

                    <td>
                        <strong>
                            ${escapeHtml(
                                column
                            )}
                        </strong>
                    </td>

                    <td>
                        ${formatNumber(
                            stats.mean
                        )}
                    </td>

                    <td>
                        ${formatNumber(
                            stats.median
                        )}
                    </td>

                    <td>
                        ${formatNumber(
                            stats.std
                        )}
                    </td>

                    <td>
                        ${formatNumber(
                            stats.min
                        )}
                    </td>

                    <td>
                        ${formatNumber(
                            stats.max
                        )}
                    </td>

                    <td>
                        ${formatNumber(
                            stats.outlier_percent
                        )}%
                    </td>

                </tr>

            `
        ).join("");
}


// ============================================================
// CORRELATIONS
// ============================================================

function renderCorrelations(
    evidence
) {

    const correlations =
        evidence
            .correlation_analysis
            ?.top_correlations
        || [];


    if (
        correlations.length === 0
    ) {

        correlationList.innerHTML = `

            <p style="color:#94a3b8;">
                No numerical correlations available.
            </p>
        `;

    } else {

        correlationList.innerHTML =
            correlations
                .slice(
                    0,
                    5
                )
                .map(
                    item => `

                        <div class="correlation-item">

                            <div class="correlation-pair">

                                ${escapeHtml(
                                    item.column_1
                                )}

                                <span>
                                    ↔
                                </span>

                                ${escapeHtml(
                                    item.column_2
                                )}

                            </div>

                            <div class="correlation-value">

                                ${formatNumber(
                                    item.correlation,
                                    3
                                )}

                            </div>

                        </div>

                    `
                ).join("");
    }


    if (
        availableCharts.includes(
            "correlation_heatmap.png"
        )
    ) {

        correlationChart.innerHTML = `

            <img
                src="${chartUrl(
                    "correlation_heatmap.png"
                )}"
                alt="Correlation Heatmap"
            >
        `;

    } else {

        correlationChart.innerHTML = "";
    }
}


// ============================================================
// TARGET
// ============================================================

function renderTarget(
    evidence
) {

    const target =
        evidence.target_analysis
        || {};


    if (
        !target.target_column
    ) {

        targetSection.classList.add(
            "hidden"
        );

        return;
    }


    targetSection.classList.remove(
        "hidden"
    );


    let targetHtml = `

        <div class="target-info">

            <div class="target-row">

                <span class="target-label">
                    Target Column
                </span>

                <span class="target-value">
                    ${escapeHtml(
                        target.target_column
                    )}
                </span>

            </div>


            <div class="target-row">

                <span class="target-label">
                    Semantic Type
                </span>

                <span class="target-value">
                    ${escapeHtml(
                        target.semantic_type
                        ||
                        target.target_semantic_type
                        || "-"
                    )}
                </span>

            </div>


            <div class="target-row">

                <span class="target-label">
                    Task Type
                </span>

                <span class="target-value">
                    ${escapeHtml(
                        target.task_type
                        || "-"
                    )}
                </span>

            </div>
    `;


    if (
        target.task_type
        === "classification"
    ) {

        const counts =
            target.class_counts
            || {};


        const percentages =
            target.class_percentages
            || {};


        Object.keys(
            counts
        ).forEach(
            className => {

                targetHtml += `

                    <div class="target-row">

                        <span class="target-label">

                            Class
                            ${escapeHtml(
                                className
                            )}

                        </span>

                        <span class="target-value">

                            ${escapeHtml(
                                counts[
                                    className
                                ]
                            )}

                            &nbsp;·&nbsp;

                            ${escapeHtml(
                                percentages[
                                    className
                                ]
                                ?? "-"
                            )}%

                        </span>

                    </div>
                `;
            }
        );
    }


    if (
        target.task_type
        === "regression"
        &&
        target.numeric_summary
    ) {

        const summary =
            target.numeric_summary;


        targetHtml += `

            <div class="target-row">

                <span class="target-label">
                    Mean
                </span>

                <span class="target-value">
                    ${formatNumber(
                        summary.mean
                    )}
                </span>

            </div>


            <div class="target-row">

                <span class="target-label">
                    Median
                </span>

                <span class="target-value">
                    ${formatNumber(
                        summary.median
                    )}
                </span>

            </div>


            <div class="target-row">

                <span class="target-label">
                    Range
                </span>

                <span class="target-value">

                    ${formatNumber(
                        summary.min
                    )}

                    –

                    ${formatNumber(
                        summary.max
                    )}

                </span>

            </div>
        `;
    }


    if (
        target.requires_confirmation
    ) {

        targetHtml += `

            <div class="target-row">

                <span class="target-label">
                    Target Check
                </span>

                <span class="target-value">
                    Confirmation recommended
                </span>

            </div>
        `;
    }


    targetHtml += `
        </div>
    `;


    targetSummary.innerHTML =
        targetHtml;


    if (
        availableCharts.includes(
            "target_distribution.png"
        )
    ) {

        targetChart.innerHTML = `

            <img
                src="${chartUrl(
                    "target_distribution.png"
                )}"
                alt="Target Distribution"
            >
        `;

    } else {

        targetChart.innerHTML =
            "";
    }
}


// ============================================================
// RELEVANT FEATURES
// ============================================================

function getRelevantFeatures(
    evidence
) {

    const target =
        evidence.target_analysis
            ?.target_column;


    if (!target) {

        return [];
    }


    const relevant = [];


    const correlations =
        evidence
            .correlation_analysis
            ?.top_correlations
        || [];


    correlations.forEach(
        item => {

            let feature = null;


            if (
                item.column_1
                === target
                &&
                item.column_2
                !== target
            ) {

                feature =
                    item.column_2;

            } else if (
                item.column_2
                === target
                &&
                item.column_1
                !== target
            ) {

                feature =
                    item.column_1;
            }


            if (
                feature
                &&
                !relevant.includes(
                    feature
                )
            ) {

                relevant.push(
                    feature
                );
            }
        }
    );


    const numericCategorical =
        evidence
            .relationship_analysis
            ?.numeric_categorical
        || [];


    numericCategorical.forEach(
        item => {

            if (
                item.categorical_column
                === target
                &&
                item.numeric_column
                &&
                !relevant.includes(
                    item.numeric_column
                )
            ) {

                relevant.push(
                    item.numeric_column
                );
            }
        }
    );


    return relevant.slice(
        0,
        2
    );
}


// ============================================================
// PRIMARY CHARTS
// ============================================================

function renderPrimaryCharts(
    evidence
) {

    primaryCharts.innerHTML = "";


    const rendered =
        new Set();


    function addChart(
        filename,
        title = null
    ) {

        if (
            !filename
            ||
            rendered.has(
                filename
            )
            ||
            !availableCharts.includes(
                filename
            )
        ) {

            return;
        }


        rendered.add(
            filename
        );


        primaryCharts.innerHTML += `

            <div class="chart-card">

                <h4>
                    ${escapeHtml(
                        title
                        || readableChartName(
                            filename
                        )
                    )}
                </h4>

                <img
                    src="${chartUrl(
                        filename
                    )}"
                    alt="${escapeHtml(
                        title
                        || readableChartName(
                            filename
                        )
                    )}"
                    loading="lazy"
                >

            </div>
        `;
    }


    // --------------------------------------------------------
    // Target-relevant numeric features
    // --------------------------------------------------------

    const features =
        getRelevantFeatures(
            evidence
        );


    features.forEach(
        feature => {

            addChart(
                `distribution_${feature}.png`,
                `Distribution of ${feature}`
            );
        }
    );


    // --------------------------------------------------------
    // Semantic charts
    // --------------------------------------------------------

    const semanticPrefixes = [
        "geospatial_",
        "multivalue_",
        "datetime_",
        "categorical_",
    ];


    availableCharts.forEach(
        filename => {

            if (
                semanticPrefixes.some(
                    prefix =>
                        String(filename)
                            .startsWith(
                                prefix
                            )
                )
            ) {

                addChart(
                    filename
                );
            }
        }
    );


    // --------------------------------------------------------
    // Generic fallback for datasets without a target
    // --------------------------------------------------------

    if (
        rendered.size === 0
    ) {

        const fallbackCharts =
            availableCharts
                .filter(
                    filename =>
                        (
                            String(filename)
                                .startsWith(
                                    "distribution_"
                                )
                            ||
                            String(filename)
                                .startsWith(
                                    "categorical_"
                                )
                            ||
                            String(filename)
                                .startsWith(
                                    "datetime_"
                                )
                            ||
                            String(filename)
                                .startsWith(
                                    "geospatial_"
                                )
                            ||
                            String(filename)
                                .startsWith(
                                    "multivalue_"
                                )
                        )
                        &&
                        filename
                        !== "target_distribution.png"
                )
                .slice(
                    0,
                    4
                );


        fallbackCharts.forEach(
            filename => {

                addChart(
                    filename
                );
            }
        );
    }


    if (
        rendered.size === 0
    ) {

        primaryCharts.innerHTML = `

            <p style="color:#94a3b8;">
                No additional charts are available for this dataset.
            </p>
        `;
    }
}


// ============================================================
// CHART EXPLORER
// ============================================================

function populateChartSelector(
    evidence
) {

    const variables =
        Object.keys(
            evidence.numerical_analysis
            || {}
        )
            .filter(
                variable =>
                    variable
                    !==
                    evidence
                        .target_analysis
                        ?.target_column
            );


    chartVariable.innerHTML = "";


    if (
        variables.length === 0
    ) {

        const option =
            document.createElement(
                "option"
            );


        option.value = "";

        option.textContent =
            "No numeric variables available";


        chartVariable.appendChild(
            option
        );


        return;
    }


    variables.forEach(
        variable => {

            const option =
                document.createElement(
                    "option"
                );


            option.value =
                variable;

            option.textContent =
                variable;


            chartVariable.appendChild(
                option
            );
        }
    );
}


showChartButton.addEventListener(
    "click",
    () => {

        const variable =
            chartVariable.value;


        const type =
            chartType.value;


        if (!variable) {

            return;
        }


        const filename =
            `${type}_${variable}.png`;


        if (
            !availableCharts.includes(
                filename
            )
        ) {

            customChartArea.innerHTML = `

                <p style="color:#fbbf24;">
                    This chart is not available.
                </p>
            `;

            return;
        }


        customChartArea.innerHTML = `

            <div class="chart-card">

                <h4>
                    ${escapeHtml(
                        readableChartName(
                            filename
                        )
                    )}
                </h4>

                <img
                    src="${chartUrl(
                        filename
                    )}"
                    alt="${escapeHtml(
                        variable
                    )}"
                >

            </div>
        `;
    }
);


clearChartButton.addEventListener(
    "click",
    () => {

        customChartArea.innerHTML =
            "";
    }
);


// ============================================================
// FILE SELECTION
// ============================================================

fileInput.addEventListener(
    "change",
    async () => {

        const file =
            fileInput.files[0];


        stopAnalysisStatusCycle();

        resetUploadState();

        resetTargetSelect();

        resetDashboard();


        if (!file) {

            showStatus(
                "Select a dataset to begin."
            );

            return;
        }


        if (
            !isSupportedFile(file)
        ) {

            showStatus(
                (
                    "Unsupported file format. "
                    +
                    "Please choose a supported dataset."
                )
            );


            fileInput.value = "";

            return;
        }


        if (
            file.size
            > MAX_UPLOAD_BYTES
        ) {

            showStatus(
                (
                    "This dataset exceeds "
                    + "the 50 MB upload limit."
                )
            );


            fileInput.value = "";

            return;
        }


        analyzeButton.disabled =
            true;


        analyzeButton.textContent =
            "Preparing Dataset...";


        showStatus(
            "Preparing your dataset..."
        );


        try {

            const uploadResult =
                await uploadDataset(
                    file
                );


            populateTargetSelect(
                uploadResult.columns
                || []
            );


            showStatus(
                (
                    "Dataset ready — "
                    +
                    `${uploadResult.rows} rows · `
                    +
                    `${uploadResult.column_count} columns`
                )
            );


        } catch (error) {

            console.error(
                error
            );


            resetUploadState();

            resetTargetSelect();


            showStatus(
                `Upload error: ${error.message}`
            );


        } finally {

            analyzeButton.disabled =
                false;


            analyzeButton.textContent =
                "Start Analysis";
        }
    }
);


// ============================================================
// RUN ANALYSIS
// ============================================================

analyzeButton.addEventListener(
    "click",
    async () => {

        if (
            !fileInput.files.length
        ) {

            showStatus(
                "Please select a dataset first."
            );

            return;
        }


        if (
            uploadInProgress
        ) {

            showStatus(
                "Preparing your dataset..."
            );

            return;
        }


        const file =
            fileInput.files[0];


        resetDashboard();


        analyzeButton.disabled =
            true;


        analyzeButton.textContent =
            "Analyzing Dataset...";


        try {

            const signature =
                createFileSignature(
                    file
                );


            if (
                !uploadedFileId
                ||
                uploadedFileSignature
                !== signature
            ) {

                showStatus(
                    "Preparing your dataset..."
                );


                const uploadResult =
                    await uploadDataset(
                        file
                    );


                populateTargetSelect(
                    uploadResult.columns
                    || []
                );
            }


            if (
                !uploadedFileId
            ) {

                throw new Error(
                    (
                        "Dataset upload "
                        + "did not complete."
                    )
                );
            }


            const targetColumn =
                targetSelect.value;


            // ------------------------------------------------
            // Calm progress messaging
            // ------------------------------------------------

            startAnalysisStatusCycle();


            const analysisUrl =
                (
                    `/api/analyze/${uploadedFileId}`
                    +
                    "?target_column="
                    +
                    encodeURIComponent(
                        targetColumn
                    )
                );


            const analysisResponse =
                await fetch(
                    analysisUrl,
                    {
                        method:
                            "POST",
                    }
                );


            const result =
                await readApiResponse(
                    analysisResponse
                );


            if (
                !analysisResponse.ok
            ) {

                throw new Error(
                    getApiErrorMessage(
                        result,
                        "Analysis failed."
                    )
                );
            }


            stopAnalysisStatusCycle();


            // ------------------------------------------------
            // STATE
            // ------------------------------------------------

            currentRunId =
                result.run_id
                || null;


            availableCharts =
                Array.isArray(
                    result.charts
                )
                    ? result.charts
                    : [];


            const evidence =
                result.evidence
                || {};


            // ------------------------------------------------
            // HEADER
            // ------------------------------------------------

            datasetName.textContent =
                (
                    uploadedFileMetadata
                        ?.filename
                    ||
                    file.name
                );


            if (
                result.approved
            ) {

                reviewStatus.textContent =
                    "✓ Approved";


                reviewStatus.className =
                    "status-badge approved";

            } else {

                reviewStatus.textContent =
                    "Needs Revision";


                reviewStatus.className =
                    "status-badge needs-revision";
            }


            // ------------------------------------------------
            // RENDER
            // ------------------------------------------------

            renderKpiCards(
                evidence,
                result.review
            );


            renderDataQuality(
                evidence
            );


            renderNumericalTable(
                evidence
            );


            renderCorrelations(
                evidence
            );


            renderTarget(
                evidence
            );


            renderPrimaryCharts(
                evidence
            );


            populateChartSelector(
                evidence
            );


            reportContent.innerHTML =
                result.report_html
                || "";


            // ------------------------------------------------
            // SHOW DASHBOARD
            // ------------------------------------------------

            dashboardSection.classList.remove(
                "hidden"
            );


            showStatus(
                "Analysis completed successfully."
            );


            dashboardSection.scrollIntoView(
                {
                    behavior:
                        "smooth",
                }
            );


        } catch (error) {

            stopAnalysisStatusCycle();


            console.error(
                error
            );


            showStatus(
                `Analysis error: ${error.message}`
            );


        } finally {

            stopAnalysisStatusCycle();


            analyzeButton.disabled =
                false;


            analyzeButton.textContent =
                "Start Analysis";
        }
    }
);