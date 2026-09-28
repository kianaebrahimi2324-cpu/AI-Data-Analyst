from __future__ import annotations

import json
import os
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

DEFAULT_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-120b",
)

MAX_EVIDENCE_CHARS = 18000
MAX_REPORT_TOKENS = 1200


# ============================================================
# LLM
# ============================================================

def create_llm():
    """
    Create the Groq language model.
    """

    api_key = os.getenv(
        "GROQ_API_KEY"
    )

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY was not found."
        )

    return ChatGroq(
        api_key=api_key,
        model=DEFAULT_MODEL,
        temperature=0,
        max_tokens=MAX_REPORT_TOKENS,
        reasoning_effort="low",
        max_retries=2,
    )


# ============================================================
# EVIDENCE HELPERS
# ============================================================

def _compact_value(
    value: Any,
    list_limit: int = 10,
    dict_limit: int = 20,
    depth: int = 0,
) -> Any:
    """
    Reduce very large nested structures while
    preserving useful analytical evidence.
    """

    if depth >= 4:
        return str(value)[:500]

    if isinstance(value, dict):

        result = {}

        for index, (key, item) in enumerate(
            value.items()
        ):

            if index >= dict_limit:
                break

            result[str(key)] = _compact_value(
                item,
                list_limit=list_limit,
                dict_limit=dict_limit,
                depth=depth + 1,
            )

        return result

    if isinstance(value, list):

        return [
            _compact_value(
                item,
                list_limit=list_limit,
                dict_limit=dict_limit,
                depth=depth + 1,
            )
            for item in value[:list_limit]
        ]

    return value


def prepare_llm_evidence(
    evidence: dict,
    max_chars: int = MAX_EVIDENCE_CHARS,
) -> str:
    """
    Prepare compact structured evidence for the LLM.

    Raw datasets are never sent to the model.

    The function preserves semantic summaries first and never
    truncates serialized JSON in the middle of an object.
    """

    if not isinstance(evidence, dict):
        return "{}"

    priority_keys = [
        # Dataset context
        "dataset_source",
        "dataset_profile",
        "dataset",

        # High-value grounding controls
        "grounding_contract",
        "column_fact_index",

        # High-value semantic summaries
        "analysis_summary",
        "data_quality",
        "target_analysis",
        "target",

        # Specialized semantic analyzers
        "geospatial_analysis",
        "multi_value_analysis",
        "metadata_analysis",

        # Relationships
        "correlation_analysis",
        "correlations",
        "relationship_analysis",
        "relationships",
        "text_target_analysis",

        # Standard analyzers
        "numerical_analysis",
        "categorical_analysis",
        "datetime_analysis",
        "text_analysis",

        # Routing/schema context
        "schema_summary",
        "analysis_plan",
        "schema",

        # Presentation inventory
        "generated_charts",
    ]

    def serialize(payload: dict) -> str:
        return json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )

    def build_compact(
        list_limit: int,
        dict_limit: int,
    ) -> dict:
        compact = {}

        for key in priority_keys:
            if key not in evidence:
                continue

            value = evidence.get(key)

            if value in (
                None,
                {},
                [],
            ):
                continue

            compact[key] = _compact_value(
                value,
                list_limit=list_limit,
                dict_limit=dict_limit,
            )

        return compact

    # First pass: preserve useful detail.
    compact = build_compact(
        list_limit=10,
        dict_limit=30,
    )

    text = serialize(
        compact
    )

    if len(text) <= max_chars:
        return text

    # Second pass: stronger but still structured compression.
    compact = build_compact(
        list_limit=6,
        dict_limit=18,
    )

    text = serialize(
        compact
    )

    if len(text) <= max_chars:
        return text

    # Final pass: fit complete JSON key-by-key.
    # Important summaries appear earlier in priority_keys,
    # so they are preserved before lower-priority detail.
    fitted = {}

    for key in priority_keys:
        if key not in evidence:
            continue

        value = evidence.get(key)

        if value in (
            None,
            {},
            [],
        ):
            continue

        compact_value = _compact_value(
            value,
            list_limit=3,
            dict_limit=8,
        )

        candidate = {
            **fitted,
            key: compact_value,
        }

        candidate_text = serialize(
            candidate
        )

        if len(candidate_text) <= max_chars:
            fitted[key] = compact_value
            continue

        # Try a tiny representation before dropping this key.
        tiny_value = _compact_value(
            value,
            list_limit=2,
            dict_limit=5,
        )

        candidate = {
            **fitted,
            key: tiny_value,
        }

        candidate_text = serialize(
            candidate
        )

        if len(candidate_text) <= max_chars:
            fitted[key] = tiny_value

    return serialize(
        fitted
    )


# ============================================================
# RESPONSE EXTRACTION
# ============================================================

def _extract_response_text(
    content: Any,
) -> str:
    """
    Convert LangChain response content to plain text.
    """

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):

        parts = []

        for item in content:

            if isinstance(item, str):
                parts.append(item)

            elif isinstance(item, dict):

                text = item.get("text")

                if text:
                    parts.append(
                        str(text)
                    )

        return "\n".join(
            parts
        ).strip()

    return str(
        content or ""
    ).strip()


# ============================================================
# REPORT PROMPT
# ============================================================

def _build_report_prompt(
    evidence_text: str,
) -> str:
    """
    Build the production report prompt.
    """

    return f"""
You are an experienced senior data and machine-learning analyst.

Your task is to produce a concise, rigorous and useful analytical report.

The dashboard already displays descriptive statistics and charts.

Do NOT repeat the dashboard.

Instead, identify:
- what matters,
- why it matters,
- what requires attention,
- and what should be done next.

Use ONLY the supplied structured evidence.

Treat the evidence package as the complete factual boundary of the report.
If a fact is not present in the evidence, do not state it as true.

============================================================
EVIDENCE RULES
============================================================

FACTUAL GROUNDING CONTRACT:

- Every numerical claim must be directly supported by a value in the evidence.
- Do not create new statistics, percentages, ratios, thresholds, significance
  claims, probabilities, rankings, or counts unless they are explicitly present.
- Do not silently perform new arithmetic in prose. If a derived comparison is
  not already provided by the evidence, describe the direction qualitatively.
- When "column_fact_index" is present, use it as the source of truth for
  per-column descriptive claims.
- When "grounding_contract" is present, obey it even if another section of the
  evidence is ambiguous.
- Never combine facts from different columns as if they belonged to one column.
- If two evidence sources appear inconsistent, report the uncertainty or omit
  the disputed claim. Do not choose a convenient value.

Never invent:
- numerical values
- column names
- units
- domain context
- medical meaning
- business meaning
- target-class meanings

If the evidence says class 0 and class 1,
write "class 0" and "class 1".

Do NOT replace them with:
- positive / negative
- diseased / healthy
- fraud / normal
- churn / retained
or any other interpretation.

Do not use domain-specific descriptions such as:
- physiological
- clinical
- demographic
- risk factor
unless that meaning is explicitly present in the evidence.

A minimum value of zero only proves that zero exists.
It does NOT prove that many zeros exist.

An outlier does NOT automatically mean:
- invalid value
- measurement error
- out-of-range value

Use terms such as "detected outlier" or "extreme observation".

Correlation and statistical association do NOT establish:
- causality
- predictive importance
- model performance

Do not call a feature:
- an important predictor
- a strong predictor
- a discriminator
- a risk factor

unless predictive modeling evidence supports that statement.

============================================================
SEMANTIC DATA RULES
============================================================

Respect the semantic routing and specialized evidence.

Missingness:
- "raw_missing_cells" describes physical empty cells.
- "analytically_relevant_missing_values" describes missing logical fields.
- "structural_slot_missing_cells" are unused structural slots and must NOT
  automatically be described as missing features or data-quality defects.
- If raw and analytical missingness differ, make the distinction explicit.

Multi-valued fields:
- A multi-value group represents ONE logical field stored across multiple
  physical columns.
- Do NOT treat its overflow columns as separate sparse features.
- Do NOT recommend deleting overflow columns merely because many cells are empty.

Geospatial data:
- Latitude and longitude are semantic coordinates, not ordinary numeric features.
- Use the geospatial evidence and legal coordinate ranges.
- A geographically distant but legally valid coordinate is NOT an error or
  outlier merely because an IQR rule would flag it.
- Do NOT recommend capping, transforming or removing valid coordinates based
  only on generic numeric outlier logic.

Metadata and entities:
- URL, email, phone, address, entity-name and identifier fields are metadata-like
  fields unless the evidence explicitly says otherwise.
- High uniqueness can be expected for these fields and is not automatically
  a categorical-cardinality problem.
- Do not treat identifiers or metadata fields as ordinary predictive features
  without explicit evidence supporting that use.

Analyzer status:
- If an analyzer reports "not_applicable", do not invent findings for it.
- If an analyzer reports "error", acknowledge the limitation only if material;
  never replace the missing analysis with guessed conclusions.


Additional grounding rules:

- Never write "many zeros", "frequent zeros", "numerous zeros", or similar
  quantity language unless the evidence contains an explicit zero_count or
  zero_percent supporting that claim.
- A minimum of 0 only establishes that at least one zero-valued observation
  exists.
- Never call a value "physiologically implausible", "clinically implausible",
  "medically invalid", or use similar domain language unless the evidence
  explicitly supplies that domain interpretation.
- Never claim that missingness is random, non-random, MCAR, MAR, or MNAR unless
  a dedicated missingness-mechanism analysis is present in the evidence.
- Do not call URL, phone, email, address, or entity-name fields "identifiers"
  unless the semantic role explicitly says identifier.
- High uniqueness is expected for many metadata/entity fields and does not by
  itself establish a modeling problem.
- Do not recommend removing, hashing, excluding, imputing, winsorizing, or
  transforming a field as an automatic action. Recommend validation or empirical
  comparison first unless the evidence directly establishes the need.
- When discussing correlations, prefer the numeric coefficient itself over
  subjective labels such as "strong" or "weak" unless a strength threshold is
  explicitly defined in the evidence.
- If no target is selected, do NOT recommend a classifier, regressor, predictive
  baseline, feature-importance analysis, target encoding, target semantics work,
  or evaluation metrics for a predictive model.
- If no target is selected, do NOT mention class labels or "unknown target
  semantics" as a limitation.
- Only recommend predictive modeling when grounding_contract explicitly says
  predictive_modeling_supported = true.
- Do not infer skewness from standard deviation, mean/median differences, or
  visual intuition unless explicit skewness/distribution evidence is supplied.
- For zero-valued observations, use zero_count and zero_percent when available.
  A minimum of zero is never sufficient to describe zero prevalence.
- Do not infer why duplicate rows, duplicate coordinates, or repeated values
  exist. Terms such as "same facility", "same address", "shared household",
  "duplicate entity", or similar causes require explicit evidence.
- Legal latitude/longitude ranges establish only range validity. They do not
  establish that coordinates are geographically correct, accurately geocoded,
  or automatically suitable for clustering.
- Do not describe missingness as causing bias unless evidence from a dedicated
  analysis supports that claim.
- Do not state that a field can be used "reliably" for a downstream activity
  merely because it has no missing values.
- Recommendations must be conditional on the actual analysis objective. Never
  invent a business, medical, scientific, or operational use case.

============================================================
REPORT STYLE
============================================================

The report should normally be approximately 450-700 words.

Use concise Markdown.

Do not include large statistical tables.

Do not list every variable.

Do not repeat the same relationship in multiple sections.

Prioritize only the strongest findings.

A simple dataset may produce an even shorter report.

============================================================
REQUIRED REPORT STRUCTURE
============================================================

## Executive Summary

Write 3-4 bullets only. Never exceed 4 bullets.

Each bullet must represent an important analytical conclusion.

Do not fill this section with basic metadata already obvious
from the dashboard unless it materially matters.

------------------------------------------------------------

## Key Findings

Include only 3-5 findings.

Do not create extra findings just to reach five.

For each finding use:

### Short finding title

**Evidence:** the minimum numerical evidence required
to support the finding.

**Interpretation:** one concise explanation of what
the evidence supports.

The interpretation must not go beyond the evidence.

------------------------------------------------------------

## Data Quality & Analytical Risks

Only discuss issues that may materially affect analysis.

Possible examples:
- missing values
- duplicates
- detected outliers
- suspicious values when evidence supports suspicion
- skewness
- target distribution
- identifiers
- low schema confidence
- sample-size limitations

Do not turn every statistic into a problem.

If data quality is generally good, say so briefly.

------------------------------------------------------------

## Limitations

Include 2-4 important limitations only.

Examples:
- association does not establish causality
- no predictive model has been evaluated
- statistical significance has not been tested
- target semantics are unknown
- data provenance is unavailable

Only include limitations relevant to this analysis.

------------------------------------------------------------

## Recommended Actions

Write 4-6 recommendations only.

DO NOT separate recommendations by professional role.

DO NOT write headings such as:
- Senior Data Analyst perspective
- Senior AI/ML Engineer perspective
- Data Scientist recommendation

Recommendations should reflect strong professional judgment
across statistics, data analysis and machine learning naturally.

Each recommendation must:
1. be concrete,
2. be justified by the evidence,
3. describe a useful next action.

Prefer recommendations such as:

- validate questionable observations before preprocessing
- perform an appropriate statistical comparison
- establish a simple predictive baseline
- compare preprocessing alternatives empirically
- use appropriate validation metrics
- clarify missing metadata or target semantics

Do NOT automatically recommend:
- deleting outliers
- SMOTE
- feature removal
- log transformation
- Box-Cox transformation
- neural networks
- deployment
- production monitoring
- feature engineering

unless the evidence provides a clear reason.

If a transformation may be useful, recommend TESTING it,
not automatically applying it.

If modeling is appropriate AND a valid target is present, recommend starting
with simple baselines before complex approaches.

If there is no target, keep recommendations focused on:
- data validation,
- descriptive analysis,
- semantic normalization,
- data-quality remediation,
- relationship exploration supported by available analyzers,
- or clarification of the intended analytical objective.

============================================================
FINAL SELF-CHECK
============================================================

Before answering verify:

- Did I invent any meaning?
- Did I invent any unit?
- Did I repeat dashboard statistics unnecessarily?
- Did I call association predictive importance?
- Did I interpret class labels?
- Are recommendations tied to evidence?
- Are recommendations specific rather than generic?
- Did I state any number that is absent from the evidence?
- Did I accidentally attach a statistic to the wrong column?
- Did I infer skewness, zero prevalence, missingness mechanism, duplicate cause,
  or geospatial correctness without explicit evidence?
- If no target exists, did I avoid predictive-model recommendations and
  target/class language?
- Is the report concise?

Return ONLY the Markdown report.

STRUCTURED EVIDENCE:

{evidence_text}
"""


# ============================================================
# INITIAL REPORT
# ============================================================

def run_analyst_agent(
    evidence: dict,
) -> str:
    """
    Generate the initial analytical report.
    """

    llm = create_llm()

    evidence_text = prepare_llm_evidence(
        evidence
    )

    prompt = _build_report_prompt(
        evidence_text
    )

    response = llm.invoke(
        [
            HumanMessage(
                content=prompt
            )
        ]
    )

    report = _extract_response_text(
        response.content
    )

    if not report:
        raise RuntimeError(
            "Analyst returned an empty response."
        )

    return report


# ============================================================
# REPORT REVISION
# ============================================================

def revise_analyst_report(
    report: str | None = None,
    review: dict | None = None,
    evidence: dict | None = None,
    current_report: str | None = None,
) -> str:
    """
    Revise the report using reviewer feedback.

    Supports both ``report=`` and the workflow-compatible
    ``current_report=`` argument.
    """

    llm = create_llm()

    evidence = (
        evidence
        if isinstance(evidence, dict)
        else {}
    )

    review = (
        review
        if isinstance(review, dict)
        else {}
    )

    source_report = (
        current_report
        if current_report is not None
        else report
    )

    evidence_text = prepare_llm_evidence(
        evidence
    )

    review_text = json.dumps(
        review,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )

    report_text = str(
        source_report or ""
    )[:7000]

    prompt = f"""
Revise the analytical report using the reviewer feedback.

The objective is NOT to make it longer.

The objective is to make it:
- more accurate
- more analytical
- more concise
- more evidence-grounded
- more useful

Use only the structured evidence.

Required structure:

## Executive Summary
## Key Findings
## Data Quality & Analytical Risks
## Limitations
## Recommended Actions

Rules:

- Executive Summary: maximum 4 bullets.
- Key Findings: maximum 5.
- Recommended Actions: 4-6.
- Do not include long statistical tables.
- Do not repeat dashboard information.
- Do not create a separate Target & Relationships section.
- Do not separate recommendations by professional role.
- Do not invent units or domain meaning.
- Do not assign meaning to class labels.
- Do not call statistical association predictive importance.
- Do not automatically prescribe preprocessing.
- Every numerical claim in the revised report must be supported by evidence.
- Do not create new arithmetic, percentages, or statistics during revision.
- Use column_fact_index as the per-column source of truth when present.
- Obey grounding_contract.
- If no target is selected, do not recommend predictive models and do not
  discuss class labels or target semantics.
- Do not infer skewness without explicit skewness evidence.
- Do not infer zero prevalence from a minimum value.
- Do not infer causes for duplicate coordinates or duplicate values.
- Do not infer geospatial correctness from legal coordinate ranges.
- Fix every valid reviewer issue.
- Keep useful supported findings.

CURRENT REPORT:

{report_text}

REVIEWER FEEDBACK:

{review_text}

STRUCTURED EVIDENCE:

{evidence_text}

Return ONLY the revised Markdown report.
"""

    response = llm.invoke(
        [
            HumanMessage(
                content=prompt
            )
        ]
    )

    revised = _extract_response_text(
        response.content
    )

    if not revised:
        raise RuntimeError(
            "Analyst revision returned an empty response."
        )

    return revised


# ============================================================
# COMPATIBILITY
# ============================================================

def revise_analyst_agent(
    report: str,
    review: dict,
    evidence: dict,
) -> str:
    """
    Compatibility wrapper.
    """

    return revise_analyst_report(
        report=report,
        review=review,
        evidence=evidence,
    )