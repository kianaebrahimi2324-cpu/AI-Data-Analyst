from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from app.agents.analyst import (
    create_llm,
    prepare_llm_evidence,
)


# ============================================================
# CONFIGURATION
# ============================================================

REVIEWER_MAX_TOKENS = 1000

MAX_SECTION_CHARS = 2600

MAX_ISSUES_PER_CATEGORY = 6

MAX_CLAIM_AUDIT_ITEMS = 5


# ============================================================
# REVIEW MODELS
# ============================================================

class ClaimAuditItem(BaseModel):

    section: str = ""

    claim: str = ""

    status: str = "UNCLEAR"

    reason: str = ""


class ReviewerAssessment(BaseModel):

    unsupported_claims: list[str] = Field(
        default_factory=list
    )

    numerical_mismatches: list[str] = Field(
        default_factory=list
    )

    invented_context: list[str] = Field(
        default_factory=list
    )

    causal_or_predictive_overreach: list[str] = Field(
        default_factory=list
    )

    missing_or_weak_sections: list[str] = Field(
        default_factory=list
    )

    recommendation_issues: list[str] = Field(
        default_factory=list
    )

    wording_issues: list[str] = Field(
        default_factory=list
    )

    required_fixes: list[str] = Field(
        default_factory=list
    )

    claim_audit: list[ClaimAuditItem] = Field(
        default_factory=list
    )


# ============================================================
# RESPONSE UTILITIES
# ============================================================

def _response_to_text(
    content: Any,
) -> str:
    """
    Convert model content into plain text.
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


def _remove_markdown_fences(
    text: str,
) -> str:
    """
    Remove accidental JSON Markdown fences.
    """

    value = text.strip()

    if value.startswith("```json"):
        value = value[7:]

    elif value.startswith("```"):
        value = value[3:]

    if value.endswith("```"):
        value = value[:-3]

    return value.strip()


def _extract_json_object(
    text: str,
) -> dict:
    """
    Extract a JSON object locally.

    Groq provider-side JSON validation is not used.
    """

    cleaned = _remove_markdown_fences(
        text
    )

    try:

        value = json.loads(
            cleaned
        )

        if isinstance(value, dict):
            return value

    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()

    for index, character in enumerate(
        cleaned
    ):

        if character != "{":
            continue

        try:

            value, _ = decoder.raw_decode(
                cleaned[index:]
            )

            if isinstance(value, dict):
                return value

        except json.JSONDecodeError:
            continue

    raise RuntimeError(
        "Reviewer did not return valid JSON."
    )


# ============================================================
# REPORT SECTION PARSING
# ============================================================

def _split_report_sections(
    report: str,
) -> dict[str, str]:
    """
    Split Markdown report into H2 sections.

    This allows the reviewer to explicitly inspect
    every major part of the report.
    """

    report = str(
        report or ""
    )

    pattern = re.compile(
        r"^##\s+(.+?)\s*$",
        flags=re.MULTILINE,
    )

    matches = list(
        pattern.finditer(
            report
        )
    )

    if not matches:

        return {
            "Full Report":
                report[:MAX_SECTION_CHARS]
        }

    sections = {}

    for index, match in enumerate(
        matches
    ):

        name = match.group(1).strip()

        start = match.end()

        if index + 1 < len(matches):
            end = matches[
                index + 1
            ].start()
        else:
            end = len(report)

        content = report[
            start:end
        ].strip()

        sections[name] = content[
            :MAX_SECTION_CHARS
        ]

    return sections


# ============================================================
# NORMALIZATION
# ============================================================

def _string_list(
    value: Any,
) -> list[str]:
    """
    Normalize arbitrary output into list[str].
    """

    if value is None:
        return []

    if isinstance(value, str):

        value = value.strip()

        return (
            [value]
            if value
            else []
        )

    if not isinstance(value, list):
        return [
            str(value)
        ]

    result = []

    for item in value:

        if item is None:
            continue

        text = str(
            item
        ).strip()

        if text:
            result.append(
                text
            )

    return result


def _claim_audit(
    value: Any,
) -> list[ClaimAuditItem]:
    """
    Normalize claim audit entries.
    """

    if not isinstance(value, list):
        return []

    allowed = {
        "SUPPORTED",
        "UNSUPPORTED",
        "OVERSTATED",
        "UNCLEAR",
    }

    result = []

    for item in value:

        if not isinstance(item, dict):
            continue

        claim = str(
            item.get(
                "claim",
                "",
            )
        ).strip()

        if not claim:
            continue

        status = str(
            item.get(
                "status",
                "UNCLEAR",
            )
        ).strip().upper()

        if status not in allowed:
            status = "UNCLEAR"

        result.append(
            ClaimAuditItem(
                section=str(
                    item.get(
                        "section",
                        "",
                    )
                ).strip(),

                claim=claim,

                status=status,

                reason=str(
                    item.get(
                        "reason",
                        "",
                    )
                ).strip(),
            )
        )

    return result[
        :MAX_CLAIM_AUDIT_ITEMS
    ]


def _normalize_assessment(
    payload: dict,
) -> ReviewerAssessment:
    """
    Normalize LLM JSON locally.
    """

    return ReviewerAssessment(
        unsupported_claims=_string_list(
            payload.get(
                "unsupported_claims"
            )
        ),

        numerical_mismatches=_string_list(
            payload.get(
                "numerical_mismatches"
            )
        ),

        invented_context=_string_list(
            payload.get(
                "invented_context"
            )
        ),

        causal_or_predictive_overreach=_string_list(
            payload.get(
                "causal_or_predictive_overreach"
            )
        ),

        missing_or_weak_sections=_string_list(
            payload.get(
                "missing_or_weak_sections"
            )
        ),

        recommendation_issues=_string_list(
            payload.get(
                "recommendation_issues"
            )
        ),

        wording_issues=_string_list(
            payload.get(
                "wording_issues"
            )
        ),

        required_fixes=_string_list(
            payload.get(
                "required_fixes"
            )
        ),

        claim_audit=_claim_audit(
            payload.get(
                "claim_audit"
            )
        ),
    )




# ============================================================
# DETERMINISTIC GROUNDING HELPERS
# ============================================================

def _append_check(
    result: dict[str, list[str]],
    field: str,
    message: str,
    required: bool = True,
) -> None:
    """Append one unique deterministic review issue."""
    bucket = result.setdefault(
        field,
        [],
    )

    if message not in bucket:
        bucket.append(
            message
        )

    if required:
        fixes = result.setdefault(
            "required_fixes",
            [],
        )

        if message not in fixes:
            fixes.append(
                message
            )


def _predictive_modeling_supported(
    evidence: dict | None,
) -> bool:
    """
    Read the deterministic modeling permission from the
    grounding contract when available.
    """
    if not isinstance(
        evidence,
        dict,
    ):
        return False

    contract = evidence.get(
        "grounding_contract"
    )

    if isinstance(
        contract,
        dict,
    ):
        return bool(
            contract.get(
                "predictive_modeling_supported",
                False,
            )
        )

    target = evidence.get(
        "target_analysis"
    )

    if not isinstance(
        target,
        dict,
    ):
        return False

    if not target.get(
        "target_column"
    ):
        return False

    if target.get(
        "requires_confirmation",
        False,
    ):
        return False

    return (
        str(
            target.get(
                "task_type",
                "",
            )
        )
        in {
            "classification",
            "regression",
        }
    )


def _has_target(
    evidence: dict | None,
) -> bool:
    if not isinstance(
        evidence,
        dict,
    ):
        return False

    contract = evidence.get(
        "grounding_contract"
    )

    if isinstance(
        contract,
        dict,
    ):
        return bool(
            contract.get(
                "has_target",
                False,
            )
        )

    target = evidence.get(
        "target_analysis"
    )

    return bool(
        isinstance(
            target,
            dict,
        )
        and target.get(
            "target_column"
        )
    )


def _numerical_facts(
    evidence: dict | None,
) -> dict[str, dict[str, Any]]:
    """
    Return per-column numerical facts from column_fact_index
    when available, otherwise numerical_analysis.
    """
    if not isinstance(
        evidence,
        dict,
    ):
        return {}

    result: dict[str, dict[str, Any]] = {}

    fact_index = evidence.get(
        "column_fact_index"
    )

    if isinstance(
        fact_index,
        dict,
    ):
        for column, payload in fact_index.items():
            if not isinstance(
                payload,
                dict,
            ):
                continue

            numerical = payload.get(
                "numerical"
            )

            if isinstance(
                numerical,
                dict,
            ):
                result[
                    str(column)
                ] = numerical

    if result:
        return result

    numerical = evidence.get(
        "numerical_analysis"
    )

    if not isinstance(
        numerical,
        dict,
    ):
        return {}

    return {
        str(column): facts
        for column, facts
        in numerical.items()
        if isinstance(
            facts,
            dict,
        )
    }


def _approximately_equal(
    observed: float,
    expected: float,
) -> bool:
    """
    Allow normal report rounding without accepting materially
    different numerical claims.
    """
    scale = max(
        1.0,
        abs(expected),
    )

    tolerance = max(
        0.005,
        0.005 * scale,
    )

    return abs(
        observed - expected
    ) <= tolerance


def _metric_grounding_checks(
    report: str,
    evidence: dict | None,
) -> list[str]:
    """
    Verify simple descriptive-statistic claims against deterministic
    per-column facts.

    This intentionally checks only unambiguous metric patterns.
    Relationship statistics such as correlations or eta-squared
    remain the LLM reviewer's responsibility.
    """
    facts_by_column = (
        _numerical_facts(
            evidence
        )
    )

    if not facts_by_column:
        return []

    issues: list[str] = []

    lines = [
        line.strip()
        for line in str(
            report or ""
        ).splitlines()
        if line.strip()
    ]

    metric_patterns = {
        "mean":
            r"\bmean\b\s*(?:=|:|is)?\s*(-?\d+(?:\.\d+)?)",

        "median":
            r"\bmedian\b\s*(?:=|:|is)?\s*(-?\d+(?:\.\d+)?)",

        "std":
            r"\b(?:std|standard deviation)\b\s*(?:=|:|is)?\s*(-?\d+(?:\.\d+)?)",

        "min":
            r"\b(?:min|minimum)\b\s*(?:=|:|is)?\s*(-?\d+(?:\.\d+)?)",

        "max":
            r"\b(?:max|maximum)\b\s*(?:=|:|is)?\s*(-?\d+(?:\.\d+)?)",

        "skewness":
            r"\bskewness\b\s*(?:=|:|is)?\s*(-?\d+(?:\.\d+)?)",
    }

    for line in lines:
        lower = line.lower()

        # Class-specific means/medians are relationship facts and
        # should not be compared with whole-column summaries.
        if (
            "class " in lower
            or "group " in lower
        ):
            continue

        mentioned = [
            column
            for column in facts_by_column
            if column.lower()
            in lower
        ]

        if not mentioned:
            continue

        for metric, pattern in metric_patterns.items():
            match = re.search(
                pattern,
                line,
                flags=re.IGNORECASE,
            )

            if not match:
                continue

            try:
                observed = float(
                    match.group(1)
                )
            except (
                TypeError,
                ValueError,
            ):
                continue

            for column in mentioned:
                expected = facts_by_column[
                    column
                ].get(
                    metric
                )

                if expected is None:
                    continue

                try:
                    expected_value = float(
                        expected
                    )
                except (
                    TypeError,
                    ValueError,
                ):
                    continue

                if not _approximately_equal(
                    observed,
                    expected_value,
                ):
                    issues.append(
                        (
                            f"Numerical mismatch for {column}: report states "
                            f"{metric}={observed}, but deterministic evidence "
                            f"reports {metric}={expected_value}."
                        )
                    )

        # Generic list claim such as:
        # "Minimum values = 0 for A, B, C"
        if (
            re.search(
                r"\b(?:min|minimum)\b",
                lower,
            )
            and re.search(
                r"(?:=|:)\s*0(?:\.0+)?\b",
                lower,
            )
        ):
            for column in mentioned:
                expected_min = facts_by_column[
                    column
                ].get(
                    "min"
                )

                if expected_min is None:
                    continue

                try:
                    expected_min = float(
                        expected_min
                    )
                except (
                    TypeError,
                    ValueError,
                ):
                    continue

                if not _approximately_equal(
                    0.0,
                    expected_min,
                ):
                    issues.append(
                        (
                            f"Numerical mismatch for {column}: the report "
                            "groups it among columns with minimum 0, but "
                            f"deterministic evidence reports min={expected_min}."
                        )
                    )

        # Zero-value claims can now be verified directly.
        if (
            "zero" in lower
            or re.search(
                r"\b0(?:\.0+)?\b",
                lower,
            )
        ):
            for column in mentioned:
                zero_count = facts_by_column[
                    column
                ].get(
                    "zero_count"
                )

                if zero_count is None:
                    continue

                try:
                    zero_count = int(
                        zero_count
                    )
                except (
                    TypeError,
                    ValueError,
                ):
                    continue

                if (
                    zero_count == 0
                    and re.search(
                        r"\b(?:zero[- ]valued|contains?\s+zero|minimum\s*(?:=|:)\s*0)",
                        lower,
                    )
                ):
                    issues.append(
                        (
                            f"Unsupported zero-value claim for {column}: "
                            "deterministic evidence reports zero_count=0."
                        )
                    )

        if "right-skew" in lower:
            for column in mentioned:
                skewness = facts_by_column[
                    column
                ].get(
                    "skewness"
                )

                if skewness is None:
                    issues.append(
                        (
                            f"Unsupported right-skew claim for {column}: "
                            "no skewness statistic is available."
                        )
                    )
                    continue

                if float(skewness) <= 0:
                    issues.append(
                        (
                            f"Direction mismatch for {column}: report claims "
                            f"right skew, but skewness={skewness}."
                        )
                    )

        if "left-skew" in lower:
            for column in mentioned:
                skewness = facts_by_column[
                    column
                ].get(
                    "skewness"
                )

                if skewness is None:
                    issues.append(
                        (
                            f"Unsupported left-skew claim for {column}: "
                            "no skewness statistic is available."
                        )
                    )
                    continue

                if float(skewness) >= 0:
                    issues.append(
                        (
                            f"Direction mismatch for {column}: report claims "
                            f"left skew, but skewness={skewness}."
                        )
                    )

    return list(
        dict.fromkeys(
            issues
        )
    )


# ============================================================
# DETERMINISTIC REPORT CHECKS
# ============================================================

def _run_deterministic_checks(
    report: str,
    sections: dict[str, str],
    evidence: dict | None = None,
) -> dict[str, list[str]]:
    """
    Run simple Python checks before asking the LLM.
    """

    result = {
        "missing_or_weak_sections": [],
        "recommendation_issues": [],
        "wording_issues": [],
        "required_fixes": [],
    }

    expected = [
        "Executive Summary",
        "Key Findings",
        "Data Quality & Analytical Risks",
        "Limitations",
        "Recommended Actions",
    ]

    normalized_names = {
        name.strip().lower()
        for name in sections
    }

    for section in expected:

        if (
            section.lower()
            not in normalized_names
        ):

            message = (
                f"Missing required section: "
                f"{section}."
            )

            result[
                "missing_or_weak_sections"
            ].append(
                message
            )

            result[
                "required_fixes"
            ].append(
                message
            )

    word_count = len(
        re.findall(
            r"\b\w+\b",
            report,
        )
    )

    if word_count > 850:

        result[
            "wording_issues"
        ].append(
            (
                "The report is longer than the "
                "intended concise format."
            )
        )

    role_labels = [
        "senior data analyst perspective",
        "senior ai/ml engineer perspective",
        "from the senior data analyst",
        "from the senior ai",
    ]

    lower_report = report.lower()

    for label in role_labels:

        if label in lower_report:

            result[
                "recommendation_issues"
            ].append(
                (
                    "Recommendations are separated "
                    "by professional role."
                )
            )

            result[
                "required_fixes"
            ].append(
                (
                    "Combine recommendations into "
                    "one unified Recommended Actions section."
                )
            )

            break

    recommended_section = ""

    for name, value in sections.items():

        if (
            name.strip().lower()
            ==
            "recommended actions"
        ):

            recommended_section = value
            break

    if recommended_section:

        actions = re.findall(
            r"(?m)^\s*(?:\d+[\.\)]|[-*+])\s+",
            recommended_section,
        )

        count = len(
            actions
        )

        if count < 4 or count > 6:

            message = (
                "Recommended Actions should "
                "contain 4-6 prioritized actions."
            )

            result[
                "recommendation_issues"
            ].append(
                message
            )

            result[
                "required_fixes"
            ].append(
                message
            )

    table_rows = len(
        re.findall(
            r"(?m)^\s*\|.*\|\s*$",
            report,
        )
    )

    if table_rows >= 5:

        result[
            "wording_issues"
        ].append(
            (
                "The report contains a large table "
                "that largely duplicates dashboard detail."
            )
        )


    # --------------------------------------------------------
    # Executive Summary length
    # --------------------------------------------------------

    executive_summary = ""

    for name, value in sections.items():
        if name.strip().lower() == "executive summary":
            executive_summary = value
            break

    if executive_summary:
        executive_bullets = re.findall(
            r"(?m)^\s*[-*+]\s+",
            executive_summary,
        )

        if len(executive_bullets) > 4:
            message = (
                "Executive Summary must contain no more than 4 bullets."
            )

            result[
                "wording_issues"
            ].append(
                message
            )

            result[
                "required_fixes"
            ].append(
                message
            )

    # --------------------------------------------------------
    # Evidence-grounding checks for recurring overclaims
    # --------------------------------------------------------

    evidence_text = json.dumps(
        evidence or {},
        ensure_ascii=False,
        default=str,
    ).lower()

    lower_report = report.lower()

    if re.search(
        r"\b(?:many|numerous|frequent|large number of)\s+zero(?:-valued)?\s+(?:values|observations)",
        lower_report,
    ):
        if (
            "zero_count" not in evidence_text
            and "zero_percent" not in evidence_text
        ):
            message = (
                "The report quantifies zero-valued observations without "
                "explicit zero-count evidence."
            )

            result.setdefault(
                "required_fixes",
                [],
            ).append(
                message
            )

    if re.search(
        r"(?:missingness\s+is\s+not\s+random|non[- ]random\s+missingness|not\s+missing\s+at\s+random|\bmcar\b|\bmar\b|\bmnAR\b)",
        lower_report,
        flags=re.IGNORECASE,
    ):
        if "missingness_mechanism" not in evidence_text:
            message = (
                "The report infers a missingness mechanism without a "
                "dedicated missingness-mechanism analysis."
            )

            result.setdefault(
                "required_fixes",
                [],
            ).append(
                message
            )

    domain_terms = [
        "physiologically",
        "physiological",
        "clinically",
        "clinical",
        "medically",
        "medical",
    ]

    for term in domain_terms:
        if (
            term in lower_report
            and term not in evidence_text
        ):
            message = (
                f"The report introduces unsupported domain context: '{term}'."
            )

            result.setdefault(
                "required_fixes",
                [],
            ).append(
                message
            )

            break

    recommended_lower = recommended_section.lower()

    prescriptive_patterns = [
        r"\bremove\s+or\s+hash\b",
        r"\bexclude\b.+\bfrom\s+model",
        r"\bwinsori[sz]",
        r"\bapply\s+(?:a\s+)?log\s+transform",
        r"\bimpute\s+or\s+exclude\b",
    ]

    for pattern in prescriptive_patterns:
        if re.search(
            pattern,
            recommended_lower,
        ):
            message = (
                "Recommended Actions contain a prescriptive preprocessing "
                "or feature-removal step that should first be validated "
                "or compared empirically."
            )

            result[
                "recommendation_issues"
            ].append(
                message
            )

            result[
                "required_fixes"
            ].append(
                message
            )

            break



    # --------------------------------------------------------
    # Deterministic numerical grounding
    # --------------------------------------------------------

    for issue in _metric_grounding_checks(
        report=report,
        evidence=evidence,
    ):
        _append_check(
            result,
            "numerical_mismatches",
            issue,
            required=True,
        )

    # --------------------------------------------------------
    # No-target / modeling safety
    # --------------------------------------------------------

    has_target = _has_target(
        evidence
    )

    predictive_supported = (
        _predictive_modeling_supported(
            evidence
        )
    )

    if not predictive_supported:

        predictive_patterns = [
            r"\blogistic regression\b",
            r"\bdecision tree\b",
            r"\bclassifier\b",
            r"\bregressor\b",
            r"\bpredictive baseline\b",
            r"\bpredictive model\b",
            r"\bfeature importance\b",
            r"\bprecision\b",
            r"\brecall\b",
            r"\bf1(?:[- ]score)?\b",
            r"\broc[- ]?auc\b",
            r"\bauc\b",
        ]

        if any(
            re.search(
                pattern,
                recommended_lower,
                flags=re.IGNORECASE,
            )
            for pattern
            in predictive_patterns
        ):
            _append_check(
                result,
                "recommendation_issues",
                (
                    "Predictive-model recommendations are not supported "
                    "because no confirmed classification/regression target "
                    "is available."
                ),
                required=True,
            )

    if not has_target:

        target_language_patterns = [
            r"\bclass labels?\b",
            r"\btarget semantics?\b",
            r"\bbinary target\b",
            r"\boutcome classes?\b",
        ]

        if any(
            re.search(
                pattern,
                lower_report,
                flags=re.IGNORECASE,
            )
            for pattern
            in target_language_patterns
        ):
            _append_check(
                result,
                "unsupported_claims",
                (
                    "The report discusses target/class semantics even "
                    "though no target column was selected."
                ),
                required=True,
            )

    # --------------------------------------------------------
    # Generic inference-overreach checks
    # --------------------------------------------------------

    duplicate_cause_patterns = [
        r"\bsame address\b",
        r"\bsame facility\b",
        r"\bsame household\b",
        r"\bco[- ]located\b",
    ]

    if (
        re.search(
            r"\bduplicate\b",
            lower_report,
        )
        and any(
            re.search(
                pattern,
                lower_report,
            )
            for pattern
            in duplicate_cause_patterns
        )
    ):
        _append_check(
            result,
            "invented_context",
            (
                "The report assigns a real-world cause to duplicated "
                "values/coordinates without explicit supporting evidence."
            ),
            required=True,
        )

    if re.search(
        r"\b(?:spatial|geospatial).{0,80}\bproceed safely\b",
        lower_report,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        _append_check(
            result,
            "unsupported_claims",
            (
                "Legal coordinate ranges do not establish geographic "
                "correctness or guarantee that spatial analysis can "
                "proceed safely."
            ),
            required=True,
        )

    if re.search(
        r"\bused reliably\b",
        lower_report,
        flags=re.IGNORECASE,
    ):
        _append_check(
            result,
            "wording_issues",
            (
                "Completeness alone does not justify describing a field "
                "as reliable for a downstream use."
            ),
            required=True,
        )


    return result


# ============================================================
# REVIEW PROMPT
# ============================================================

def _build_review_prompt(
    sections: dict[str, str],
    evidence_text: str,
) -> str:
    """
    Build a section-aware reviewer request.
    """

    sections_text = json.dumps(
        sections,
        ensure_ascii=False,
        separators=(",", ":"),
    )

    return f"""
You are a strict Senior Analytics Quality Reviewer.

Review EVERY report section below against the structured evidence.

The report has already been split into sections by Python.
Do not review only the beginning.

Your job is to detect material problems.

============================================================
CHECK FOR
============================================================

1. Numerical claims not supported by evidence.

2. Invented:
   - units
   - domain meanings
   - class meanings
   - medical/business context

3. Claims that turn association into causation.

4. Claims that turn correlation or association into
   predictive importance.

5. Unsupported claims about:
   - model performance
   - feature importance
   - multicollinearity
   - interaction effects
   - stability
   - generalization

6. Invalid interpretation of outliers or zero values.

7. Repetition or unnecessary descriptive detail.

8. Weak recommendations.

Recommendations should:
- be concrete
- be evidence-driven
- be prioritized
- represent useful next analytical or modeling actions

Recommendations should NOT automatically prescribe:
- outlier deletion
- SMOTE
- transformations
- feature removal
- complex models
- deployment

unless evidence supports the decision.

9. Semantic-routing mistakes.

Explicitly check whether the report:
- confuses raw physical missing cells with analytically relevant missing fields,
- describes unused structural overflow slots as ordinary missing features,
- treats physical columns of one multi-valued field as independent features,
- applies generic IQR/outlier reasoning to latitude or longitude,
- calls legally valid but geographically distant coordinates invalid,
- recommends capping/removing/transformation of valid coordinates without
  geospatial evidence,
- treats URL/email/phone/address/entity/identifier fields as ordinary
  categorical or predictive features without evidence,
- treats metadata uniqueness as a data-quality defect merely because it is high,
- invents findings when a specialized analyzer says "not_applicable" or "error".

Use "geospatial_analysis", "multi_value_analysis", "metadata_analysis",
"data_quality", "analysis_summary", "schema_summary" and "analysis_plan"
when those keys are present.

10. Unsupported quantitative/domain language.

Flag as material problems when the report:
- says there are "many", "frequent", or "numerous" zero values when only a
  minimum of zero is evidenced,
- labels values physiological, clinical, medical, impossible, or invalid without
  explicit domain evidence,
- claims missingness is random or non-random without a missingness-mechanism test,
- labels URL/phone/email/address/entity-name fields as identifiers when their
  semantic role says otherwise,
- recommends automatic feature removal, hashing, winsorization, imputation,
  exclusion, or transformation without evidence demonstrating the need.

Prefer neutral descriptions tied directly to measured evidence.

11. Target-aware recommendation safety.

- If grounding_contract.has_target is false, flag recommendations for
  classification, regression, predictive baselines, feature importance,
  predictive metrics, or class/target semantics.
- If predictive_modeling_supported is false, predictive modeling may be
  mentioned only as a future possibility after the analytical objective
  or target has been established.
- A dataset without a target must not receive a recommendation such as
  logistic regression, decision tree classification, or model evaluation.

12. Numerical fact ownership.

- Verify that every stated mean, median, minimum, maximum, standard deviation,
  skewness, zero frequency, missing percentage, outlier percentage, correlation,
  effect size, and class count belongs to the correct column/relationship.
- Use column_fact_index as the source of truth for per-column facts when present.
- Do not accept a correct number attached to the wrong variable.
- A minimum of zero does not establish how many zero-valued observations exist.
- Standard deviation does not establish skewness.
- Legal coordinate range does not establish geographic correctness.
- Duplicate coordinates do not establish the reason records share a location.

============================================================
IMPORTANT
============================================================

Do not invent problems.

Do not require unnecessary detail.

A concise report is desirable.

A future action may be recommended even if it has not yet
been performed.

Example:

"Evaluate precision and recall"
is valid.

"The model has poor recall"
requires model evidence.

"Test whether transforming Insulin improves validation"
can be valid.

"Apply a log transformation to Insulin"
is too prescriptive unless evidence establishes the need.

============================================================
OUTPUT
============================================================

Return ONE JSON object only.

No Markdown.

Use exactly these keys:

{{
  "unsupported_claims": [],
  "numerical_mismatches": [],
  "invented_context": [],
  "causal_or_predictive_overreach": [],
  "missing_or_weak_sections": [],
  "recommendation_issues": [],
  "wording_issues": [],
  "required_fixes": [],
  "claim_audit": [
    {{
      "section": "Key Findings",
      "claim": "example",
      "status": "SUPPORTED",
      "reason": "short explanation"
    }}
  ]
}}

Allowed claim status:

SUPPORTED
UNSUPPORTED
OVERSTATED
UNCLEAR

Audit only the most important claims.
Maximum 5 claim_audit entries.

REPORT SECTIONS:

{sections_text}

STRUCTURED EVIDENCE:

{evidence_text}
"""


# ============================================================
# LLM REVIEW
# ============================================================

def _invoke_reviewer(
    prompt: str,
) -> ReviewerAssessment:
    """
    Call the reviewer with resilient local JSON parsing.

    The first response is parsed locally. If the model returns
    malformed or truncated JSON, one compact repair retry is made.

    If both attempts fail, return a conservative deterministic-only
    assessment instead of crashing the entire analysis pipeline.
    """

    llm = create_llm().bind(
        max_tokens=REVIEWER_MAX_TOKENS
    )

    response = llm.invoke(
        [
            HumanMessage(
                content=prompt
            )
        ]
    )

    response_text = _response_to_text(
        response.content
    )

    if response_text:

        try:

            payload = _extract_json_object(
                response_text
            )

            return _normalize_assessment(
                payload
            )

        except RuntimeError:
            pass

    # --------------------------------------------------------
    # One compact repair retry
    # --------------------------------------------------------

    repair_prompt = f"""
Return ONLY one compact valid JSON object.

Do not use Markdown.
Do not use code fences.
Do not include explanatory text before or after the JSON.

Use exactly these keys:

{{
  "unsupported_claims": [],
  "numerical_mismatches": [],
  "invented_context": [],
  "causal_or_predictive_overreach": [],
  "missing_or_weak_sections": [],
  "recommendation_issues": [],
  "wording_issues": [],
  "required_fixes": [],
  "claim_audit": []
}}

Rules:
- Every field must be valid JSON.
- Use double quotes for all strings and keys.
- Maximum 3 items per issue list.
- Maximum 4 claim_audit items.
- Each claim_audit item may contain only:
  "section", "claim", "status", "reason".
- Keep every reason short.
- Do not invent new findings.
- Preserve only conclusions supported by the original review task.

ORIGINAL REVIEW REQUEST:

{prompt}
"""

    retry_llm = create_llm().bind(
        max_tokens=800
    )

    retry_response = retry_llm.invoke(
        [
            HumanMessage(
                content=repair_prompt
            )
        ]
    )

    retry_text = _response_to_text(
        retry_response.content
    )

    if retry_text:

        try:

            payload = _extract_json_object(
                retry_text
            )

            return _normalize_assessment(
                payload
            )

        except RuntimeError:
            pass

    # --------------------------------------------------------
    # Safe fallback
    # --------------------------------------------------------

    return ReviewerAssessment(
        wording_issues=[
            (
                "The LLM reviewer response could not be parsed as valid "
                "JSON after two attempts. Deterministic review checks "
                "were used as a fallback."
            )
        ]
    )


# ============================================================
# MERGE CHECKS
# ============================================================

def _append_unique(
    target: list[str],
    items: list[str],
) -> None:
    """
    Add unique issue strings.
    """

    existing = {
        item.strip().lower()
        for item in target
    }

    for item in items:

        cleaned = str(
            item
        ).strip()

        if not cleaned:
            continue

        key = cleaned.lower()

        if key not in existing:

            target.append(
                cleaned
            )

            existing.add(
                key
            )


def _merge_deterministic_checks(
    assessment: ReviewerAssessment,
    checks: dict[str, list[str]],
) -> ReviewerAssessment:
    """
    Merge Python checks with LLM review.
    """

    for field_name, issues in (
        checks.items()
    ):

        current = getattr(
            assessment,
            field_name,
            None,
        )

        if current is None:
            continue

        _append_unique(
            current,
            issues,
        )

    return assessment


# ============================================================
# LIMIT REVIEW SIZE
# ============================================================

def _limit_assessment(
    assessment: ReviewerAssessment,
) -> ReviewerAssessment:
    """
    Limit feedback size.
    """

    fields = [
        "unsupported_claims",
        "numerical_mismatches",
        "invented_context",
        "causal_or_predictive_overreach",
        "missing_or_weak_sections",
        "recommendation_issues",
        "wording_issues",
        "required_fixes",
    ]

    for field_name in fields:

        value = getattr(
            assessment,
            field_name,
        )

        setattr(
            assessment,
            field_name,
            value[
                :MAX_ISSUES_PER_CATEGORY
            ],
        )

    assessment.claim_audit = (
        assessment.claim_audit[
            :MAX_CLAIM_AUDIT_ITEMS
        ]
    )

    return assessment


# ============================================================
# QUALITY SCORE
# ============================================================

def _calculate_quality_score(
    assessment: ReviewerAssessment,
) -> int:
    """
    Calculate quality score in Python.
    """

    critical = (
        len(
            assessment.unsupported_claims
        )
        +
        len(
            assessment.numerical_mismatches
        )
        +
        len(
            assessment.invented_context
        )
        +
        len(
            assessment
            .causal_or_predictive_overreach
        )
    )

    structure = len(
        assessment.missing_or_weak_sections
    )

    recommendations = len(
        assessment.recommendation_issues
    )

    wording = len(
        assessment.wording_issues
    )

    required_fixes = len(
        assessment.required_fixes
    )

    score = 10

    score -= min(
        6,
        critical * 2,
    )

    score -= min(
        2,
        structure,
    )

    score -= min(
        2,
        recommendations,
    )

    if wording >= 1:
        score -= 1

    score -= min(
        3,
        required_fixes,
    )

    return max(
        1,
        min(
            10,
            score,
        ),
    )


# ============================================================
# VERDICT
# ============================================================

def _determine_verdict(
    assessment: ReviewerAssessment,
    quality_score: int,
) -> str:
    """
    Determine approval deterministically.
    """

    critical = (
        len(
            assessment.unsupported_claims
        )
        +
        len(
            assessment.numerical_mismatches
        )
        +
        len(
            assessment.invented_context
        )
        +
        len(
            assessment
            .causal_or_predictive_overreach
        )
    )

    blocking = (
        len(
            assessment.missing_or_weak_sections
        )
        +
        len(
            assessment.recommendation_issues
        )
        +
        len(
            assessment.required_fixes
        )
    )

    if (
        critical == 0
        and blocking == 0
        and quality_score >= 8
    ):
        return "APPROVED"

    return "NEEDS_REVISION"


# ============================================================
# CRITIQUE
# ============================================================

def _build_critique(
    assessment: ReviewerAssessment,
) -> list[str]:
    """
    Build concise reviewer feedback.
    """

    result = []

    groups = [
        (
            "Unsupported claim",
            assessment.unsupported_claims,
        ),
        (
            "Numerical mismatch",
            assessment.numerical_mismatches,
        ),
        (
            "Invented context",
            assessment.invented_context,
        ),
        (
            "Interpretation overreach",
            assessment
            .causal_or_predictive_overreach,
        ),
        (
            "Weak section",
            assessment.missing_or_weak_sections,
        ),
        (
            "Recommendation issue",
            assessment.recommendation_issues,
        ),
        (
            "Writing issue",
            assessment.wording_issues,
        ),
    ]

    for label, issues in groups:

        for issue in issues[:4]:

            result.append(
                f"{label}: {issue}"
            )

    if not result:

        result.append(
            (
                "No material evidence-grounding "
                "or report-quality problems detected."
            )
        )

    return result[:15]


# ============================================================
# MAIN REVIEWER
# ============================================================

def run_reviewer_agent(
    report: str,
    evidence: dict,
) -> dict:
    """
    Review the complete report section by section.

    Only one LLM request is used to reduce TPM usage.
    """

    sections = _split_report_sections(
        report
    )

    deterministic_checks = (
        _run_deterministic_checks(
            report=report,
            sections=sections,
            evidence=evidence,
        )
    )

    evidence_text = (
        prepare_llm_evidence(
            evidence,
            max_chars=16000,
        )
    )

    prompt = _build_review_prompt(
        sections=sections,
        evidence_text=evidence_text,
    )

    assessment = _invoke_reviewer(
        prompt
    )

    assessment = (
        _merge_deterministic_checks(
            assessment,
            deterministic_checks,
        )
    )

    assessment = _limit_assessment(
        assessment
    )

    quality_score = (
        _calculate_quality_score(
            assessment
        )
    )

    verdict = _determine_verdict(
        assessment=assessment,
        quality_score=quality_score,
    )

    return {
        "verdict":
            verdict,

        "quality_score":
            quality_score,

        "critique":
            _build_critique(
                assessment
            ),

        "required_fixes":
            assessment.required_fixes,

        "unsupported_claims":
            assessment.unsupported_claims,

        "numerical_mismatches":
            assessment.numerical_mismatches,

        "invented_context":
            assessment.invented_context,

        "causal_or_predictive_overreach":
            (
                assessment
                .causal_or_predictive_overreach
            ),

        "missing_or_weak_sections":
            assessment
            .missing_or_weak_sections,

        "recommendation_issues":
            assessment
            .recommendation_issues,

        "wording_issues":
            assessment.wording_issues,

        "claim_audit":
            [
                item.model_dump()
                for item
                in assessment.claim_audit
            ],

        "sections_reviewed":
            list(
                sections.keys()
            ),
    }