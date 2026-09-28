from __future__ import annotations

from typing import TypedDict

from langgraph.graph import (
    END,
    START,
    StateGraph,
)

from app.agents.analyst import (
    revise_analyst_report,
    run_analyst_agent,
)

from app.agents.reviewer import (
    run_reviewer_agent,
)


# ============================================================
# CONFIGURATION
# ============================================================

MAX_REVISIONS = 2


# ============================================================
# STATE
# ============================================================

class AnalysisState(
    TypedDict,
    total=False,
):

    evidence: dict

    report: str

    review: dict

    iteration: int

    approved: bool

    final_report: str


# ============================================================
# ANALYST
# ============================================================

def analyst_node(
    state: AnalysisState,
) -> AnalysisState:
    """
    Create the initial analysis report.
    """

    evidence = state.get(
        "evidence",
        {},
    )


    report = (
        run_analyst_agent(
            evidence=evidence,
        )
    )


    return {
        "report":
            report,

        "iteration":
            0,

        "approved":
            False,
    }


# ============================================================
# REVIEWER
# ============================================================

def reviewer_node(
    state: AnalysisState,
) -> AnalysisState:
    """
    Review the current report against evidence.
    """

    report = state.get(
        "report",
        "",
    )


    evidence = state.get(
        "evidence",
        {},
    )


    review = (
        run_reviewer_agent(
            report=report,
            evidence=evidence,
        )
    )


    approved = (
        review.get(
            "verdict"
        )
        == "APPROVED"
    )


    return {
        "review":
            review,

        "approved":
            approved,
    }


# ============================================================
# REVISION
# ============================================================

def revision_node(
    state: AnalysisState,
) -> AnalysisState:
    """
    Revise the report using reviewer feedback.
    """

    evidence = state.get(
        "evidence",
        {},
    )


    report = state.get(
        "report",
        "",
    )


    review = state.get(
        "review",
        {},
    )


    current_iteration = int(
        state.get(
            "iteration",
            0,
        )
    )


    revised_report = (
        revise_analyst_report(
            evidence=evidence,
            current_report=report,
            review=review,
        )
    )


    return {
        "report":
            revised_report,

        "iteration":
            current_iteration
            + 1,
    }


# ============================================================
# ROUTING
# ============================================================

def route_after_review(
    state: AnalysisState,
) -> str:
    """
    Decide whether to approve, revise, or finalize.
    """

    approved = bool(
        state.get(
            "approved",
            False,
        )
    )


    iteration = int(
        state.get(
            "iteration",
            0,
        )
    )


    if approved:

        return "finalize"


    if (
        iteration
        >= MAX_REVISIONS
    ):

        return "finalize"


    return "revise"


# ============================================================
# FINALIZER
# ============================================================

def finalizer_node(
    state: AnalysisState,
) -> AnalysisState:
    """
    Return the final reviewed report.
    """

    report = state.get(
        "report",
        "",
    )


    return {
        "final_report":
            report,
    }


# ============================================================
# BUILD WORKFLOW
# ============================================================

def build_workflow():
    """
    Build the Analyst -> Reviewer -> Revision workflow.
    """

    workflow = StateGraph(
        AnalysisState
    )


    workflow.add_node(
        "analyst",
        analyst_node,
    )


    workflow.add_node(
        "reviewer",
        reviewer_node,
    )


    workflow.add_node(
        "revise",
        revision_node,
    )


    workflow.add_node(
        "finalize",
        finalizer_node,
    )


    workflow.add_edge(
        START,
        "analyst",
    )


    workflow.add_edge(
        "analyst",
        "reviewer",
    )


    workflow.add_conditional_edges(
        "reviewer",
        route_after_review,
        {
            "revise":
                "revise",

            "finalize":
                "finalize",
        },
    )


    workflow.add_edge(
        "revise",
        "reviewer",
    )


    workflow.add_edge(
        "finalize",
        END,
    )


    return workflow.compile()