# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Typed models for the Customer Support Triage environment.

The environment simulates a real support queue where an agent must:
1. Triage the ticket
2. Choose the best operational next action
3. Draft a safe customer response
"""

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


BENCHMARK_NAME = "customer_support_triage"
STAGE_ORDER = ("triage", "plan", "reply")

ALLOWED_CATEGORIES = (
    "account_access",
    "billing_refund",
    "security_incident",
    "technical_support",
    "general_inquiry",
)

ALLOWED_PRIORITIES = ("low", "medium", "high", "urgent")

ALLOWED_TEAMS = (
    "account_access",
    "billing",
    "security",
    "technical_support",
    "customer_success",
)

ALLOWED_NEXT_ACTIONS = (
    "request_identity_verification",
    "send_password_reset_instructions",
    "issue_refund",
    "request_order_confirmation",
    "lock_account_and_escalate",
    "troubleshoot_steps",
    "escalate_to_engineering",
)


class CustomerSupportTriageAction(Action):
    """Single action used across the staged workflow."""

    decision_type: str = Field(
        default="triage",
        description="Current workflow stage: triage, plan, or reply.",
    )
    category: str = Field(
        default="",
        description=f"Ticket category. Suggested values: {', '.join(ALLOWED_CATEGORIES)}.",
    )
    priority: str = Field(
        default="",
        description=f"Priority label. Suggested values: {', '.join(ALLOWED_PRIORITIES)}.",
    )
    assigned_team: str = Field(
        default="",
        description=f"Owning team. Suggested values: {', '.join(ALLOWED_TEAMS)}.",
    )
    next_action: str = Field(
        default="",
        description=f"Operational next action. Suggested values: {', '.join(ALLOWED_NEXT_ACTIONS)}.",
    )
    internal_note: str = Field(
        default="",
        description="Internal operational note summarizing why the action is correct.",
    )
    customer_reply: str = Field(
        default="",
        description="Reply that would be sent to the customer.",
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Agent confidence in the chosen action.",
    )


class CustomerSupportTriageObservation(Observation):
    """Observation returned by the support triage environment."""

    benchmark: str = Field(default=BENCHMARK_NAME)
    task_name: str = Field(default="")
    difficulty: str = Field(default="")
    ticket_id: str = Field(default="")
    title: str = Field(default="")
    customer_tier: str = Field(default="")
    customer_name: str = Field(default="")
    subject: str = Field(default="")
    customer_message: str = Field(default="")
    conversation_history: list[str] = Field(default_factory=list)
    policy_snippets: list[str] = Field(default_factory=list)
    current_stage: str = Field(default="triage")
    stage_description: str = Field(default="")
    allowed_categories: list[str] = Field(default_factory=lambda: list(ALLOWED_CATEGORIES))
    allowed_priorities: list[str] = Field(default_factory=lambda: list(ALLOWED_PRIORITIES))
    allowed_teams: list[str] = Field(default_factory=lambda: list(ALLOWED_TEAMS))
    allowed_next_actions: list[str] = Field(default_factory=lambda: list(ALLOWED_NEXT_ACTIONS))
    feedback: str = Field(default="")
    stage_score: float = Field(default=0.0, ge=0.0, le=1.0)
    progress_score: float = Field(default=0.0, ge=0.0, le=1.0)
    score_breakdown: dict[str, float] = Field(default_factory=dict)
    completed_stages: list[str] = Field(default_factory=list)
    last_action_error: str | None = Field(default=None)
    final_score: float | None = Field(default=None, ge=0.0, le=1.0)


class CustomerSupportTriageState(State):
    """Internal state exposed through the standard OpenEnv state endpoint."""

    task_name: str = Field(default="")
    difficulty: str = Field(default="")
    ticket_id: str = Field(default="")
    current_stage: str = Field(default="triage")
    current_stage_index: int = Field(default=0, ge=0)
    cumulative_reward: float = Field(default=0.0, ge=0.0, le=1.0)
    score_breakdown: dict[str, float] = Field(default_factory=dict)
    completed_stages: list[str] = Field(default_factory=list)
    action_history: list[dict[str, str]] = Field(default_factory=list)
    last_action_error: str | None = Field(default=None)
