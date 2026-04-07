# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Environment logic for customer support ticket triage."""

from __future__ import annotations

import os
import re
from typing import Any
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata

try:
    from ..models import (
        ALLOWED_CATEGORIES,
        ALLOWED_NEXT_ACTIONS,
        ALLOWED_PRIORITIES,
        ALLOWED_TEAMS,
        BENCHMARK_NAME,
        STAGE_ORDER,
        CustomerSupportTriageAction,
        CustomerSupportTriageObservation,
        CustomerSupportTriageState,
    )
    from .task_bank import KeywordGroup, get_task
except ImportError:
    from models import (
        ALLOWED_CATEGORIES,
        ALLOWED_NEXT_ACTIONS,
        ALLOWED_PRIORITIES,
        ALLOWED_TEAMS,
        BENCHMARK_NAME,
        STAGE_ORDER,
        CustomerSupportTriageAction,
        CustomerSupportTriageObservation,
        CustomerSupportTriageState,
    )
    from server.task_bank import KeywordGroup, get_task


TRIAGE_STAGE_MAX = 0.35
PLAN_STAGE_MAX = 0.35
REPLY_STAGE_MAX = 0.30
TASK_SUCCESS_THRESHOLD = 0.75


def _normalize_choice(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def _contains_phrase(text: str, phrase: str) -> bool:
    return phrase.lower() in text.lower()


def _score_keyword_groups(text: str, groups: tuple[KeywordGroup, ...], weight: float) -> tuple[float, list[str], list[str]]:
    if not groups:
        return 0.0, [], []

    matched_labels: list[str] = []
    missing_labels: list[str] = []
    lowered = text.lower()

    for group in groups:
        if any(phrase.lower() in lowered for phrase in group.phrases):
            matched_labels.append(group.label)
        else:
            missing_labels.append(group.label)

    score = weight * (len(matched_labels) / len(groups))
    return score, matched_labels, missing_labels


def _penalty_for_forbidden_phrases(text: str, phrases: tuple[str, ...], max_penalty: float) -> tuple[float, list[str]]:
    lowered = text.lower()
    hits: list[str] = []
    for phrase in phrases:
        needle = phrase.lower()
        start = lowered.find(needle)
        if start == -1:
            continue
        window_start = max(0, start - 18)
        prefix = lowered[window_start:start]
        if any(token in prefix for token in ("do not ", "don't ", "never ", "should not ")):
            continue
        hits.append(phrase)
    if not hits:
        return 0.0, []
    penalty = min(max_penalty, 0.08 * len(hits))
    return penalty, hits


class CustomerSupportTriageEnvironment(Environment):
    """
    A multi-step environment that simulates support ticket handling.

    Each episode spans three deterministic workflow stages:
    - triage: classify the ticket and assign urgency/owner
    - plan: choose the best operational next action
    - reply: draft a safe customer-facing response
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._default_task_name = os.getenv("CUSTOMER_SUPPORT_TRIAGE_TASK")
        self._task = get_task(self._default_task_name)
        self._state = CustomerSupportTriageState()
        self._reset_episode(self._task.task_name, episode_id=None)

    def _reset_episode(self, task_name: str | None, episode_id: str | None) -> None:
        self._task = get_task(task_name)
        self._state = CustomerSupportTriageState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_name=self._task.task_name,
            difficulty=self._task.difficulty,
            ticket_id=self._task.ticket_id,
            current_stage=STAGE_ORDER[0],
            current_stage_index=0,
            cumulative_reward=0.0,
            score_breakdown={stage: 0.0 for stage in STAGE_ORDER},
            completed_stages=[],
            action_history=[],
            last_action_error=None,
        )

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        task_name: str | None = None,
        **kwargs: Any,
    ) -> CustomerSupportTriageObservation:
        del seed, kwargs
        self._reset_episode(task_name or self._default_task_name, episode_id=episode_id)
        return self._build_observation(
            stage_score=0.0,
            feedback="Start with triage: choose category, priority, and owning team.",
            last_action_error=None,
            done=False,
        )

    def step(
        self,
        action: CustomerSupportTriageAction,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> CustomerSupportTriageObservation:
        del timeout_s, kwargs

        current_stage = STAGE_ORDER[min(self._state.current_stage_index, len(STAGE_ORDER) - 1)]
        stage_score, feedback, last_action_error = self._grade_stage(current_stage, action)

        self._state.step_count += 1
        self._state.score_breakdown[current_stage] = stage_score
        self._state.cumulative_reward = round(sum(self._state.score_breakdown.values()), 4)
        self._state.completed_stages = list(STAGE_ORDER[: self._state.current_stage_index + 1])
        self._state.action_history.append(
            {
                "stage": current_stage,
                "decision_type": action.decision_type,
                "category": action.category,
                "priority": action.priority,
                "assigned_team": action.assigned_team,
                "next_action": action.next_action,
            }
        )
        self._state.last_action_error = last_action_error

        done = self._state.current_stage_index >= len(STAGE_ORDER) - 1
        if done:
            return self._build_observation(
                stage_score=stage_score,
                feedback=feedback,
                last_action_error=last_action_error,
                done=True,
            )

        self._state.current_stage_index += 1
        self._state.current_stage = STAGE_ORDER[self._state.current_stage_index]
        return self._build_observation(
            stage_score=stage_score,
            feedback=feedback,
            last_action_error=last_action_error,
            done=False,
        )

    def _grade_stage(
        self,
        stage: str,
        action: CustomerSupportTriageAction,
    ) -> tuple[float, str, str | None]:
        if stage == "triage":
            return self._grade_triage(action)
        if stage == "plan":
            return self._grade_plan(action)
        return self._grade_reply(action)

    def _decision_penalty(self, expected_stage: str, action: CustomerSupportTriageAction) -> tuple[float, str | None]:
        if _normalize_choice(action.decision_type) == expected_stage:
            return 0.0, None
        return 0.05, f"Expected decision_type '{expected_stage}' but received '{action.decision_type}'."

    def _grade_triage(self, action: CustomerSupportTriageAction) -> tuple[float, str, str | None]:
        score = 0.0
        messages: list[str] = []
        error_messages: list[str] = []

        decision_penalty, decision_error = self._decision_penalty("triage", action)
        if decision_error:
            error_messages.append(decision_error)

        if _normalize_choice(action.category) == self._task.expected_category:
            score += 0.15
            messages.append("Category is correct.")
        else:
            messages.append(f"Category should be '{self._task.expected_category}'.")
            if not action.category.strip():
                error_messages.append("Category cannot be empty during triage.")

        if _normalize_choice(action.priority) == self._task.expected_priority:
            score += 0.10
            messages.append("Priority is appropriate.")
        else:
            messages.append(f"Priority should be '{self._task.expected_priority}'.")
            if not action.priority.strip():
                error_messages.append("Priority cannot be empty during triage.")

        if _normalize_choice(action.assigned_team) == self._task.expected_team:
            score += 0.10
            messages.append("Owning team is correct.")
        else:
            messages.append(f"Owning team should be '{self._task.expected_team}'.")
            if not action.assigned_team.strip():
                error_messages.append("Assigned team cannot be empty during triage.")

        stage_score = max(0.0, min(TRIAGE_STAGE_MAX, score - decision_penalty))
        return (
            round(stage_score, 4),
            " ".join(messages),
            "; ".join(error_messages) if error_messages else None,
        )

    def _grade_plan(self, action: CustomerSupportTriageAction) -> tuple[float, str, str | None]:
        score = 0.0
        messages: list[str] = []
        error_messages: list[str] = []

        decision_penalty, decision_error = self._decision_penalty("plan", action)
        if decision_error:
            error_messages.append(decision_error)

        if _normalize_choice(action.next_action) in self._task.accepted_next_actions:
            score += 0.20
            messages.append("Operational next action is appropriate.")
        else:
            expected = " or ".join(self._task.accepted_next_actions)
            messages.append(f"Next action should be '{expected}'.")
            if not action.next_action.strip():
                error_messages.append("Next action cannot be empty during planning.")

        note_score, matched, missing = _score_keyword_groups(
            action.internal_note,
            self._task.note_requirements,
            weight=0.15,
        )
        score += note_score
        if matched:
            messages.append(f"Internal note covered: {', '.join(matched)}.")
        if missing:
            messages.append(f"Internal note should also mention: {', '.join(missing)}.")
        if not action.internal_note.strip():
            error_messages.append("Internal note cannot be empty during planning.")

        safety_penalty, safety_hits = _penalty_for_forbidden_phrases(
            action.internal_note,
            self._task.forbidden_note_phrases,
            max_penalty=0.15,
        )
        if safety_hits:
            messages.append("Internal note includes unsafe handling guidance.")
            error_messages.append(f"Unsafe note content detected: {', '.join(safety_hits)}.")

        stage_score = max(0.0, min(PLAN_STAGE_MAX, score - decision_penalty - safety_penalty))
        return (
            round(stage_score, 4),
            " ".join(messages),
            "; ".join(error_messages) if error_messages else None,
        )

    def _grade_reply(self, action: CustomerSupportTriageAction) -> tuple[float, str, str | None]:
        score = 0.0
        messages: list[str] = []
        error_messages: list[str] = []

        decision_penalty, decision_error = self._decision_penalty("reply", action)
        if decision_error:
            error_messages.append(decision_error)

        reply_score, matched, missing = _score_keyword_groups(
            action.customer_reply,
            self._task.reply_requirements,
            weight=REPLY_STAGE_MAX,
        )
        score += reply_score
        if matched:
            messages.append(f"Reply covered: {', '.join(matched)}.")
        if missing:
            messages.append(f"Reply should also include: {', '.join(missing)}.")
        if not action.customer_reply.strip():
            error_messages.append("Customer reply cannot be empty during the reply stage.")

        safety_penalty, safety_hits = _penalty_for_forbidden_phrases(
            action.customer_reply,
            self._task.forbidden_reply_phrases,
            max_penalty=0.20,
        )
        if safety_hits:
            messages.append("Reply asks for sensitive or policy-unsafe information.")
            error_messages.append(f"Unsafe reply content detected: {', '.join(safety_hits)}.")

        stage_score = max(0.0, min(REPLY_STAGE_MAX, score - decision_penalty - safety_penalty))
        return (
            round(stage_score, 4),
            " ".join(messages),
            "; ".join(error_messages) if error_messages else None,
        )

    def _build_observation(
        self,
        stage_score: float,
        feedback: str,
        last_action_error: str | None,
        done: bool,
    ) -> CustomerSupportTriageObservation:
        current_stage = "done" if done else self._state.current_stage
        final_score = round(self._state.cumulative_reward, 4) if done else None
        next_description = (
            "Episode complete. Review the final score and stage breakdown."
            if done
            else self._task.stage_descriptions[current_stage]
        )

        return CustomerSupportTriageObservation(
            benchmark=BENCHMARK_NAME,
            task_name=self._task.task_name,
            difficulty=self._task.difficulty,
            ticket_id=self._task.ticket_id,
            title=self._task.title,
            customer_tier=self._task.customer_tier,
            customer_name=self._task.customer_name,
            subject=self._task.subject,
            customer_message=self._task.customer_message,
            conversation_history=list(self._task.conversation_history),
            policy_snippets=list(self._task.policy_snippets),
            current_stage=current_stage,
            stage_description=next_description,
            allowed_categories=list(ALLOWED_CATEGORIES),
            allowed_priorities=list(ALLOWED_PRIORITIES),
            allowed_teams=list(ALLOWED_TEAMS),
            allowed_next_actions=list(ALLOWED_NEXT_ACTIONS),
            feedback=feedback,
            stage_score=round(stage_score, 4),
            progress_score=round(self._state.cumulative_reward, 4),
            score_breakdown=dict(self._state.score_breakdown),
            completed_stages=list(self._state.completed_stages),
            last_action_error=last_action_error,
            final_score=final_score,
            reward=round(stage_score, 4),
            done=done,
            metadata={
                "task_success_threshold": TASK_SUCCESS_THRESHOLD,
                "available_tasks": [
                    "password_reset_easy",
                    "duplicate_charge_medium",
                    "account_takeover_hard",
                ],
            },
        )

    @property
    def state(self) -> CustomerSupportTriageState:
        return self._state

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="CustomerSupportTriageEnvironment",
            description=(
                "A deterministic support operations environment where an agent must "
                "triage tickets, choose safe next actions, and draft compliant customer replies."
            ),
            version="1.0.0",
            author="Rahul Dadwal + Codex",
        )
