# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Deterministic task bank for the Customer Support Triage environment."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KeywordGroup:
    label: str
    phrases: tuple[str, ...]


@dataclass(frozen=True)
class TicketTask:
    task_name: str
    difficulty: str
    title: str
    ticket_id: str
    customer_name: str
    customer_tier: str
    subject: str
    customer_message: str
    conversation_history: tuple[str, ...]
    policy_snippets: tuple[str, ...]
    expected_category: str
    expected_priority: str
    expected_team: str
    accepted_next_actions: tuple[str, ...]
    note_requirements: tuple[KeywordGroup, ...]
    reply_requirements: tuple[KeywordGroup, ...]
    forbidden_note_phrases: tuple[str, ...]
    forbidden_reply_phrases: tuple[str, ...]
    stage_descriptions: dict[str, str]


def _group(label: str, *phrases: str) -> KeywordGroup:
    return KeywordGroup(label=label, phrases=tuple(phrases))


TASKS: dict[str, TicketTask] = {
    "password_reset_easy": TicketTask(
        task_name="password_reset_easy",
        difficulty="easy",
        title="Recover a locked-out self-serve account",
        ticket_id="CST-1001",
        customer_name="Neha Patel",
        customer_tier="standard",
        subject="Locked out after reset links expired",
        customer_message=(
            "Hi team, I have tried the password reset link twice and both links expired "
            "before I could log in. I have a client demo tomorrow morning and I still "
            "cannot access the dashboard. Please help me get back in."
        ),
        conversation_history=(
            "Customer: The reset link keeps expiring before I can use it.",
            "Auto-reply: We received your ticket and will respond shortly.",
        ),
        policy_snippets=(
            "Account access issues belong to the account_access team.",
            "Never ask the customer to share their password, OTP, or MFA code.",
            "For account recovery, first verify identity and then issue reset instructions.",
            "A looming customer demo justifies medium priority unless a security breach is reported.",
        ),
        expected_category="account_access",
        expected_priority="medium",
        expected_team="account_access",
        accepted_next_actions=("request_identity_verification",),
        note_requirements=(
            _group("identity verification", "identity verification", "verify identity", "confirm the account owner"),
            _group("password reset flow", "password reset", "reset link", "reset instructions"),
            _group("access issue", "locked out", "login issue", "account access"),
        ),
        reply_requirements=(
            _group("empathy", "sorry", "understand", "frustrating"),
            _group("verification step", "verify your identity", "identity verification"),
            _group("reset support", "password reset link", "reset instructions", "reset your password"),
            _group("safe handling", "do not share your password", "never send us your otp", "don't send your password"),
        ),
        forbidden_note_phrases=(
            "ask for password",
            "share password",
            "send otp",
        ),
        forbidden_reply_phrases=(
            "tell us your password",
            "share your password",
            "send your password",
            "share the otp",
            "send the otp",
        ),
        stage_descriptions={
            "triage": "Choose category, priority, and owning team for the ticket.",
            "plan": "Choose the next operational action and write an internal note.",
            "reply": "Draft the customer-facing reply with empathy and safe instructions.",
        },
    ),
    "duplicate_charge_medium": TicketTask(
        task_name="duplicate_charge_medium",
        difficulty="medium",
        title="Handle a duplicate annual subscription charge",
        ticket_id="CST-2044",
        customer_name="Karan Mehta",
        customer_tier="pro",
        subject="Charged twice for annual plan renewal",
        customer_message=(
            "I renewed our Pro plan this morning and now I see two charges for the same "
            "invoice in my card statement. The order ID in the portal is INV-77821. "
            "Please reverse the duplicate charge quickly because finance is closing books today."
        ),
        conversation_history=(
            "Customer: I can see the same annual renewal twice on the same card.",
            "CRM note: Customer has been active for 3 years with no prior refund abuse.",
        ),
        policy_snippets=(
            "Duplicate subscription charges are owned by the billing team.",
            "When a valid invoice or order ID is provided, billing may initiate a refund without asking for full card details.",
            "Customers blocked by finance deadlines should be treated as high priority.",
            "Never request full card number, CVV, or banking PIN over support channels.",
        ),
        expected_category="billing_refund",
        expected_priority="high",
        expected_team="billing",
        accepted_next_actions=("issue_refund",),
        note_requirements=(
            _group("duplicate charge", "duplicate charge", "charged twice", "double charge"),
            _group("invoice reference", "inv-77821", "invoice", "order id"),
            _group("refund action", "refund", "reverse the duplicate charge", "billing refund"),
        ),
        reply_requirements=(
            _group("empathy", "sorry", "understand", "inconvenience"),
            _group("refund confirmation", "refund", "reverse the duplicate charge", "billing team is processing the refund"),
            _group("timeline", "3-5 business days", "5 business days", "a few business days"),
            _group("safe payment handling", "no need to send your full card number", "please do not share card details"),
        ),
        forbidden_note_phrases=(
            "ask for cvv",
            "ask for card pin",
            "decline refund",
        ),
        forbidden_reply_phrases=(
            "send your full card number",
            "share your cvv",
            "share your card pin",
            "we cannot help",
        ),
        stage_descriptions={
            "triage": "Decide the ticket category, urgency, and owning team.",
            "plan": "Choose the operational next action and capture the finance rationale internally.",
            "reply": "Draft a billing-safe response that confirms the next step and refund timeline.",
        },
    ),
    "account_takeover_hard": TicketTask(
        task_name="account_takeover_hard",
        difficulty="hard",
        title="Contain a possible account takeover with billing impact",
        ticket_id="CST-3307",
        customer_name="Riya Sethi",
        customer_tier="enterprise",
        subject="Urgent: suspicious MFA prompts and unauthorized invoices",
        customer_message=(
            "I am the workspace admin for our enterprise account. Since 6 AM I have received "
            "multiple unexpected MFA prompts and password reset emails. I also noticed two invoices "
            "for add-on seats that nobody on my team purchased. I can still access the admin console "
            "but I am worried the account is compromised."
        ),
        conversation_history=(
            "Customer: Unexpected MFA prompts started early this morning.",
            "Security log: New IP login from an unrecognized region 47 minutes ago.",
            "Billing log: Two seat expansion invoices created after the suspicious login.",
        ),
        policy_snippets=(
            "Potential account takeover cases are owned by the security team and must be marked urgent.",
            "Immediately lock or contain the account before processing any billing adjustments.",
            "Use out-of-band identity verification through the registered admin contact.",
            "Never ask for passwords, OTPs, backup codes, or MFA secrets.",
            "Do not promise an immediate refund before the security investigation confirms the fraudulent activity.",
        ),
        expected_category="security_incident",
        expected_priority="urgent",
        expected_team="security",
        accepted_next_actions=("lock_account_and_escalate",),
        note_requirements=(
            _group("takeover risk", "account takeover", "security incident", "compromised account"),
            _group("containment", "lock account", "secure the workspace", "contain access"),
            _group("verification", "registered admin contact", "out-of-band verification", "verify the admin"),
            _group("billing follow-up", "billing review", "review unauthorized invoices", "billing follow up"),
        ),
        reply_requirements=(
            _group("empathy", "sorry", "understand", "serious"),
            _group("containment", "secure the account", "lock the account", "security team is containing access"),
            _group("verification", "registered admin contact", "verify your identity", "out-of-band verification"),
            _group("billing caveat", "review the invoices after the investigation", "billing will review the charges after verification"),
        ),
        forbidden_note_phrases=(
            "ask for otp",
            "ask for password",
            "promise refund now",
        ),
        forbidden_reply_phrases=(
            "tell us your password",
            "send your otp",
            "send your mfa code",
            "we have already refunded you",
            "ignore the invoices",
        ),
        stage_descriptions={
            "triage": "Classify and prioritize the security-sensitive ticket.",
            "plan": "Choose the containment action and leave an internal note that reflects security + billing coordination.",
            "reply": "Draft a safe customer response that focuses on containment, identity verification, and careful billing follow-up.",
        },
    ),
}


DEFAULT_TASK = "password_reset_easy"


def list_task_names() -> list[str]:
    return list(TASKS.keys())


def get_task(task_name: str | None) -> TicketTask:
    if task_name is None:
        return TASKS[DEFAULT_TASK]
    return TASKS.get(task_name, TASKS[DEFAULT_TASK])
