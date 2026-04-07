---
title: Customer Support Triage Environment
emoji: 📨
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Customer Support Triage Environment

A real-world OpenEnv environment for training and  evaluating agents on customer support operations. The agent must handle realistic support tickets across three workflow stages:

1. Triage the ticket
2. Choose the next operational action
3. Draft a safe customer reply

## Why this is useful

Support operations teams regularly need agents that can route tickets correctly, respect policy constraints, and respond safely under time pressure. This environment models exactly that workflow using deterministic graders and partial rewards.

## Tasks

The environment ships with 3 graded tasks:

- `password_reset_easy`: recover a locked-out account safely
- `duplicate_charge_medium`: handle a duplicate annual subscription charge
- `account_takeover_hard`: contain a likely account takeover with billing impact

Each task returns a final score in `[0.0, 1.0]`.

## Action Space

`CustomerSupportTriageAction`

- `decision_type`: `triage` | `plan` | `reply`
- `category`: support category label
- `priority`: ticket urgency
- `assigned_team`: owning team
- `next_action`: operational next step
- `internal_note`: internal support note
- `customer_reply`: customer-facing response
- `confidence`: scalar confidence in `[0, 1]`

## Observation Space

`CustomerSupportTriageObservation` includes:

- ticket details: `task_name`, `ticket_id`, `title`, `subject`, `customer_message`
- support context: `conversation_history`, `policy_snippets`
- workflow guidance: `current_stage`, `stage_description`
- reward signals: `stage_score`, `progress_score`, `score_breakdown`, `final_score`
- feedback: `feedback`, `last_action_error`

## Reward Design

The reward is shaped over the full trajectory.

- `triage` stage: rewards correct category, priority, and team
- `plan` stage: rewards correct next action plus key operational details in the note
- `reply` stage: rewards empathy, policy-safe handling, and correct customer guidance

Unsafe behaviors reduce reward, including asking for passwords, OTPs, MFA codes, CVVs, or promising outcomes that violate policy.

## Quick Start

### Install dependencies

```bash
uv sync
```

### Run locally

```bash
uv run server
```

### Validate the environment

```bash
openenv validate
```

### Build Docker image

```bash
docker build -t customer_support_triage-env:latest .
```

## Inference Script

The required submission script is [`inference.py`](/Users/rahuldadwal/Desktop/meta/customer_support_triage/inference.py). It:

- uses the OpenAI client for LLM calls
- runs all 3 tasks by default
- prints strict `[START]`, `[STEP]`, and `[END]` logs
- produces deterministic fallback actions if the model response is unavailable

### Required environment variables

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`
- `LOCAL_IMAGE_NAME`

Example:

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="hf_xxx"
export LOCAL_IMAGE_NAME="customer_support_triage-env:latest"
uv run python inference.py
```

## Submission Notes

- `openenv.yaml` is present at repo root
- root `Dockerfile` supports validator-compatible builds
- `server/app.py` exposes `reset`, `step`, `state`, and `ws`
- `inference.py` is in the repo root as required

## Project Structure

```text
customer_support_triage/
├── Dockerfile
├── README.md
├── inference.py
├── openenv.yaml
├── pyproject.toml
├── client.py
├── models.py
└── server/
    ├── app.py
    ├── customer_support_triage_environment.py
    └── task_bank.py
```
