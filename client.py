# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Customer Support Triage environment client."""

from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from .models import (
    CustomerSupportTriageAction,
    CustomerSupportTriageObservation,
    CustomerSupportTriageState,
)


class CustomerSupportTriageEnv(
    EnvClient[
        CustomerSupportTriageAction,
        CustomerSupportTriageObservation,
        CustomerSupportTriageState,
    ]
):
    """Async OpenEnv client for the customer support triage environment."""

    def _step_payload(self, action: CustomerSupportTriageAction) -> Dict[str, Any]:
        return action.model_dump()

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[CustomerSupportTriageObservation]:
        obs_data = payload.get("observation", {})
        obs_payload = dict(obs_data)
        obs_payload["done"] = payload.get("done", obs_payload.get("done", False))
        obs_payload["reward"] = payload.get("reward", obs_payload.get("reward"))
        observation = CustomerSupportTriageObservation(
            **obs_payload,
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> CustomerSupportTriageState:
        return CustomerSupportTriageState(**payload)
