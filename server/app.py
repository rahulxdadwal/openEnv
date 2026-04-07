# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FastAPI application for the Customer Support Triage environment."""

import os

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with 'uv sync'."
    ) from e

try:
    from ..models import CustomerSupportTriageAction, CustomerSupportTriageObservation
    from .customer_support_triage_environment import CustomerSupportTriageEnvironment
except (ImportError, ModuleNotFoundError):
    from models import CustomerSupportTriageAction, CustomerSupportTriageObservation
    from server.customer_support_triage_environment import CustomerSupportTriageEnvironment


app = create_app(
    CustomerSupportTriageEnvironment,
    CustomerSupportTriageAction,
    CustomerSupportTriageObservation,
    env_name="customer_support_triage",
    max_concurrent_envs=4,
)


def main() -> None:
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
