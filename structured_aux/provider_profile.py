"""The ``structured-aux`` provider profile.

Only ``auth_type="external_process"`` is viable here. In
``agent/auxiliary_client.py`` the ``api_key`` branch always constructs a plain
``OpenAI(api_key=..., base_url=...)`` and never consults ``profile.create_client()``;
the ``external_process`` branch does, and its own comment states that out-of-tree
providers reach the auxiliary path (compression, vision, background review) exactly like
in-tree ones.

``process_command`` is set to an always-present binary purely to satisfy
``resolve_external_process_provider_credentials``, which refuses a provider whose
command does not resolve via ``shutil.which``. Nothing is ever spawned: ``create_client``
returns the in-process shim, so the subprocess path is never reached.
"""

from __future__ import annotations

from typing import Any

from providers.base import ProviderProfile

from . import config, contracts
from .shim import StructuredAuxClient

# Only used to satisfy the external-process availability gate; never executed.
PLACEHOLDER_COMMAND = "python3"


class StructuredAuxProfile(ProviderProfile):
    """Profile whose client is the decision shim rather than an HTTP endpoint."""

    def create_client(self, **client_kwargs: Any) -> Any:
        return StructuredAuxClient(
            api_key=str(client_kwargs.get("api_key") or "") or config.api_key(),
            base_url=str(client_kwargs.get("base_url") or "") or config.decision_base_url(),
        )


def build_profile() -> StructuredAuxProfile:
    return StructuredAuxProfile(
        name=config.PROVIDER_NAME,
        aliases=("structured_aux",),
        display_name="Structured Aux (Jev decisions)",
        description="Bounded Jev decision calls on OpenRouter for selected Hermes auxiliary tasks.",
        auth_type="external_process",
        base_url=config.DEFAULT_BASE_URL,
        env_vars=(config.CREDENTIAL_ENV_VAR,),
        process_command=PLACEHOLDER_COMMAND,
        default_aux_model=f"{config.MODEL_PREFIX}/{contracts.TASK_APPROVAL}",
        supports_health_check=False,
    )
