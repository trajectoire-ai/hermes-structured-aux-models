"""Hermes model-provider plugin entry point.

Loaded by ``providers/__init__.py`` via ``importlib`` with
``submodule_search_locations=[<this directory>]``, so relative imports below resolve
against this directory as a package. The module-level ``register_provider`` call is the
plugin contract — there is no ``register(ctx)`` for model-provider plugins.

The import is tolerant on purpose. Because this directory is a Python package, a test
collector that walks up to the repository root will import this file outside the Hermes
runtime, where ``providers`` does not exist. Inside Hermes the module is imported *by*
``providers``, so the import always succeeds there and registration always runs; the
guard only covers the out-of-Hermes case, where there is no registry to register into.
"""

from __future__ import annotations

try:
    from providers import register_provider
except ModuleNotFoundError:  # pragma: no cover - only reachable outside the Hermes runtime
    register_provider = None


if register_provider is not None:  # pragma: no branch - always taken inside Hermes
    from .structured_aux.provider_profile import build_profile

    register_provider(build_profile())
