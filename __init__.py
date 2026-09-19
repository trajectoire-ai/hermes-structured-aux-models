"""Hermes model-provider plugin entry point.

Loaded by ``providers/__init__.py`` via ``importlib`` with
``submodule_search_locations=[<this directory>]``, so relative imports below resolve
against this directory as a package. The module-level ``register_provider`` call is the
plugin contract — there is no ``register(ctx)`` for model-provider plugins.
"""

from providers import register_provider

from .structured_aux.provider_profile import build_profile

register_provider(build_profile())
