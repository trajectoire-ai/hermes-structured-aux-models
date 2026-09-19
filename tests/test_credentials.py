"""Credential resolution for the decision transport.

An operator who already ran ``hermes auth add openrouter`` has the key in Hermes' credential
store, so the plugin reads it from there instead of forcing a duplicate ``.env`` entry. These
tests cover the precedence and the degradation path; none touches a real credential or network.
"""

from __future__ import annotations

import sys
import types

import pytest

from structured_aux import config


@pytest.fixture(autouse=True)
def _no_env(monkeypatch):
    """Every test states its own credential sources; start from none of them."""
    monkeypatch.delenv(config.CREDENTIAL_ENV_VAR, raising=False)
    monkeypatch.setattr(config, "_dotenv_value", lambda name: "")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)


@pytest.fixture
def fake_pool(monkeypatch):
    """Install a stub ``agent.credential_pool`` module with a chosen ``load_pool``."""

    def install(*, api_key="", has_credentials=True, raises=None, pool=None):
        module = types.ModuleType("agent.credential_pool")

        class _Entry:
            runtime_api_key = api_key

        class _Pool:
            def has_credentials(self):
                return has_credentials

            def select(self):
                return _Entry()

        def load_pool(provider):
            if raises is not None:
                raise raises
            if pool is not None:
                return pool
            return _Pool()

        module.load_pool = load_pool
        monkeypatch.setitem(sys.modules, "agent.credential_pool", module)
        return module

    return install


def test_pool_key_is_used_when_env_and_dotenv_are_empty(fake_pool):
    fake_pool(api_key="pool-key-123")
    assert config.api_key() == "pool-key-123"


def test_process_env_wins_over_the_pool(fake_pool, monkeypatch):
    fake_pool(api_key="pool-key-123")
    monkeypatch.setenv(config.CREDENTIAL_ENV_VAR, "env-key")
    assert config.api_key() == "env-key"


def test_profile_dotenv_wins_over_the_pool(fake_pool, monkeypatch):
    fake_pool(api_key="pool-key-123")
    monkeypatch.setattr(config, "_dotenv_value", lambda name: "dotenv-key")
    assert config.api_key() == "dotenv-key"


def test_empty_pool_degrades_to_no_credential(fake_pool):
    fake_pool(has_credentials=False)
    assert config.api_key() == ""


def test_missing_pool_object_degrades_to_no_credential(fake_pool):
    fake_pool(pool=None)
    assert config.api_key() == ""


def test_unresolvable_pool_degrades_to_no_credential(fake_pool):
    # A missing Hermes module (or a credential-store error) must never raise out of a
    # credential lookup: no key means "cannot serve this request", and Hermes falls back.
    fake_pool(raises=RuntimeError("no store"))
    assert config.api_key() == ""
