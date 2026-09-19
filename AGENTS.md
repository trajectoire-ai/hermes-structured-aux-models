# Agent instructions — hermes-structured-aux-models

Instructions for agents working in this repository.

## What this is

A Hermes Agent **model-provider plugin** (`kind: model-provider`). It registers one
provider, `structured-aux`, whose client is an in-process shim: it accepts the
OpenAI-shaped `chat.completions.create()` call that Hermes' auxiliary client makes,
translates it into a bounded Jev decision request on OpenRouter
(`/api/alpha/decisions`), and maps the typed answer back into an OpenAI-shaped
response.

It is **not** a general chat provider. It only answers requests it can express as a
bounded decision; everything else must raise so Hermes falls back to the operator's
real auxiliary provider.

## Hard rules

- **Never edit the Hermes core.** This plugin works only through documented plugin
  surfaces. If it needs more than they provide, stop and report — do not patch
  `hermes-agent`.
- **No live paid calls in tests or CI.** Every test injects a fake transport. A live
  provider run needs explicit per-run approval from the operator.
- **Fail open, never fail closed.** If the request cannot be expressed as a decision,
  or the provider errors, raise `UnsupportedRequest` / let the transport error
  propagate. Hermes' auxiliary fallback then does its job. Never invent an answer.
- **Never fabricate a decision.** If the provider returns a label outside the
  contract's criteria set, raise. Do not map unknown labels onto a default.
- **Keep the tree clean for Hermes' install-time plugin scanner** (`tools/plugin_guard.py`):
  do not introduce literal destructive-delete command strings (even inside test
  fixtures), an environment-variable subscript, an environment-dumping command name, or
  unpinned package-install lines. Read credentials with `os.getenv` only. Re-run the
  scanner before pushing.
- **Text artifacts are UTF-8 without BOM, LF line endings.**

## Verified Hermes seams this depends on

Documented so a future maintainer can re-verify instead of guessing. All line
references were true against Hermes `v0.21.0`.

- `providers/__init__.py:_import_plugin_dir` loads a plugin dir with
  `submodule_search_locations=[plugin_dir]`, so the plugin root is a real package and
  relative imports work. `plugin.yaml` must declare `kind: model-provider` to be
  discovered from the flat `$HERMES_HOME/plugins/<name>/` install location.
- `hermes_cli/auth.py:_register_plugin_provider` adds an `external_process` profile to
  `PROVIDER_REGISTRY`, which is what makes `auxiliary.<task>.provider` resolvable.
- `agent/auxiliary_client.py:_resolve_external_process_branch` is the **only** branch
  that honours `profile.create_client()`; the `api_key` branch always builds a plain
  `OpenAI(...)`. This is why the profile uses `auth_type="external_process"`.
- `resolve_external_process_provider_credentials` gates on a resolvable
  `process_command` (`shutil.which`). We set it to `python3` purely to satisfy that
  gate — the shim never spawns it, because `create_client()` supersedes the subprocess.
- `HERMES_SKIP_TRANSPORT_WRAP` / `HERMES_SKIP_ASYNC_WRAP` class attributes keep the shim
  out of the wire adapters (`_client_declares`).
- `call_llm` invokes `client.chat.completions.create(**kwargs)` and reads
  `.choices[0].message.content`.
- Aux call sites: `tools/approval_smart.py` (`task="approval"`), `tools/mcp_tool_sampling.py`
  (`task="mcp"`), `agent/context_compressor.py` (`task="compression"`).
  `skills_hub` is a registered aux key with **no call site** — deliberately out of scope.
- `agent/credential_pool.py:load_pool(provider)` + `CredentialPool.has_credentials()` /
  `select().runtime_api_key` is how Hermes itself reads an authenticated provider's key.
  It is the plugin's third credential source, so an operator's `hermes auth add openrouter`
  entry is used without duplicating the secret into `.env`. Note `hermes_cli.auth` exposes no
  API-key resolver for `openrouter` (it has a dedicated chat branch, not `auth_type="api_key"`),
  so the pool is the correct seam rather than `resolve_api_key_provider_credentials`.

## Testing

Install the dev extra from `pyproject.toml` (`[project.optional-dependencies] dev`),
then:

```bash
python -m pytest
```

The suite is offline: every test injects a fake transport, and no test makes a network
call. A live provider run needs explicit operator approval.

Fixtures live in `tests/conftest.py`. Fake transports record calls as
`{"url", "headers", "body", "timeout"}`.

The root `__init__.py` is a package entry point, so pytest imports it from outside the
Hermes runtime while collecting; that is why its `providers` import is guarded. Inside
Hermes the module is imported *by* `providers`, so registration always runs. Do not
"clean up" that guard — it is load-bearing for the test run.

## Delivery

- Remote is GitHub (`trajectoire-ai/hermes-structured-aux-models`), HTTPS only.
- Commit identity is `trajectoire-regis[bot] <325471804+trajectoire-regis[bot]@users.noreply.github.com>`.
- Authenticate Git with the profile-local `gh auth git-credential` helper. Never put a
  token in a remote URL.
- Commit often; do not merge or release without explicit authorization.
