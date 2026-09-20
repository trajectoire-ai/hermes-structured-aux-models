# hermes-structured-aux-models

A Hermes Agent **model-provider plugin** that routes selected auxiliary tasks through
bounded [Jev](https://openrouter.ai) **decision calls** on OpenRouter instead of
free-form chat prompts.

Hermes' auxiliary slots — compression, smart approval, MCP sampling — are normally
served by pointing `auxiliary.<task>` at a chat model. Weak models are unreliable there,
and pointing a *structured-output* model at a chat-shaped prompt is worse: the request
is still a chat completion, so the model still has to generate prose, and it still
fails in prose-shaped ways.

This plugin changes the request itself. It registers a provider, `structured-aux`,
whose client is an in-process shim. When Hermes' auxiliary client asks that provider for
a chat completion, the shim translates the request into a **decision request**: one
shared state plus questions whose entire answer set is enumerated in advance. The
provider returns a label; the shim returns it to Hermes as an ordinary completion.

A decision model cannot answer a question whose outcomes are not bounded. That is a
constraint, not a limitation to work around — and it is why the plugin **raises instead
of guessing** for anything it cannot express, letting Hermes fall back to your real
auxiliary provider.

## Scope

| Task | Contract | Status |
|---|---|---|
| `approval` | bounded `choice` over `APPROVE` / `DENY` / `ESCALATE` | supported |
| `mcp` | bounded `choice` over the MCP server's supplied tool names | supported, with a caveat (below) |
| `compression` | bounded `KEEP` / `DROP` per context block, then an extractive digest | supported |
| `skills_hub` | — | **out of scope** |

`skills_hub` is deliberately excluded. Hermes registers it as an auxiliary task and
lists it in the model picker, but nothing in the shipped code calls it — there is no
call site to intercept. It will be revisited if Hermes wires one.

## Install

```bash
hermes plugins install trajectoire-ai/hermes-structured-aux-models
```

The tree scans clean under Hermes' install-time plugin scanner
(`tools/plugin_guard.py`): verdict `safe`, no findings.

Confirm the provider actually registered. `hermes doctor` reports a **false** error for
model-provider plugins (`Plugin registration failed: no register() function`) — they
self-register at import and have no `register()`, so ignore that line. Check the way that
matters instead, in a fresh process under the target `HERMES_HOME`:

```bash
HERMES_HOME=<home> python -c "
from agent import auxiliary_client as A
c, m = A.resolve_provider_client('structured-aux', 'structured-aux/approval')
print(type(c).__name__, m, bool(getattr(c, 'api_key', '')))"
```

A `StructuredAuxClient` with a non-empty key is the proof. Anything else means the plugin
is not reachable from that home — see below.

### Where to install

Hermes discovers provider plugins **once per process, at process start**, by scanning
`$HERMES_HOME/plugins/`. Discovery is process-global and memoized: it does not re-run when a
session later switches profiles. So the plugin must live in the `HERMES_HOME` that the
**process making the auxiliary calls** starts with:

| Consuming process | `HERMES_HOME` at start | Install into |
|---|---|---|
| Profile-scoped CLI or messaging gateway (`hermes -p <p> …`) | `~/.hermes/profiles/<p>` | the profile: `hermes -p <p> plugins install …` |
| Machine-level desktop / remote backend (`hermes serve`, no profile) | `~/.hermes` (root) | the root home: `hermes plugins install …` |

The machine-level `hermes serve` backend is the easy one to get wrong: it is started once
with the **root** home and only binds a session's profile `HERMES_HOME` later, when it builds
that session's agent. A plugin installed only under `profiles/<p>/plugins/` is therefore
never imported by it — even though that same profile's CLI and gateway resolve it fine.

Symptoms of installing into the wrong home:

- `Auxiliary …: using openai-codex (…)` / any provider other than `using structured-aux (structured-aux/<task>)`;
- `Smart approvals: LLM call failed … (RuntimeError: Provider 'structured-aux' is set in config.yaml but no API key was found …), escalating`;
- no app page appears at `https://openrouter.ai/apps?url=https://trajectoire.ai`.

If one machine runs both a profile-scoped gateway **and** a machine-level `serve`, install in
**both** homes — and configure in both (see [Where to configure](#where-to-configure); install
location and configuration are separate decisions, and both are per home). Then restart every
long-lived process that makes auxiliary calls: discovery is memoized, so a process started
before the install keeps its old registry until it restarts.

## Configure

Point only the tasks you want at the provider. Each task's `model` names the contract:

```bash
hermes config set auxiliary.approval.provider structured-aux --force
hermes config set auxiliary.approval.model structured-aux/approval --force

hermes config set auxiliary.mcp.provider structured-aux --force
hermes config set auxiliary.mcp.model structured-aux/mcp --force

hermes config set auxiliary.compression.provider structured-aux --force
hermes config set auxiliary.compression.model structured-aux/compression --force
```

The `structured-aux/<task>` model string is how the shim knows which contract applies;
Hermes passes it through untouched. `structured-aux:approval`, `structured-aux.approval`
and `structured-aux-approval` are accepted too. The approval contract additionally
recognises Hermes' guardian prompt by its wording, so it still routes if you leave the
model blank.

That string is a **routing token, not a model id**. It never reaches OpenRouter — the
Jev model actually sent to the decisions endpoint is `decision_model` below.

### Where to configure

Configuration is per home, exactly like install. The mapping lives in that home's own
`config.yaml`, and Hermes resolves it **per auxiliary call** from whichever home is bound to
the caller — so the home you installed into is not necessarily the home that reads it:

| Caller | Home the mapping is read from |
|---|---|
| a conversation turn — tool use, smart approval, in-turn compaction | the session profile's home (bound for the turn) |
| work **outside** a turn — manual `/compress`, title generation, background review, memory flush | the **launcher** home the process started with |

The machine-level `hermes serve` backend is the easy one to get wrong here too. A manual
`/compress` is dispatched as an RPC on the launcher home, not inside a turn, so
`auxiliary.compression` is read from the **root** config (`~/.hermes/config.yaml`) — never from
the session profile's. With it unset there, compression resolves `provider: auto` and quietly
runs on your main model: the exact fallback this plugin exists to avoid. The tell is an
`Auxiliary compression: using <your main model>` line with no
`Auxiliary compression: using structured-aux (structured-aux/compression)` beside it. In-turn
tasks are unaffected — that is why smart approval can reach the plugin while a manual `/compress`
on the same session does not.

So on a host that runs both a profile-scoped gateway and a machine-level `serve`, run the block
above twice — once per home:

```bash
# launcher / root home (~/.hermes/config.yaml)
hermes config set auxiliary.compression.provider structured-aux --force
hermes config set auxiliary.compression.model structured-aux/compression --force

# session profile home (~/.hermes/profiles/<p>/config.yaml)
hermes -p <p> config set auxiliary.compression.provider structured-aux --force
hermes -p <p> config set auxiliary.compression.model structured-aux/compression --force
```

Repeat for `approval` and `mcp` if you want those routed in that home too.

### Credential

Only the OpenRouter credential is used. It is resolved in this order:

1. `OPENROUTER_API_KEY` in the process environment;
2. `OPENROUTER_API_KEY` in the active profile's `.env`;
3. Hermes' own `openrouter` credential entry (`agent.credential_pool.load_pool`).

Step 3 exists so an operator who already ran `hermes auth add openrouter` needs no duplicate
secret: the plugin bills the same key as the rest of Hermes. If none of the three resolves, the
provider reports "no OpenRouter credential is configured" and Hermes falls back to your real
auxiliary provider — a missing key is never fatal. No other credential is required, and none is
forwarded anywhere except the OpenRouter decisions endpoint.

### Plugin settings

Optional, under `plugins.entries.hermes-structured-aux-models.settings`:

| Key | Default | Meaning |
|---|---|---|
| `tasks` | `[approval, mcp, compression]` | which task keys this provider may serve |
| `decision_model` | `~typesafe/jev-latest` | Jev model id sent to OpenRouter. The `~` prefix marks a moving alias, so the default tracks the newest Jev release instead of a dated id that has to be bumped by hand. Pin an exact version here if you need reproducibility. |
| `decision_base_url` | `https://openrouter.ai` | must be HTTPS |
| `app_referer` | `https://trajectoire.ai` | OpenRouter `HTTP-Referer`. The app's primary attribution identifier. |
| `app_title` | `hermes-structured-aux` | OpenRouter `X-OpenRouter-Title`. The app's display name in rankings. |
| `timeout_seconds` | `15.0` | per decision request |
| `compression_max_blocks` | `48` | segmentation cap per compression prompt |
| `compression_blocks_per_call` | `16` | blocks asked about per provider call |
| `compression_output_budget_chars` | `6000` | digest size ceiling |
| `compression_min_block_chars` | `120` | minimum size before a block is closed |

### App attribution

Every decision request carries OpenRouter's [app
attribution](https://openrouter.ai/docs/app-attribution) headers: `HTTP-Referer:
https://trajectoire.ai` — the value OpenRouter keys the app's page and rankings on — and
`X-OpenRouter-Title: hermes-structured-aux`, its display name. Without a `HTTP-Referer`,
OpenRouter files the usage under "Unknown" and no app page is created. Both are plain,
non-secret strings; override them with `app_referer` and `app_title` above.

## Per-task behaviour

### approval

Hermes' smart-approval guardian asks for exactly one word and maps anything it does not
recognise to `escalate`. The contract therefore uses three labels, not two — collapsing
`ESCALATE` into `DENY` would silently turn "unknown risk" into a refusal. The guardian's
system prompt is forwarded as `guardian_policy` and the flagged command as `request`.

### mcp

The `mcp` auxiliary task serves **MCP sampling**: an MCP server asks Hermes to run a
completion, optionally supplying tools. When the server supplies at least two
*argument-free* tools, the plugin asks which one to call and returns a tool call.

**Caveat:** a decision returns a label, never synthesised arguments. Tools that declare
required parameters are therefore excluded from the candidate set rather than guessed
at, and if fewer than two answerable tools remain the request is handed back to your
real provider. This makes MCP support narrower than the other two tasks — it is a
genuine property of decision models, not a missing feature.

### compression

Hermes' compressor expects a prose summary. A decision model cannot write prose, so the
plugin does the other thing: it segments the prompt into deterministic blocks, asks
`KEEP`/`DROP` per block in bounded batches, and emits a digest assembled **only** from
retained original text plus a provenance header. Nothing is generated.

Two invariants matter here:

- Segmentation never loses text. When a prompt exceeds the block cap the overflow is
  merged into the final block rather than dropped.
- The digest is never empty. If the provider retains nothing, the final block is kept
  anyway — an empty result would make Hermes treat compression as failed and fall back
  to your main model, which is worse than retaining the most recent context.

## Cost posture

Only the tasks you explicitly point at this provider make provider calls. Nothing runs
per turn: approval fires only for commands Hermes already flagged, compression only at
compaction boundaries, MCP sampling only when a server asks. There is no background
supervision loop and no turn admission.

## Privacy

Every payload is redacted before it leaves the process, reusing Hermes' own redactor
(`agent.redact`) with a conservative local fallback. Decision receipts are not written
to disk by this plugin; nothing is persisted outside Hermes' own logs.

## Testing

Install the dev extra declared in `pyproject.toml`
(`[project.optional-dependencies] dev`), then:

```bash
python -m pytest
```

The suite is fully offline — every test injects a fake transport, and no test makes a
network call. A live provider run requires explicit operator approval.

Two things about the layout are worth knowing before you touch the tests:

- The repository root is a Python package (Hermes requires `__init__.py` there), so a
  test collector imports the plugin entry point from outside the Hermes runtime. That
  entry point therefore tolerates a missing `providers` module; inside Hermes the import
  always succeeds and registration always runs.
- `tests/conftest.py` holds every fixture. Fake transports record calls as
  `{"url", "headers", "body", "timeout"}`, so assert on `transport.calls[0]["body"]`.

## How it works

The interception point is a documented plugin surface, not a core patch:

1. `plugin.yaml` declares `kind: model-provider`, so `providers/__init__.py` discovers
   the plugin from `$HERMES_HOME/plugins/<name>/` and imports it with the plugin
   directory as the package root.
2. The module-level `register_provider(profile)` call registers `structured-aux` with
   `auth_type="external_process"`. `hermes_cli/auth.py` projects that into
   `PROVIDER_REGISTRY`, which is what makes `auxiliary.<task>.provider` resolvable.
3. `external_process` is the **only** auth type whose auxiliary branch consults
   `profile.create_client()`; the `api_key` branch always builds a plain HTTP client.
   That branch is therefore what lets an in-process shim answer auxiliary calls.
4. `HERMES_SKIP_TRANSPORT_WRAP` and `HERMES_SKIP_ASYNC_WRAP` keep the shim out of the
   Anthropic / Codex / Bedrock wire adapters.

The repository's agent instructions record each of these seams with the symbols to
re-verify, so a future maintainer can check them against a newer Hermes instead of
guessing.

## License

MIT — see `LICENSE`.
