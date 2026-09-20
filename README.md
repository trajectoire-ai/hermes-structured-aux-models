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

### Compression window

Hermes resolves the compression model's context window on its own — from a catalog, a live
`/models` probe, or the persistent context-length cache — and it never consults the session
model's window for that. `structured-aux/compression` is a **routing token**, not a catalog
model, so every resolution step misses and Hermes falls back to a generic **256,000**-token
assumption.

That number matters more than it looks. Hermes' default compaction threshold is **50 % of the
main model's window** (524,288 tokens for a 1,048,576-token main model), and
`check_compression_model_feasibility` refuses to leave a compression model smaller than the
threshold. With 256,000 it prints

```
Auxiliary compression model structured-aux/compression has 256000 token context, below the
main model's compression threshold of 524288 tokens — auto-lowered session threshold to 256000
```

and **halves the session's compaction threshold for its whole lifetime** — the session then
compacts roughly twice as often, each time retaining a smaller window. Nothing is broken; the
plugin is asked for a summary it can serve. The session just carries less context than it
should.

The route chunks — every decision call stays inside Jev's 32,000-token window, and a prompt of
any size becomes as many calls as it needs — so it accepts a **full session-window prompt**.
Declare that explicitly, in every home whose agents do the compacting:

```bash
hermes -p <p> config set auxiliary.compression.context_length 1048576 --force
```

One value, no per-model bookkeeping: use the main model's window (it is what the threshold is
derived from, so `aux >= main` always clears the check). `--force` because the key is not in the
shipped config schema. The equivalent through Hermes' documented override surface is
`model_overrides.structured-aux['structured-aux/compression'].context_window`.

Verify it took, in a fresh process under that home:

```bash
HERMES_HOME=<home> python -c "
from agent.model_metadata import get_model_context_length
print(get_model_context_length('structured-aux/compression',
      base_url='https://openrouter.ai', provider='structured-aux'))"
```

`1048576`, not `256000`. A long-lived session keeps the threshold it already computed until its
agent is rebuilt, so expect the new value on the next session (or after a restart), not
mid-conversation.

### Credential

Only the OpenRouter credential is used. It is resolved in this order:

1. `OPENROUTER_API_KEY` in the process environment;
2. `OPENROUTER_API_KEY` in the active profile's `.env`;
3. Hermes' own credential entry (`agent.credential_pool.load_pool`) for the provider named by
   `credential_pool_provider` — `openrouter` unless you say otherwise.

Step 3 exists so an operator who already ran `hermes auth add openrouter` needs no duplicate
secret: the plugin bills the same key as the rest of Hermes. That lookup is keyed on the provider
name, so an install holding the OpenRouter key under a different name sets
`credential_pool_provider` to it; otherwise the lookup finds nothing and looks exactly like "no
credential configured". If none of the three resolves, the
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
| `credential_pool_provider` | `openrouter` | provider key whose Hermes credential-pool entry supplies the decision credential, when `OPENROUTER_API_KEY` is not set in the environment or the profile `.env` |
| `timeout_seconds` | `15.0` | per decision request |
| `decision_max_attempts` | `3` | total attempts for one decision request, first try included; a transient failure is retried, a payload/credential/contract failure is not |
| `decision_retry_backoff_seconds` | `0.5` | base delay before a retry; doubles per attempt (0.5 s, then 1.0 s) |
| `compression_max_blocks` | `48` | segmentation cap per compression prompt |
| `compression_blocks_per_call` | `16` | blocks asked about per provider call |
| `compression_call_budget_tokens` | `8000` | estimated-token ceiling for one decision call; capped at `28000` so no call can exceed Jev's 32,000-token window |
| `compression_output_budget_chars` | `18000` | character ceiling for the assembled digest. The digest is the concatenation of every retained chunk, so it is not bounded by one call's window — see [Compression window](#compression-window) |
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

Three invariants matter here:

- Segmentation never loses text. When a prompt exceeds the block cap the overflow is
  merged into the final block rather than dropped.
- No call exceeds the call budget. Jev's context window is **32,000 tokens** (OpenRouter
  states it on the model page) and the decisions endpoint is not a chat endpoint, so a
  compaction prompt cannot be handed over whole. A block larger than
  `compression_call_budget_tokens` is split at a paragraph, then a line, boundary; pieces
  are packed into however many calls the budget requires; and a block is retained if **any**
  of its pieces is, because retaining the whole block is the safe reading of a partial
  answer. Nothing is truncated to make something fit: a merged block that is the whole
  transcript becomes a couple of dozen calls, each inside the window.
- The digest is never empty. If the provider retains nothing, the final block is kept
  anyway — an empty result would make Hermes treat compression as failed and fall back
  to your main model, which is worse than retaining the most recent context.
- The digest budget bounds the **sum** of the retained blocks, not one call's payload. Each
  call is a separate decision, so the digest is assembled from however many chunks came back;
  `compression_output_budget_chars` (18,000 by default) only has to stay smaller than the
  turns it replaces, and Hermes injects the result as the session's context summary.

## Retries and logging

A decision request that fails **transiently** — a connection error, a read timeout, or a
retryable HTTP status (`408, 425, 429, 5xx`) — is retried up to `decision_max_attempts` (3 by
default) with a doubling backoff (`0.5 s`, then `1.0 s`). Failures the provider has already
judged are **not** retried: any other HTTP status, a non-JSON body, missing answers, or an
out-of-contract label. Repeating those only repeats the same answer.

The retry exists because Hermes will not do it for you. A compression call is critical-path
work, so on a full-budget timeout Hermes **skips its own same-provider retry and falls back to
your main model** — one stalled decision call turns a compaction into a prose summary and
loses the extractive guarantee. A second attempt inside the plugin is the cheap way to keep
the digest.

Failures are logged with enough context to find them, and no payload or credential is ever in
the message:

| Line | Level | Says |
|---|---|---|
| `decision call failed transiently: attempt 1/3 after 15041 ms (model=…, questions=16, status=-), retrying in 0.5 s: …` | WARNING | one attempt of three failed and why |
| `decision call recovered on attempt 2/3 after 812 ms …` | WARNING | the retry worked — this call was slow/flaky and is worth knowing about |
| `decision call slow: 7400 ms for attempt 1/3 …` | WARNING | a call over half its timeout that still succeeded: the shape of a stall before it costs you the digest |
| `decision call failed: attempt 3/3 after 45032 ms (model=…, questions=16, status=503): …` | ERROR | all attempts spent; Hermes will fall back |
| `compression plan: 3 block(s), 3 piece(s), 3 decision call(s), ~21000 estimated payload tokens` | DEBUG | how many calls this compaction will take |
| `compression decision call 2/3 failed (16 question(s), ~8000 estimated payload tokens): …` | ERROR | **which** call of the batch failed |
| `decision call answered in 412 ms on attempt 1/3 …` | DEBUG | per-call latency, for the healthy case |

WARNING and ERROR land in `errors.log` (and `agent.log`); DEBUG needs a verbose log level
(`hermes logs --level debug`). `hermes logs --follow` while a compaction runs is the way to
watch a flaky provider.

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
