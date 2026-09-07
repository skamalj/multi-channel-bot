# multi-channel-bot

Backend for the Protec bot demo: **BOT-05 Health Customer**, **BOT-02 Motor
Agent** and **BOT-06 Motor Customer** on one platform.

Built backend-first and channel-agnostic. The test surface is a **web chat
with a glass-box trace**; WhatsApp is bolted on later by registering an
adapter — nothing below the channel layer changes.

---

## Run it

Package management is `uv`.

```bash
uv sync --group dev
cp .env.example .env
```

Credentials are **never** read from `.env`. boto3 uses its default chain, so
whatever your shell already has — `AWS_PROFILE`, SSO, env vars, an instance
role — is what gets used.

```bash
uv run python scripts/build_corpus.py     # 74 documents from products.yaml
uv run uvicorn app.main:app --port 8000   # http://localhost:8000
```

`NO_AWS=1` (the default in `.env.example`) keeps every store in memory, so
only Bedrock needs credentials. To use DynamoDB and S3 instead:

```bash
uv run python scripts/preflight.py        # tells you exactly what is missing
uv run python scripts/bootstrap_aws.py    # 4 DynamoDB tables + 1 S3 bucket
```

No AWS at all? Both halves stub independently:

```bash
MOCK_LLM=1 NO_AWS=1 uv run uvicorn app.main:app --port 8000
```

### Models

The default is **Kimi K2.5** on Bedrock, with **Nova Lite** for the small
classifier and reranker calls. Nothing in the code is vendor-specific — the
agent loop is the Converse API with `toolConfig` — and the live test suite
passes on both:

| Model | live suite | p95 turn |
|---|---|---|
| `moonshotai.kimi-k2.5` | 11/11, stable over repeated runs | ~2.4 s |
| `apac.amazon.nova-pro-v1:0` | 11/11 on 2 runs in 3 | ~3.0 s |

Swap with `BEDROCK_MODEL_ID`. `scripts/preflight.py` lists what your region
actually exposes.

The intermittent Nova failure is a *capability* assertion — whether the model
chooses to search the corpus on a given phrasing — not a safety one. Nova
declines more readily than Kimi, and declining is a safe outcome. The
assertions that must never fail are the ones about what reaches the customer
(`test_a_knowledge_answer_is_either_grounded_or_a_refusal`,
`test_an_unanswerable_question_refuses_rather_than_inventing`), and those have
held on every run of both models.

Running the suite against a **second** model, and walking a whole issuance in
the console, is what found most of the bugs in `docs/architecture.md` §10-11.
A control that only holds for the model you developed against - or only on
the happy path of a knowledge question - is not a control.

---

## Test it

```bash
uv run pytest -m "not live"      # 151 tests, no AWS, no Bedrock, ~5 s
uv run pytest -m live            # 11 tests against real Bedrock
```

The offline suite is not a mock of the system — it drives `POST /api/chat`
through the real orchestrator, resolver, tool registry and guardrails, with
only the model and the stores swapped. The live suite is the part that can
only be true against a real model: that the generated tool schemas are
accepted, that the model calls the right tool, that citations survive a real
generation, and that a turn fits the latency budget.

---

## What to try in the console

| Type this | What it demonstrates |
|---|---|
| `hi` | Ambiguous first turn — the intent model has no opinion, so the resolver **asks** |
| `what is the waiting period for a pre-existing disease` | LOB from an entity; a cited answer, with the **rejected** chunks and their scores shown beside it |
| `and what about maternity` | The prior holds through an ambiguous turn |
| `actually my car renewal is due` | Mid-conversation switch; a different bot, a different tool set, the health thread untouched |
| `will you reimburse my Lisbon holiday and my new laptop` | Below the retrieval floor — it refuses and offers a human rather than composing something confident |
| `grant consent`, then `please give me a health quote` | A mutating tool is **proposed**, not performed |
| `yes` | Confirmed — and the confirmation token *is* the idempotency key |
| **capabilities** | Exactly what each bot can *ever* call — build time, no conversation simulated |
| **memory** | Every store, its key, TTL, sensitivity and erasure handler |
| **audit** | One record per turn, keyed by person, spanning every line of business |

Set the user id to `919820000001` to be recognised as a producer by the
directory stub; `919820000009` holds both a health and a motor policy;
`919820000002` holds only health, which is what makes holdings-based routing
visible. Anything else is an unknown customer.

---

## Layout

```
app/
  main.py            FastAPI: /api/chat, /api/bots, /api/tools, /api/memory,
                     /api/audit, /api/erase, /api/consent, /api/stepup
  orchestrator.py    one turn end to end: channel → resolver → session → agent → audit
  channels/          IngestEvent + webchat (live) + whatsapp (parsing only)
  resolver/          resolver GRAPH (thread = user); route ledger; step-up
  agents/            bot GRAPH (thread = user#lob), citations, confirmation
  mcpserver/         ONE server, tagged tools, build-time filter, server authz
  knowledge/         corpus generation, hybrid retrieval, rerank, floor
  coremock/          InsureMO-shaped mock: catalogue, rating, gates, claims
  memory/            two checkpointers, shared profile, long-term stores, registry
  llm/               Bedrock binding, intent classifier, scripted stub
  obs/               trace events and the person-keyed audit store
data/products.yaml   single source of truth for every product figure
data/generated/      74 documents + chunks.json, rendered from products.yaml
web/index.html       chat + glass box
scripts/             preflight, bootstrap, build_corpus
docs/architecture.md the decisions and why
```

---

## The five rules this code is built around

1. **The model proposes, deterministic code commits.** A premium comes from
   `coremock/rating.py`, never from a prompt. `tests/test_rating.py` is why.
2. **Tags are build time.** An agent *is* its tag set; tools are bound once at
   construction. That is what makes `capability_matrix()` printable.
3. **Tags select, they do not authorize.** `authorize()` re-checks every call
   *with its arguments*, because a model can name a tool it was never offered
   — and can name somebody else's producer id.
4. **Threads are keyed apart.** Two LangGraph checkpointers: the resolver
   thread is `user`, the bot thread is `user#lob`. The thread id IS the
   compartment boundary — isolation is a fact about addressing, not a
   function someone must remember to call. Per-turn context (identity,
   consent, the idempotency key) travels in `configurable` and is never
   checkpointed.
5. **An unsupported claim does not reach the customer.** Every answer is
   checked: material claims need a retrieved chunk, and figures need to match
   the tool result behind them. An empty answer is a correct answer.

---

## Not built yet

See `requirements.md` for the full list and the known limits.

- WhatsApp outbound and webhook signature checking (parsing is done)
- A real vector store — retrieval is BM25 + TF-IDF over the generated corpus
- Document AI / IDP, vision, payments, KYC provider, e-sign
- The DynamoDB checkpoint saver has not yet run against real AWS (its logic
  is covered against an in-memory table double; `NO_AWS=1` uses `InMemorySaver`)
