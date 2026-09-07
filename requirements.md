# Requirements — multi-channel-bot backend

Scope: the backend platform and three bot configurations, testable through a
web chat. Traceability column refers to the Protec RFP (PROTEC-RFP-BOTS-2026-01).

Status: `done` in this repo · `stub` wired with a placeholder · `todo` not started.

**Verified** (2026-09-06):

| | |
|---|---|
| `uv run pytest -m "not live"` | 164 tests, no AWS, no Bedrock, ~8 s |
| the same suite, real DynamoDB + S3 | 164/164 with `NO_AWS=0` |
| `uv run pytest -m live` | 11 tests against real Bedrock |
| Kimi K2.5 (`moonshotai.kimi-k2.5`) | 11/11, stable across repeated runs, p95 2.4 s |
| Nova Pro (`apac.amazon.nova-pro-v1:0`) | 11/11 on 2 runs in 3, p95 3.0 s |
| Console, driven end to end | knowledge, LOB switch, refusal, and a full health issuance |
| Real AWS | 4 DynamoDB tables + S3 bucket created; saver verified against both |

The offline suite is not a mock of the system: it drives `POST /api/chat`
through the real orchestrator, resolver, tool registry and guardrails, with
only the model and the stores swapped.

---

## 1. Channel

| ID | Requirement | Status | RFP |
|---|---|---|---|
| CH-1 | Every inbound message normalises to a channel-agnostic `IngestEvent` before anything else sees it | done | §4 Channels |
| CH-2 | Caption and text body normalise to a single `text` field so routing has one input | done | §4 Conversation |
| CH-3 | Web chat adapter returns replies in the HTTP response for testing | done | — |
| CH-4 | WhatsApp payload parsing: all three arrays iterated; `statuses[]` distinguished from `messages[]` | done | §4 Channels |
| CH-5 | WhatsApp outbound: `to` copied verbatim from `from`; sender is `phone_number_id` from the same payload | todo — deferred with the rest of WhatsApp | §4 |
| CH-6 | Webhook signature verified over the **raw** body; ack within 300 ms; dedupe on message id | todo — deferred with the rest of WhatsApp | §7 Security |

## 2. Resolver

| ID | Requirement | Status | RFP |
|---|---|---|---|
| RS-1 | `persona × lob → configuration`; the bots are a 2-D lookup, not N applications | done | §3 |
| RS-2 | Persona from a producer-directory lookup on identity — **never** inferred from conversation | done | §3, §7 |
| RS-3 | LOB decided by: entry context → explicit entity → holdings → route-ledger prior → intent model → ask | done — the intent model is a small Bedrock call that fails closed | §4 |
| RS-4 | Route ledger persisted per user; decisions carry `decided_by`, confidence and evidence | done | §7 Analytics |
| RS-5 | Asymmetric thresholds: establish 0.60, leave idle route 0.70, leave in-flight journey 0.85 | done | — |
| RS-6 | Below threshold the bot asks rather than routes | done | §4 |
| RS-7 | `corrected_to` back-annotation when the next turn proves a mis-route | done — only weak decisions, only on the next turn; writes to the learning store | §8.2 golden sets |
| RS-8 | Mid-conversation LOB switch preserves the other session untouched | done | §3 |
| RS-9 | Persona switch requires step-up auth and carries no journey state | done — and an entry link can never *grant* a persona the directory refuses | §7 |

## 3. Agent and tools

| ID | Requirement | Status | RFP |
|---|---|---|---|
| AG-1 | ONE MCP server hosting every tool | done — in-process registry plus a real MCP stdio server over the same manifest | — |
| AG-2 | Tools carry selection tags (`lob`, `persona`) and behaviour metadata (`effect`, `authority`, `pii`, `consent_purpose`, `auth`) | done — plus `subject` for entitlement | §6 |
| AG-3 | Tool binding happens once at agent construction from the tag manifest — never per turn | done | §6 |
| AG-4 | A static, printable capability matrix per bot | done — `GET /api/bots`, rendered in the console | §6, §9.3 |
| AG-5 | Server-side authorization on every call, independent of what was bound | done — takes the **arguments** too: producer id, policy id and application id are each checked against the caller, so "may you call it for THAT subject" is a separate question | §6 |
| AG-6 | Mutating tools require confirmation and an idempotency key | done — the confirmation token *is* the idempotency key, and the system-written prompt is kept out of the model's context so it cannot imitate the template instead of calling the tool | §4 Transactions |
| AG-7 | Tool results are LOB-scoped — the tool is the third leak path after sessions and retrieval | done | §3 |
| AG-8 | Tool-round limit with a human-handoff fallback | done — and a handoff before any lookup has succeeded is refused | §4 Handoff |
| AG-9 | Narrow typed business operations, never a generic `execute_rest(payload)` | done — closed sets are `enum`s in the schema; entitlements (`_scopes`, `_customer_id`, `_user_id`, the idempotency key) are injected, never modelled | §6 |

## 4. Knowledge and RAG

| ID | Requirement | Status | RFP |
|---|---|---|---|
| KB-1 | Chunk metadata: product, lob, doc_type, authority, scope, version, effective dates, section, page, source | done | §6 RAG |
| KB-2 | Metadata pre-filter on lob and scope **before** ranking — entitlement is structural | done — the corpus scope is injected from the bound agent, so a model cannot ask for a scope it was not given | §6 |
| KB-3 | Hybrid retrieval with reranking | done — BM25 + TF-IDF cosine, RRF fusion, a deterministic feature reranker modulating an absolute score, and query-coverage weighting (optional LLM reranker behind a flag) | §6 |
| KB-4 | Every answer cites at least one chunk; uncited claims dropped before composing | done — on **every** answer, including one produced with no tool call. A material claim is a quantity or a promise, and a gate decision counts as a promise | §6 |
| KB-5 | Refusal path below the score threshold, with handoff offered | done — the score is absolute, so the floor actually fires; an action-shaped turn refuses in action words rather than claiming a knowledge gap | §6 |
| KB-6 | Two effective-dated versions of one wording, answered per policy date | done — PHS V1 (PED 48m) and V2 (PED 36m); an ambiguous versionless citation is not silently resolved to one of them | §3 |
| KB-7 | Corpus generated from `products.yaml` so no two documents disagree | done — 74 documents, 197 chunks, `scripts/build_corpus.py` | — |

## 5. Memory

| ID | Requirement | Status | RFP |
|---|---|---|---|
| ME-1 | Resolver session keyed `user` — ledger, shared profile, auth state | done — a LangGraph checkpointer on the resolver graph, `thread_id = user` | §3 |
| ME-2 | Bot session keyed `user#lob` — messages, slots, document **metadata** | done — a second LangGraph checkpointer on the bot graph, `thread_id = user#lob`; RS-9 clears the journey on a persona switch rather than adding a third compartment | §3 |
| ME-3 | Shared profile carries facts *about* a line of business, never *from* one (enforced) | done | §7 Privacy |
| ME-4 | Bot session written before the resolver ledger — the two writes are not atomic | done — two invocations, and each checkpoint is written when its own returns | §4 |
| ME-5 | Separate TTLs: bot session 30 d, resolver 180 d, audit 7 y | done — one saver per table, TTL per store | §7 Retention |
| ME-6 | Long-term stores: consent ledger, suppression, interaction summary, LOB profile, producer profile, learning store | done | §7 |
| ME-7 | Memory writes typed and evidenced — never model inference about a person | done — `model_inference` raises rather than being dropped quietly | §6 |
| ME-9 | A thread stays inside what a checkpoint can hold: documents to object storage, old turns reduced to a rolling summary | done — `storage/documents.py` and `agents/reduce.py`; ten live turns leave a 27 KB thread against a 1 MB cap | §7 |
| ME-8 | Memory registry: every store declared with key, TTL, sensitivity, erasure handler | done — `GET /api/memory`, `POST /api/erase/{user}`; the bot and resolver handlers call the checkpointer's own `delete_thread` | §7 |

## 6. Core (mocked)

| ID | Requirement | Status | RFP |
|---|---|---|---|
| CO-1 | InsureMO-shaped interfaces: product, rating, quote, policy, endorsement, claims | done — all six, with idempotency on every write | §5.x.3 |
| CO-2 | Premium computed by a deterministic rating engine, never by the model | done — verified end to end: every figure in a live quote matches `rating.py` | §3 |
| CO-3 | Quotes carry a validity date; an expired quote is re-priced, never resurrected | done | §4 |
| CO-4 | Realistic behaviours: latency, pending states, a 429, one "not available" interface | done — endorsement is the unavailable one | §4 |
| CO-5 | Gate chain — KYC, UW/inspection, payment — enforced before issuance | done — the refusal names the blocking gate, and `inspection_result` closes the motor chain (producer-only: a customer cannot clear their own inspection) | §4 |

## 7. Observability

| ID | Requirement | Status | RFP |
|---|---|---|---|
| OB-1 | Every turn emits a trace: resolution, binding, retrieval, tool calls, gates, guardrails, model and prompt version | done | §7 |
| OB-2 | Glass-box console renders the trace beside the conversation | done | §7, §8.3 |
| OB-3 | Audit store keyed by person, spanning every LOB — separation governs runtime context, not the record | done — `GET /api/audit/{user}`; not erased by a subject request, and the report says so | §7 |
| OB-4 | Retrieval scores shown including **rejected** chunks | done — with the reason each was rejected | §6 |

## 8. Non-functional

| ID | Requirement | Status |
|---|---|---|
| NF-1 | No credentials in the repo; boto3 default chain only | done |
| NF-2 | Runs without AWS (`NO_AWS=1`) and without Bedrock (`MOCK_LLM=1`) for wiring tests | done |
| NF-3 | Rating, tag filtering and routing covered by tests that need no cloud | done — 164 of them, and the same 164 pass against real DynamoDB and S3 |
| NF-4 | Config effective-dated and versioned; `PROMPT_VERSION` in every trace | done — `config_version` is read from `products.yaml`, so a catalogue edit cannot silently detach from the answers it produced |
| NF-5 | p95 turn under 20 s with a real Bedrock call | done — Kimi K2.5 p95 ≈ 2.4 s, Nova Pro p95 ≈ 3.0 s over ten turns |
| NF-6 | Nothing in the design is vendor-specific | done — the same suite passes on two unrelated model families; running it against the second one is what exposed three "controls" that only held for the first |

---

## 9. Journeys verified end to end

Not a claim that the code exists — a record of what was actually driven
through `POST /api/chat` against real Bedrock.

| Journey | Outcome |
|---|---|
| Ambiguous first turn | The intent model returns no opinion; the resolver asks |
| Knowledge question | Cited answer; accepted **and** rejected chunks with scores in the trace |
| Ambiguous follow-up | The prior holds |
| Mid-conversation LOB switch | A different bot, a different tool set, the health thread untouched |
| Out-of-corpus question | Below the floor: refuses and offers a human |
| Health issuance, complete | quote → application → KYC → underwriting → payment → **policy issued**; every step confirmation-gated, premium ₹29,240.40 matching `rating.py` |
| Motor issuance | Blocks at pre-inspection for a customer, by design; completes for a producer once the surveyor reports |
| Erasure | Every store reported; audit reported as retained |

Four things the guardrail caught in a **live** conversation, each of which
would otherwise have reached a customer:

- a fabricated policy issuance — "Policy ID: POL-2025-8847361 … you are covered" — with no tool call behind it
- a fabricated underwriting decision — "Underwriting Decision: CLEARED"
- a wrong vehicle — "Maruti Swift VXI" over a lookup that returned a Baleno
- an invented benefit on an out-of-corpus question

---

## Known limits

Named so nobody assumes otherwise.

- **WhatsApp outbound and webhook security** (CH-5, CH-6) are deliberately
  deferred. Inbound parsing is real and tested.
- **The retrieval index is in-process** — BM25 and a TF-IDF cosine over 197
  chunks. The metadata contract and the pre-filter are the parts that matter,
  and they are what a real vector store would inherit; the ranking itself
  would be replaced.
- **Uncited *procedural* prose can still reach the customer.** The guardrail
  refuses a material claim — a figure, a money amount, a promise of cover, a
  gate decision — with nothing behind it. It does not refuse "get admitted to
  a network hospital, then submit a pre-authorisation" written from the
  model's own knowledge. Separating "here is how a claim works" from "here is
  what I need from you" well enough to refuse the first and keep the second
  is not something a sentence test does reliably, and refusing both makes the
  bot unusable. Observed on Nova; named here rather than implied to be closed.
- **Non-numeric drift is narrowed, not eliminated.** Every figure in an
  answer is checked against the tool results behind it — including facts the
  journey established on earlier turns — which catches premiums, IDVs, years
  and gate states. A wrong proper noun in a sentence with no figure at all can
  still pass. Closing it needs the responder to render core fields itself
  rather than let the model restate them. A proper-noun filter was considered
  and rejected: the stoplist misfires on customer names, cities and hospitals.
- **Model capability varies, and the suite shows it.** Nova declines to search
  the corpus on some phrasings where Kimi searches, so one *capability*
  assertion fails on roughly one Nova run in three. Declining is a safe
  outcome; the assertions about what reaches the customer have held on every
  run of both models.
- **No document AI, vision, payments, KYC provider, or e-sign.** The gate
  chain is real; the integrations behind each gate are mocked.
- **The live Bedrock suite runs on in-memory stores.** `NO_AWS=1` in its
  fixture, so it varies one thing at a time: the model. The full offline
  suite covers the DynamoDB path separately with `NO_AWS=0`, and both have
  been run green - but no single run exercises real stores and a real model
  together beyond the manual end-to-end conversations.
