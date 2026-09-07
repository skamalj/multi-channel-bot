# Architecture — decisions and why

Companion to `requirements.md`. Each section records a decision that is easy
to get wrong and expensive to reverse.

## 1. No interrupts anywhere - which is not the same as no checkpointer

Waiting is not a paused execution. On Lambda there is no in-flight run to
suspend — the process ends when the reply is sent and hours pass before the
next message. So "waiting for a job id / a vehicle number / an answer" is a
**field that is still empty**, and the router re-reads the session every turn.

This also removes a real bug class: LangGraph restarts the whole node on
resume, so any send *before* an `interrupt()` fires twice.

**Declining interrupts does not mean declining LangGraph.** The two are
independent: a graph compiles with a checkpointer and persists per
`thread_id` whether or not anything ever calls `interrupt()`. An earlier
version of this code took "no interrupts" as licence to hand-roll the session
store as well, and paid for it in ~100 lines of message serialisation, TTL
arithmetic and get/put that the checkpointer does for free. The graph is now
real; the interrupts are still absent, and every invocation runs start to
finish in one turn.

The same holds for a multi-question screening round. The next question is a
fold over a list, and the questions belong to the *product*, not to the code:

```python
todo = [q for q in job["screening_questions"] if q["key"] not in answers]
```

## 2. Two routers

| | Decides | Where | When |
|---|---|---|---|
| Resolver | `persona × lob → configuration` | `resolver/resolver.py` | once per turn, sticky |
| Journey router | which value stream this turn is | inside the agent | per turn |

One WhatsApp number serves every bot, so something must decide *which bot*
before anything can route an intent.

**Persona is never inferred from conversation.** It comes from the producer
directory every turn. Without that line, a long enough conversation becomes a
privilege-escalation path: drift the topic toward agent-shaped questions,
accumulate agent-shaped history, and let the prior grant what an
authorization check would have refused.

## 3. The route ledger

History is a signal, not just a record. A strong prior raises the bar for
leaving it:

| Situation | Threshold |
|---|---|
| No prior route | 0.60 |
| Prior route, nothing captured | 0.70 |
| Prior route, work in flight | 0.85 |
| Explicit statement or hard entity | bypass |
| Below the bar | ask |

`corrected_to` back-annotation turns every mis-route into labelled data that
nobody had to annotate — the golden set §8.2 asks for, for free.

Two limits on it, both deliberate. Only **weak** decisions are labelled —
history, holdings, the intent model — because a customer who says "motor" and
then says "health" changed their mind, and recording that as a mis-route
manufactures training data out of an ordinary conversation. And only the
**immediately** following turn may correct one, for the same reason.

## 4. Two session stores, keyed apart

Two LangGraph checkpointers, one per thread shape:

```
resolver graph   thread_id = user        ledger · shared profile · auth   180 d
bot graph        thread_id = user#lob    messages · slots · documents      30 d
audit            person#user             one record per turn, every LOB     7 y
```

The thread id IS the compartment boundary. There is no store class left in
this codebase: `app/memory/checkpoint.py` holds a `DynamoDBSaver` written
against the v4 `BaseCheckpointSaver` contract, and the graphs are compiled
with it. Partition key is the thread and sort key orders checkpoints within
it, so a thread's history is one query and a thread's erasure is one query
plus deletes - and ME-8 can promise erasure rather than approximate it by
writing an empty state over the top.

RS-9 - a persona switch carries no journey state - is satisfied by clearing
the journey fields on the switch rather than by adding persona to the key.
Keying by persona would give every customer three compartments to reason
about; clearing costs one `update_state` call on a switch that is rare.

**What must never be checkpointed.** Everything in a graph's state is
persisted, so per-turn context does not go there. The request identity, the
consent snapshot, the idempotency key and the trace travel in
`config["configurable"]`. Put the idempotency key in state and it returns on
the next turn, and the double-write protection inverts into a double-write
cause.

Holding one thread with LOB compartments inside it makes non-contamination
depend on every prompt-assembly path filtering correctly, forever. Keyed
apart, the leak requires deliberately constructing another session's id.

Three things follow: two TTLs (the ledger improves with age, the bot session
holds the most sensitive data), clean erasure, and bots that are written as
if each were the only bot.

**The rule for the shared profile:** facts *about* a line of business, never
facts *from* one. `holdings={"health": True}` yes; `has_diabetes` never.
`memory/profile.py` enforces it.

**Write order:** bot session first, ledger second. They are not atomic, and a
ledger one turn stale self-corrects, whereas a ledger pointing at a session
that never advanced strands the customer.

## 5. Three leak paths, not one

A motor turn's context comes from three places:

- **Session** — closed by the key.
- **Retrieval** — closed by the metadata pre-filter on `lob`, before ranking.
- **Tools** — closed by *neither*. `policy_get(customer_id)` returning every
  policy puts health cover into a motor prompt through the front door, with a
  clean session and a clean retriever. `coremock/store.py:get_policy()` takes
  `lob` and refuses out of scope.

## 6. Tags are build time

An agent *is* its tag set. At construction it lists the server's tools,
filters on its declared tags, and binds that set for life. Nothing about a
conversation changes it.

Consequence worth the whole design: `capability_matrix()` prints exactly what
each bot can *ever* call, with no conversation simulated. An agent whose tool
set is computed per turn cannot produce that document.

Adding a capability is therefore a versioned change — tag an existing tool, or
write a new tool carrying the tag — reviewable under the maker-checker control
§4 requires.

### Tags select, they do not authorize

| Question | Decided | By |
|---|---|---|
| May this agent ever call this tool? | build time | tags / manifest |
| May this call proceed, now, for this subject? | per call | `authorize()` + request context |

A model can name a tool it was never offered. Without the second check, a
correctly-built agent still reads another producer's book by passing a
different id. The two fail differently: the first is a design error, the
second is a breach.

## 7. One MCP server

One deployment, one Cloud Map entry, one CI pipeline, one middleware for auth,
idempotency, rate limiting and audit. Six servers is six connections and six
handshakes per Lambda cold start, paid on every conversation.

Divergence from the Salesforce-agent pattern this borrows from: there, the
agent composes SOQL and REST payloads and the tools execute them. For a CRM
that is right. For a regulated insurer it inverts the RFP — a generic
`execute_rest(payload)` over InsureMO hands the model the ability to compose
any core transaction. So the hosting shape stays and the grain changes:
narrow, typed, business-meaningful operations with their own schema,
authorization check and idempotency key.

## 8. The model proposes, the code commits

The model chooses *which* tool and fills *declared* fields. It never composes
a call, never computes a premium, never decides eligibility, and never decides
whether a journey is complete.

`coremock/rating.py` is the whole argument: a premium is arithmetic, so it is
testable, tunable and defensible six months later. `tests/test_rating.py`
asserts things a customer might one day ask you to explain — that NCB never
touches third-party premium, that GST applies once, that the floater discount
skips the eldest member.


## 9. The floor has to be absolute

The retrieval score started out normalised to the best hit in the query. That
is the natural thing to write and it silently disables the refusal path: the
best of five bad chunks always scores 1.0, so the floor never fires and the
bot composes a confident answer out of nothing.

So the score is absolute — a saturating BM25 plus a cosine — and the reranker
is a **multiplier** on it rather than an addition. Metadata modulates
relevance; it does not manufacture it. A chunk with exactly the right
doc_type and no term overlap still scores near zero, which is correct.

One more term earns its place: **query coverage**. A term the corpus has
never seen is the *most* informative term in a question, not the least, so
absent terms are weighted at maximum idf. "Does my plan cover a holiday in
Lisbon" is mostly about Lisbon, and a brochure matching "plan" and "cover"
has answered none of it. Without that, every question containing two common
insurance words retrieves something.

## 10. What a citation check is actually for

The naive rule — a citation per sentence — produces answers that are safe and
unreadable. It deletes headings, hedges, offers of help and the lead-in that
introduces a bulleted list, and what survives is a wall of cited fragments.

The rule that works is narrower: a **material claim** needs a source. A
figure, a percentage, a money amount, a date, a promise that something is
covered, excluded, free or unlimited. "It depends on when your policy
started" is not a material claim and citing it improves nothing.

Three things this file learned the hard way, each of which was a real answer
that reached the console:

- **A sentence with no tool behind it must be checked too.** Running the
  guardrail only on turns that retrieved leaves the obvious hole open: a
  model that skips retrieval and answers from general knowledge is exactly
  the answer the rule exists to catch.
- **A denied tool call is not evidence.** A refused `policy_get` counted as
  "a core tool ran" and licensed every sentence in the answer.
- **A core tool result is not a blanket exemption.** One true sentence about
  the premium was licensing every other sentence in the same answer, and a
  quote summary said "Maruti Swift VXI" over a lookup that had returned a
  Baleno. Every figure in an answer is now checked against the tool results
  behind it. Non-numeric drift in a sentence with no figures is the residual
  gap, and `requirements.md` says so rather than implying it is closed.

## 11. Controls, not instructions

Three behaviours started as sentences in the system prompt and had to become
code, because a model that does not read an instruction is not a bug you can
fix by writing the instruction more firmly:

| Wanted | As a prompt | As a control |
|---|---|---|
| Look before handing off | "search the sources first" | `human_handoff` is refused until a read tool has succeeded this turn |
| Use real product ids | "use the code from the catalogue" | the ids are an `enum` in the tool schema |
| Do not invent figures | "only quote tool results" | every figure is checked against the tool results |

The first two were found by running the same suite against a second model.
Nova and Kimi fail differently, and a control that only holds for the model
you developed against is not a control.


## 12. What walking a whole journey found

Everything in §10 came from knowledge questions. Driving an **issuance** end
to end in the console - quote, application, KYC, underwriting, payment, issue
- found a different class of bug, and a worse one.

- **The confirmation prompt was teaching the model to fake it.** "I am about
  to issue the policy. Shall I go ahead?" is written by `confirm.py`, not by
  the model. Replaying it as an assistant message taught the pattern, so the
  model started writing that sentence *instead of* calling the tool. The
  customer said yes, nothing was parked, and the journey stalled with the bot
  describing an action it never took. System-authored messages are now kept
  out of the model's context - the customer saw them, the audit records them,
  the model is not shown its own words that it never wrote.
- **"Any digit" was the wrong test for a material claim.** An application id
  and a numbered step both contain digits, so a bot walking somebody through
  issuance had every sentence treated as an unsourced claim and refused its
  way out of its own flow. A claim that hurts somebody is a *quantity* or a
  *promise*, not a character that happens to be a digit.
- **Facts the journey established were being thrown away.** A turn that ends
  in a confirmation had usually just looked something up. Dropping that meant
  the next turn could not check what the model said about it - which is how
  "Maruti Swift VXI" survived over a lookup that returned a Baleno.
- **A fabricated gate decision had no figure in it.** The model wrote
  "Underwriting Decision: CLEARED" without calling underwriting. The gate
  caught it two turns later, by which point the customer had been told the
  most consequential thing in the journey, wrongly. A decision belongs to the
  core and its vocabulary is closed, so those words are now claims.
- **The gate chain was missing its second half.** Booking an inspection only
  makes the gate `pending`, and nothing could ever clear it - a motor policy
  could not be issued at all. That was an absent operation, not strictness.
  `inspection_result` is producer-only, because a customer cannot clear their
  own inspection.
- **An application id was not checked against its owner.** Same class as the
  producer id: having the tool bound is not permission to act on somebody
  else's application.

The pattern across all six: a control tested only on the path you designed it
for is a control you have not tested.


## 13. What real DynamoDB found that an in-memory double could not

The checkpointer's logic was covered by tests against an in-memory table
double, and every one of them passed before it had executed a single
statement against AWS. The first real run found two defects, and both are
the kind a double cannot have.

- **Queries were not paginated.** DynamoDB caps a query response at 1 MB, so
  an unpaginated read reported a thread erased while rows remained, and
  showed a partial history as if it were the whole one. `erase()` returned
  `bot_session: 2` with fourteen rows still in the table. `_query_prefix` now
  follows `LastEvaluatedKey`, and `delete_thread` loops until the thread is
  empty. Note that this is a **safety net, not the fix** - see §14.
- **Reads were eventually consistent.** The default. For a checkpointer that
  means a turn can read a stale checkpoint and lose the previous one - the
  customer says "yes" and the confirmation they were answering is not there
  yet. Both read paths now pass `ConsistentRead=True`.

The double was then taught to page by row count, so the truncation is
reproducible in the offline suite without writing a megabyte. That is the
useful shape: the double proves the logic, the real service proves the
assumptions, and the fix goes back into the double so it stays proved.

A third, smaller finding came from the same run. LangGraph warns when it
deserialises a type it does not know, and will block it in a future version -
rightly, since a checkpoint that can rehydrate arbitrary classes is a
deserialisation gadget. Its permissive default cannot be added to, so the
answer was not to register `ResolverSession` but to stop persisting a class
at all: the resolver thread holds `model_dump()` and hydrates on read. The
checkpoint holds data, and a dependency bump cannot break it.


## 14. A thread that never approaches the cap

Pagination stops a large thread being read wrongly. It does not make a large
thread a good idea. For a text conversation 1 MB is enormous, so the design
intent is that a checkpoint never gets close, and two rules do that.

**Documents never enter the message list.** `storage/documents.py` writes the
bytes to S3 under `documents/{user}/{lob}/{sha256}` - addressed by content, so
a resend is not a second copy - and the thread gets one line:

```
[document] policy_schedule.pdf - application/pdf, 214 KB, ref DOC-3f9a12c4
```

One scanned schedule inline would exceed the cap on its own. It is also the
same argument as the rating engine: a tool that needs the content fetches it
by reference, and the model names the document rather than carrying it.

**Old turns are reduced.** `agents/reduce.py` runs `agentstate-reducer` as a
graph node between `entry` and the rest, pruning to a recent window and
folding what it drops into a rolling summary on the small model.

The reason for that library rather than a hand-rolled window is one setting:
`cascade_tool_messages`. An AI message carrying `tool_calls` and its
`ToolMessage` results are a single unit - drop the call, keep the result, and
Bedrock rejects the whole turn with "toolResult blocks exceeds toolUse
blocks". A naive "keep the last N" gets that wrong about half the time, and
it is the same error this codebase already hit once when replaying a
confirmation. It also has no dependencies at all, which matters on a
codebase whose claim is that nothing is vendor-specific: the alternative
prebuilt node pulled in fifty packages including two LLM vendor SDKs.

The summary is injected as a `system_authored` pair, so it is context the
model reads and never words the model is told it said - the same rule as the
confirmation prompt in §12. And it is context, never evidence: the citation
guardrail still applies to every answer composed after it.

Measured on a real conversation: ten turns against Bedrock and DynamoDB
leaves a 27 KB thread, bounded, against a 1,048,576-byte cap.
