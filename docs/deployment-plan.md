# Deployment plan — AWS build

Against `builds-requirements.md`. Everything below is a change of *substrate*,
not of behaviour: `orchestrator.handle()` stays the core, the two checkpointers
stay keyed apart, and the 164 offline tests must stay green throughout.

## Decisions taken (and what they override)

| # | Decision | Supersedes |
| --- | --- | --- |
| D-1 | Caller identity to the tool Lambda is a **JWT**, verified at the Gateway | — |
| D-2 | Redshift reached with **psycopg**, not the Data API. All 22 record/transaction tools move; RAG does not | — |
| D-3 | **FastAPI for both** agent and console, one codebase, two modes | — |
| D-4 | KB is **semantic only, top 5, no reranking** | KB-3, KB-5, OB-4 |
| D-5 | Console talks to the Runtime over **AG-UI** (SSE), not a bespoke JSON reply | — |
| D-6 | Redshift credential in **SSM Parameter Store**, not Secrets Manager | — |
| D-7 | **GitHub Actions deploys everything**, via OIDC. No CodeBuild, no local Docker | — |

## Phase 0 — spike, before any restructuring

Four unknowns that would each invalidate a phase if discovered late.

1. **JWT reaches the Lambda.** Gateway with inbound JWT auth, one trivial
   target, assert the claims arrive in the Lambda event. If they do not, the
   `authorize()` contract has no subject to check and Phase 2 changes shape.
2. **AG-UI SSE survives the wire.** A stub AG-UI runtime emitting five events,
   invoked from a local console through SSO credentials. Assert the events
   arrive *incrementally*, not buffered into one chunk at the end.
3. **Redshift from a Lambda with no NAT.** Lambda in the workgroup's VPC, S3
   and DynamoDB via gateway endpoints. Assert `SELECT 1`, and assert no NAT
   Gateway was created — this is the one item that silently turns a
   sub-$1/month demo into a $35/month one.
4. **The Actions pipeline works end to end.** Repository created, bootstrap
   stack applied, OIDC assume-role succeeding from a workflow, and a trivial
   ARM64 image cross-built, pushed to ECR and launched as a Runtime. Prove the
   pipeline while there is nothing complicated in it.

## Phase 1 — Redshift Serverless

- Namespace + workgroup, base RPU set to the **lowest the region allows**.
- Schema and seed generated from the existing `products.yaml` and
  `coremock/store.py` fixtures, so the data the tools return does not change.
- `app/coremock/store.py` becomes `app/core/redshift.py`: the same functions,
  parameterised SQL underneath. **No SQL is ever model-composed** — the model
  picks a tool, the tool owns its statement.
- **Idempotency needs an explicit design.** Redshift does not enforce unique or
  primary key constraints and has no `ON CONFLICT`, so the guarantee behind
  AG-6/CO-1 (12 write tools, a retry returns the first receipt) cannot rest on
  a unique index. Approach: `MERGE` keyed on the idempotency key, proven by a
  test that issues the same `payment_collect` twice and asserts one row and one
  receipt number.

## Phase 2 — tools to Lambda behind AgentCore Gateway

- One Lambda, all 27 tools, dispatching on
  `context.client_context.custom['bedrockAgentCoreToolName']` (prefix delimiter
  `___`).
- `ToolDefinition[]` is **generated from the existing registry**, not written by
  hand — `mcpserver/registry.py` already produces JSON Schema from real type
  hints via `get_type_hints`.
- Inbound JWT (D-1), outbound IAM.
- The tags/authority model does not move. Tags still select at build time;
  `authorize(spec, ctx, args)` still runs inside the Lambda, and the injected
  entitlements (`_customer_id`, `_user_id`, `_lob`, `_scopes`,
  `_idempotency_key`) still come from the caller's verified claims, never from
  model-supplied arguments.

## Phase 3 — RAG to Bedrock Knowledge Base on S3 Vectors

- Source of truth: the 74-document corpus in S3.
- Vector store: **S3 Vectors**, not OpenSearch Serverless (see the cost note —
  this choice is worth roughly $350/month).
- Embeddings: a Bedrock embedding model.
- Retrieval: semantic, top 5, **no reranking** (D-4). The KB-5 score floor and
  the OB-4 low-confidence reject retire with it; `_figures_supported()` and the
  citation guardrail do not, and remain the thing standing between a retrieved
  chunk and a claim reaching a customer.
- **Kept:** the metadata filter on `lob` + `corpus_scope` (KB-2). Scope is still
  injected, never a model argument.

## Phase 4 — agent to AgentCore Runtime, spoken to over AG-UI

ARM64 container, host `0.0.0.0`, port 8080, `POST /invocations` (SSE), `/ws`,
`GET /ping` → `{"status": "Healthy"}`.

```
browser ──POST /api/chat──► console FastAPI       (holds the credentials)
                                 │ invoke_agent_runtime(...)  ← SigV4
                                 ▼
                         AgentCore Runtime :8080 /invocations
                                 │ AG-UI server
                                 ▼
                         orchestrator.handle() → resolver → bot
                                 │
                         SSE events ◄──── streamed back unbuffered
```

The browser cannot call the Runtime itself — SigV4 needs credentials, and a page
cannot hold them. The console signs. That is the whole reason "FastAPI for both"
works, and it is the shape AWS's own AG-UI sample uses, with a Lambda Function
URL in `RESPONSE_STREAM` mode in place of the local process.

### Thread mapping

| Layer | Key | Set by |
| --- | --- | --- |
| AG-UI `threadId` | `user_id` | client |
| `runtimeSessionId` header | `sha256(user_id)` — the API requires ≥33 chars | client |
| Resolver checkpoint (ME-1) | `user_id` | server |
| Bot checkpoint (ME-2) | `user_id#lob` | **server only** |

The LOB is not known until the resolver has run, so a client cannot name a bot
thread. The compartment guarantee survives the network boundary structurally —
the id is not constructible from outside, so nothing needs to validate it.

### The glass box becomes live

`Trace` stops being a JSON blob returned after the reply:

| Trace kind | AG-UI event |
| --- | --- |
| `channel`, `bind`, `resolve`, `route` | `STEP_STARTED` / `STEP_FINISHED` |
| `tool` | `TOOL_CALL_START` → `ARGS` → `RESULT` |
| `retrieve`, `gate`, `guardrail` | `CUSTOM` |
| `llm` | `TEXT_MESSAGE_START` / `CONTENT` / `END` |
| `error` | `RUN_ERROR` |

### The answer text is NOT streamed

The citation guardrail rewrites an answer *after* the model finishes — an
uncited premium becomes a refusal. Streaming `TEXT_MESSAGE_CONTENT` deltas live
would put a fabricated figure on the customer's screen and retract it a second
later, which is worse than not streaming at all.

So: **the process streams, the claims do not.** Steps, retrieval scores, tool
calls and latencies go out live; the answer is emitted as a single
`TEXT_MESSAGE_CONTENT` once the guardrail has passed it.

### Session lifetime is a billing decision, not just a timeout

Runtime **memory** is billed for every second a session is alive, including idle
seconds — only CPU stops during I/O wait. Two consequences:

- Set the idle timeout **low**; the default is 15 minutes.
- Never let `/ping` return a `time_of_last_update` that advances on each call. A
  timestamp that moves every ping reads as a continuous status change, the idle
  timeout never fires, and sessions live to `MaxLifetime` (8 h) billing memory
  the whole way.

### Checkpointing does not change

DynamoDB, our own `DynamoDBSaver`, both threads, TTLs as they are (req 5). Not
AgentCore Memory.

## Phase 5 — SAM, fully parameterised

One `template.yaml` per stack; every name, region and model id a parameter; no
hardcoded ARNs. A `teardown` target that removes everything. No `pause` target
is needed, because nothing in the deployed set has a meaningful standing cost —
see below.

### Deploying from GitHub Actions (D-7)

AgentCore Runtime requires an ARM64 (Graviton) container and this laptop is
x86, with no Docker. Rather than work around that locally, **every artifact is
built and deployed by GitHub Actions** — Linux runners, Docker present, and a
deploy that does not depend on what happens to be installed on one machine.
That is the strongest reading of req 6.

It also fixes a second, quieter problem: `sam build` on Windows resolves
**Windows** wheels for any dependency with a binary component (`psycopg` among
them), and the usual remedy `--use-container` needs Docker. Building on a Linux
runner produces Linux wheels because it *is* Linux.

**Authentication is OIDC — no access keys in the repository.** The workflow
requests a short-lived token (`permissions: id-token: write`) and assumes an
IAM deploy role whose trust policy is scoped to this repository and branch.

There is a bootstrap ordering problem worth naming: the OIDC provider and the
deploy role cannot be created by the workflow that needs them. One small stack,
`mcb-bootstrap`, is applied once from the laptop's SSO session; everything after
that is Actions.

**Architecture split.** The AgentCore container is ARM64 because the service
requires it; the tools Lambda is **x86_64**, matching the default runner, so its
wheels resolve natively with no cross-compilation anywhere in the Lambda path.

**Verified before planning any of this:** the SSO role is `AdministratorAccess`
on account `719030485523` in `ap-south-1`, so IAM writes are permitted, and no
GitHub OIDC provider exists yet — the bootstrap stack creates the first one.
Note there is no `[default]` AWS profile; everything runs with
`AWS_PROFILE=AdministratorAccess-719030485523`.

**The repository is public**, so `runs-on: ubuntu-24.04-arm` is free: native
ARM64 runners, no QEMU, no emulation anywhere.

Being public has three consequences that are design constraints, not notes:

1. **No account identifiers in the tree.** The account id is currently baked
   into names such as `mcb-artifacts-skamalj-719030485523`. Not a credential,
   but not something to publish either. Every such name becomes a stack
   parameter resolved from `AWS::AccountId` at deploy time — which req 6 asks
   for regardless.
2. **A fork PR must never reach AWS.** Anyone can open one. `test.yml` runs on
   `pull_request` with no credentials and **no `id-token` permission** at all;
   the deploy workflows trigger only on push to `main`, which a fork cannot do.
3. **The role's trust condition must be exact.** `sub` is pinned to
   `repo:skamalj/multi-channel-bot:ref:refs/heads/main` and the `production`
   environment. No wildcards — `repo:skamalj/*` would let any repository on the
   account assume the deploy role, which on a public account is the whole ball
   game.

### Workflows

| Workflow | Trigger | Does |
| --- | --- | --- |
| `test.yml` | push, PR | The 164 offline tests, `NO_AWS=1`. No AWS credentials |
| `deploy-infra.yml` | push to `main`, dispatch | SAM stacks: Redshift, DynamoDB, S3, KB, Gateway, tools Lambda |
| `deploy-agent.yml` | push to `main`, dispatch | ARM64 image → ECR → update the AgentCore Runtime |
| `teardown.yml` | **manual dispatch only**, typed confirmation | Deletes every stack |

Teardown is never automatic and never runs on a push.

### The repository does not exist yet

`git init`, first commit, `gh repo create skamalj/multi-channel-bot --public`.
`gh` is authenticated as **skamalj** with `repo` and `workflow` scopes, which
covers creating the repository, pushing workflow files and setting the role ARN
as a repository variable. With OIDC there are no secrets to store.
Checked before proposing it: the
existing `.gitignore` already excludes `.env`, `.venv/`, `dist/` and
`data/generated/`; `.env` holds configuration only, no credentials (NF-1); the
tree is 1.5 MB. Nothing sensitive is staged. Git identity is unset globally and
will be set on the repository rather than the machine.

### Iteration does not go through any of this

The AG-UI server is plain FastAPI and runs locally as a normal process
(`uv run uvicorn`), x86, no container. ARM64 matters at deploy time, not in the
edit-test loop.

## Phase 6 — verification

- 164 offline tests stay green, in-memory and against real DynamoDB.
- New: idempotent double-write against Redshift; JWT claim propagation;
  AG-UI event ordering; the guardrail-before-emit rule.
- The 11 live Bedrock tests re-pointed at the deployed Runtime.

## Cost shape — what can be left deployed

| Service | Standing cost | Note |
| --- | --- | --- |
| Redshift Serverless — compute | **none** | Idle bills nothing. 60-second minimum per query wakes it |
| Redshift Serverless — storage | ~$0.024/GB-mo | Our dataset is well under 1 GB |
| S3 (corpus, documents, artifacts) | pennies | |
| S3 Vectors | $0.06/GB-mo | A few MB of vectors |
| DynamoDB on-demand | storage only | TTLs already expire rows |
| Lambda | **none** | Per invocation |
| AgentCore Gateway | $0.02/100 tools/mo | 27 tools ≈ half a cent |
| AgentCore Runtime | **none idle** | But memory bills per live session — see above |
| Bedrock inference + embeddings | **none idle** | Per token |
| ECR | ~$0.10/GB-mo | The ARM64 image; the largest single line |
| CloudWatch Logs | ingest + storage | Set 7-day retention |

**Two things that would break this, both avoided by choice, not by luck:**

1. **OpenSearch Serverless** as the vector store — a ~2 OCU floor, roughly
   $350/month whether or not a query is ever run. S3 Vectors (D-4/Phase 3) has
   no floor.
2. **A NAT Gateway** for the VPC Lambda that reaches Redshift — ~$32/month plus
   data, and it appears by default in most VPC-Lambda tutorials. Avoided by
   using *gateway* endpoints for S3 and DynamoDB, which are free, and by
   keeping the tools Lambda's only other dependency inside the VPC. Interface
   endpoints (~$7.30/month each) are also avoided for the same reason. This is
   the assertion in Phase 0 spike 3.

## Open risks

1. Idempotency on Redshift without unique constraints (Phase 1) — designed, not
   yet proven.
2. `runtimeSessionId` ≥33 characters — believed correct, confirmed in Phase 0.
3. Retiring the KB score floor (D-4) removes one of two defences against a
   low-relevance chunk being cited. The citation guardrail is now load-bearing
   alone.

---

# Deployment artifacts — what covers what

Six stacks, split by **churn rate and dependency direction**, not by service
family. The agent stack changes on every commit; Redshift takes minutes to
create and then never changes. One monolithic stack would put the slow, stable
resources at risk on every deploy.

Confirmed by schema lookup, all of it is native CloudFormation — **no custom
resources are needed for any component**:
`AWS::BedrockAgentCore::Runtime` accepts `ProtocolConfiguration: AGUI`,
`AWS::BedrockAgentCore::Gateway` accepts `AuthorizerType: CUSTOM_JWT`, and
`AWS::S3Vectors::VectorBucket` / `::Index` and
`AWS::Bedrock::KnowledgeBase` with `StorageConfiguration.Type: S3_VECTORS`
all exist.

## Stacks

| Stack | Contents | Mechanism | Churn |
| --- | --- | --- | --- |
| `mcb-bootstrap` | GitHub OIDC provider, `mcb-github-deploy`, `mcb-cfn-exec` | **CFN**, applied once from the laptop | never |
| `mcb-foundation` | 4 DynamoDB tables, 2 S3 buckets, ECR repo, SSM parameters | **CFN** | rare |
| `mcb-data` | VPC, subnets, SG, S3+DynamoDB gateway endpoints, Redshift namespace + workgroup, migrate Lambda | **SAM** (Lambda) + CFN | rare |
| `mcb-knowledge` | S3 Vectors bucket + index, Bedrock KB, DataSource | **CFN** | occasional |
| `mcb-tools` | tools Lambda, AgentCore Gateway, GatewayTarget | **SAM** (Lambda) + CFN | per commit |
| `mcb-agent` | AgentCore Runtime, RuntimeEndpoint | **CFN** | per commit |

SAM is a superset of CloudFormation, so `AWS::Serverless::Function` and raw CFN
resources sit in the same template. SAM is used only where its transform earns
its keep — the two Lambdas — and plain CFN everywhere else.

## What cannot be a template, and why

Five things are genuinely imperative. Each is a CI step, idempotent, and
re-runnable.

| # | Step | Why not CFN | Runs |
| --- | --- | --- | --- |
| S-1 | Redshift schema + seed | DDL/DML is not a resource | `mcb-migrate` Lambda, invoked by CI |
| S-2 | Corpus upload (74 docs) | Bucket contents are not resources | `aws s3 sync` |
| S-3 | KB ingestion | CFN creates the DataSource; it does not run a job | `StartIngestionJob` after S-2 |
| S-4 | ARM64 image build + push | — | `docker buildx` on `ubuntu-24.04-arm` |
| S-5 | `ToolDefinition[]` generation | Derived from the tool registry at build time | script → S3 → GatewayTarget |

**S-1 deserves a note.** The Redshift workgroup is in a private VPC, so a
GitHub runner cannot reach it directly, and making the workgroup publicly
accessible to fix that would be the wrong trade. The migration therefore runs
as a Lambda **inside** the VPC that CI invokes. It is deliberately not a CFN
custom resource: migrations should be re-runnable without a stack update.

## IAM roles

**Six explicit roles.** The two Lambdas do not get one: `AWS::Serverless::Function`
generates its own execution role from the `Policies:` list and attaches
`AWSLambdaBasicExecutionRole` for logging by itself. A hand-written role for a
SAM function is code to maintain that buys nothing.

```yaml
ToolsFunction:
  Type: AWS::Serverless::Function
  Properties:
    Architectures: [x86_64]
    VpcConfig: { ... }
    Policies:
      - VPCAccessPolicy: {}                      # ENI create/delete
      - SSMParameterReadPolicy:
          ParameterName: !Sub "${ParamPrefix}/redshift/*"
      - S3ReadPolicy:
          BucketName: !Ref DocumentsBucket
```

A role is written by hand only where the service demands a `RoleArn` it will
assume itself — AgentCore Runtime, AgentCore Gateway, Bedrock Knowledge Bases,
Redshift — or where the trust policy is the point, as in the two bootstrap
roles.

| Role | Trusted by | Grants |
| --- | --- | --- |
| `mcb-github-deploy` | GitHub OIDC, `sub` pinned to this repo + `main` | `cloudformation:*` on `mcb-*`; `iam:PassRole` **only** to `mcb-cfn-exec`; S3 artifacts; ECR push; `StartIngestionJob`; invoke `mcb-migrate` |
| `mcb-cfn-exec` | `cloudformation.amazonaws.com` | Creates every resource. Broad — but assumable only by CloudFormation |
| `mcb-agent-runtime` | `bedrock-agentcore.amazonaws.com` | `bedrock:InvokeModel*` on the two model ARNs; `bedrock:Retrieve` on the KB; DynamoDB CRUD on 4 tables; S3 `documents/*`; Gateway invoke; ECR pull; Logs |
| `mcb-gateway` | `bedrock-agentcore.amazonaws.com` | `lambda:InvokeFunction` on the tools Lambda, nothing else |
| `mcb-kb` | `bedrock.amazonaws.com` | S3 read on the corpus prefix; `bedrock:InvokeModel` on the embedding model; `s3vectors:PutVectors`/`QueryVectors` |
| `mcb-redshift` | `redshift.amazonaws.com` | S3 read, for `COPY` during seeding |
| *(tools Lambda)* | — | **Generated by SAM** from `Policies:` |
| *(migrate Lambda)* | — | **Generated by SAM** from `Policies:` |

### The privilege-escalation point

`mcb-cfn-exec` still creates IAM roles — SAM-generated ones included — and a
role that can create roles
can create an administrator. On a **public** repository that is not a
theoretical concern.

The mitigation is the two-role split above: the GitHub role holds almost no
direct resource permissions. It can call CloudFormation and pass exactly one
role. CloudFormation performs every creation using `mcb-cfn-exec`, which GitHub
cannot assume. A compromised workflow can therefore deploy this template — it
cannot mint itself a role.

Hardening available if wanted: a permissions boundary on every role
`mcb-cfn-exec` creates.

## Teardown

Delete is designed, not assumed. Four things block a naive `delete-stack`:

| Blocker | Handling |
| --- | --- |
| S3 buckets containing objects | `teardown.yml` empties them first |
| ECR repository containing images | `EmptyOnDelete: true` on the repository — declarative |
| Redshift final snapshot | `FinalSnapshotName` explicitly unset, or a snapshot survives and keeps billing |
| Auto-created `/aws/lambda/*` log groups | Log groups declared explicitly so CloudFormation owns and deletes them |

Order is the reverse of creation — `mcb-agent`, `mcb-tools`, `mcb-knowledge`,
`mcb-data`, `mcb-foundation` — and `mcb-bootstrap` is deliberately left, since
it is what the workflow authenticates with.

**Every stack carries `Project=mcb` at stack level**, so the last teardown step
is a verification, not a hope: query the tagging API for surviving `Project=mcb`
resources and fail the job if any are found.

## Orphans from earlier sessions

`mcb-checkpoints`, `mcb-resolver`, `mcb-profiles`, `mcb-audit` and
`mcb-artifacts-skamalj-719030485523` were created imperatively in earlier
sessions and belong to no stack. They must be either imported into
`mcb-foundation` or deleted before it is created, or the stack will fail on
name collision.

Recommendation: **delete and let CloudFormation recreate them.** The only loss
is existing checkpoints and audit rows, all of it demo data, and it proves the
create path rather than papering over it. Needs a decision before Phase 1.

---

# What building it actually taught us

Everything below replaced an assumption with a measurement.

## Corrections to decisions

**D-2 is now psycopg2, not psycopg3.** The Lambda reached Redshift and then
failed on its first query with `NotSupportedError: codec not available in
Python: 'UNICODE'`. Redshift reports its client encoding as `UNICODE`, which
is not a codec name Python knows; psycopg2 ships a mapping from `UNICODE` to
`utf_8` and psycopg3 does not. The pg-library decision stands - the driver
within it does not. All 12 write tools inherit this.

**D-6 needed a second look.** Parameter Store is the store of record, but the
migrate Lambda sits in a VPC with no NAT and no interface endpoints, so it
cannot reach the SSM API at all. The workflow reads the parameter and passes
it as a `NoEcho` override; the function receives it as an environment
variable. Reaching SSM from inside that VPC would have cost ~$7.30/month per
AZ, which is more than the rest of the stack combined.

## Seven failures, none of them typos

| Failure | Cause |
| --- | --- |
| `Credentials could not be loaded` | Repository variables did not exist yet |
| `Not authorized to perform sts:AssumeRoleWithWebIdentity` | GitHub sends **ID-qualified** subjects |
| `not authorized ... transform/Serverless-2016-10-31` | The SAM transform is a resource the service role needs |
| `REDSHIFT_PORT: expected String, found Integer` | Lambda environment values must be strings |
| `variable ... does not resolve to a string` | `Fn::Sub` will not coerce a number either |
| `codec not available in Python: 'UNICODE'` | psycopg3 cannot talk to Redshift |
| `schema "mcb" does not exist` | Redshift will not see a schema made in the same batch |

The OIDC one is worth keeping. The subject GitHub actually sends is

    repo:skamalj@32254183/multi-channel-bot@1360189167:environment:production

not the `repo:owner/repo:...` form in both AWS's and GitHub's documentation.
The trust policy pins both. The ID-qualified form is the *stricter* of the
two, because deleting and recreating a repository changes its id.

## Teardown is a ~25 minute operation, and that is fine

Measured, on the first real teardown:

| Stack | Time | Why |
| --- | --- | --- |
| `mcb-data` | **22m 35s** | Lambda Hyperplane ENIs |
| `mcb-foundation` | 34s | |

The function was already deleted while three ENIs described as
`AWS Lambda VPC ENI-mcb-migrate` sat `in-use`. AWS releases them on its own
schedule, and the subnets and security group cannot go until it does. Nothing
to fix; the waiter allows 60 minutes. Three further ENIs belonged to the
managed VPC endpoint Redshift Serverless creates for a private workgroup -
also service-managed, also unbilled.

The cost conclusion is unchanged. The *speed* conclusion is not: a teardown
is not something to start five minutes before you need the account clean.

## Two things the delete path exposed that the create path never would

1. **A non-empty bucket is the only thing that blocks a stack delete.** Proven
   by deliberately putting an object in each bucket and watching the delete
   fail with `409 ... not empty` on exactly two resources. DynamoDB tables and
   the ECR repository (`EmptyOnDelete: true`) clean themselves up.
2. **The teardown verification failed on success.** The tag sweep printed
   "these resources outlived teardown:" followed by nothing - an empty but
   non-zero-length string. It now writes to a file and tests `[ -s ]`. An
   alarm that fires on every successful run is worse than no alarm.

## Out of scope, and checked

`default-namespace` and `default-workgroup` predate this project and belong to
no stack. Teardown snapshots every non-`mcb-` namespace and workgroup before
it starts and diffs them at the end, with `if: always()`. Deleting something
the project does not own is a worse outcome than leaving something behind, so
it is checked last and fails loudly.
