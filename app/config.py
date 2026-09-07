"""Configuration. Credentials are never held here - boto3 uses its default chain.

Model defaults are Bedrock models that this account can actually reach and
that support the Converse tool-use API: Kimi K2.5 for the agent loop, Nova
Lite for the cheap classifier and reranker calls. Both are overridable, and
nothing in the code assumes a vendor.
"""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    aws_region: str = "ap-south-1"

    # Agent loop. Must support the Converse API with toolConfig.
    bedrock_model_id: str = "moonshotai.kimi-k2.5"
    # Intent classification and reranking - small, fast, called off the hot path
    # of the answer itself.
    bedrock_small_model_id: str = "apac.amazon.nova-lite-v1:0"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 1024

    ddb_checkpoint_table: str = "mcb-checkpoints"
    ddb_resolver_table: str = "mcb-resolver"
    ddb_profile_table: str = "mcb-profiles"
    ddb_audit_table: str = "mcb-audit"
    s3_bucket: str = "mcb-artifacts-CHANGE-ME"

    session_ttl_days: int = 30
    resolver_ttl_days: int = 180
    audit_ttl_days: int = 2555          # 7 years - the record outlives the session
    msg_history_to_keep: int = 12

    # NF-4: both versions travel in every trace. config_version is read back
    # from data/products.yaml so a catalogue edit cannot silently detach from
    # the answers it produced.
    prompt_version: str = "v0.3"

    # Bedrock Guardrails (deployed by infra/guardrails). Empty means no
    # guardrail is attached, which is the offline and test posture - the
    # deploy asserts it is set in a real environment rather than letting a
    # missing id silently disable the control.
    guardrail_id: str = ""
    guardrail_version: str = "DRAFT"

    # Bedrock Knowledge Base. Empty falls back to the local corpus, which is
    # what the offline tests use.
    knowledge_base_id: str = ""

    # Redshift. Empty means the in-memory core store (NF-2 / offline tests).
    redshift_host: str = ""
    redshift_port: int = 5439
    redshift_db: str = "mcb"
    redshift_user: str = "mcbadmin"
    redshift_password: str = ""
    redshift_schema: str = "mcb"

    # Retrieval (KB-3/KB-5). Below the floor the bot refuses and offers a human.
    retrieval_k: int = 4
    retrieval_candidates: int = 12
    retrieval_floor: float = 0.30
    rerank_with_llm: bool = False       # deterministic reranker by default

    # RS-3: the intent model is the LAST resort before asking, and only runs
    # when the deterministic signals produced nothing.
    intent_model_enabled: bool = True

    # Keeping a thread small enough that a checkpoint never approaches the
    # 1 MB DynamoDB query cap. Documents go to object storage (only metadata
    # reaches the message list) and old turns are reduced to a summary.
    reduce_enabled: bool = True
    reduce_after_messages: int = 24     # prune once the thread passes this
    reduce_keep_messages: int = 10      # recent turns kept verbatim

    # CO-4: the mock core behaves like a core - latency, pending states, a 429.
    core_latency_ms: int = 120
    core_chaos: bool = True

    mock_llm: bool = False
    no_aws: bool = False
    log_level: str = "INFO"

    @property
    def config_version(self) -> str:
        from app.coremock.catalog import catalog

        return str(catalog().get("meta", {}).get("config_version", "unversioned"))


@lru_cache
def settings() -> Settings:
    return Settings()
