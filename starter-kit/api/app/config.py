"""Runtime configuration, read once from environment variables."""
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    portkey_api_key: str
    portkey_base_url: str
    llm_model: str
    qdrant_url: str
    qdrant_collection: str
    knowledge_base_dir: Path
    tools_dir: Path
    db_path: Path
    embedding_model: str
    similarity_strong: float
    similarity_floor: float
    max_agent_turns: int
    llm_reasoning_effort: str
    llm_image_detail: str
    llm_max_output_tokens: int
    llm_timeout_seconds: float
    llm_strict_tools: bool

    @property
    def llm_configured(self) -> bool:
        return bool(self.portkey_api_key and self.portkey_base_url and self.llm_model)


def load_settings() -> Settings:
    env = os.environ.get
    return Settings(
        portkey_api_key=env("PORTKEY_API_KEY", ""),
        portkey_base_url=env("PORTKEY_BASE_URL", "").rstrip("/"),
        llm_model=env("LLM_MODEL", ""),
        qdrant_url=env("QDRANT_URL", "http://localhost:6333"),
        qdrant_collection=env("QDRANT_COLLECTION", "historical_claims"),
        knowledge_base_dir=Path(env("KNOWLEDGE_BASE_DIR", "/app/data/knowledge-base")),
        tools_dir=Path(env("TOOLS_DIR", "/app/tools")),
        db_path=Path(env("DB_PATH", "/app/state/claims.db")),
        embedding_model=env("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
        # Calibrated on development queries: unrelated text scored <=0.53, genuine precedents >=0.69.
        similarity_strong=float(env("SIMILARITY_STRONG", "0.70")),
        similarity_floor=float(env("SIMILARITY_FLOOR", "0.60")),
        max_agent_turns=int(env("MAX_AGENT_TURNS", "6")),
        # gpt-5.5 on Chat Completions rejects tools combined with reasoning; "none" is required there. Empty = omit.
        llm_reasoning_effort=env("LLM_REASONING_EFFORT", "none"),
        llm_image_detail=env("LLM_IMAGE_DETAIL", "auto"),
        llm_max_output_tokens=int(env("LLM_MAX_OUTPUT_TOKENS", "4000")),
        llm_timeout_seconds=float(env("LLM_TIMEOUT_SECONDS", "60")),
        llm_strict_tools=env("LLM_STRICT_TOOLS", "true").lower() == "true",
    )
