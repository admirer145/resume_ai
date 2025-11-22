import os
from typing import Literal

# Model Provider Configuration
MODEL_PROVIDER: Literal["openai", "ollama"] = os.getenv("MODEL_PROVIDER", "ollama").lower()

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Ollama Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Embedding Model Configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "embeddinggemma:latest")  # Default to Ollama model
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"

# LLM Model Configuration
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-coder:1.3b")  # Default to Ollama model
OPENAI_LLM_MODEL = "gpt-4o-mini"

# Weaviate Configuration
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")

# Get the appropriate embedding model based on provider
def get_embedding_model() -> str:
    if MODEL_PROVIDER == "openai":
        return OPENAI_EMBEDDING_MODEL
    return EMBEDDING_MODEL

# Get the appropriate LLM model based on provider
def get_llm_model() -> str:
    if MODEL_PROVIDER == "openai":
        return OPENAI_LLM_MODEL
    return LLM_MODEL

# Configuration summary for logging
def get_config_summary() -> dict:
    return {
        "provider": MODEL_PROVIDER,
        "embedding_model": get_embedding_model(),
        "llm_model": get_llm_model(),
        "ollama_url": OLLAMA_BASE_URL if MODEL_PROVIDER == "ollama" else None,
        "weaviate_url": WEAVIATE_URL
    }
