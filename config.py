"""
Simpsons Quote Search Engine - Configuration
=============================================
Author: César Adrián Delgado Díaz
License: MIT

Configuración centralizada del proyecto.
"""

import os
from typing import Optional
from functools import lru_cache

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Configuración de la aplicación."""
    
    # ==========================================================================
    # General
    # ==========================================================================
    APP_NAME: str = "Simpsons Quote Search Engine"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = Field(default=False, env="DEBUG")
    ENVIRONMENT: str = Field(default="development", env="ENVIRONMENT")
    
    # ==========================================================================
    # API
    # ==========================================================================
    API_HOST: str = Field(default="0.0.0.0", env="API_HOST")
    API_PORT: int = Field(default=8000, env="API_PORT")
    API_PREFIX: str = "/api/v1"
    CORS_ORIGINS: list = ["*"]
    
    # ==========================================================================
    # Database (PostgreSQL + pgvector)
    # ==========================================================================
    DATABASE_URL: str = Field(
        default="postgresql://postgres:postgres@localhost:5432/simpsons_quotes",
        env="DATABASE_URL"
    )
    DATABASE_POOL_SIZE: int = Field(default=5, env="DATABASE_POOL_SIZE")
    
    # ==========================================================================
    # Vector Store
    # ==========================================================================
    VECTOR_DIMENSION: int = 384  # all-MiniLM-L6-v2 dimension
    VECTOR_INDEX_TYPE: str = "ivfflat"  # ivfflat or hnsw
    VECTOR_LISTS: int = 100  # Para ivfflat
    
    # ==========================================================================
    # Embeddings
    # ==========================================================================
    EMBEDDING_MODEL: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        env="EMBEDDING_MODEL"
    )
    EMBEDDING_BATCH_SIZE: int = 32
    EMBEDDING_DEVICE: str = Field(default="cpu", env="EMBEDDING_DEVICE")
    
    # ==========================================================================
    # BM25
    # ==========================================================================
    BM25_K1: float = 1.5
    BM25_B: float = 0.75
    BM25_INDEX_PATH: str = "data/bm25_index"
    
    # ==========================================================================
    # Retrieval
    # ==========================================================================
    DEFAULT_TOP_K: int = 5
    MAX_TOP_K: int = 20
    HYBRID_ALPHA: float = 0.5  # Weight for semantic vs BM25 (0=BM25, 1=semantic)
    RRF_K: int = 60
    
    # ==========================================================================
    # LLM (OpenAI)
    # ==========================================================================
    OPENAI_API_KEY: str = Field(default="", env="OPENAI_API_KEY")
    LLM_MODEL: str = Field(default="gpt-3.5-turbo", env="LLM_MODEL")
    LLM_TEMPERATURE: float = 0.7
    LLM_MAX_TOKENS: int = 500
    
    # ==========================================================================
    # MLflow
    # ==========================================================================
    MLFLOW_TRACKING_URI: str = Field(
        default="sqlite:///mlruns.db",
        env="MLFLOW_TRACKING_URI"
    )
    MLFLOW_EXPERIMENT_NAME: str = "simpsons-quote-search"
    
    # ==========================================================================
    # Observability
    # ==========================================================================
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    LOG_FORMAT: str = Field(default="json", env="LOG_FORMAT")  # json or text
    
    OTLP_ENDPOINT: Optional[str] = Field(default=None, env="OTLP_ENDPOINT")
    ENABLE_TRACING: bool = Field(default=True, env="ENABLE_TRACING")
    
    PROMETHEUS_PORT: int = Field(default=9090, env="PROMETHEUS_PORT")
    ENABLE_METRICS: bool = Field(default=True, env="ENABLE_METRICS")
    
    # ==========================================================================
    # Data
    # ==========================================================================
    DATA_DIR: str = "data"
    QUOTES_FILE: str = "data/simpsons_quotes.csv"
    
    # ==========================================================================
    # Cache
    # ==========================================================================
    ENABLE_CACHE: bool = True
    CACHE_TTL_SECONDS: int = 3600
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """
    Obtiene la configuración (cacheada).
    
    Returns:
        Instancia de Settings
    """
    return Settings()


# Prompts del sistema
SYSTEM_PROMPTS = {
    "rag_answer": """Eres un experto en Los Simpsons y debes responder preguntas 
usando ÚNICAMENTE la información de las quotes proporcionadas.

Reglas:
1. SIEMPRE cita la quote exacta entre comillas
2. SIEMPRE menciona el personaje que dijo la frase
3. SIEMPRE incluye el episodio si está disponible
4. Si no encuentras información relevante, di que no tienes suficiente contexto
5. Responde en el mismo idioma de la pregunta
6. Sé conciso pero informativo

Formato de respuesta:
- Respuesta breve
- Quote: "[frase exacta]"
- Personaje: [nombre]
- Episodio: [código y nombre si disponible]
""",

    "evaluate_faithfulness": """Evalúa si la respuesta generada es fiel a los 
documentos fuente proporcionados.

Puntuación (0-1):
- 1.0: Completamente fiel, toda la información proviene de las fuentes
- 0.5: Parcialmente fiel, mezcla información de fuentes con inferencias
- 0.0: No fiel, contiene información no presente en las fuentes

Responde SOLO con un número entre 0 y 1.
""",

    "evaluate_relevance": """Evalúa qué tan relevante es la respuesta para la 
pregunta del usuario.

Puntuación (0-1):
- 1.0: Completamente relevante, responde exactamente lo preguntado
- 0.5: Parcialmente relevante, responde pero de forma tangencial
- 0.0: No relevante, no responde la pregunta

Responde SOLO con un número entre 0 y 1.
"""
}


# Exportar
__all__ = ['Settings', 'get_settings', 'SYSTEM_PROMPTS']
