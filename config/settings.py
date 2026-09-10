"""
StartupGenie – Application Configuration
Loads and validates all environment variables.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache


class Settings(BaseSettings):
    # IBM watsonx credentials
    watsonx_api_key: str = Field(..., env="WATSONX_API_KEY")
    watsonx_project_id: str = Field(..., env="WATSONX_PROJECT_ID")
    watsonx_url: str = Field(
        default="https://us-south.ml.cloud.ibm.com", env="WATSONX_URL"
    )

    # IBM Granite model
    granite_model_id: str = Field(
        default="ibm/granite-13b-instruct-v2", env="GRANITE_MODEL_ID"
    )

    # Application settings
    app_host: str = Field(default="0.0.0.0", env="APP_HOST")
    app_port: int = Field(default=8000, env="APP_PORT")
    debug: bool = Field(default=False, env="DEBUG")

    # RAG settings
    chunk_size: int = Field(default=512, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=64, env="CHUNK_OVERLAP")
    top_k_results: int = Field(default=5, env="TOP_K_RESULTS")
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2", env="EMBEDDING_MODEL"
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()
