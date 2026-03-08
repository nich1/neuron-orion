from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    OLLAMA_URL: str = "http://localhost:11434"
    OPEN_WEBUI_URL: str = "http://localhost:3000"
    N8N_URL: str = "http://localhost:5678"
    QDRANT_URL: str = "http://localhost:6333"
    AUTH_URL: str = "http://localhost:8081"
    SEQ_URL: str = "http://localhost:5341"
    SEQ_API_KEY: str = ""
    API_TOKEN: str = ""
    DB_PATH: str = "data/memory.db"
    DEFAULT_MODEL: str = "qwen2.5:7b"
    EMBEDDING_MODEL: str = "nomic-embed-text"


settings = Settings()