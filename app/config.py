import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 500))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 75))

    TOP_K_RETRIEVE: int = int(os.getenv("TOP_K_RETRIEVE", 20))
    TOP_N_RERANK: int = int(os.getenv("TOP_N_RERANK", 5))

    DATA_DIR: str = os.getenv("DATA_DIR", "data/uploads")
    INDEX_DIR: str = os.getenv("INDEX_DIR", "data/index")


settings = Settings()
