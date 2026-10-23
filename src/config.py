"""Environment-backed paths and settings for Movie Lab."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_GMS_BASE_URL = "https://gms.ssafy.io/gmsapi/api.openai.com/v1/"


class ConfigurationError(ValueError):
    """Raised when required project configuration is invalid."""


@dataclass(frozen=True)
class Settings:
    gms_key: str
    gms_base_url: str
    chat_model: str
    embedding_model: str
    wiki_contact: str
    movies_path: Path
    wiki_dir: Path
    chroma_dir: Path
    chunk_size: int
    chunk_overlap: int
    top_k: int
    min_relevance: float


def get_settings(environ: dict[str, str] | None = None) -> Settings:
    load_dotenv(ROOT_DIR / ".env")
    env = os.environ if environ is None else environ

    try:
        chunk_size = int(env.get("CHUNK_SIZE", "1000"))
        chunk_overlap = int(env.get("CHUNK_OVERLAP", "200"))
        top_k = int(env.get("TOP_K", "5"))
        min_relevance = float(env.get("MIN_RELEVANCE", "0.15"))
    except ValueError as exc:
        raise ConfigurationError("CHUNK_SIZE, CHUNK_OVERLAP, TOP_K, and MIN_RELEVANCE must be numeric.") from exc

    if chunk_size < 1 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ConfigurationError("CHUNK_SIZE must be positive and CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")
    if top_k < 1:
        raise ConfigurationError("TOP_K must be at least 1.")
    if not 0 <= min_relevance <= 1:
        raise ConfigurationError("MIN_RELEVANCE must be between 0 and 1.")

    gms_key = env.get("GMS_KEY", "").strip()
    wiki_contact = env.get("WIKI_CONTACT", "").strip()
    if gms_key.casefold() in {"your_gms_api_key", "your_gms_api_key_here"}:
        gms_key = ""
    if wiki_contact.casefold() == "your-email@example.com":
        wiki_contact = ""

    gms_base_url = env.get("GMS_BASE_URL", DEFAULT_GMS_BASE_URL).strip()
    chat_model = env.get("GMS_MODEL", "gpt-5.4-nano").strip()
    embedding_model = env.get("GMS_EMBEDDING_MODEL", "text-embedding-3-small").strip()
    if not gms_base_url or not chat_model or not embedding_model:
        raise ConfigurationError("GMS_BASE_URL, GMS_MODEL, and GMS_EMBEDDING_MODEL cannot be empty.")

    return Settings(
        gms_key=gms_key,
        gms_base_url=gms_base_url,
        chat_model=chat_model,
        embedding_model=embedding_model,
        wiki_contact=wiki_contact,
        movies_path=Path(env.get("MOVIES_PATH", ROOT_DIR / "data" / "movies.json")),
        wiki_dir=Path(env.get("WIKI_DIR", ROOT_DIR / "data" / "movielab_wiki" / "wikipedia")),
        chroma_dir=Path(env.get("CHROMA_DIR", ROOT_DIR / "chroma_db")),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=top_k,
        min_relevance=min_relevance,
    )


def require_api_key(settings: Settings) -> str:
    if not settings.gms_key:
        raise ConfigurationError("GMS_KEY is not set. Create a local .env file and add your API key.")
    return settings.gms_key
