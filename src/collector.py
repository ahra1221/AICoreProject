"""Plan evidence searches from the supplied corpus and optional Wikipedia API."""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from datetime import date
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.config import Settings
from src.data_loader import Movie, WikiDocument
from src.indexer import MovieIndexer
from src.prompts import DOCUMENT_SELECTION_PROMPT, QUERY_EXTRACTION_PROMPT

logger = logging.getLogger(__name__)
WIKIPEDIA_API = "https://ko.wikipedia.org/w/api.php"


class CollectorError(ValueError):
    """Raised when query planning or external document collection fails."""


@dataclass(frozen=True)
class SearchPlan:
    queries: tuple[str, ...]
    selected_doc_keys: tuple[str, ...]


def _model_text(response: Any) -> str:
    content = response.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    raise CollectorError("The language model returned an unsupported response format.")


def _parse_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end < start:
        raise CollectorError("The language model did not return a JSON object.")
    try:
        value = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError as exc:
        raise CollectorError("The language model returned invalid JSON.") from exc
    if not isinstance(value, dict):
        raise CollectorError("The language model returned an invalid JSON structure.")
    return value


class DocumentCollector:
    def __init__(self, settings: Settings, model: Any, indexer: MovieIndexer):
        self.settings = settings
        self.model = model
        self.indexer = indexer
        self._last_request_at = 0.0
        self._request_lock = threading.Lock()
        self._cache_lock = threading.Lock()
        self._cache_path = settings.movies_path.parent / ".cache" / "wikipedia_search.json"
        self._http = requests.Session()
        retry = Retry(
            total=3,
            connect=3,
            read=3,
            backoff_factor=0.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET"}),
            respect_retry_after_header=True,
        )
        self._http.mount("https://", HTTPAdapter(max_retries=retry))
        contact = settings.wiki_contact or "contact-not-configured"
        self._http.headers.update(
            {"User-Agent": f"MovieLab/1.0 (Wikipedia research; contact: {contact})"}
        )

    def prepare(self, movie: Movie, question: str) -> SearchPlan:
        candidates = self.indexer.movie_candidates(movie.movie_id)
        queries = self._extract_queries(movie, question)
        if not candidates:
            if not self.settings.wiki_contact:
                raise CollectorError(
                    "No pre-collected documents exist for this movie. Set WIKI_CONTACT to enable Wikipedia fallback."
                )
            selected = self._collect_wikipedia_documents(movie, question, queries)
            candidates = self.indexer.movie_candidates(movie.movie_id)
            if candidates:
                available = {item["doc_key"] for item in candidates}
                selected = [key for key in selected if key in available]
            return SearchPlan(tuple(queries), tuple(selected))
        else:
            selected = self._select_documents(movie, question, candidates)
        return SearchPlan(tuple(queries), tuple(selected))

    def _extract_queries(self, movie: Movie, question: str) -> list[str]:
        prompt = (
            f"{QUERY_EXTRACTION_PROMPT}\n영화 제목: {movie.title}\n원제: {movie.original_title}\n"
            f"질문: {question}"
        )
        value = _parse_json(_model_text(self.model.invoke(prompt)))
        raw_queries = value.get("queries")
        if not isinstance(raw_queries, list):
            raise CollectorError("The language model did not return a query list.")
        queries = [str(query).strip() for query in raw_queries if str(query).strip()]
        required = [movie.title]
        if movie.original_title:
            required.append(movie.original_title)
        for query in required:
            if query.casefold() not in {item.casefold() for item in queries}:
                queries.insert(0, query)
        return list(dict.fromkeys(queries))[:8]

    def _select_documents(
        self, movie: Movie, question: str, candidates: list[dict[str, str]]
    ) -> list[str]:
        candidate_map = {item["doc_key"]: item for item in candidates}
        body = json.dumps(candidates, ensure_ascii=False)
        prompt = (
            f"{DOCUMENT_SELECTION_PROMPT}\n영화: {movie.title}\n질문: {question}\n"
            f"후보 JSON 데이터:\n{body}"
        )
        value = _parse_json(_model_text(self.model.invoke(prompt)))
        raw_keys = value.get("doc_keys")
        if not isinstance(raw_keys, list):
            raise CollectorError("The language model did not return a document key list.")
        selected = [
            str(key) for key in raw_keys
            if isinstance(key, str) and key in candidate_map
        ]
        for doc_key, candidate in candidate_map.items():
            candidate_title = candidate["title"].strip().casefold()
            if candidate_title.split(" (", 1)[0] in {
                movie.title.strip().casefold(),
                movie.original_title.strip().casefold(),
            }:
                if doc_key not in selected:
                    selected.append(doc_key)
        return selected

    def _read_cache(self) -> dict[str, list[dict[str, str]]]:
        if not self._cache_path.exists():
            return {}
        try:
            value = json.loads(self._cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not read Wikipedia search cache: %s", exc)
            return {}
        return value if isinstance(value, dict) else {}

    def _write_cache(self, cache: dict[str, list[dict[str, str]]]) -> None:
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self._cache_path.with_suffix(".tmp")
        temporary_path.write_text(
            json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary_path.replace(self._cache_path)

    def _cached_wikipedia_search(
        self, movie: Movie, query: str
    ) -> list[WikiDocument]:
        with self._cache_lock:
            return self._fetch_cached_wikipedia_search(movie, query)

    def _fetch_cached_wikipedia_search(
        self, movie: Movie, query: str
    ) -> list[WikiDocument]:
        cache = self._read_cache()
        cache_key = query.casefold()
        if cache_key in cache:
            return [
                WikiDocument(
                    page_id=str(item["page_id"]),
                    title=str(item["title"]),
                    url=str(item["url"]),
                    text=str(item["text"]),
                    source=str(item.get("source") or "한국어 위키백과"),
                    movie_id=movie.movie_id,
                    related_movie=movie.title,
                    search_query=query,
                    collected_at=str(item.get("collected_at") or ""),
                )
                for item in cache[cache_key]
            ]

        response = self._request(
            {
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": 5,
                "format": "json",
            },
            timeout=20,
        )
        response.raise_for_status()
        search_results = response.json().get("query", {}).get("search", [])
        page_ids = [str(item["pageid"]) for item in search_results if item.get("pageid")]
        if not page_ids:
            cache[cache_key] = []
            self._write_cache(cache)
            return []

        response = self._request(
            {
                "action": "query",
                "pageids": "|".join(page_ids),
                "prop": "extracts|info",
                "explaintext": 1,
                "exchars": 30000,
                "inprop": "url",
                "format": "json",
                "formatversion": 2,
            },
            timeout=30,
        )
        response.raise_for_status()
        pages = response.json().get("query", {}).get("pages", [])
        documents = [
            WikiDocument(
                page_id=str(page["pageid"]),
                title=str(page.get("title") or ""),
                url=str(page.get("fullurl") or ""),
                text=str(page.get("extract") or ""),
                source="한국어 위키백과",
                movie_id=movie.movie_id,
                related_movie=movie.title,
                search_query=query,
                collected_at=date.today().isoformat(),
            )
            for page in pages
            if page.get("pageid") and page.get("title") and page.get("extract")
        ]
        cache[cache_key] = [
            {
                "page_id": document.page_id,
                "title": document.title,
                "url": document.url,
                "text": document.text,
                "source": document.source,
                "movie_id": document.movie_id,
                "related_movie": document.related_movie,
                "search_query": document.search_query,
                "collected_at": document.collected_at,
            }
            for document in documents
        ]
        self._write_cache(cache)
        return documents

    def _request(self, params: dict[str, object], timeout: int) -> requests.Response:
        with self._request_lock:
            wait = 1.0 - (time.monotonic() - self._last_request_at)
            if wait > 0:
                time.sleep(wait)
            response = self._http.get(WIKIPEDIA_API, params=params, timeout=timeout)
            self._last_request_at = time.monotonic()
        return response

    def _collect_wikipedia_documents(
        self, movie: Movie, question: str, queries: list[str]
    ) -> list[str]:
        documents: dict[str, WikiDocument] = {}
        for query in queries[:4]:
            for document in self._cached_wikipedia_search(movie, query):
                documents[f"{document.movie_id}:{document.page_id}"] = document
        if not documents:
            return []

        candidates = [
            {
                "doc_key": f"{document.movie_id}:{document.page_id}",
                "title": document.title,
                "summary": document.text[:800],
                "related_movie": document.related_movie,
            }
            for document in documents.values()
        ]
        selected_keys = self._select_documents(movie, question, candidates)
        selected_documents = [
            document
            for doc_key, document in documents.items()
            if doc_key in selected_keys
        ]
        if selected_documents:
            report = self.indexer.index_external_documents(selected_documents, prune=False)
            logger.info(
                "Added %d Wikipedia documents for movie %s",
                report.added_or_updated,
                movie.movie_id,
            )
        return selected_keys
