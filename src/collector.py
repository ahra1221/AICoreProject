"""P1 -> MediaWiki search -> P2 -> accepted bodies -> index, inside retrieve."""

from __future__ import annotations

import json
import re
import threading
import time
import unicodedata
from dataclasses import asdict, dataclass
from typing import Any

from src.config import Settings
from src.data_loader import Movie, WikiDocument
from src.indexer import MovieIndexer
from src.prompts import DOCUMENT_SELECTION_PROMPT, QUERY_EXTRACTION_PROMPT
from src.wikipedia import JsonCache, WikipediaClient


class CollectorError(ValueError):
    """A collection/planning error safe for a generic UI message."""


@dataclass(frozen=True)
class SearchPlan:
    queries: tuple[str, ...]
    selected_doc_keys: tuple[str, ...]
    decisions: tuple[dict[str, Any], ...] = ()
    cache_hit: bool = False
    collection_mode: str = "snapshot"
    api_requests: int = 0
    elapsed_seconds: float = 0.0


def _model_text(response: Any) -> str:
    content = response.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    raise CollectorError("The language model returned an unsupported response format.")


def _parse_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        raise CollectorError("The language model did not return valid JSON.") from None
    if not isinstance(value, dict):
        raise CollectorError("The language model did not return a JSON object.")
    return value


def _normalize(value: str) -> str:
    return re.sub(r"[^\w]", "", unicodedata.normalize("NFKC", value).casefold())


def identity_check(movie: Movie, candidate: dict[str, Any]) -> tuple[str, str]:
    """Hard veto for homonyms; P2 still decides relevance of related concepts."""
    title = str(candidate.get("title", ""))
    summary = str(candidate.get("summary", ""))[:1600]
    if candidate.get("disambiguation") or "동음이의" in title:
        return "reject", "동음이의 문서"
    base_title = re.sub(r"\s*\([^)]*\)\s*$", "", title)
    aliases = {_normalize(movie.title), _normalize(movie.original_title)} - {""}
    if _normalize(base_title) not in aliases:
        return "related", "배경 문서의 질문 관련성을 P2에서 판단"
    title_year = re.search(r"\((\d{4})\s*(?:년)?\s*(?:영화|film)\)", title, re.I)
    lead = summary.split("\n", 1)[0][:500]
    film = bool(re.search(r"영화(?!배우|감독|인)|\bfilm\b|\bmovie\b", title + " " + lead, re.I))
    if not film:
        return "reject", "같은 제목이지만 영화 문서임을 확인하지 못함"
    lead_years = re.findall(r"(\d{4})\s*년(?:에)?\s*(?:개봉|공개|제작)?", lead)
    if not lead_years:
        lead_years = re.findall(r"\b(\d{4})\b(?=[^.]{0,80}\bfilm\b)", lead, re.I)
    years = {int(title_year.group(1))} if title_year else {int(y) for y in lead_years}
    if movie.year and years and movie.year not in years:
        return "reject", f"동명 영화의 개봉연도 불일치: 선택 {movie.year}, 후보 {sorted(years)}"
    if movie.year and not years and not any(_normalize(d) in _normalize(lead) for d in movie.director):
        return "reject", "제목만 같고 연도·감독 일치를 확인하지 못함"
    return "movie", "영화 제목과 연도 또는 감독 일치 확인"


def movie_queries(movie: Movie) -> list[str]:
    year = f" {movie.year}" if movie.year else ""
    queries = [f"{movie.title}{year} 영화"]
    if movie.original_title and movie.original_title.casefold() != movie.title.casefold():
        queries.append(f"{movie.original_title}{year} film")
    return queries


class DocumentCollector:
    def __init__(self, settings: Settings, model: Any, indexer: MovieIndexer,
                 wiki: WikipediaClient | None = None):
        self.settings, self.model, self.indexer = settings, model, indexer
        self.cache_dir = settings.movies_path.parent / ".cache"
        self.cache = JsonCache(self.cache_dir / "question-plans-v2", settings.wiki_cache_ttl)
        self.wiki = wiki
        self._lock = threading.Lock()

    def _cache_key(self, movie: Movie, question: str) -> str:
        return json.dumps(["P1-P2-v2", asdict(movie), " ".join(question.split()),
                           self.settings.wiki_mode, self.settings.chat_model,
                           self.settings.chunk_size, self.settings.chunk_overlap,
                           self.settings.wiki_max_queries, self.settings.wiki_max_documents,
                           str(self.settings.chroma_dir.resolve()), str(self.settings.wiki_dir.resolve()),
                           QUERY_EXTRACTION_PROMPT, DOCUMENT_SELECTION_PROMPT],
                          sort_keys=True, ensure_ascii=False)

    def prepare(self, movie: Movie, question: str) -> SearchPlan:
        with self._lock:
            return self._prepare(movie, question)

    def _prepare(self, movie: Movie, question: str) -> SearchPlan:
        started = time.perf_counter()
        key = self._cache_key(movie, question)
        cached = self.cache.get(key)
        if isinstance(cached, dict):
            try:
                selected = tuple(cached["selected_doc_keys"])
                available = {item["doc_key"] for item in self.indexer.movie_candidates(movie.movie_id)}
                if not set(selected).issubset(available):
                    self.indexer.index_external_documents(
                        [WikiDocument(**item) for item in cached["documents"]], prune=False)
                return SearchPlan(tuple(cached["queries"]), selected, tuple(cached["decisions"]),
                                  True, self.settings.wiki_mode, 0, time.perf_counter() - started)
            except (KeyError, TypeError):
                pass

        if self.settings.wiki_mode == "api" and self.wiki is None:
            self.wiki = WikipediaClient(self.settings.wiki_contact, self.cache_dir,
                                        interval=self.settings.wiki_interval,
                                        cache_ttl=self.settings.wiki_cache_ttl)
        requests_before = self.wiki.request_count if self.wiki else 0
        queries = self._extract_queries(movie, question)
        source_map: dict[str, WikiDocument] = {}
        if self.settings.wiki_mode == "api":
            candidates_by_key = {}
            for query in queries[:self.settings.wiki_max_queries]:
                for page in self.wiki.search(query):
                    doc_key = f"{movie.movie_id}:{page['page_id']}"
                    candidates_by_key.setdefault(doc_key, {**page, "doc_key": doc_key,
                                                          "related_movie": movie.title})
            candidates = list(candidates_by_key.values())
        else:
            for path in sorted(self.settings.wiki_dir.glob(f"{movie.movie_id}_*.json")):
                document = WikiDocument.from_dict(json.loads(path.read_text(encoding="utf-8")), path)
                source_map[f"{movie.movie_id}:{document.page_id}"] = document
            candidates = [{"doc_key": doc_key, "title": doc.title, "summary": doc.text[:1600],
                           "related_movie": doc.related_movie, "url": doc.url}
                          for doc_key, doc in source_map.items()]
            if not candidates:
                candidates = self.indexer.movie_candidates(movie.movie_id)

        decisions = self._select_documents(movie, question, candidates) if candidates else []
        selected_documents = []
        selected_keys = []
        candidate_map = {item["doc_key"]: item for item in candidates}
        for decision in decisions:
            if not decision["adopt"]:
                continue
            doc_key = decision["doc_key"]
            if self.settings.wiki_mode == "api":
                page = self.wiki.fetch(candidate_map[doc_key]["page_id"])
                if page is None:
                    decision.update(adopt=False, reason="본문 없음 또는 동음이의 문서로 변경됨")
                    continue
                verdict, reason = identity_check(movie, {**page, "summary": page["text"][:1600]})
                if verdict == "reject":
                    decision.update(adopt=False, reason=reason)
                    continue
                document = WikiDocument(**page, movie_id=movie.movie_id, related_movie=movie.title,
                                        search_query=candidate_map[doc_key].get("search_query", ""))
                doc_key = f"{movie.movie_id}:{document.page_id}"
                decision["doc_key"] = doc_key
            else:
                document = source_map.get(doc_key)
            if doc_key not in selected_keys:
                selected_keys.append(doc_key)
                if document:
                    selected_documents.append(document)
        if selected_documents:
            self.indexer.index_external_documents(selected_documents, prune=False)
        self.cache.put(key, {"queries": queries, "selected_doc_keys": selected_keys,
                             "decisions": decisions,
                             "documents": [asdict(doc) for doc in selected_documents]})
        return SearchPlan(tuple(queries), tuple(selected_keys), tuple(decisions), False,
                          self.settings.wiki_mode,
                          (self.wiki.request_count if self.wiki else 0) - requests_before,
                          time.perf_counter() - started)

    def _extract_queries(self, movie: Movie, question: str) -> list[str]:
        prompt = (f"{QUERY_EXTRACTION_PROMPT}\n영화 데이터: {json.dumps(asdict(movie), ensure_ascii=False)}"
                  f"\n질문 데이터: {json.dumps(question, ensure_ascii=False)}")
        value = _parse_json(_model_text(self.model.invoke(prompt)))
        raw = value.get("queries")
        if not isinstance(raw, list):
            raise CollectorError("P1 did not return a query list.")
        queries = movie_queries(movie) + [item.strip()[:250] for item in raw
                                         if isinstance(item, str) and item.strip()]
        return list(dict.fromkeys(queries))[:self.settings.wiki_max_queries]

    def _select_documents(self, movie: Movie, question: str,
                          candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        prompt = (f"{DOCUMENT_SELECTION_PROMPT}\n영화 데이터: {json.dumps(asdict(movie), ensure_ascii=False)}"
                  f"\n질문 데이터: {json.dumps(question, ensure_ascii=False)}"
                  f"\n후보 JSON 데이터:\n{json.dumps(candidates, ensure_ascii=False)}")
        value = _parse_json(_model_text(self.model.invoke(prompt)))
        choices: dict[str, tuple[bool, str]] = {}
        if isinstance(value.get("decisions"), list):
            for choice in value["decisions"]:
                if isinstance(choice, dict) and isinstance(choice.get("adopt"), bool):
                    choices[str(choice.get("doc_key", ""))] = (choice["adopt"], str(choice.get("reason", ""))[:240])
        elif isinstance(value.get("doc_keys"), list):
            choices = {item: (True, "P2 채택") for item in value["doc_keys"] if isinstance(item, str)}
        else:
            raise CollectorError("P2 did not return document decisions.")
        decisions = []
        accepted = 0
        ranked = sorted(candidates, key=lambda item: identity_check(movie, item)[0] != "movie")
        for candidate in ranked:
            verdict, identity_reason = identity_check(movie, candidate)
            adopt, reason = choices.get(candidate["doc_key"], (False, "P2 미채택"))
            if verdict == "reject":
                adopt, reason = False, identity_reason
            elif verdict == "movie" and adopt:
                reason = f"{reason} · {identity_reason}"
            if adopt and accepted >= self.settings.wiki_max_documents:
                adopt, reason = False, "질문당 채택 문서 상한"
            accepted += int(adopt)
            decisions.append({"doc_key": candidate["doc_key"], "title": candidate["title"],
                              "adopt": adopt, "reason": reason, "identity": verdict})
        return decisions
