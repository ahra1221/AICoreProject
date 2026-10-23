"""Read-only MediaWiki client: bounded requests, serial retries and expiring cache."""

from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

import requests

API_URL = "https://ko.wikipedia.org/w/api.php"
LICENSE_URL = "https://creativecommons.org/licenses/by-sa/4.0/"
_REQUEST_LOCK = threading.Lock()
_LAST_REQUEST_AT = 0.0


class WikipediaError(ValueError):
    """A safe, actionable API error, without server response bodies."""


class JsonCache:
    def __init__(self, directory: Path, ttl: float):
        self.directory, self.ttl = directory, ttl

    def path(self, key: str) -> Path:
        return self.directory / (hashlib.sha256(key.encode("utf-8")).hexdigest() + ".json")

    def get(self, key: str) -> Any | None:
        try:
            record = json.loads(self.path(key).read_text(encoding="utf-8"))
            if 0 <= time.time() - float(record["created_at"]) < self.ttl:
                return record["value"]
        except (OSError, ValueError, TypeError, KeyError):
            pass
        return None

    def put(self, key: str, value: Any) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.path(key)
        temporary = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(json.dumps({"created_at": time.time(), "value": value},
                                            ensure_ascii=False), encoding="utf-8")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


class WikipediaClient:
    def __init__(self, contact: str, cache_dir: Path, *, interval: float = 1.0,
                 cache_ttl: float = 86400, session: Any = None):
        if not contact.strip() or any(char in contact for char in "\r\n"):
            raise WikipediaError("Wikipedia 수집에는 연락 가능한 WIKI_CONTACT(이메일 또는 프로젝트 URL)가 필요합니다.")
        self.interval = max(1.0, interval)
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": f"MovieLab/2.0 ({contact.strip()})",
                                     "Accept": "application/json"})
        self.cache = JsonCache(cache_dir / "mediawiki-v2", cache_ttl)
        self.request_count = 0
        self.cache_hits = 0

    def query(self, params: dict[str, Any]) -> dict[str, Any]:
        params = {**params, "action": "query", "format": "json", "formatversion": 2, "maxlag": 5}
        key = json.dumps([API_URL, params], sort_keys=True, ensure_ascii=False)
        cached = self.cache.get(key)
        if isinstance(cached, dict) and isinstance(cached.get("query"), dict):
            self.cache_hits += 1
            return cached
        result = self._request(params)
        result["_retrieved_at"] = datetime.now(timezone.utc).isoformat()
        self.cache.put(key, result)
        return result

    def _request(self, params: dict[str, Any]) -> dict[str, Any]:
        global _LAST_REQUEST_AT
        # Shared across collectors in one process, including all retry attempts.
        with _REQUEST_LOCK:
            delay = 0.0
            for attempt in range(3):
                wait = max(delay, self.interval - (time.monotonic() - _LAST_REQUEST_AT))
                if wait > 0:
                    time.sleep(wait)
                response = None
                retryable = False
                try:
                    self.request_count += 1
                    response = self.session.get(API_URL, params=params, timeout=(5, 25))
                    retryable = response.status_code in {429, 500, 502, 503, 504}
                    if not retryable:
                        response.raise_for_status()
                        payload = response.json()
                        if not isinstance(payload, dict):
                            raise WikipediaError("Wikipedia 응답 형식을 확인할 수 없습니다.")
                        error = payload.get("error", {})
                        if error:
                            retryable = error.get("code") in {"maxlag", "ratelimited", "readonly"}
                            if not retryable:
                                raise WikipediaError("Wikipedia API 요청을 처리하지 못했습니다. 설정과 요청 조건을 확인하세요.")
                        elif isinstance(payload.get("query"), dict):
                            return payload
                        elif "batchcomplete" in payload:
                            return {**payload, "query": {}}
                        else:
                            raise WikipediaError("Wikipedia 응답에 query 데이터가 없습니다.")
                except (requests.Timeout, requests.ConnectionError):
                    retryable = True
                except (requests.RequestException, ValueError) as exc:
                    if isinstance(exc, WikipediaError):
                        raise
                    raise WikipediaError("Wikipedia 연결 또는 응답에 문제가 있습니다. 잠시 후 재시도하세요.") from None
                finally:
                    _LAST_REQUEST_AT = time.monotonic()
                if not retryable or attempt == 2:
                    break
                delay = max(2.0 ** (attempt + 1), self._retry_after(response))
                if delay > 60:
                    raise WikipediaError("Wikipedia가 긴 대기를 요청했습니다. 요청을 중단했으니 나중에 재시도하세요.")
        raise WikipediaError("Wikipedia가 일시적으로 응답하지 않습니다. 재시도를 중단했습니다.")

    @staticmethod
    def _retry_after(response: Any) -> float:
        if response is None:
            return 0.0
        value = response.headers.get("Retry-After", "0")
        try:
            return max(0, float(value))
        except ValueError:
            try:
                return max(0, parsedate_to_datetime(value).timestamp() - time.time())
            except (ValueError, TypeError, OverflowError):
                return 0.0

    def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        # exintro previews only. Full body is requested AFTER P2 accepts a page.
        result = self.query({"generator": "search", "gsrsearch": query[:250],
                             "gsrnamespace": 0, "gsrlimit": min(max(limit, 1), 5),
                             "prop": "extracts|info|pageprops", "exintro": 1,
                             "explaintext": 1, "exlimit": "max", "inprop": "url",
                             "ppprop": "disambiguation"})
        pages = result["query"].get("pages", [])
        return [{"page_id": str(page["pageid"]), "title": str(page["title"]),
                 "summary": str(page.get("extract", ""))[:1600],
                 "url": str(page.get("fullurl", "")),
                 "disambiguation": "disambiguation" in page.get("pageprops", {}),
                 "search_query": query}
                for page in sorted(pages, key=lambda item: item.get("index", 0))
                if page.get("pageid", 0) > 0 and not page.get("missing")]

    def fetch(self, page_id: str) -> dict[str, Any] | None:
        if not str(page_id).isdigit():
            raise WikipediaError("유효한 Wikipedia 문서 ID가 필요합니다.")
        result = self.query({"pageids": str(page_id), "redirects": 1,
                             "prop": "extracts|info|pageprops|revisions", "explaintext": 1,
                             "exlimit": 1, "inprop": "url", "ppprop": "disambiguation",
                             "rvprop": "ids|timestamp", "rvlimit": 1})
        for page in result["query"].get("pages", []):
            if page.get("missing") or "disambiguation" in page.get("pageprops", {}):
                continue
            if not page.get("extract", "").strip():
                continue
            revision = page.get("revisions", [{}])[0]
            return {"page_id": str(page["pageid"]), "title": page["title"],
                    "url": page.get("fullurl", f"https://ko.wikipedia.org/?curid={page['pageid']}"),
                    "text": page["extract"], "source": "한국어 위키백과",
                    "revision_id": str(revision.get("revid", "")),
                    "revision_timestamp": revision.get("timestamp", ""),
                    "license": "CC BY-SA 4.0", "license_url": LICENSE_URL,
                    "collected_at": result.get("_retrieved_at", "")}
        return None
