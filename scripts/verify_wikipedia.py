"""Small live API smoke test, with no LLM, embeddings or vector DB writes."""
from __future__ import annotations
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.collector import identity_check, movie_queries
from src.config import get_settings
from src.data_loader import load_movies
from src.wikipedia import WikipediaClient, WikipediaError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--movie-id", default="157336")
    parser.add_argument("--contact", help="Contact URL or email for the identifying User-Agent")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/evidence/wikipedia-api.json")
    args = parser.parse_args()
    settings = get_settings()
    movie = next((m for m in load_movies(settings.movies_path) if m.movie_id == args.movie_id), None)
    if movie is None:
        parser.error("movie-id is not in movies.json")
    started = time.perf_counter()
    report = {"recorded_at": datetime.now(timezone.utc).isoformat(), "mode": "live_public_api",
              "movie_id": movie.movie_id, "movie_title": movie.title, "year": movie.year,
              "paid_api_calls": 0, "selection": "deterministic identity smoke; P1/P2 LLM not executed"}
    client = None
    try:
        client = WikipediaClient(args.contact or settings.wiki_contact, settings.movies_path.parent / ".cache")
        query = movie_queries(movie)[0]
        candidates = client.search(query)
        decisions = [{"title": page["title"], "page_id": page["page_id"],
                      "identity": identity_check(movie, page)[0]} for page in candidates]
        selected = next((page for page in candidates if identity_check(movie, page)[0] == "movie"), None)
        document = client.fetch(selected["page_id"]) if selected else None
        report.update(query=query, candidates=decisions, success=bool(document))
        if document:
            report.update(document={key: value for key, value in document.items() if key != "text"},
                          body_characters=len(document["text"]))
            before = client.request_count
            self_cached = client.search(query)
            cached_document = client.fetch(selected["page_id"])
            report["repeat_added_http_requests"] = client.request_count - before
            report["repeat_equal"] = self_cached == candidates and cached_document == document
    except WikipediaError as exc:
        report.update(success=False, reason=str(exc))
    report.update(http_requests=client.request_count if client else 0,
                  http_cache_hits=client.cache_hits if client else 0,
                  duration_seconds=time.perf_counter() - started)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
