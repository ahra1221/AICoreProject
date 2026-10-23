"""Retrieve scoped movie evidence and connect citations to their source documents."""

from __future__ import annotations

import re
from typing import Any

from langchain_core.documents import Document

from src.data_loader import Movie
from src.indexer import MovieIndexer

CITATION_PATTERN = re.compile(r"\[근거\s*([\d,\s]+)\]")


def cited_sources(
    answer: str, sources: list[dict[str, Any]]
) -> list[tuple[int, dict[str, Any]]]:
    cited_numbers = {
        int(value)
        for group in CITATION_PATTERN.findall(answer)
        for value in group.split(",")
        if value.strip().isdigit()
    }
    return [
        (int(source.get("number", number)), source)
        for number, source in enumerate(sources, start=1)
        if int(source.get("number", number)) in cited_numbers
    ]


def _source_from_document(
    document: Document, number: int, source_type: str
) -> dict[str, Any]:
    metadata = document.metadata
    return {
        "number": number,
        "title": metadata.get("title", ""),
        "url": metadata.get("url") or metadata.get("tmdb_url", ""),
        "source": metadata.get("source", source_type),
        "source_type": source_type,
        "movie_id": str(metadata.get("movie_id", "")),
        "page_id": str(metadata.get("page_id", "")),
        "doc_key": str(metadata.get("doc_key", "")),
        "related_movie": metadata.get("related_movie", ""),
        "text": document.page_content,
        "license": metadata.get("license", ""),
        "license_url": metadata.get("license_url", ""),
        "revision_id": metadata.get("revision_id", ""),
        "collected_at": metadata.get("collected_at", ""),
    }


class EvidenceRetriever:
    def __init__(self, indexer: MovieIndexer):
        self.indexer = indexer

    def retrieve(
        self,
        movie: Movie,
        queries: list[str],
        selected_doc_keys: list[str],
        top_k: int,
        min_relevance: float,
    ) -> list[dict[str, Any]]:
        movie_document = self.indexer.get_movie_document(movie.movie_id)
        if movie_document is None:
            raise ValueError(
                f"Movie {movie.movie_id} is not indexed. Run scripts/build_index.py first."
            )
        results = [_source_from_document(movie_document, 1, "영화 메타데이터 (TMDB)")]
        ranked: dict[str, tuple[Document, float]] = {}

        if not selected_doc_keys:
            return results
        for query in queries:
            matches = self.indexer.search_external(
                query,
                movie.movie_id,
                top_k,
                selected_doc_keys=selected_doc_keys,
            )
            for document, relevance in matches:
                if relevance < min_relevance:
                    continue
                if (str(document.metadata.get("movie_id")) != movie.movie_id
                    or document.metadata.get("doc_key") not in selected_doc_keys):
                    continue
                document_id = str(document.metadata.get("doc_key", "")) + ":" + str(
                    document.metadata.get("chunk_index", "")
                )
                if document_id not in ranked or relevance > ranked[document_id][1]:
                    ranked[document_id] = (document, relevance)
        for document, relevance in sorted(ranked.values(), key=lambda item: item[1], reverse=True)[:top_k]:
            source = _source_from_document(document, len(results) + 1, "한국어 위키백과")
            source["relevance"] = relevance
            results.append(source)
        return results
