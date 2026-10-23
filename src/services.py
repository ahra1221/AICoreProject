"""Natural-language movie discovery and explicit metadata filtering."""

from __future__ import annotations

import re

from src.data_loader import Movie
from src.indexer import MovieIndexer

YEAR_CONDITION = re.compile(r"(?P<year>\d{4})\s*년\s*(?P<direction>이후|이전|부터|까지)")


def movie_chat_history(
    chat_store: dict[str, list[dict[str, object]]], movie_id: str
) -> list[dict[str, object]]:
    return chat_store.setdefault(movie_id, [])


def find_movies(
    query: str,
    movies: list[Movie],
    indexer: MovieIndexer,
    limit: int = 12,
) -> list[Movie]:
    if not query.strip():
        return []

    semantic_documents = indexer.search_movies(query, limit=max(limit * 6, 60))
    by_id = {movie.movie_id: movie for movie in movies}
    candidates: dict[str, Movie] = {
        str(document.metadata["movie_id"]): by_id[str(document.metadata["movie_id"])]
        for document in semantic_documents
        if str(document.metadata.get("movie_id", "")) in by_id
    }

    query_folded = query.casefold()
    known_genres = {
        genre
        for movie in movies
        for genre in movie.genres
        if genre.casefold() in query_folded
    }
    known_directors = {
        director
        for movie in movies
        for director in movie.director
        if director.casefold() in query_folded
    }
    year_match = YEAR_CONDITION.search(query)

    if known_genres or known_directors or year_match:
        candidates = {movie.movie_id: movie for movie in movies}

    def matches(movie: Movie) -> bool:
        if known_genres and not known_genres.intersection(movie.genres):
            return False
        if known_directors and not known_directors.intersection(movie.director):
            return False
        if year_match:
            if movie.year is None:
                return False
            year = int(year_match.group("year"))
            direction = year_match.group("direction")
            if direction == "이후" and movie.year <= year:
                return False
            if direction == "이전" and movie.year >= year:
                return False
            if direction == "부터" and movie.year < year:
                return False
            if direction == "까지" and movie.year > year:
                return False
        return True

    results = [movie for movie in candidates.values() if matches(movie)]
    if not (known_genres or known_directors or year_match):
        semantic_order = {str(doc.metadata["movie_id"]): i for i, doc in enumerate(semantic_documents)}
        results.sort(key=lambda movie: semantic_order.get(movie.movie_id, len(semantic_order)))
    else:
        results.sort(
            key=lambda movie: (
                -(movie.vote_average or 0.0),
                -(movie.year or 0),
                movie.title.casefold(),
            )
        )
    return results[:limit]
