"""Load and validate the supplied movie metadata and Wikipedia snapshots."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class DataFileError(ValueError):
    """Raised when a supplied data file is missing or malformed."""


def _string_list(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(item).strip() for item in value if str(item).strip())


@dataclass(frozen=True)
class Movie:
    movie_id: str
    title: str
    original_title: str = ""
    release_date: str = ""
    year: int | None = None
    genres: tuple[str, ...] = ()
    director: tuple[str, ...] = ()
    cast: tuple[str, ...] = ()
    overview: str = ""
    runtime: int | None = None
    original_language: str = ""
    production_countries: tuple[str, ...] = ()
    vote_average: float | None = None
    vote_count: int | None = None
    tmdb_url: str = ""

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Movie:
        movie_id = str(value.get("movie_id", "")).strip()
        title = str(value.get("title", "")).strip()
        if not movie_id or not title:
            raise DataFileError("Every movie must have a non-empty movie_id and title.")
        return cls(
            movie_id=movie_id,
            title=title,
            original_title=str(value.get("original_title") or ""),
            release_date=str(value.get("release_date") or ""),
            year=_optional_int(value.get("year")),
            genres=_string_list(value.get("genres")),
            director=_string_list(value.get("director")),
            cast=_string_list(value.get("cast")),
            overview=str(value.get("overview") or ""),
            runtime=_optional_int(value.get("runtime")),
            original_language=str(value.get("original_language") or ""),
            production_countries=_string_list(value.get("production_countries")),
            vote_average=_optional_float(value.get("vote_average")),
            vote_count=_optional_int(value.get("vote_count")),
            tmdb_url=str(value.get("tmdb_url") or ""),
        )

    def search_text(self) -> str:
        fields = [
            f"제목: {self.title}",
            f"원제: {self.original_title}" if self.original_title else "",
            f"개봉연도: {self.year or self.release_date}",
            f"장르: {', '.join(self.genres)}" if self.genres else "",
            f"감독: {', '.join(self.director)}" if self.director else "",
            f"출연: {', '.join(self.cast)}" if self.cast else "",
            f"줄거리: {self.overview}" if self.overview else "",
            f"러닝타임: {self.runtime}분" if self.runtime else "",
        ]
        return "\n".join(field for field in fields if field)

    def as_metadata(self) -> dict[str, str | int | float]:
        metadata: dict[str, str | int | float] = {
            "movie_id": self.movie_id,
            "title": self.title,
            "original_title": self.original_title,
            "release_date": self.release_date,
            "genres": ", ".join(self.genres),
            "director": ", ".join(self.director),
            "cast": ", ".join(self.cast),
            "overview": self.overview,
            "original_language": self.original_language,
            "production_countries": ", ".join(self.production_countries),
            "tmdb_url": self.tmdb_url,
        }
        for key, value in (
            ("year", self.year),
            ("runtime", self.runtime),
            ("vote_average", self.vote_average),
            ("vote_count", self.vote_count),
        ):
            if value is not None:
                metadata[key] = value
        return metadata


@dataclass(frozen=True)
class WikiDocument:
    page_id: str
    title: str
    url: str
    text: str
    source: str
    movie_id: str
    related_movie: str = ""
    search_query: str = ""
    collected_at: str = ""

    @classmethod
    def from_dict(cls, value: dict[str, Any], file_path: Path) -> WikiDocument:
        fields = ("title", "url", "text", "movie_id")
        missing = [field for field in fields if not str(value.get(field, "")).strip()]
        if missing:
            raise DataFileError(f"{file_path} is missing required fields: {', '.join(missing)}")
        page_id = str(value.get("page_id") or file_path.stem).strip()
        return cls(
            page_id=page_id,
            title=str(value["title"]).strip(),
            url=str(value["url"]).strip(),
            text=str(value["text"]),
            source=str(value.get("source") or "한국어 위키백과"),
            movie_id=str(value["movie_id"]).strip(),
            related_movie=str(value.get("related_movie") or ""),
            search_query=str(value.get("search_query") or ""),
            collected_at=str(value.get("collected_at") or ""),
        )


def _optional_int(value: Any) -> int | None:
    try:
        return int(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError as exc:
        raise DataFileError(f"Data file was not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise DataFileError(f"Invalid JSON in {path}: {exc}") from exc


def load_movies(path: Path) -> list[Movie]:
    raw_movies = _read_json(path)
    if not isinstance(raw_movies, list):
        raise DataFileError(f"Expected a JSON array in {path}.")
    movies = [
        Movie.from_dict(item) for item in raw_movies
        if isinstance(item, dict)
    ]
    if len(movies) != len(raw_movies):
        raise DataFileError(f"Every item in {path} must be a JSON object.")
    ids = [movie.movie_id for movie in movies]
    if len(ids) != len(set(ids)):
        raise DataFileError(f"Duplicate movie_id values were found in {path}.")
    logger.info("Loaded %d movies from %s", len(movies), path)
    return movies


def load_wiki_documents(directory: Path) -> list[WikiDocument]:
    if not directory.is_dir():
        raise DataFileError(f"Wikipedia data directory was not found: {directory}")
    paths = sorted(directory.glob("*.json"))
    documents: list[WikiDocument] = []
    for path in paths:
        value = _read_json(path)
        if not isinstance(value, dict):
            raise DataFileError(f"Expected a JSON object in {path}.")
        documents.append(WikiDocument.from_dict(value, path))
    logger.info("Loaded %d Wikipedia documents from %s", len(documents), directory)
    return documents


def load_wiki_manifest(path: Path) -> dict[str, dict[str, Any]]:
    value = _read_json(path)
    if not isinstance(value, dict):
        raise DataFileError(f"Expected a JSON object in {path}.")
    return value
