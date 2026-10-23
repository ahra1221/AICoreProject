"""Create and incrementally update the persistent ChromaDB collections."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Iterable

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import Settings, require_api_key
from src.data_loader import Movie, WikiDocument

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IndexReport:
    collection: str
    before: int
    after: int
    added_or_updated: int
    removed: int


@dataclass(frozen=True)
class IndexTestSelection:
    movies: list[Movie]
    external_ids: list[str]
    external_documents: list[Document]


def external_document_key(movie_id: str, page_id: str, chunk_index: int) -> str:
    return f"{movie_id}:{page_id}:{chunk_index}"


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def create_external_chunks(
    source_documents: Iterable[WikiDocument],
    splitter: RecursiveCharacterTextSplitter,
) -> tuple[list[str], list[Document]]:
    ids: list[str] = []
    documents: list[Document] = []
    for source in source_documents:
        doc_key = f"{source.movie_id}:{source.page_id}"
        for chunk_index, chunk in enumerate(splitter.split_text(source.text)):
            chunk_id = external_document_key(source.movie_id, source.page_id, chunk_index)
            metadata = {
                "movie_id": source.movie_id,
                "page_id": source.page_id,
                "doc_key": doc_key,
                "title": source.title,
                "url": source.url,
                "source": source.source,
                "related_movie": source.related_movie,
                "search_query": source.search_query,
                "collected_at": source.collected_at,
                "chunk_index": chunk_index,
                "content_hash": _content_hash(chunk),
            }
            documents.append(Document(page_content=chunk, metadata=metadata))
            ids.append(chunk_id)
    return ids, documents


def select_index_test_data(
    movies: list[Movie],
    source_documents: Iterable[WikiDocument],
    splitter: RecursiveCharacterTextSplitter,
    movie_limit: int = 100,
    chunk_limit: int = 50,
) -> IndexTestSelection:
    if not 5 <= movie_limit <= 100:
        raise ValueError("Test mode must include between 5 and 100 movies.")
    if not 20 <= chunk_limit <= 50:
        raise ValueError("Test mode must include between 20 and 50 Wikipedia chunks.")

    selected_movies = movies[:movie_limit]
    selected_movie_ids = {movie.movie_id for movie in selected_movies}
    sources_by_movie: dict[str, list[WikiDocument]] = {
        movie.movie_id: [] for movie in selected_movies
    }
    for source in source_documents:
        if source.movie_id in selected_movie_ids:
            sources_by_movie[source.movie_id].append(source)

    external_ids: list[str] = []
    external_documents: list[Document] = []
    for movie in selected_movies:
        ids, chunks = create_external_chunks(sources_by_movie[movie.movie_id], splitter)
        remaining = chunk_limit - len(external_documents)
        external_ids.extend(ids[:remaining])
        external_documents.extend(chunks[:remaining])
        if len(external_documents) == chunk_limit:
            break

    if len(external_documents) < 20:
        raise ValueError(
            f"Test mode found only {len(external_documents)} wiki chunks for "
            f"{len(selected_movies)} movies; at least 20 are required."
        )
    return IndexTestSelection(selected_movies, external_ids, external_documents)


class MovieIndexer:
    def __init__(
        self, settings: Settings, embeddings: OpenAIEmbeddings | None = None
    ):
        settings.chroma_dir.mkdir(parents=True, exist_ok=True)
        embedding_function = embeddings
        if embedding_function is None and settings.gms_key:
            embedding_function = OpenAIEmbeddings(
                model=settings.embedding_model,
                api_key=settings.gms_key,
                base_url=settings.gms_base_url,
            )
        self.settings = settings
        self.embeddings = embedding_function
        self.movie_collection = Chroma(
            collection_name="movie_meta",
            embedding_function=embedding_function,
            persist_directory=str(settings.chroma_dir),
        )
        self.external_collection = Chroma(
            collection_name="external_docs",
            embedding_function=embedding_function,
            persist_directory=str(settings.chroma_dir),
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            add_start_index=True,
        )

    @property
    def has_embeddings(self) -> bool:
        return self.embeddings is not None

    def require_embeddings(self) -> None:
        require_api_key(self.settings)
        if self.embeddings is None:
            raise ValueError("An embedding model is not configured.")

    def counts(self) -> dict[str, int]:
        return {
            "movie_meta": self.movie_collection._collection.count(),
            "external_docs": self.external_collection._collection.count(),
        }

    def indexed_movie_ids(self) -> set[str]:
        result = self.movie_collection.get(include=[])
        return {str(movie_id) for movie_id in result["ids"]}

    def index_movies(
        self, movies: Iterable[Movie], prune: bool = True
    ) -> IndexReport:
        self.require_embeddings()
        documents: list[Document] = []
        ids: list[str] = []
        for movie in movies:
            text = movie.search_text()
            metadata = movie.as_metadata()
            metadata["content_hash"] = _content_hash(text)
            documents.append(Document(page_content=text, metadata=metadata))
            ids.append(movie.movie_id)
        return self._upsert(self.movie_collection, ids, documents, prune=prune)

    def index_external_documents(
        self, source_documents: Iterable[WikiDocument], prune: bool = True
    ) -> IndexReport:
        self.require_embeddings()
        ids, documents = create_external_chunks(source_documents, self.splitter)
        return self._upsert(self.external_collection, ids, documents, prune=prune)

    def _upsert(
        self,
        collection: Chroma,
        ids: list[str],
        documents: list[Document],
        prune: bool,
    ) -> IndexReport:
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate document IDs were generated for a collection.")
        existing = collection.get(include=["metadatas"])
        existing_ids = existing["ids"]
        existing_metadata = dict(zip(existing_ids, existing["metadatas"], strict=True))
        wanted = set(ids)
        changed_documents = [
            (document_id, document)
            for document_id, document in zip(ids, documents, strict=True)
            if existing_metadata.get(document_id, {}).get("content_hash")
            != document.metadata["content_hash"]
        ]
        stale = [document_id for document_id in existing_ids if prune and document_id not in wanted]
        before = len(existing_ids)
        if stale:
            collection.delete(ids=stale)
        batch_size = 128
        for start in range(0, len(changed_documents), batch_size):
            batch = changed_documents[start : start + batch_size]
            collection.add_documents(
                [document for _, document in batch],
                ids=[document_id for document_id, _ in batch],
            )
            logger.info(
                "%s indexing progress: %d/%d",
                collection._collection.name,
                min(start + batch_size, len(changed_documents)),
                len(changed_documents),
            )
        after = collection._collection.count()
        logger.info(
            "%s index updated: %d -> %d documents (%d changed, %d removed)",
            collection._collection.name,
            before,
            after,
            len(changed_documents),
            len(stale),
        )
        return IndexReport(
            collection=collection._collection.name,
            before=before,
            after=after,
            added_or_updated=len(changed_documents),
            removed=len(stale),
        )

    def get_movie_document(self, movie_id: str) -> Document | None:
        result = self.movie_collection.get(
            ids=[movie_id], include=["documents", "metadatas"]
        )
        if not result["ids"]:
            return None
        return Document(
            page_content=result["documents"][0],
            metadata=result["metadatas"][0],
        )

    def movie_candidates(self, movie_id: str) -> list[dict[str, str]]:
        result = self.external_collection.get(
            where={"movie_id": movie_id},
            include=["documents", "metadatas"],
        )
        candidates: dict[str, dict[str, str]] = {}
        for text, metadata in zip(result["documents"], result["metadatas"], strict=True):
            doc_key = metadata["doc_key"]
            if doc_key not in candidates:
                candidates[doc_key] = {
                    "doc_key": doc_key,
                    "title": metadata["title"],
                    "summary": text[:800],
                    "related_movie": metadata.get("related_movie", ""),
                }
        return list(candidates.values())

    def search_movies(self, query: str, limit: int) -> list[Document]:
        self.require_embeddings()
        return self.movie_collection.similarity_search(query, k=limit)

    def search_external(
        self,
        query: str,
        movie_id: str,
        limit: int,
        selected_doc_keys: list[str] | None = None,
    ) -> list[tuple[Document, float]]:
        self.require_embeddings()
        conditions: list[dict[str, object]] = [{"movie_id": movie_id}]
        if selected_doc_keys:
            conditions.append({"doc_key": {"$in": selected_doc_keys}})
        filters: dict[str, object]
        if len(conditions) == 1:
            filters = conditions[0]
        else:
            filters = {"$and": conditions}
        return self.external_collection.similarity_search_with_relevance_scores(
            query,
            k=limit,
            filter=filters,
        )
