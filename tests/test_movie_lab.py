from __future__ import annotations

import unittest
import tempfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from app import _indexed_movie_ids
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from chromadb.api.types import validate_where

from src.collector import DocumentCollector, SearchPlan
from src.config import ConfigurationError, get_settings, require_api_key
from src.data_loader import Movie, WikiDocument, load_movies, load_wiki_documents
from src.graph import create_rag_graph
from src.indexer import (
    MovieIndexer,
    create_external_chunks,
    external_document_key,
    select_index_test_data,
)
from src.retriever import EvidenceRetriever, cited_sources
from src.services import find_movies, movie_chat_history

ROOT_DIR = Path(__file__).resolve().parents[1]


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        settings = get_settings(environ={})
        cls.movies = load_movies(settings.movies_path)
        cls.wiki_documents = load_wiki_documents(settings.wiki_dir)

    def test_movie_dataset_has_964_unique_ids(self) -> None:
        self.assertEqual(len(self.movies), 964)
        ids = [movie.movie_id for movie in self.movies]
        self.assertEqual(len(ids), len(set(ids)))

    def test_wikipedia_documents_have_required_metadata(self) -> None:
        self.assertEqual(len(self.wiki_documents), 3725)
        document = self.wiki_documents[0]
        self.assertTrue(document.page_id)
        self.assertTrue(document.movie_id)
        self.assertTrue(document.title)
        self.assertTrue(document.url.startswith("https://"))
        self.assertTrue(document.text)


class ConfigurationTests(unittest.TestCase):
    def test_api_key_is_required_only_when_requested(self) -> None:
        settings = get_settings(environ={})
        with self.assertRaises(ConfigurationError):
            require_api_key(settings)
        self.assertTrue(require_api_key(get_settings(environ={"GMS_KEY": "test-key"})))
        self.assertEqual(
            get_settings(environ={"GMS_KEY": "your_gms_api_key"}).gms_key,
            "",
        )

    def test_invalid_chunk_overlap_is_rejected(self) -> None:
        with self.assertRaises(ConfigurationError):
            get_settings(environ={"CHUNK_SIZE": "100", "CHUNK_OVERLAP": "100"})


class ChunkAndIndexTests(unittest.TestCase):
    def setUp(self) -> None:
        self.wiki_document = WikiDocument(
            page_id="42",
            title="영화와 과학",
            url="https://ko.wikipedia.org/wiki/example",
            text=("사실 자료입니다. " * 30),
            source="한국어 위키백과",
            movie_id="100",
            related_movie="영화",
            search_query="과학",
            collected_at="2026-09-01",
        )

    def test_chunk_metadata_and_ids_are_stable_and_movie_scoped(self) -> None:
        splitter = RecursiveCharacterTextSplitter(chunk_size=40, chunk_overlap=8)
        ids, chunks = create_external_chunks([self.wiki_document], splitter)
        self.assertGreater(len(chunks), 1)
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(ids[0], external_document_key("100", "42", 0))
        self.assertTrue(all(chunk.metadata["url"] == self.wiki_document.url for chunk in chunks))
        self.assertTrue(all(chunk.metadata["movie_id"] == "100" for chunk in chunks))
        self.assertTrue(all("content_hash" in chunk.metadata for chunk in chunks))
        other_movie_id = external_document_key("200", "42", 0)
        self.assertNotEqual(ids[0], other_movie_id)

    def test_reindex_skips_unchanged_and_prunes_stale_documents(self) -> None:
        collection = FakeCollection()
        indexer = MovieIndexer.__new__(MovieIndexer)
        document = Document(
            page_content="영화 제목: 예시",
            metadata={"content_hash": "hash-1"},
        )
        first = indexer._upsert(collection, ["movie-1"], [document], prune=True)
        second = indexer._upsert(collection, ["movie-1"], [document], prune=True)
        self.assertEqual(first.after, 1)
        self.assertEqual(first.added_or_updated, 1)
        self.assertEqual(second.added_or_updated, 0)
        self.assertEqual(collection.add_calls, 1)

        updated = Document(
            page_content="수정된 영화",
            metadata={"content_hash": "hash-2"},
        )
        third = indexer._upsert(collection, ["movie-2"], [updated], prune=True)
        self.assertEqual(third.removed, 1)
        self.assertEqual(third.added_or_updated, 1)
        self.assertEqual(collection.get(include=["metadatas"])["ids"], ["movie-2"])

    def test_retry_after_partial_failure_only_reindexes_missing_items(self) -> None:
        collection = FakeCollection(fail_after_first_add=True)
        indexer = MovieIndexer.__new__(MovieIndexer)
        documents = [
            Document(page_content=f"Movie {number}", metadata={"content_hash": f"hash-{number}"})
            for number in range(2)
        ]
        with self.assertRaises(RuntimeError):
            indexer._upsert(collection, ["movie-0", "movie-1"], documents, prune=False)

        report = indexer._upsert(
            collection, ["movie-0", "movie-1"], documents, prune=False
        )
        self.assertEqual(report.added_or_updated, 1)
        self.assertEqual(report.after, 2)
        self.assertEqual(collection.add_calls, 2)

    def test_test_sample_obeys_movie_and_chunk_limits(self) -> None:
        movies = [
            Movie(movie_id=str(number), title=f"영화 {number}")
            for number in range(1, 106)
        ]
        documents = [
            WikiDocument(
                page_id=f"{movie.movie_id}-{page}",
                title=f"문서 {movie.movie_id}-{page}",
                url="https://ko.wikipedia.org/wiki/example",
                text="영화에 관한 근거 자료입니다. " * 12,
                source="한국어 위키백과",
                movie_id=movie.movie_id,
                related_movie=movie.title,
            )
            for movie in movies
            for page in range(2)
        ]
        selection = select_index_test_data(
            movies,
            documents,
            RecursiveCharacterTextSplitter(chunk_size=40, chunk_overlap=8),
            movie_limit=100,
            chunk_limit=20,
        )
        self.assertEqual(len(selection.movies), 100)
        self.assertEqual(len(selection.external_documents), 20)
        self.assertTrue(
            {
                str(document.metadata["movie_id"])
                for document in selection.external_documents
            }.issubset({movie.movie_id for movie in selection.movies})
        )
        with self.assertRaises(ValueError):
            select_index_test_data(
                movies,
                documents,
                RecursiveCharacterTextSplitter(chunk_size=40, chunk_overlap=8),
                movie_limit=101,
                chunk_limit=20,
            )

    def test_scoped_retrieval_filter_uses_valid_chroma_and_syntax(self) -> None:
        indexer = MovieIndexer.__new__(MovieIndexer)
        indexer.embeddings = object()
        indexer.settings = get_settings(environ={"GMS_KEY": "test-key"})
        collection = FakeSearchCollection()
        indexer.external_collection = collection

        indexer.search_external(
            "블랙홀",
            movie_id="157336",
            limit=5,
            selected_doc_keys=["157336:21878"],
        )

        validate_where(collection.where)
        self.assertEqual(
            collection.where,
            {
                "$and": [
                    {"movie_id": "157336"},
                    {"doc_key": {"$in": ["157336:21878"]}},
                ]
            },
        )


class CitationAndRetrievalTests(unittest.TestCase):
    def test_citations_only_return_valid_source_numbers(self) -> None:
        sources = [{"title": "하나"}, {"title": "둘"}]
        self.assertEqual(
            cited_sources("주장 [근거 1, 99] 및 [근거 2]", sources),
            [(1, sources[0]), (2, sources[1])],
        )
        self.assertEqual(cited_sources("근거 인용이 없습니다.", sources), [])

    def test_retrieval_always_includes_movie_and_deduplicates_chunks(self) -> None:
        movie = Movie(movie_id="1", title="예시 영화", year=2020)
        movie_document = Document(
            page_content="제목: 예시 영화",
            metadata={"movie_id": "1", "title": "예시 영화", "tmdb_url": "https://www.themoviedb.org/movie/1"},
        )
        wiki_document = Document(
            page_content="관련 위키 근거",
            metadata={
                "movie_id": "1",
                "page_id": "42",
                "doc_key": "1:42",
                "chunk_index": 0,
                "title": "예시 문서",
                "url": "https://ko.wikipedia.org/wiki/example",
                "source": "한국어 위키백과",
            },
        )
        fake_indexer = FakeRetrieverIndexer(movie_document, wiki_document)
        sources = EvidenceRetriever(fake_indexer).retrieve(
            movie,
            queries=["첫 검색어", "다른 검색어"],
            selected_doc_keys=["1:42"],
            top_k=3,
            min_relevance=0.1,
        )
        self.assertEqual(len(sources), 2)
        self.assertEqual(sources[0]["movie_id"], "1")
        self.assertEqual(sources[1]["title"], "예시 문서")
        self.assertEqual(fake_indexer.requested_movie_ids, ["1", "1"])

    def test_metadata_remains_the_only_evidence_without_selected_documents(self) -> None:
        movie = Movie(movie_id="1", title="예시 영화")
        document = Document(
            page_content="제목: 예시 영화",
            metadata={"movie_id": "1", "title": "예시 영화"},
        )
        sources = EvidenceRetriever(
            FakeRetrieverIndexer(document, document)
        ).retrieve(
            movie,
            queries=["관련 질문"],
            selected_doc_keys=[],
            top_k=5,
            min_relevance=0.1,
        )
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0]["title"], "예시 영화")

class CollectorAndGraphTests(unittest.TestCase):
    def test_query_plan_always_contains_movie_and_movie_document(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        settings = replace(get_settings(environ={}), wiki_mode="snapshot", movies_path=root / "movies.json",
                           wiki_dir=root / "wikipedia", chroma_dir=root / "index")
        movie = Movie(
            movie_id="157336",
            title="인터스텔라",
            original_title="Interstellar",
        )
        model = FakeModel(
            [
                '{"queries":["블랙홀"]}',
                '{"doc_keys":["157336:42"]}',
            ]
        )
        fake_indexer = FakeCandidateIndexer(
            [
                {
                    "doc_key": "157336:42",
                    "title": "인터스텔라",
                    "summary": "영화 설명",
                    "related_movie": "인터스텔라",
                },
                {
                    "doc_key": "157336:99",
                    "title": "웜홀",
                    "summary": "배경 설명",
                    "related_movie": "인터스텔라",
                },
            ]
        )
        plan = DocumentCollector(settings, model, fake_indexer).prepare(
            movie, "블랙홀의 묘사는 어떤가요?"
        )
        self.assertIn("인터스텔라 영화", plan.queries)
        self.assertIn("Interstellar film", plan.queries)
        self.assertEqual(plan.selected_doc_keys, ("157336:42",))

    def test_graph_executes_separate_retrieve_and_generate_nodes(self) -> None:
        settings = get_settings(environ={"GMS_KEY": "test-key"})
        movie = Movie(movie_id="1", title="예시")
        source = {
            "number": 1,
            "title": "예시",
            "url": "https://www.themoviedb.org/movie/1",
            "source": "TMDB",
            "text": "영화 메타데이터",
        }
        graph = create_rag_graph(
            model=FakeModel(['답변 [근거 1]']),
            collector=FakeGraphCollector(),
            retriever=FakeGraphRetriever([source]),
            settings=settings,
        )
        updates = list(
            graph.stream(
                {"movie": movie, "question": "질문", "mode": "general"},
                stream_mode="updates",
            )
        )
        self.assertEqual([next(iter(update)) for update in updates], ["retrieve", "generate"])
        final = {}
        for update in updates:
            final.update(next(iter(update.values())))
        self.assertEqual(final["answer"], "답변 [근거 1]")
        self.assertEqual(final["used_evidence_numbers"], [1])


class MovieDiscoveryAndChatTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        settings = get_settings(environ={})
        cls.movies = load_movies(settings.movies_path)

    def test_explicit_year_and_genre_conditions_use_real_metadata(self) -> None:
        results = find_movies(
            "2010년 이후 개봉한 애니메이션",
            self.movies,
            FakeMovieSearchIndexer(),
        )
        self.assertTrue(results)
        self.assertTrue(
            all(movie.year is not None and movie.year > 2010 for movie in results)
        )
        self.assertTrue(
            all("애니메이션" in movie.genres for movie in results)
        )

    def test_director_condition_matches_movie_metadata(self) -> None:
        results = find_movies(
            "크리스토퍼 놀란 감독의 영화",
            self.movies,
            FakeMovieSearchIndexer(),
        )
        self.assertTrue(results)
        self.assertTrue(
            all("크리스토퍼 놀란" in movie.director for movie in results)
        )

    def test_movie_chat_histories_are_isolated(self) -> None:
        histories: dict[str, list[dict[str, object]]] = {}
        first = movie_chat_history(histories, "movie-1")
        first.append({"role": "assistant", "content": "one"})
        second = movie_chat_history(histories, "movie-2")
        self.assertEqual(second, [])
        self.assertEqual(len(movie_chat_history(histories, "movie-1")), 1)

    def test_app_reads_indexed_ids_without_relying_on_a_cached_indexer_method(self) -> None:
        legacy_indexer = SimpleNamespace(
            movie_collection=SimpleNamespace(
                get=lambda include: {"ids": ["157336", "27205"]}
            )
        )
        self.assertEqual(_indexed_movie_ids(legacy_indexer), {"157336", "27205"})


class FakeCollection:
    def __init__(self, fail_after_first_add: bool = False) -> None:
        self._collection = self
        self.name = "fake"
        self.items: dict[str, Document] = {}
        self.add_calls = 0
        self.fail_after_first_add = fail_after_first_add

    def count(self) -> int:
        return len(self.items)

    def get(self, include: list[str]) -> dict[str, list]:
        return {
            "ids": list(self.items),
            "metadatas": [document.metadata for document in self.items.values()],
        }

    def delete(self, ids: list[str]) -> None:
        for document_id in ids:
            self.items.pop(document_id, None)

    def add_documents(self, documents: list[Document], ids: list[str]) -> None:
        self.add_calls += 1
        if self.fail_after_first_add:
            self.items[ids[0]] = documents[0]
            self.fail_after_first_add = False
            raise RuntimeError("Simulated partial batch failure")
        self.items.update(zip(ids, documents, strict=True))


class FakeSearchCollection:
    def __init__(self) -> None:
        self.where = None

    def similarity_search_with_relevance_scores(
        self, query: str, k: int, filter: dict[str, object]
    ) -> list[tuple[Document, float]]:
        self.where = filter
        return []


class FakeRetrieverIndexer:
    def __init__(self, movie_document: Document, wiki_document: Document) -> None:
        self.movie_document = movie_document
        self.wiki_document = wiki_document
        self.requested_movie_ids: list[str] = []

    def get_movie_document(self, movie_id: str) -> Document:
        return self.movie_document

    def search_external(
        self, query: str, movie_id: str, limit: int, selected_doc_keys: list[str]
    ) -> list[tuple[Document, float]]:
        self.requested_movie_ids.append(movie_id)
        return [(self.wiki_document, 0.9)]


class FakeCandidateIndexer:
    def __init__(self, candidates: list[dict[str, str]]) -> None:
        self.candidates = candidates

    def movie_candidates(self, movie_id: str) -> list[dict[str, str]]:
        return self.candidates


class FakeGraphCollector:
    def prepare(self, movie: Movie, question: str) -> SearchPlan:
        return SearchPlan(("예시",), ("1:42",))


class FakeGraphRetriever:
    def __init__(self, sources: list[dict[str, object]]) -> None:
        self.sources = sources

    def retrieve(self, **kwargs: object) -> list[dict[str, object]]:
        return self.sources


class FakeModel:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls = 0

    def invoke(self, prompt: str) -> SimpleNamespace:
        response = self.responses[self.calls]
        self.calls += 1
        return SimpleNamespace(content=response)


class FakeMovieSearchIndexer:
    def search_movies(self, query: str, limit: int) -> list[Document]:
        return []


if __name__ == "__main__":
    unittest.main()
