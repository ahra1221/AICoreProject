"""Actual local Chroma persistence with deterministic, nonsemantic test vectors."""
import json
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from offline_support import LocalTestEmbeddings
from src.config import get_settings
from src.data_loader import WikiDocument, load_movies, load_wiki_documents
from src.indexer import MovieIndexer
from src.retriever import EvidenceRetriever

ROOT = Path(__file__).resolve().parents[1]


class LocalIndexTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(tempfile.mkdtemp(prefix="local-chroma-"))
        cls.settings = replace(get_settings({}), chroma_dir=cls.root / "db")
        cls.index = MovieIndexer(cls.settings, embeddings=LocalTestEmbeddings())
        cls.movies = load_movies(cls.settings.movies_path)
        cls.report = {"mode": "real_local_chroma_with_test_vectors", "network": "blocked",
                      "semantic_quality_tested": False}

    @classmethod
    def tearDownClass(cls):
        output = ROOT / "docs/evidence"
        output.mkdir(parents=True, exist_ok=True)
        (output / "local-index.json").write_text(json.dumps(cls.report, ensure_ascii=False, indent=2), encoding="utf-8")
        # Chroma holds Windows file handles; ignored test-temp is left intact.

    def test_metadata_count_and_idempotency_for_all_964_movies(self):
        first = self.index.index_movies(self.movies)
        second = self.index.index_movies(self.movies)
        self.assertEqual(first.after, 964)
        self.assertEqual(second.after, 964)
        self.assertEqual(second.added_or_updated, 0)
        sample = self.index.get_movie_document(self.movies[0].movie_id)
        for label in ("제목:", "원제:", "개봉연도:", "장르:", "감독:", "출연:", "줄거리:"):
            self.assertIn(label, sample.page_content)
        self.report.update(metadata_items=len(self.movies), first_index=asdict(first), second_index=asdict(second),
                           sample_metadata_text=sample.page_content)

    def test_chunk_change_reindexes_without_stale_tails_and_preserves_other_movie(self):
        first_doc = WikiDocument("7", "예시", "https://ko.wikipedia.org/?curid=7", "자료에 관한 설명 문장입니다. " * 70,
                                 "한국어 위키백과", "1")
        second_doc = replace(first_doc, movie_id="2")
        self.index.splitter = RecursiveCharacterTextSplitter(chunk_size=120, chunk_overlap=20)
        first = self.index.index_external_documents([first_doc, second_doc], prune=False)
        before = len(self.index.external_collection.get(where={"movie_id": "1"})["ids"])
        self.index.splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)
        second = self.index.index_external_documents([first_doc], prune=False)
        after = len(self.index.external_collection.get(where={"movie_id": "1"})["ids"])
        self.assertLess(after, before)
        self.assertEqual(len(self.index.external_collection.get(where={"movie_id": "2"})["ids"]), before)
        repeated = self.index.index_external_documents([first_doc], prune=False)
        self.assertEqual(repeated.added_or_updated, 0)
        self.report["chunk_comparison"] = {"before": {"size": 120, "overlap": 20, "chunks": before},
                                            "after": {"size": 400, "overlap": 40, "chunks": after},
                                            "other_movie_chunks_preserved": before,
                                            "repeat_added_or_updated": repeated.added_or_updated}
