"""Offline contracts: HTTP, model and embeddings are controlled test doubles."""
from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import requests
from langchain_core.documents import Document

from src.collector import DocumentCollector, SearchPlan, identity_check
from src.config import ConfigurationError, get_settings
from src.data_loader import Movie
from src.graph import INSUFFICIENT_EVIDENCE, create_rag_graph, validate_answer
from src.retriever import EvidenceRetriever
from src.wikipedia import API_URL, JsonCache, WikipediaClient, WikipediaError


class Response:
    def __init__(self, payload=None, status=200, headers=None):
        self.payload, self.status_code, self.headers = payload, status, headers or {}
    def json(self):
        return self.payload
    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError("Private server body")


class Session:
    def __init__(self, responses):
        self.responses, self.headers, self.calls = list(responses), {}, []
    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class ClientTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.sleep = patch("src.wikipedia.time.sleep").start()
        self.addCleanup(patch.stopall)

    def client(self, responses):
        session = Session(responses)
        return WikipediaClient("https://github.com/ahra1221/AICoreProject", self.root, session=session), session

    def test_search_preview_body_separation_and_cache(self):
        preview = {"query": {"pages": [{"pageid": 7, "title": "킹콩 (2005년 영화)", "extract": "2005년 영화"}]}}
        body = {"query": {"pages": [{"pageid": 7, "title": "킹콩 (2005년 영화)", "extract": "긴 본문 " * 300,
                 "revisions": [{"revid": 9, "timestamp": "2026-10-23T00:00:00Z"}]}]}}
        client, session = self.client([Response(preview), Response(body)])
        candidates = client.search("킹콩 2005 영화")
        document = client.fetch(candidates[0]["page_id"])
        self.assertEqual(document, client.fetch("7"))
        self.assertEqual(len(session.calls), 2)
        self.assertEqual(session.calls[0][0], API_URL)
        self.assertEqual(session.calls[0][1]["params"]["exintro"], 1)
        self.assertNotIn("exintro", session.calls[1][1]["params"])
        self.assertNotIn("exchars", session.calls[1][1]["params"])
        self.assertEqual(document["revision_id"], "9")
        self.assertEqual(document["license"], "CC BY-SA 4.0")
        self.assertEqual(client.cache_hits, 1)
        self.assertTrue(session.headers["User-Agent"].startswith("MovieLab/2.0 (https://github.com/"))
        self.assertEqual(session.calls[0][1]["params"]["maxlag"], 5)

    def test_retry_after_and_http200_maxlag(self):
        client, session = self.client([Response(status=429, headers={"Retry-After": "7"}),
                                      Response({"error": {"code": "maxlag"}}), Response({"query": {"pages": []}})])
        self.assertEqual(client.search("영화"), [])
        self.assertEqual(len(session.calls), 3)
        self.assertIn(7, [call.args[0] for call in self.sleep.call_args_list])

    def test_errors_are_safe_not_cached_and_not_retried_on_403(self):
        client, session = self.client([Response(status=403)])
        with self.assertRaises(WikipediaError) as context:
            client.search("영화")
        self.assertNotIn("Private", str(context.exception))
        self.assertEqual(len(session.calls), 1)
        self.assertFalse(list(self.root.rglob("*.json")))

    def test_long_retry_after_stops(self):
        client, session = self.client([Response(status=429, headers={"Retry-After": "120"})])
        with self.assertRaises(WikipediaError):
            client.search("영화")
        self.assertEqual(len(session.calls), 1)

    def test_empty_missing_and_disambiguation(self):
        client, _ = self.client([Response({"batchcomplete": True}),
                                Response({"query": {"pages": [{"pageid": 7, "missing": True}]}}),
                                Response({"query": {"pages": [{"pageid": 8, "extract": "뜻",
                                          "pageprops": {"disambiguation": ""}}]}})])
        self.assertEqual(client.search("없음"), [])
        self.assertIsNone(client.fetch("7"))
        self.assertIsNone(client.fetch("8"))

    def test_cache_expiry_and_corruption(self):
        cache = JsonCache(self.root, 60)
        with patch("src.wikipedia.time.time", return_value=100):
            cache.put("item", {"ok": True})
        with patch("src.wikipedia.time.time", return_value=120):
            self.assertEqual(cache.get("item"), {"ok": True})
        with patch("src.wikipedia.time.time", return_value=161):
            self.assertIsNone(cache.get("item"))
        cache.path("item").write_text("broken", encoding="utf-8")
        self.assertIsNone(cache.get("item"))

    def test_configuration_limits_and_empty_example(self):
        for contact in ("", "test\r\nInjected: yes"):
            with self.assertRaises(WikipediaError):
                WikipediaClient(contact, self.root)
        for env in ({"WIKI_REQUEST_INTERVAL": "0"}, {"WIKI_MAX_QUERIES": "40"}, {"WIKI_MODE": "other"}):
            with self.assertRaises(ConfigurationError):
                get_settings(env)
        self.assertEqual(get_settings({"CHUNK_SIZE": "", "GMS_MODEL": ""}).chunk_size, 1000)


class FakeWiki:
    def __init__(self, events):
        self.events, self.request_count, self.fetched = events, 0, []
    def search(self, query):
        self.events.append("search")
        self.request_count += 1
        return [{"page_id": "5", "title": "킹콩 (2005년 영화)", "summary": "2005년 영화이다."},
                {"page_id": "3", "title": "킹콩 (1933년 영화)", "summary": "1933년 영화이다."},
                {"page_id": "9", "title": "킹콩", "summary": "여러 뜻", "disambiguation": True},
                {"page_id": "8", "title": "특수 효과", "summary": "시각 효과"}]
    def fetch(self, page_id):
        self.events.append("body")
        self.request_count += 1
        self.fetched.append(page_id)
        return {"page_id": page_id, "title": "킹콩 (2005년 영화)", "text": "2005년 피터 잭슨 감독의 영화이다.",
                "url": "https://ko.wikipedia.org/?curid=5", "source": "한국어 위키백과"}


class FakeModel:
    def __init__(self, events):
        self.events, self.calls = events, 0
    def invoke(self, prompt):
        self.calls += 1
        if "프롬프트 1(P1)" in prompt:
            self.events.append("P1")
            return SimpleNamespace(content='{"queries":["킹콩 특수 효과"]}')
        self.events.append("P2")
        return SimpleNamespace(content=json.dumps({"decisions": [
            {"doc_key": "1:5", "adopt": True, "reason": "선택 영화"},
            {"doc_key": "1:3", "adopt": True, "reason": "잘못된 동명 채택"},
            {"doc_key": "1:9", "adopt": True, "reason": "동음이의"},
            {"doc_key": "1:8", "adopt": False, "reason": "이번 질문과 무관"}]}))


class FakeIndex:
    def __init__(self, events):
        self.events, self.documents = events, {}
    def movie_candidates(self, movie_id):
        return [{"doc_key": key, "title": doc.title, "summary": doc.text}
                for key, doc in self.documents.items() if doc.movie_id == movie_id]
    def index_external_documents(self, docs, prune=False):
        self.events.append("index")
        self.documents.update({f"{doc.movie_id}:{doc.page_id}": doc for doc in docs})


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.settings = replace(get_settings({}), movies_path=root / "movies.json", chroma_dir=root / "db")
        self.events = []
        self.model, self.wiki, self.index = FakeModel(self.events), FakeWiki(self.events), FakeIndex(self.events)
        self.movie = Movie("1", "킹콩", "King Kong", year=2005, director=("피터 잭슨",))

    def test_order_identity_and_persistent_question_cache(self):
        collector = DocumentCollector(self.settings, self.model, self.index, self.wiki)
        first = collector.prepare(self.movie, "어떤 영화인가요?")
        events_after_first = list(self.events)
        second = collector.prepare(self.movie, "어떤 영화인가요?")
        self.assertIn("킹콩 2005 영화", first.queries)
        self.assertEqual(first.selected_doc_keys, ("1:5",))
        self.assertEqual(self.wiki.fetched, ["5"])
        for left, right in zip(["P1", "search", "P2", "body"], ["search", "P2", "body", "index"]):
            self.assertLess(self.events.index(left), self.events.index(right))
        self.assertEqual(self.events, events_after_first)
        self.assertTrue(second.cache_hit)
        self.assertEqual(second.api_requests, 0)
        self.assertEqual(len(first.decisions), 4)
        third = DocumentCollector(self.settings, self.model, self.index, self.wiki).prepare(self.movie, "어떤 영화인가요?")
        self.assertTrue(third.cache_hit)
        self.assertEqual(self.model.calls, 2)

    def test_cache_rebuilds_missing_index_from_saved_body(self):
        collector = DocumentCollector(self.settings, self.model, self.index, self.wiki)
        collector.prepare(self.movie, "설명")
        self.index.documents.clear()
        requests_before = self.wiki.request_count
        second = collector.prepare(self.movie, "설명")
        self.assertTrue(second.cache_hit)
        self.assertEqual(self.wiki.request_count, requests_before)
        self.assertIn("1:5", self.index.documents)

    def test_year_homonym_and_unknown_identity(self):
        for candidate in [{"title": "킹콩 (1933년 영화)", "summary": "1933년 영화"},
                          {"title": "킹콩", "summary": "동명이인 가수이다."},
                          {"title": "킹콩", "summary": "유명 영화이다."}]:
            self.assertEqual(identity_check(self.movie, candidate)[0], "reject")
        self.assertEqual(identity_check(self.movie, {"title": "킹콩 (2005년 영화)", "summary": "영화"})[0], "movie")
        self.assertEqual(identity_check(self.movie, {"title": "특수 효과", "summary": "1933년 기술"})[0], "related")

    def test_matching_movie_still_requires_p2_acceptance(self):
        model = SimpleNamespace(invoke=lambda prompt: SimpleNamespace(content='{"decisions":[{"doc_key":"1:5","adopt":false,"reason":"무관"}]}'))
        collector = DocumentCollector(self.settings, model, self.index, self.wiki)
        decisions = collector._select_documents(self.movie, "질문", [{"doc_key": "1:5", "title": "킹콩 (2005년 영화)", "summary": "영화"}])
        self.assertFalse(decisions[0]["adopt"])

    def test_missing_and_invalid_citations_abstain(self):
        sources = [{"number": 1, "title": "영화"}]
        for answer in ("추측한 답변", "답 [근거 99]", "답 [근거 1, 99]"):
            self.assertEqual(validate_answer(answer, sources), INSUFFICIENT_EVIDENCE)
        self.assertEqual(validate_answer("확인 [근거 1]", sources), "확인 [근거 1]")

    def test_zero_evidence_does_not_call_generation(self):
        graph = create_rag_graph(SimpleNamespace(invoke=lambda p: self.fail("model called")),
                                 SimpleNamespace(prepare=lambda m, q: SearchPlan(("q",), ())),
                                 SimpleNamespace(retrieve=lambda **kw: []), self.settings)
        result = graph.invoke({"movie": self.movie, "question": "미확인 정보"})
        self.assertEqual(result["answer"], INSUFFICIENT_EVIDENCE)
        self.assertEqual(result["used_evidence_numbers"], [])

    def test_top_k_aggregates_queries_and_excludes_other_movies(self):
        def doc(i, movie="1"):
            return Document(page_content=f"근거 {i}", metadata={"movie_id": movie, "doc_key": f"{movie}:{i}", "chunk_index": 0})
        index = SimpleNamespace(get_movie_document=lambda _: Document(page_content="영화", metadata={"movie_id": "1", "tmdb_url": "https://www.themoviedb.org/movie/1"}),
                                search_external=lambda q, *a, **kw: [(doc(5), .5), (doc(6), .9), (doc(7, "2"), 1.)] if q == "a" else [(doc(8), .95)])
        retriever = EvidenceRetriever(index)
        small = retriever.retrieve(self.movie, ["a", "b"], ["1:5", "1:6", "1:8"], 1, .1)
        large = retriever.retrieve(self.movie, ["a", "b"], ["1:5", "1:6", "1:8"], 3, .1)
        self.assertEqual([s["doc_key"] for s in small[1:]], ["1:8"])
        self.assertEqual(len(large), 4)
        self.assertTrue(small[0]["url"].startswith("https://www.themoviedb.org/"))


if __name__ == "__main__":
    unittest.main()
