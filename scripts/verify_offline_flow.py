"""Record real graph/index execution with clearly labelled mock P1/P2/generation."""
from __future__ import annotations
import json
import os
import socket
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
from offline_support import LocalTestEmbeddings
from src.collector import DocumentCollector, identity_check
from src.config import get_settings
from src.data_loader import load_movies
from src.graph import create_rag_graph
from src.indexer import MovieIndexer
from src.retriever import EvidenceRetriever


class RecordedMockModel:
    def __init__(self, movie):
        self.movie, self.calls = movie, []
    def invoke(self, prompt):
        if "프롬프트 1(P1)" in prompt:
            self.calls.append("P1")
            return SimpleNamespace(content='{"queries":["인터스텔라 블랙홀", "웜홀"]}')
        if "프롬프트 2(P2)" in prompt:
            self.calls.append("P2")
            candidates = json.loads(prompt.split("후보 JSON 데이터:\n", 1)[1])
            return SimpleNamespace(content=json.dumps({"decisions": [
                {"doc_key": c["doc_key"], "adopt": identity_check(self.movie, c)[0] != "reject",
                 "reason": "모의 P2: 영화 또는 배경 개념"} for c in candidates]}, ensure_ascii=False))
        self.calls.append("generate")
        return SimpleNamespace(content="모의 답변: 인터스텔라의 개봉연도는 2014년입니다. [근거 1]")


def main():
    scratch = ROOT / "data/.cache/test-temp"
    scratch.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="flow-", dir=scratch))
    settings = replace(get_settings({}), chroma_dir=run_dir / "db", movies_path=run_dir / "movies.json",
                       wiki_mode="snapshot", min_relevance=0.0)
    movies = load_movies(ROOT / "data/movies.json")
    movie = next(m for m in movies if m.movie_id == "157336")
    with patch.object(socket.socket, "connect", side_effect=AssertionError("Offline network forbidden")):
        index = MovieIndexer(settings, embeddings=LocalTestEmbeddings())
        index.index_movies([movie])
        model = RecordedMockModel(movie)
        collector = DocumentCollector(settings, model, index)
        retriever = EvidenceRetriever(index)
        graph = create_rag_graph(model, collector, retriever, settings)
        runs = []
        for number in (1, 2):
            started = time.perf_counter()
            result = graph.invoke({"movie": movie, "question": "인터스텔라의 개봉연도와 블랙홀은?", "mode": "general"})
            runs.append({"run": number, "seconds": time.perf_counter() - started,
                         "collection": result["collection"], "queries": result["queries"],
                         "answer": result["answer"], "used_evidence_numbers": result["used_evidence_numbers"],
                         "sources": [{k: v for k, v in s.items() if k != "text"} for s in result["sources"]]})
        top_k = {}
        for k in (1, 3):
            sources = retriever.retrieve(movie, result["queries"], result["selected_doc_keys"], k, 0.0)
            top_k[str(k)] = [{"number": s["number"], "title": s["title"], "doc_key": s["doc_key"]} for s in sources]
        nodes = [name for name in graph.get_graph().nodes if not name.startswith("__")]
    report = {"mode": "offline_mock_models_and_embeddings", "paid_api_calls": 0,
              "live_llm_quality_verified": False, "latency_is_not_live_service_latency": True,
              "graph_nodes": nodes, "model_calls": model.calls, "runs": runs, "top_k_comparison": top_k}
    assert nodes == ["retrieve", "generate"]
    assert runs[1]["collection"]["cache_hit"]
    assert model.calls.count("P1") == model.calls.count("P2") == 1
    assert len(top_k["1"]) == 2 and len(top_k["3"]) == 4
    output = ROOT / "docs/evidence"
    output.mkdir(parents=True, exist_ok=True)
    (output / "offline-flow.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "graph.mmd").write_text(graph.get_graph().draw_mermaid(), encoding="utf-8")
    print(json.dumps({"nodes": nodes, "model_calls": model.calls, "seconds": [r["seconds"] for r in runs],
                      "top_k_counts_including_metadata": {key: len(value) for key, value in top_k.items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
