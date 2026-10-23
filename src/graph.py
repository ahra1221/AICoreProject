"""LangGraph retrieve/generate pipeline for grounded Movie Lab answers."""

from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from src.collector import DocumentCollector
from src.config import Settings
from src.data_loader import Movie
from src.prompts import ANSWER_SYSTEM_PROMPT, FACT_CHECK_PROMPT
from src.retriever import EvidenceRetriever, cited_sources


class RagState(TypedDict, total=False):
    movie: Movie
    question: str
    mode: str
    queries: list[str]
    selected_doc_keys: list[str]
    sources: list[dict[str, Any]]
    answer: str
    used_evidence_numbers: list[int]


def _response_text(response: Any) -> str:
    content = response.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        ).strip()
    raise ValueError("The language model returned an unsupported response format.")


def create_rag_graph(
    model: Any,
    collector: DocumentCollector,
    retriever: EvidenceRetriever,
    settings: Settings,
):
    def retrieve_node(state: RagState) -> dict[str, Any]:
        movie = state["movie"]
        plan = collector.prepare(movie, state["question"])
        sources = retriever.retrieve(
            movie=movie,
            queries=list(plan.queries),
            selected_doc_keys=list(plan.selected_doc_keys),
            top_k=settings.top_k,
            min_relevance=settings.min_relevance,
        )
        return {
            "queries": list(plan.queries),
            "selected_doc_keys": list(plan.selected_doc_keys),
            "sources": sources,
        }

    def generate_node(state: RagState) -> dict[str, Any]:
        sources = state.get("sources", [])
        evidence = "\n\n".join(
            f"[근거 {source['number']}] 제목: {source['title']}\n"
            f"출처: {source['source']} | URL: {source['url']}\n"
            f"<untrusted_source>\n{source['text'][:12000]}\n</untrusted_source>"
            for source in sources
        )
        mode_prompt = FACT_CHECK_PROMPT if state.get("mode") == "fact_check" else ""
        user_prompt = (
            f"{mode_prompt}\n선택한 영화: {state['movie'].title} "
            f"({state['movie'].year or '연도 미상'})\n"
            f"질문: {state['question']}\n\n"
            f"다음 근거 자료:\n{evidence or '(검색된 외부 문서가 없습니다.)'}"
        )
        response = model.invoke(
            f"{ANSWER_SYSTEM_PROMPT}\n\n{user_prompt}"
        )
        answer = _response_text(response)
        used_numbers = [number for number, _ in cited_sources(answer, sources)]
        return {"answer": answer, "used_evidence_numbers": used_numbers}

    workflow = StateGraph(RagState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    return workflow.compile()
