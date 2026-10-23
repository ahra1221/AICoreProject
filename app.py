"""Streamlit entry point for the Movie Lab RAG application."""

from __future__ import annotations

import logging
import time
from dataclasses import replace
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
import streamlit as st
from chromadb.errors import ChromaError
from langchain_openai import ChatOpenAI
from openai import OpenAIError

from src.collector import CollectorError, DocumentCollector
from src.config import ConfigurationError, Settings, get_settings
from src.data_loader import DataFileError, Movie, load_movies
from src.graph import create_rag_graph
from src.indexer import MovieIndexer
from src.retriever import EvidenceRetriever, cited_sources
from src.services import find_movies, movie_chat_history
from src.wikipedia import WikipediaError

logger = logging.getLogger(__name__)


@st.cache_data(show_spinner=False)
def cached_movies(path: str) -> list[Movie]:
    return load_movies(Path(path))


@st.cache_resource(show_spinner=False)
def get_indexer(settings: Settings) -> MovieIndexer:
    return MovieIndexer(settings)


@st.cache_resource(show_spinner=False)
def get_rag_runtime(settings: Settings):
    if not settings.gms_key:
        raise ConfigurationError("GMS_KEY is not set. Create a local .env file and add your API key.")
    model = ChatOpenAI(
        model=settings.chat_model,
        api_key=settings.gms_key,
        base_url=settings.gms_base_url,
    )
    indexer = MovieIndexer(settings)
    collector = DocumentCollector(settings, model, indexer)
    retriever = EvidenceRetriever(indexer)
    graph = create_rag_graph(model, collector, retriever, settings)
    return graph, indexer


def _safe_source_url(url: str) -> bool:
    parsed = urlparse(url)
    host = (parsed.hostname or "").casefold()
    return parsed.scheme == "https" and (
        host == "themoviedb.org"
        or host.endswith(".themoviedb.org")
        or host == "wikipedia.org"
        or host.endswith(".wikipedia.org")
    )


def _indexed_movie_ids(indexer: MovieIndexer) -> set[str]:
    result = indexer.movie_collection.get(include=[])
    return {str(movie_id) for movie_id in result["ids"]}


def render_sources(answer: str, sources: list[dict[str, Any]]) -> None:
    used = cited_sources(answer, sources)
    if not used:
        st.caption("답변에 인용된 근거가 없습니다.")
        return
    with st.expander(f"답변에 사용한 근거 {len(used)}건"):
        for number, source in used:
            st.write(f"**[근거 {number}] {source.get('title') or '제목 없음'}**")
            st.caption(source.get("source_type") or source.get("source") or "출처 미상")
            if source.get("related_movie"):
                st.caption(f"관련 영화: {source['related_movie']}")
            url = str(source.get("url") or "")
            if _safe_source_url(url):
                st.link_button("원문 보기", url, key=f"source-{number}-{source.get('doc_key', '')}")
            elif url:
                st.caption("원문 링크를 확인할 수 없습니다.")
            if source.get("license"):
                st.caption(f"이용 조건: {source['license']} · 본문 발췌/청킹")
            revision = str(source.get("revision_id") or "")
            if revision.isdigit():
                st.markdown(f"[수집한 판본](https://ko.wikipedia.org/w/index.php?oldid={revision})")


def render_collection(result: dict[str, Any]) -> None:
    collection = result.get("collection", {})
    with st.expander("검색어와 문서 선별 결과"):
        st.write("P1 검색어", result.get("queries", []))
        mode = "Wikipedia API" if collection.get("mode") == "api" else "제공 문서 스냅샷"
        st.caption(f"자료: {mode} · 질문 캐시: {'사용' if collection.get('cache_hit') else '새 수집'} · "
                   f"Wikipedia 요청: {collection.get('api_requests', 0)}회")
        rows = [{"문서": row["title"], "P2 결과": "채택" if row["adopt"] else "탈락",
                 "이유": row["reason"]} for row in collection.get("decisions", [])]
        if rows:
            st.dataframe(rows, hide_index=True, width="stretch")
        else:
            st.caption("검색된 문서 후보가 없습니다.")
        st.caption(f"수집·검색 {result.get('retrieve_seconds', 0):.2f}초 · "
                   f"답변 {result.get('generate_seconds', 0):.2f}초 · "
                   f"전체 {result.get('total_seconds', 0):.2f}초")


def _format_movie(movie: Movie) -> str:
    year = movie.year or "연도 미상"
    return f"{movie.title} ({year}) · {', '.join(movie.genres) or '장르 미상'}"


def _render_movie_summary(movie: Movie) -> None:
    st.subheader(_format_movie(movie))
    if movie.director:
        st.caption(f"감독: {', '.join(movie.director)}")
    if movie.cast:
        st.caption(f"출연: {', '.join(movie.cast)}")
    if movie.overview:
        st.write(movie.overview)


def _render_discovery(movies: list[Movie], indexer: MovieIndexer, has_key: bool) -> None:
    st.header("자연어 영화 탐색")
    st.caption("영화 메타데이터에서 조건에 맞는 작품을 찾습니다.")
    with st.form("movie-discovery"):
        query = st.text_input(
            "찾고 싶은 영화 조건",
            placeholder="예: 2010년 이후 개봉한 애니메이션",
        )
        submitted = st.form_submit_button("영화 찾기")
    if submitted:
        if not has_key:
            st.error("영화 탐색에는 임베딩 API 키가 필요합니다. .env의 GMS_KEY를 설정하세요.")
            return
        if not indexer.counts()["movie_meta"]:
            st.warning("영화 색인이 없습니다. README의 실제 RAG 사용 절차로 메타데이터를 먼저 색인하세요.")
            return
        try:
            matches = find_movies(query, movies, indexer)
        except OpenAIError:
            logger.exception("Movie discovery failed")
            st.error("GMS 영화 탐색 요청에 실패했습니다. API 키, 모델명과 연결 상태를 확인하세요.")
            return
        except (ChromaError, OSError) as exc:
            logger.exception("Movie discovery index failed")
            st.error("영화 검색 색인을 사용할 수 없습니다. 색인 경로를 확인하세요.")
            return
        except ValueError as exc:
            logger.exception("Movie discovery failed")
            st.error("영화 탐색에 실패했습니다. 검색 조건과 색인을 확인하세요.")
            return
        indexed_ids = _indexed_movie_ids(indexer)
        st.session_state["discovery_results"] = [
            movie.movie_id for movie in matches if movie.movie_id in indexed_ids
        ]
        st.session_state["discovery_query"] = query
        st.session_state["discovery_performed"] = True

    result_ids = st.session_state.get("discovery_results", [])
    by_id = {movie.movie_id: movie for movie in movies}
    matches = [by_id[movie_id] for movie_id in result_ids if movie_id in by_id]
    if st.session_state.get("discovery_query"):
        st.caption(f"검색 조건: {st.session_state['discovery_query']}")
    if st.session_state.get("discovery_performed") and not matches:
        st.info("조건에 맞는 영화를 찾지 못했습니다.")
        return
    if matches:
        selected_id = st.selectbox(
            "검색 결과에서 영화 선택",
            options=[movie.movie_id for movie in matches],
            format_func=lambda movie_id: _format_movie(by_id[movie_id]),
            key="discovery-selection",
        )
        for movie in matches:
            with st.expander(_format_movie(movie)):
                if movie.director:
                    st.caption(f"감독: {', '.join(movie.director)}")
                if movie.overview:
                    st.write(movie.overview)
                if _safe_source_url(movie.tmdb_url):
                    st.link_button("영화 메타데이터 출처 (TMDB)", movie.tmdb_url)
        if st.button("선택한 영화 질문하기"):
            st.session_state["selected_movie_id"] = selected_id
            st.session_state["chat-movie-selection"] = selected_id
            st.session_state["view"] = "영화 질문"
            st.rerun()


def _render_chat(movies: list[Movie], settings: Settings, indexer: MovieIndexer) -> None:
    indexed_ids = _indexed_movie_ids(indexer)
    available_movies = [movie for movie in movies if movie.movie_id in indexed_ids]
    if not available_movies:
        st.warning("색인 데이터가 없습니다. README의 실제 RAG 사용 절차로 영화 메타데이터를 먼저 색인하세요.")
        return
    movies_by_id = {movie.movie_id: movie for movie in available_movies}
    with st.sidebar:
        st.header("영화 선택")
        movie_id = st.selectbox(
            "질문할 영화",
            options=[movie.movie_id for movie in available_movies],
            format_func=lambda value: _format_movie(movies_by_id[value]),
            index=next(
                (
                    index for index, movie in enumerate(available_movies)
                    if movie.movie_id == st.session_state.get("selected_movie_id")
                ),
                0,
            ),
            key="chat-movie-selection",
        )
        st.session_state["selected_movie_id"] = movie_id
        if st.button("이 영화의 대화 지우기", width="stretch"):
            st.session_state.setdefault("movie_chats", {})[movie_id] = []
            st.rerun()

    movie = movies_by_id[movie_id]
    _render_movie_summary(movie)
    counts = indexer.counts()
    if counts["movie_meta"] == 0:
        st.warning("색인 데이터가 없습니다. 이 프로젝트의 키를 설정한 뒤 README의 색인 절차를 확인하세요.")
        return
    if not settings.gms_key:
        st.warning("GMS_KEY가 없습니다. 로컬 `.env` 파일을 만들고 키를 설정하세요.")
        return
    if settings.wiki_mode == "api" and not settings.wiki_contact:
        st.warning("Wikipedia 수집을 위해 .env의 WIKI_CONTACT에 연락처나 프로젝트 URL을 입력하세요. 제공 문서는 WIKI_MODE=snapshot으로 사용할 수 있습니다.")
        return

    movie_chats = st.session_state.setdefault("movie_chats", {})
    messages = movie_chat_history(movie_chats, movie_id)
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_sources(message["content"], message.get("sources", []))
                if message.get("audit"):
                    render_collection(message["audit"])

    mode = st.radio(
        "답변 방식",
        options=["영화 질문", "과학·역사 사실 검증"],
        horizontal=True,
        label_visibility="collapsed",
    )
    question = st.chat_input(f"{movie.title}에 대해 질문해 보세요")
    if not question:
        return

    messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        answer = ""
        sources: list[dict[str, Any]] = []
        try:
            started = time.perf_counter()
            graph, _ = get_rag_runtime(settings)
            initial_state = {
                "movie": movie,
                "question": question,
                "mode": "fact_check" if mode == "과학·역사 사실 검증" else "general",
            }
            result: dict[str, Any] = dict(initial_state)
            with st.status("검색어와 관련 문서를 준비하는 중...", expanded=False) as status:
                for update in graph.stream(initial_state, stream_mode="updates"):
                    for node, node_result in update.items():
                        result.update(node_result)
                        if node == "retrieve":
                            sources = result.get("sources", [])
                            status.update(
                                label=f"근거 {len(sources)}건 검색 완료 · 답변 생성 중...",
                                state="running",
                            )
                        elif node == "generate":
                            status.update(label="답변 생성 완료", state="complete")
            answer = str(result.get("answer", "")).strip()
            result["total_seconds"] = time.perf_counter() - started
            sources = result.get("sources", [])
            if not answer:
                raise ValueError("The language model returned an empty answer.")
            st.markdown(answer)
            render_sources(answer, sources)
            render_collection(result)
            audit = {key: result.get(key) for key in ("queries", "collection", "retrieve_seconds",
                                                       "generate_seconds", "total_seconds")}
            messages.append(
                {"role": "assistant", "content": answer, "sources": sources, "audit": audit}
            )
        except ConfigurationError as exc:
            logger.exception("Movie question failed")
            st.error("프로젝트 설정을 확인하세요. API 키와 모델명은 .env에 설정합니다.")
        except (CollectorError, WikipediaError) as exc:
            logger.exception("Document collection failed")
            st.error("문서 수집 또는 선별에 실패했습니다. 연락처와 네트워크를 확인하거나 WIKI_MODE=snapshot을 사용하세요.")
        except OpenAIError:
            logger.exception("GMS request failed")
            st.error("GMS 요청에 실패했습니다. API 키, 모델명, 임베딩 지원 여부와 네트워크를 확인하세요.")
        except ChromaError:
            logger.exception("ChromaDB request failed")
            st.error("ChromaDB 색인을 읽지 못했습니다. README의 색인 경로와 갱신 절차를 확인하세요.")
        except OSError:
            logger.exception("Movie Lab storage operation failed")
            st.error("로컬 데이터 또는 벡터 저장소에 접근하지 못했습니다. 경로와 권한을 확인하세요.")
        except requests.RequestException:
            logger.exception("Wikipedia request failed")
            st.error("위키백과 문서 요청에 실패했습니다. 잠시 후 다시 시도하거나 제공된 문서 색인을 확인하세요.")
        except (DataFileError, ValueError) as exc:
            logger.exception("Movie question failed")
            st.error("질문을 처리하지 못했습니다. 데이터 형식과 설정을 확인하세요.")


def main() -> None:
    st.set_page_config(page_title="Movie Lab", page_icon="🎬", layout="wide")
    st.title("🎬 Movie Lab")
    st.caption("영화와 관련 문서를 근거로 답하고 출처를 함께 보여 주는 영화 지식 서비스")

    try:
        settings = get_settings()
        movies = cached_movies(str(settings.movies_path))
        indexer = get_indexer(settings)
    except (ConfigurationError, DataFileError, ChromaError, OSError, ValueError):
        logger.exception("Movie Lab startup failed")
        st.error("데이터 또는 색인을 열 수 없습니다. .env의 경로·숫자 설정과 데이터 파일을 확인하세요.")
        st.stop()

    if len(movies) != 964:
        st.warning(f"제공 데이터에서 영화 {len(movies)}편을 확인했습니다. 기대 데이터는 964편입니다.")
    if not settings.gms_key:
        st.info("LLM 및 벡터 검색을 사용하려면 `.env`에 GMS_KEY를 설정하세요.")

    with st.sidebar:
        counts = indexer.counts()
        st.caption(f"영화 메타데이터 {counts['movie_meta']}건 / 외부 문서 청크 {counts['external_docs']}개")
        top_k = st.number_input("외부 근거 수 (Top-k)", min_value=1, max_value=max(20, settings.top_k),
                                value=settings.top_k, step=1)
        settings = replace(settings, top_k=int(top_k))
        view = st.radio("서비스", ["영화 질문", "자연어 영화 탐색"], key="view")
        with st.expander("자료 출처와 이용 조건"):
            st.markdown("[TMDB](https://www.themoviedb.org/) 영화 메타데이터와 "
                        "[한국어 위키백과](https://ko.wikipedia.org/) 본문을 사용합니다.")
            st.caption("This product uses the TMDB API but is not endorsed or certified by TMDB.")
            st.markdown("Wikipedia: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). "
                        "본문은 평문으로 추출하고 청킹했습니다. 문서·판본 링크에서 원문과 이력을 확인할 수 있습니다.")

    if view == "자연어 영화 탐색":
        _render_discovery(movies, indexer, bool(settings.gms_key))
    else:
        _render_chat(movies, settings, indexer)


if __name__ == "__main__":
    main()
