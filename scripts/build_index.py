"""Load the provided datasets and build or incrementally update ChromaDB indexes."""

from __future__ import annotations

import logging
import argparse
import sys
from pathlib import Path

from openai import OpenAIError
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import get_settings
from src.data_loader import load_movies, load_wiki_documents
from src.indexer import MovieIndexer, create_external_chunks, select_index_test_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Index only a small sample and never run a full-corpus index automatically.",
    )
    parser.add_argument(
        "--test-movies",
        type=int,
        default=100,
        help="Test-mode movie limit (5-100; default: 100).",
    )
    parser.add_argument(
        "--test-wiki-chunks",
        type=int,
        default=50,
        help="Test-mode wiki chunk limit (20-50; default: 50).",
    )
    parser.add_argument(
        "--estimate-only",
        action="store_true",
        help="Print full-corpus workload without making API requests or writing an index.",
    )
    return parser.parse_args()


def check_embedding_connection(settings) -> tuple[OpenAIEmbeddings, int]:
    if not settings.gms_key:
        raise ValueError("GMS_KEY is not configured. Add it to the local .env file.")
    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.gms_key,
        base_url=settings.gms_base_url,
    )
    vector = embeddings.embed_query("Movie Lab GMS embedding connection check")
    if not vector:
        raise ValueError("The configured GMS embedding model returned an empty vector.")
    logger.info(
        "GMS embedding connection succeeded: model=%s, dimensions=%d",
        settings.embedding_model,
        len(vector),
    )
    return embeddings, len(vector)


def main() -> None:
    args = parse_args()
    settings = get_settings()
    movies = load_movies(settings.movies_path)
    wiki_documents = load_wiki_documents(settings.wiki_dir)
    if len(movies) != 964:
        raise ValueError(f"Expected 964 movies, found {len(movies)} in {settings.movies_path}.")
    if not wiki_documents:
        raise ValueError(f"No Wikipedia documents found in {settings.wiki_dir}.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        add_start_index=True,
    )

    if args.estimate_only:
        chunks = sum(len(splitter.split_text(document.text)) for document in wiki_documents)
        print(f"Full index estimate: movies={len(movies)}, wiki_documents={len(wiki_documents)}, wiki_chunks={chunks}")
        print(f"Embedding inputs on a clean index: {len(movies) + chunks}")
        print("No API request was sent and no index was written.")
        return

    if args.test_mode:
        selection = select_index_test_data(
            movies,
            wiki_documents,
            splitter,
            movie_limit=args.test_movies,
            chunk_limit=args.test_wiki_chunks,
        )
        logger.info(
            "Test sample: %d movies, %d wiki chunks; maximum new embeddings=%d "
            "(unchanged items are skipped)",
            len(selection.movies),
            len(selection.external_documents),
            len(selection.movies) + len(selection.external_documents),
        )
        embeddings, dimensions = check_embedding_connection(settings)
        indexer = MovieIndexer(settings, embeddings=embeddings)
        before = indexer.counts()
        movie_report = indexer.index_movies(selection.movies, prune=False)
        wiki_report = indexer._upsert(
            indexer.external_collection,
            selection.external_ids,
            selection.external_documents,
            prune=False,
        )
        after = indexer.counts()
        print(f"GMS embedding model: {settings.embedding_model} (supported; {dimensions} dimensions)")
        print(f"Test-mode sample: movies={len(selection.movies)}, wiki_chunks={len(selection.external_documents)}")
        print(f"Maximum new embedding inputs: {len(selection.movies) + len(selection.external_documents)}")
        print(f"Actually embedded: movies={movie_report.added_or_updated}, wiki_chunks={wiki_report.added_or_updated}")
        print(
            "Embedding preflight probes: 1; total embedding inputs including "
            f"preflight: {1 + movie_report.added_or_updated + wiki_report.added_or_updated}"
        )
        print(f"ChromaDB counts: before={before}; after={after}")
        print(f"Test index location: {settings.chroma_dir}")
        print("Full-corpus indexing was not started.")
        return

    embeddings, dimensions = check_embedding_connection(settings)
    indexer = MovieIndexer(settings, embeddings=embeddings)
    before = indexer.counts()
    logger.info("Embedding model preflight passed with %d dimensions.", dimensions)
    movie_report = indexer.index_movies(movies)
    wiki_report = indexer.index_external_documents(wiki_documents)
    logger.info(
        "movie_meta: %d -> %d (%d added/updated, %d removed)",
        movie_report.before,
        movie_report.after,
        movie_report.added_or_updated,
        movie_report.removed,
    )
    logger.info(
        "external_docs: %d -> %d (%d added/updated, %d removed)",
        wiki_report.before,
        wiki_report.after,
        wiki_report.added_or_updated,
        wiki_report.removed,
    )
    logger.info("Final collection counts: %s", indexer.counts())


if __name__ == "__main__":
    try:
        main()
    except (OpenAIError, OSError, ValueError) as exc:
        logger.error("Indexing failed (%s). Successful items remain indexed; rerunning retries only missing or changed items.", type(exc).__name__)
        raise SystemExit(1) from exc
