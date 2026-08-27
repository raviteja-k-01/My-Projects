import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

import bm25s
from bm25s.tokenization import Tokenizer

from src.indexing.chunk_loader import load_indexing_chunks


BM25_INDEX_PATH = Path("data/bm25")

# Keeps useful enterprise terms together:
# SEC-IR-004
# SEV-1
# HR-LV-07
TOKEN_PATTERN = re.compile(
    r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*"
)


def split_text(text: str) -> list[str]:
    """
    Convert text into lowercase BM25 tokens while
    preserving hyphenated enterprise terms.
    """

    return TOKEN_PATTERN.findall(
        text.lower()
    )


def build_search_text(chunk) -> str:
    """
    Create the text representation that BM25 searches.

    We include useful structural metadata once,
    followed by the actual chunk text.
    """

    parts = [
        chunk.document_id,
        chunk.document_name,
        chunk.section_number,
        chunk.section_title,
        chunk.subsection_number,
        chunk.subsection_title,
        chunk.table_caption,
        chunk.text,
    ]

    return " ".join(
        str(part)
        for part in parts
        if part is not None
        and str(part).strip()
    )


def make_payload(chunk) -> dict[str, Any]:
    """
    Convert a ChunkRecord into JSON-safe metadata
    that BM25 can return after retrieval.
    """

    payload = asdict(chunk)

    for key, value in payload.items():
        if isinstance(value, Path):
            payload[key] = str(value)

    return payload


class BM25Index:
    """
    Builds, saves, loads, and searches the project's
    sparse BM25 retrieval index.
    """

    def __init__(
        self,
        index_path: Path = BM25_INDEX_PATH,
    ) -> None:
        self.index_path = index_path
        self.tokenizer: Tokenizer | None = None
        self.retriever: bm25s.BM25 | None = None

    def build(self) -> int:
        """
        Build BM25 from all retrieval-ready chunks.
        """

        chunks = load_indexing_chunks()

        if not chunks:
            raise ValueError(
                "No chunks available for BM25 indexing"
            )

        search_texts = [
            build_search_text(chunk)
            for chunk in chunks
        ]

        corpus = [
            make_payload(chunk)
            for chunk in chunks
        ]

        self.tokenizer = Tokenizer(
            splitter=split_text
        )

        corpus_tokens = self.tokenizer.tokenize(
            search_texts,
            return_as="tuple",
        )

        self.retriever = bm25s.BM25(
            corpus=corpus
        )

        self.retriever.index(
            corpus_tokens
        )

        return len(chunks)

    def save(self) -> None:
        """
        Persist the BM25 index, corpus, and tokenizer
        vocabulary to disk.
        """

        if self.retriever is None:
            raise ValueError(
                "BM25 index has not been built"
            )

        if self.tokenizer is None:
            raise ValueError(
                "BM25 tokenizer has not been built"
            )

        self.index_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.retriever.save(
            self.index_path
        )

        self.tokenizer.save_vocab(
            save_dir=self.index_path
        )

    def load(self) -> None:
        """
        Load an already-saved BM25 index.
        """

        if not self.index_path.exists():
            raise ValueError(
                f"BM25 index does not exist: "
                f"{self.index_path}"
            )

        self.retriever = bm25s.BM25.load(
            self.index_path,
            load_corpus=True,
        )

        self.tokenizer = Tokenizer(
            splitter=split_text
        )

        self.tokenizer.load_vocab(
            self.index_path
        )

    def search(
        self,
        query: str,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """
        Search BM25 and return ranked chunks
        with their BM25 scores.
        """

        if not query.strip():
            raise ValueError(
                "BM25 query cannot be empty"
            )

        if limit <= 0:
            raise ValueError(
                "BM25 limit must be greater than zero"
            )

        if self.retriever is None:
            raise ValueError(
                "BM25 index is not loaded"
            )

        if self.tokenizer is None:
            raise ValueError(
                "BM25 tokenizer is not loaded"
            )

        query_tokens = self.tokenizer.tokenize(
            [query],
            update_vocab=False,
            return_as="tuple",
        )

        results, scores = self.retriever.retrieve(
            query_tokens,
            k=limit,
        )

        ranked_results: list[dict[str, Any]] = []

        for rank in range(
            results.shape[1]
        ):
            document = results[0, rank]
            score = float(
                scores[0, rank]
            )

            ranked_results.append(
                {
                    "rank": rank + 1,
                    "score": score,
                    "chunk": document,
                }
            )

        return ranked_results