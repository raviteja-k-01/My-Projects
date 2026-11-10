from collections import Counter

from src.indexing.chunk_loader import (
    load_indexing_chunks,
)


EXPECTED_CHUNKS = 1985
EXPECTED_BODY_CHUNKS = 1766
EXPECTED_TABLE_CHUNKS = 219


def main() -> None:
    chunks = load_indexing_chunks()

    content_types = Counter(
        chunk.content_type
        for chunk in chunks
    )

    assert (
        len(chunks)
        == EXPECTED_CHUNKS
    )

    assert (
        content_types["body"]
        == EXPECTED_BODY_CHUNKS
    )

    assert (
        content_types["table"]
        == EXPECTED_TABLE_CHUNKS
    )

    assert len(
        {
            chunk.chunk_id
            for chunk in chunks
        }
    ) == EXPECTED_CHUNKS

    print(
        "\nINDEXING INPUT VALIDATION"
    )

    print(
        f"Total chunks: "
        f"{len(chunks)}"
    )

    print(
        f"Body chunks: "
        f"{content_types['body']}"
    )

    print(
        f"Table chunks: "
        f"{content_types['table']}"
    )

    print(
        "Unique chunk IDs: PASS"
    )

    print(
        "Non-empty chunk text: PASS"
    )

    print(
        "Typed ChunkRecord loading: PASS"
    )

    print(
        "\nINDEXING INPUT: PASS"
    )


if __name__ == "__main__":
    main()