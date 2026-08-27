import pytest

from src.retrieval.rrf import reciprocal_rank_fusion


def test_rrf_combines_dense_and_sparse_results():
    dense_results = [
        {
            "chunk_id": "A",
            "rank": 1,
            "score": 0.90,
            "chunk": {"text": "Chunk A"},
        },
        {
            "chunk_id": "B",
            "rank": 2,
            "score": 0.80,
            "chunk": {"text": "Chunk B"},
        },
    ]

    sparse_results = [
        {
            "chunk_id": "B",
            "rank": 1,
            "score": 12.5,
            "chunk": {"text": "Chunk B"},
        },
        {
            "chunk_id": "C",
            "rank": 2,
            "score": 10.0,
            "chunk": {"text": "Chunk C"},
        },
    ]

    results = reciprocal_rank_fusion(
        dense_results=dense_results,
        sparse_results=sparse_results,
    )

    assert len(results) == 3

    assert results[0]["chunk_id"] == "B"
    assert results[0]["rank"] == 1

    assert results[1]["chunk_id"] == "A"
    assert results[1]["rank"] == 2

    assert results[2]["chunk_id"] == "C"
    assert results[2]["rank"] == 3

    expected_b_score = (
        1 / (60 + 2)
        + 1 / (60 + 1)
    )

    assert results[0]["rrf_score"] == pytest.approx(
        expected_b_score
    )

    assert results[0]["dense_rank"] == 2
    assert results[0]["sparse_rank"] == 1
    assert results[0]["dense_score"] == 0.80
    assert results[0]["sparse_score"] == 12.5


def test_rrf_rejects_invalid_rank():
    dense_results = [
        {
            "chunk_id": "A",
            "rank": 0,
            "score": 0.90,
            "chunk": {"text": "Chunk A"},
        }
    ]

    with pytest.raises(
        ValueError,
        match="Result rank must be greater than zero",
    ):
        reciprocal_rank_fusion(
            dense_results=dense_results,
            sparse_results=[],
        )