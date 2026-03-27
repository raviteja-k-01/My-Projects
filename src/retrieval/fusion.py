from typing import List, Dict


def reciprocal_rank_fusion(
    dense_chunks: List[Dict],
    sparse_chunks: List[Dict],
    top_k: int = 20,
    k: int = 60
) -> List[Dict]:
    """
    Combine dense and sparse results using Reciprocal Rank Fusion (RRF)
    """

    scores = {}

    # Assign RRF scores for dense results
    for rank, chunk in enumerate(dense_chunks):
        key = chunk.get("chunk_id") or chunk.get("text")

        if key not in scores:
            scores[key] = {"score": 0, "chunk": chunk}

        scores[key]["score"] += 1 / (k + rank + 1)

    # Assign RRF scores for sparse results
    for rank, chunk in enumerate(sparse_chunks):
        key = chunk.get("chunk_id") or chunk.get("text")

        if key not in scores:
            scores[key] = {"score": 0, "chunk": chunk}

        scores[key]["score"] += 1 / (k + rank + 1)

    # Sort by fused score
    fused = sorted(
        scores.values(),
        key=lambda x: x["score"],
        reverse=True
    )

    # Extract chunks
    results = [item["chunk"] for item in fused[:top_k]]

    print(f"[FUSION] {len(results)} fused results")

    return results