from typing import List, Dict
from sentence_transformers.cross_encoder import CrossEncoder


def load_reranker(model_name: str) -> CrossEncoder:
    """
    Load cross-encoder model
    """
    print(f"[RERANK] Loading model: {model_name}")
    model = CrossEncoder(model_name)
    print("[RERANK] Model ready\n")
    return model


def rerank(
    query: str,
    chunks: List[Dict],
    model: CrossEncoder,
    top_k: int = 5
) -> List[Dict]:
    """
    Rerank chunks using cross-encoder
    """

    if not chunks:
        return []

    # Prepare pairs
    pairs = [[query, c["text"]] for c in chunks]

    # Predict scores
    try:
        scores = model.predict(pairs, show_progress_bar=False)
    except Exception as e:
        print(f"[RERANK ERROR] {e}")
        return chunks[:top_k]  # fallback
    # Attach scores
    scored_chunks = []
    for chunk, score in zip(chunks, scores):
        c = chunk.copy()
        c["rerank_score"] = float(score)
        scored_chunks.append(c)

    # Sort by rerank score
    scored_chunks.sort(
       key=lambda x: x["rerank_score"],
       reverse=True
    )

    # -------------------------
    # Threshold filtering (ADD THIS)
    # -------------------------
    THRESHOLD = 0.35

    filtered_chunks = [
        c for c in scored_chunks
        if c["rerank_score"] >= THRESHOLD
    ]

    # If enough strong chunks → use them
    if len(filtered_chunks) >= top_k:
        top_chunks = filtered_chunks[:top_k]
    else:
        # fallback → avoid empty results
        top_chunks = scored_chunks[:top_k]

    print(f"[RERANK] {len(chunks)} → {len(top_chunks)}")

    return top_chunks