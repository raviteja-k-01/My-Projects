from typing import List, Dict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def deduplicate(
    chunks: List[Dict],
    threshold: float = 0.85
) -> List[Dict]:
    """
    Remove near-duplicate chunks based on text similarity.

    Keeps highest-ranked chunks first (greedy approach).
    """

    if len(chunks) <= 1:
        return chunks

    texts = [c["text"] for c in chunks]

    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(texts)
        sim_matrix = cosine_similarity(tfidf_matrix)
    except Exception:
        # fallback if something fails
        return chunks

    selected_indices = []

    for i in range(len(chunks)):
        is_duplicate = any(
            sim_matrix[i, j] > threshold
            for j in selected_indices
        )

        if not is_duplicate:
            selected_indices.append(i)

    result = [chunks[i] for i in selected_indices]

    removed = len(chunks) - len(result)

    if removed > 0:
        print(f"[DEDUP] Removed {removed} duplicate chunks")
    else:
        print("[DEDUP] No duplicates")

    return result