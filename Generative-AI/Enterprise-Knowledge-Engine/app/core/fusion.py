def normalize(scores):
    min_score = min(scores)
    max_score = max(scores)

    if max_score - min_score == 0:
        return [0 for _ in scores]

    return [(s - min_score) / (max_score - min_score) for s in scores]


def fuse(bm25_results, dense_results, alpha=0.5):
    bm25_ids = [i for i, _ in bm25_results]
    bm25_scores = [s for _, s in bm25_results]

    dense_ids = [i for i, _ in dense_results]
    dense_scores = [-d for _, d in dense_results]  # invert distance

    bm25_norm = normalize(bm25_scores)
    dense_norm = normalize(dense_scores)

    score_dict = {}

    for i, score in zip(bm25_ids, bm25_norm):
        score_dict[i] = score_dict.get(i, 0) + alpha * score

    for i, score in zip(dense_ids, dense_norm):
        score_dict[i] = score_dict.get(i, 0) + (1 - alpha) * score

    ranked = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)

    return ranked