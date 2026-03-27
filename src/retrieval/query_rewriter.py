import litellm

litellm.suppress_debug_info = True


def rewrite_query(query: str, model_name: str) -> str:
    """
    Rewrite short queries to improve retrieval WITHOUT changing meaning.

    Safety design:
    - Only rewrite short queries
    - Strict prompt (no answering, no assumptions)
    - Fallback to original query if anything fails
    """

    #  Skip rewriting for detailed queries (prevents over-modification)
    if len(query.split()) > 12:
        return query

    prompt = (
        "You are a search query optimizer for enterprise documents.\n\n"
        "Rewrite the query to improve document retrieval.\n\n"
        "STRICT RULES:\n"
        "- Keep the meaning EXACTLY the same\n"
        "- Only add relevant keywords and synonyms\n"
        "- Do NOT introduce new concepts\n"
        "- Do NOT assume anything\n"
        "- Do NOT answer the question\n"
        "- Output ONLY the rewritten query\n\n"
        f"Query: {query}"
    )

    try:
        response = litellm.completion(
            model=model_name,
            messages=[
                {"role": "system", "content": "You rewrite queries for search improvement."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=50,
            temperature=0.0,   #  deterministic, reduces hallucination
        )

        rewritten = response.choices[0].message.content.strip()

        #  Safety check — if output looks wrong, fallback
        if not rewritten or len(rewritten) < 3:
            return query

        print(f"[REWRITE] {query} → {rewritten}")

        return rewritten

    except Exception as e:
        print(f"[REWRITE] Failed ({e}) — using original query")
        return query