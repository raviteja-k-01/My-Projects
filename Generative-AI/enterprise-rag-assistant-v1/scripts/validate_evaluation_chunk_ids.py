"""Check saved evaluation references against the regenerated local corpus.

This is an offline compatibility check, not a live Qdrant comparison.
Run after scripts.build_retrieval_artifacts has generated chunks.jsonl.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.indexing.chunk_loader import load_indexing_chunks

EVALUATION_DIR = Path("data/evaluation")
REFERENCES = {
    "gold_questions.jsonl": "expected_chunk_ids",
    "hybrid_retrieval_results.jsonl": "retrieved_chunk_ids",
}


def main() -> None:
    existing_chunks = {chunk.chunk_id for chunk in load_indexing_chunks()}
    invalid: dict[str, list[tuple[str, str]]] = {}
    checked = {}

    for filename, key in REFERENCES.items():
        path = EVALUATION_DIR / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        missing = []
        refs = 0
        for line_number, text in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not text.strip():
                continue
            question = json.loads(text)
            question_id = question.get("question_id", f"line {line_number}")
            for chunk_id in question.get(key, []):
                refs += 1
                if chunk_id not in existing_chunks:
                    missing.append((question_id, chunk_id))
        invalid[filename] = missing
        checked[filename] = refs
        print(f"{filename}: {refs} IDs checked, {len(missing)} missing locally")

    if any(invalid.values()):
        for filename, missing in invalid.items():
            if missing:
                print(f"{filename} missing sample: {missing[:12]}")
        raise SystemExit(
            "Historical evaluation references do not align with the rebuilt local corpus."
        )

    print("Historical evaluation chunk-ID presence: PASSED")
    print("Live Qdrant payload and embedding alignment: NOT VERIFIED")


if __name__ == "__main__":
    main()
