from scripts.verify_qdrant_alignment import compare_payloads


def test_alignment_returns_no_differences_for_identical_chunks():
    original = {
        "X-B0001": {"chunk_id": "X-B0001", "text": "first"},
        "X-B0002": {"chunk_id": "X-B0002", "text": "second"},
    }
    stored = {
        "X-B0002": {"chunk_id": "X-B0002", "text": "second"},
        "X-B0001": {"chunk_id": "X-B0001", "text": "first"},
    }
    assert compare_payloads(original, stored) == ([], [], [])


def test_alignment_detects_missing_extra_and_changed_payloads():
    original = {
        "A": {"chunk_id": "A", "text": "correct"},
        "B": {"chunk_id": "B", "text": "expected"},
    }
    stored = {
        "A": {"chunk_id": "A", "text": "changed"},
        "C": {"chunk_id": "C", "text": "extra"},
    }
    assert compare_payloads(original, stored) == (
        ["B"],
        ["C"],
        ["A"],
    )
