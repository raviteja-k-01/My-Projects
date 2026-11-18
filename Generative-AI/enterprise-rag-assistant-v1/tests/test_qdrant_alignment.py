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


def test_load_local_payloads_rejects_duplicate_ids(tmp_path):
    import json
    import pytest
    from scripts.verify_qdrant_alignment import load_local_payloads

    chunks = tmp_path / "chunks.jsonl"
    record = {"chunk_id": "X-B0001", "text": "test"}
    chunks.write_text(json.dumps(record) + "\n" + json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate chunk ID"):
        load_local_payloads(chunks)


def test_read_only_qdrant_scroll_paginates_and_does_not_expose_key(monkeypatch):
    import io
    import json
    from scripts import verify_qdrant_alignment as audit

    requests = []

    def fake_urlopen(request, timeout):
        requests.append(request)
        body = json.loads(request.data.decode("utf-8"))
        if "offset" not in body:
            result = {
                "status": "ok",
                "result": {
                    "points": [{"id": 1, "payload": {"chunk_id": "X1", "text": "one"}}],
                    "next_page_offset": 1,
                },
            }
        else:
            assert body["offset"] == 1
            result = {
                "status": "ok",
                "result": {
                    "points": [{"id": 2, "payload": {"chunk_id": "X2", "text": "two"}}],
                    "next_page_offset": None,
                },
            }

        class FakeResponse(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.close()

        return FakeResponse(json.dumps(result).encode("utf-8"))

    monkeypatch.setattr(audit, "urlopen", fake_urlopen)
    observed = audit.fetch_qdrant_payloads("https://qdrant.example", "secret-do-not-print")
    assert sorted(observed) == ["X1", "X2"]
    assert len(requests) == 2
    assert all(request.get_method() == "POST" for request in requests)
    assert all(request.get_header("Api-key") == "secret-do-not-print" for request in requests)
    assert all(b"with_vector" in request.data for request in requests)
    assert all(b"false" in request.data for request in requests)
