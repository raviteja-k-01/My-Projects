from pathlib import Path
from types import SimpleNamespace

from src.ingestion import pipeline


def test_process_document_parses_only_after_updating_all_page_titles(monkeypatch):
    pages = [SimpleNamespace(document_name="Original") for _ in range(3)]
    tables = [object()]
    metadata = SimpleNamespace(document_name="Controlled document name")
    calls = []

    monkeypatch.setattr(
        pipeline, "get_document_info",
        lambda pdf_path: ("DOC-01", "Original"),
    )
    monkeypatch.setattr(
        pipeline, "read_pdf", lambda **kwargs: pages,
    )
    monkeypatch.setattr(
        pipeline, "_extract_document_tables", lambda **kwargs: tables,
    )
    monkeypatch.setattr(
        pipeline, "extract_document_metadata", lambda **kwargs: metadata,
    )

    def parse_document_structure(actual_pages):
        assert all(page.document_name == metadata.document_name for page in actual_pages)
        calls.append("parse")
        return ["heading"]

    def attach_table_heading_states(*, pages, tables):
        assert pages is not None and tables is not None
        calls.append("attach")

    def reconstruct_body_content(*, pages, tables):
        calls.append("reconstruct")
        return ["body"]

    monkeypatch.setattr(pipeline, "parse_document_structure", parse_document_structure)
    monkeypatch.setattr(pipeline, "attach_table_heading_states", attach_table_heading_states)
    monkeypatch.setattr(pipeline, "reconstruct_body_content", reconstruct_body_content)

    result = pipeline.process_document(Path("01_DOC-01_Test.pdf"))

    assert calls == ["parse", "attach", "reconstruct"]
    assert result.metadata is metadata
    assert result.pages is pages
    assert result.headings == ["heading"]
    assert result.body_content == ["body"]
