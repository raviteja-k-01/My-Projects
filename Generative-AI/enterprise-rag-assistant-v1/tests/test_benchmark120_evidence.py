"""Offline tests for generic source-anchor validation. Run with unittest."""
import json
import tempfile
import unittest
from pathlib import Path

from src.evaluation.benchmark120_evidence import Chunk, assess_snippet, audit, match_component, table_cells


class TableEvidenceTests(unittest.TestCase):
    def make_chunk(self, text, doc='DOC-A', page=1):
        return Chunk.from_row({'chunk_id': f'{doc}-{page}', 'document_id': doc,
                               'page_start': page, 'page_end': page,
                               'content_type': 'table', 'text': text})

    def test_table_same_row_field_value(self):
        chunk = self.make_chunk('Row 1: Field: Approver | Controlled value: Jane Doe (CFO)\n'
                                'Row 2: Field: Owner | Controlled value: Alex Doe')
        self.assertEqual(match_component('Approver Jane Doe (CFO)', chunk), 'table_field_value')

    def test_different_rows_must_not_match(self):
        chunk = self.make_chunk('Row 1: Field: Approver | Controlled value: Jane Doe\n'
                                'Row 2: Field: Owner | Controlled value: Alex Doe')
        self.assertIsNone(match_component('Approver Alex Doe', chunk))

    def test_unrelated_doc_not_counted(self):
        row = {'expected_documents': ['DOC-A'], 'evidence_documents': ['DOC-A'],
               'evidence_locations': [{'document_id': 'DOC-A', 'page': 1}]}
        chunk = self.make_chunk('Row 1: Field: Approver | Controlled value: Jane Doe', 'DOC-B')
        resolved = assess_snippet(row, 'Approver Jane Doe', {'DOC-B': [chunk]})
        self.assertEqual(resolved['status'], 'unresolved')

    def test_composite_must_co_locate_within_document(self):
        row = {'expected_documents': ['DOC-A', 'DOC-B'], 'evidence_documents': ['DOC-A', 'DOC-B'],
               'evidence_locations': [{'document_id': 'DOC-A', 'page': 1},
                                      {'document_id': 'DOC-B', 'page': 1}]}
        a = self.make_chunk('Document ID: DOC-A', 'DOC-A')
        b = self.make_chunk('Row 1: Field: Approver | Controlled value: Jane Doe', 'DOC-B')
        resolved = assess_snippet(row, 'Document ID DOC-A && Approver Jane Doe',
                                  {'DOC-A': [a], 'DOC-B': [b]})
        self.assertEqual(resolved['status'], 'unresolved')

    def test_page_mismatch_labeled_not_as_page_match(self):
        row = {'expected_documents': ['DOC-A'], 'evidence_documents': ['DOC-A'],
               'evidence_locations': [{'document_id': 'DOC-A', 'page': 3}]}
        a = self.make_chunk('Row 1: Field: Approver | Controlled value: Jane Doe', 'DOC-A', 1)
        resolved = assess_snippet(row, 'Approver Jane Doe', {'DOC-A': [a]})
        self.assertEqual(resolved['status'], 'same_chunk_other_page')

    def test_real_document_control_normalized_fields(self):
        text = 'Row 1: Field: Classification | Controlled value: CONFIDENTIAL'
        self.assertEqual(table_cells(text), (('classification', 'confidential'),))

    def test_refuse_overwrite_without_changing_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset = root / 'data.jsonl'
            chunks = root / 'chunks.jsonl'
            doc = {'question_id': 'Q1', 'split': 'dev', 'answerable': True,
                   'expected_documents': ['DOC-A'], 'evidence_documents': ['DOC-A'],
                   'evidence_locations': [{'document_id': 'DOC-A', 'page': 1}],
                   'expected_facts': ['Approver Jane Doe'],
                   'evidence': [['Approver Jane Doe']]}
            ch = {'chunk_id': 'CH1', 'document_id': 'DOC-A', 'page_start': 1,
                  'page_end': 1, 'content_type': 'table',
                  'text': 'Row 1: Field: Approver | Controlled value: Jane Doe'}
            dataset.write_text(json.dumps(doc)+'\n')
            chunks.write_text(json.dumps(ch)+'\n')
            before = (dataset.read_bytes(), chunks.read_bytes())
            summary = audit(dataset, chunks, root/'result')
            self.assertEqual(summary['unresolved_snippet_count'], 0)
            with self.assertRaisesRegex(ValueError, 'already exists'):
                audit(dataset, chunks, root/'result')
            self.assertEqual(before, (dataset.read_bytes(), chunks.read_bytes()))


if __name__ == '__main__':
    unittest.main()


def test_does_not_match_partial_token_in_table_value():
    text = 'Row 1: Field: Approver | Controlled value: Jane Doe'
    chunk = Chunk.from_row({'chunk_id': 'T1', 'document_id': 'DOC-A',
                            'page_start': 1, 'page_end': 1, 'content_type': 'table',
                            'text': text})
    assert match_component('Approver Jan', chunk) is None
    assert match_component('Approver Jane', chunk) == 'table_field_value'
