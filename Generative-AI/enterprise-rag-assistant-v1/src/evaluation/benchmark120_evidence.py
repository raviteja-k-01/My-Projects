#!/usr/bin/env python3
"""Offline, generic source-anchor validation for RAG benchmark datasets.

This tool does NOT generate or modify gold labels. It uses only source text and
source metadata (including structured table cells). A match is not entailment.
No model, embedding, retrieval API, or network dependencies.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path


def norm(value: object) -> str:
    return ' '.join(re.findall(r'\w+', unicodedata.normalize('NFKC', str(value)).casefold(), re.UNICODE))


def sha256(path: Path) -> str:
    dig = hashlib.sha256()
    with path.open('rb') as stream:
        for piece in iter(lambda: stream.read(1 << 20), b''):
            dig.update(piece)
    return dig.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    found = []
    with path.open(encoding='utf-8-sig') as stream:
        for i, raw in enumerate(stream, 1):
            if not raw.strip():
                continue
            try:
                item = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f'{path}:{i} invalid JSON: {exc}') from exc
            if not isinstance(item, dict):
                raise ValueError(f'{path}:{i} must be a JSON object')
            found.append(item)
    return found


def snippet_list(evidence: object) -> list[str]:
    if not isinstance(evidence, list):
        raise ValueError('evidence must be a list')
    snippets = []
    for group in evidence:
        if isinstance(group, str):
            snippets.append(group)
        elif isinstance(group, list) and all(isinstance(p, str) for p in group):
            snippets.extend(group)
        else:
            raise ValueError(f'Unsupported evidence group: {type(group).__name__}')
    return snippets


# These fields are discovered from the actual table, never enumerated by policy.
TABLE_ROW = re.compile(r'^Row\s+\d+:\s+Field:\s*(.*?)\s*\|\s*Controlled value:\s*(.*)$', re.I)


def table_cells(text: str) -> tuple[tuple[str, str], ...]:
    cells = []
    for line in text.splitlines():
        result = TABLE_ROW.match(line.strip())
        if result:
            cells.append((norm(result.group(1)), norm(result.group(2))))
    return tuple(cells)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    page_start: int
    page_end: int
    content_type: str
    text: str
    normalized: str
    fields: tuple[tuple[str, str], ...]
    terms: frozenset[str]

    @classmethod
    def from_row(cls, row: dict) -> 'Chunk':
        keys = ('chunk_id', 'document_id', 'page_start', 'page_end', 'text')
        if any(k not in row for k in keys):
            raise ValueError(f'Corpus chunk is missing required fields: {row}')
        text = row['text']
        if not isinstance(text, str):
            raise ValueError(f"Chunk {row['chunk_id']} text is not a string")
        start, end = row['page_start'], row['page_end']
        if type(start) is not int or type(end) is not int or start < 1 or end < start:
            raise ValueError(f"Bad page range in {row['chunk_id']}")
        normalized = norm(text)
        return cls(row['chunk_id'], row['document_id'], start, end,
                   row.get('content_type', ''), text, normalized, table_cells(text),
                   frozenset(normalized.split()))


def contains_token_phrase(haystack: str, needle: str) -> bool:
    """Whole-token contiguous phrase, not an arbitrary substring of a token."""
    return bool(needle) and f" {needle} " in f" {haystack} "


def match_component(component: str, chunk: Chunk) -> str | None:
    """Exact normalized phrase OR a full field-value pair within one table row.

    It is unsafe to treat field and value appearing *anywhere* in a table as a
    match; both must be linked to the same parsed row.
    """
    candidate = norm(component)
    if not candidate:
        return None
    if contains_token_phrase(chunk.normalized, candidate):
        return 'normalized_text'
    for field, value in chunk.fields:
        if candidate.startswith(field + ' ') and contains_token_phrase(value, candidate[len(field) + 1:]):
            return 'table_field_value'
    return None


def listed_pages(row: dict, doc: str) -> set[int]:
    return {loc.get('page') for loc in row.get('evidence_locations', [])
            if isinstance(loc, dict) and loc.get('document_id') == doc and
            type(loc.get('page')) is int}


def assess_snippet(row: dict, snippet: str, by_doc: dict[str, list[Chunk]]) -> dict:
    components = [x.strip() for x in snippet.split('&&')]
    if any(not norm(c) for c in components):
        return {'snippet': snippet, 'status': 'invalid_component', 'parts': components,
                'document_id': None, 'chunk_ids': [], 'match_methods': []}
    documents = sorted(set(row.get('expected_documents', [])) |
                       set(row.get('evidence_documents', [])))
    candidates = []
    for doc in documents:
        chunks = by_doc.get(doc, [])
        if not chunks:
            continue
        per_component = []
        for part in components:
            matches = [(c, kind) for c in chunks if (kind := match_component(part, c))]
            if not matches:
                break
            per_component.append(matches)
        if len(per_component) != len(components):
            continue
        pages = listed_pages(row, doc)
        page_matches = [[(c, kind) for c, kind in matches if any(
            c.page_start <= page <= c.page_end for page in pages)] for matches in per_component]
        on_pages = bool(pages) and all(page_matches)
        useful = page_matches if on_pages else per_component
        shared = set(c.chunk_id for c, _ in useful[0])
        for matched in useful[1:]:
            shared &= {c.chunk_id for c, _ in matched}
        if shared:
            # A single chunk proves co-location of all clauses, not entailment.
            best_id = sorted(shared)[0]
            chosen = [(next((c, how) for c, how in matched if c.chunk_id == best_id))
                      for matched in useful]
            same_chunk = True
        else:
            chosen = [sorted(matched, key=lambda v: (v[0].page_start, v[0].chunk_id))[0]
                      for matched in useful]
            same_chunk = False
        # Favor same chunk, listed page, then fewer distinct chunks, with stable IDs.
        candidates.append((same_chunk, on_pages, -len(set(c.chunk_id for c, _ in chosen)), doc, chosen))
    if not candidates:
        return {'snippet': snippet, 'status': 'unresolved', 'parts': components,
                'document_id': None, 'chunk_ids': [], 'match_methods': []}
    # Prefer page accuracy over same-chunk match in a wrong page.
    same_chunk, on_pages, _neg_unique, doc, chosen = sorted(
        candidates, key=lambda x: (x[1], x[0], x[2], x[3]), reverse=True)[0]
    if on_pages and same_chunk:
        status = 'same_chunk_on_listed_page'
    elif on_pages:
        status = 'same_document_on_listed_pages'
    elif same_chunk:
        status = 'same_chunk_other_page'
    else:
        status = 'same_document_other_page'
    return {'snippet': snippet, 'status': status, 'parts': components,
            'document_id': doc, 'chunk_ids': list(dict.fromkeys(c.chunk_id for c, _ in chosen)),
            'chunk_pages': [f'{c.page_start}-{c.page_end}' for c, _ in chosen],
            'match_methods': [how for _, how in chosen]}


def write_csv(path: Path, fieldnames: list[str], records: list[dict]) -> None:
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)


def document_control(by_doc: dict[str, list[Chunk]]) -> tuple[list[dict], list[dict]]:
    """Extract structured document control metadata for ALL corpus documents."""
    rows, issues = [], []
    for doc, chunks in sorted(by_doc.items()):
        table_chunks = [c for c in chunks if c.content_type == 'table' and c.fields
                        and c.page_start in (1, 2)]
        values = collections.defaultdict(set)
        witnesses = collections.defaultdict(list)
        for c in table_chunks:
            for field, value in c.fields:
                values[field].add(value)
                witnesses[field].append(c.chunk_id)
        for field, variants in sorted(values.items()):
            for value in sorted(variants):
                rows.append({'document_id': doc, 'field': field, 'value_normalized': value,
                             'chunk_ids': '; '.join(sorted(set(witnesses[field]))),
                             'source_type': 'structured_table'})
            if len(variants) > 1:
                issues.append({'document_id': doc, 'field': field,
                               'issue': 'different_control_values_between_pages',
                               'values': '; '.join(sorted(variants))})
    return rows, issues


STOP = frozenset('the a an of to and or in on at with is are for that this how which what who when does do as be by from after must have has require required a specific policy policies records record approval'.split())


def lexical_hints(query: str, chunks: list[Chunk], max_items: int = 4) -> list[dict]:
    """Non-authoritative search aids for manual reviewers, not gold relevance."""
    words = [word for word in norm(query).split() if word not in STOP and len(word) > 2]
    if not words:
        return []
    counts = collections.Counter(words)
    ranked = []
    for ch in chunks:
        shared = sum(value for term, value in counts.items() if term in ch.terms)
        if shared:
            ranked.append((shared / sum(counts.values()), shared, ch))
    ranked.sort(key=lambda x: (-x[0], -x[1], x[2].chunk_id))
    return [{'chunk_id': ch.chunk_id, 'document_id': ch.document_id,
             'page': ch.page_start, 'token_overlap_fraction': round(frac, 3)}
            for frac, _shared, ch in ranked[:max_items]]


def audit(dataset: Path, corpus: Path, out: Path) -> dict:
    if out.exists():
        raise ValueError(f'Output already exists (no overwrite): {out}')
    questions = read_jsonl(dataset)
    chunks = [Chunk.from_row(c) for c in read_jsonl(corpus)]
    by_doc = collections.defaultdict(list)
    chunk_ids = set()
    for c in chunks:
        if c.chunk_id in chunk_ids:
            raise ValueError(f'Duplicate chunk ID: {c.chunk_id}')
        chunk_ids.add(c.chunk_id)
        by_doc[c.document_id].append(c)
    question_ids = [q.get('question_id') for q in questions]
    if len(question_ids) != len(set(question_ids)):
        raise ValueError('Duplicate question IDs')
    for q in questions:
        if not isinstance(q.get('expected_documents'), list) or not isinstance(q.get('evidence_documents'), list):
            raise ValueError(f'Invalid evidence document metadata for {q.get("question_id")}')
        if not isinstance(q.get('expected_facts'), list):
            raise ValueError(f'Invalid expected_facts for {q.get("question_id")}')
    per_q, flat, metadata, fact_review, abstention = [], [], [], [], []
    for row in questions:
        qid = row['question_id']
        anchors = [assess_snippet(row, snippet, by_doc)
                   for snippet in snippet_list(row.get('evidence', []))]
        located_docs = set()
        for index, match in enumerate(anchors, 1):
            if match['document_id']:
                located_docs.add(match['document_id'])
            flat.append({'question_id': qid, 'split': row.get('split'), 'snippet_number': index,
                         'status': match['status'], 'source_document': match['document_id'] or '',
                         'candidate_chunk_ids': '; '.join(match['chunk_ids']),
                         'match_methods': '; '.join(match['match_methods']),
                         'snippet': match['snippet']})
        missing_meta = sorted(set(row['expected_documents']) - set(row['evidence_documents']))
        missing_anchor = sorted(set(row['expected_documents']) - located_docs)
        for doc in sorted(set(missing_meta) | set(missing_anchor)) if row.get('answerable', True) else []:
            metadata.append({'question_id': qid, 'split': row.get('split'),
                             'document_id': doc,
                             'not_in_evidence_documents': doc in missing_meta,
                             'no_matched_snippet_in_document': doc in missing_anchor,
                             'review_required': 'yes'})
        expected_chunks = [c for doc in row['expected_documents'] for c in by_doc.get(doc, [])]
        for fact_index, fact in enumerate(row['expected_facts'], 1):
            hints = lexical_hints(str(fact), expected_chunks, 3)
            fact_review.append({'question_id': qid, 'split': row.get('split'),
                                'fact_number': fact_index, 'expected_fact': fact,
                                'candidate_chunk_ids_unverified': '; '.join(c['chunk_id'] for c in hints),
                                'human_fact_verified': '', 'reviewer_notes': ''})
        if not row.get('answerable', True):
            hints = lexical_hints(str(row.get('question', '')), expected_chunks, 5)
            abstention.append({'question_id': qid, 'split': row.get('split'),
                               'question': row.get('question'),
                               'expected_documents': '; '.join(row['expected_documents']),
                               'candidate_chunks_for_manual_negative_review': '; '.join(
                                   c['chunk_id'] for c in hints),
                               'verified_unanswerable': '', 'reviewer_notes': ''})
        per_q.append({'question_id': qid, 'split': row.get('split'),
                      'answerable': row.get('answerable'),
                      'expected_documents': row['expected_documents'],
                      'evidence_snippets': anchors,
                      'docs_without_anchored_evidence': missing_anchor,
                      'source_anchor_status': ('unanswerable_requires_negative_review'
                          if not row.get('answerable', True) else
                          'missing_document_anchor_review' if missing_anchor else
                          'all_snippets_located' if anchors and all(
                              m['status'].startswith('same_') for m in anchors) else
                          'needs_manual_source_review')})
    controls, control_issues = document_control(by_doc)
    status_counts = collections.Counter(row['status'] for row in flat)
    manual = sorted(set(r['question_id'] for r in metadata) | set(r['question_id'] for r in abstention) |
                    {r['question_id'] for r in flat if r['status'] in ('unresolved', 'invalid_component')})
    summary = {'dataset_sha256': sha256(dataset), 'corpus_sha256': sha256(corpus),
               'question_count': len(questions), 'corpus_chunk_count': len(chunks),
               'source_document_count': len(by_doc),
               'evidence_snippet_count': len(flat), 'status_counts': dict(status_counts),
               'unresolved_snippet_count': status_counts.get('unresolved', 0) + status_counts.get('invalid_component', 0),
               'missing_expected_document_anchor_rows': len(metadata),
               'unanswerable_review_rows': len(abstention),
               'expected_fact_review_rows': len(fact_review),
               'document_control_rows': len(controls),
               'document_control_inconsistencies': len(control_issues),
               'questions_with_priority_review_flags': manual,
               'limitations': ['Text and table-cell matches do not verify entailment.',
                               'Lexical hints are not gold chunk labels.',
                               'Unanswerability requires qualified source review.',
                               'No reference answer or dataset field was changed.',
                               'Test split must not be used for prompt tuning.']}
    out.mkdir(parents=True, exist_ok=False)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    with (out / 'anchor_candidates.jsonl').open('w', encoding='utf-8') as stream:
        for row in per_q:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    write_csv(out / 'snippet_matches.csv', ['question_id', 'split', 'snippet_number', 'status',
              'source_document', 'candidate_chunk_ids', 'match_methods', 'snippet'], flat)
    write_csv(out / 'missing_document_anchors.csv', ['question_id', 'split', 'document_id',
              'not_in_evidence_documents', 'no_matched_snippet_in_document', 'review_required'], metadata)
    write_csv(out / 'fact_review.csv', ['question_id', 'split', 'fact_number', 'expected_fact',
              'candidate_chunk_ids_unverified', 'human_fact_verified', 'reviewer_notes'], fact_review)
    write_csv(out / 'abstention_review.csv', ['question_id', 'split', 'question', 'expected_documents',
              'candidate_chunks_for_manual_negative_review', 'verified_unanswerable', 'reviewer_notes'], abstention)
    write_csv(out / 'document_control.csv', ['document_id', 'field', 'value_normalized', 'chunk_ids',
              'source_type'], controls)
    write_csv(out / 'document_control_conflicts.csv', ['document_id', 'field', 'issue', 'values'], control_issues)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--chunks', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if not args.dataset.is_file() or not args.chunks.is_file():
        parser.error('Both input files must exist')
    try:
        summary = audit(args.dataset, args.chunks, args.out)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f'ERROR: {exc}\n')
    print('AUDIT_OK', 'QUESTIONS', summary['question_count'],
          'SNIPPETS', summary['evidence_snippet_count'],
          'UNRESOLVED', summary['unresolved_snippet_count'],
          'MISSING_DOCUMENT_ANCHORS', summary['missing_expected_document_anchor_rows'],
          'ABSTENTION_REVIEWS', summary['unanswerable_review_rows'])
    print('RESULTS:', args.out)
    print('OFFLINE_ONLY: dataset and corpus unchanged; candidate chunk IDs are NOT gold labels')


if __name__ == '__main__':
    main()
