import re

def split_sections(text):
    lines = text.split("\n")
    sections = []
    current_section = []

    for line in lines:
        line = line.strip()

        if re.match(r"^(Section\s+)?\d+(\.\d+)+", line):
            if current_section:
                sections.append("\n".join(current_section))
                current_section = []

        current_section.append(line)

    if current_section:
        sections.append("\n".join(current_section))

    return sections


def detect_table_rows(text):
    rows = []
    lines = text.split("\n")

    for line in lines:
        line = line.strip()

        if len(line) < 25:
            continue

        has_numbers = bool(re.search(r"\d", line))
        has_keywords = any(k in line.lower() for k in [
            "tier", "band", "limit", "per", "class", "hours", "days"
        ])

        is_not_sentence = len(line.split(".")) <= 2

        if has_numbers and has_keywords and is_not_sentence:
            rows.append(line)

    return rows


def chunk_pages_structured(pages,chunk_size=None, overlap=None):
    chunks = []

    for page in pages:
        text = page["text"]

        sections = split_sections(text)

        for sec in sections:
            content = sec.strip()

            if len(content) < 80:
                continue

            title = content.split("\n")[0][:120]

            # ✅ Section chunk
            chunks.append({
                "doc_id": page["doc_id"],
                "page": page["page"],
                "section": title,
                "text": content,
                "type": "section",
                "length": len(content)
            })

            # ✅ Table row chunks
            for row in detect_table_rows(content):
                chunks.append({
                    "doc_id": page["doc_id"],
                    "page": page["page"],
                    "section": title,
                    "text": row,
                    "type": "table_row",
                    "length": len(row)
                })

    return chunks