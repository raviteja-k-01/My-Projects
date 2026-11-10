import os
import re

RAW_PATH = "data/raw"
CLEAN_PATH = "data/cleaned"

def clean_text(text):
    # remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # remove Source lines
    text = re.sub(r"Source:.*", "", text, flags=re.IGNORECASE)

    # remove URL labels and links
    text = re.sub(r"URL:.*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"http\S+|www\S+", "", text)

    # remove headings
    text = re.sub(
        r"\b(Definition|How It Works|Key Features|Common Use Cases|Overview)\b",
        "",
        text,
        flags=re.IGNORECASE
    )

    # remove standalone titles (like "Amazon S3 Overview")
    text = re.sub(r"^[A-Z][A-Za-z0-9\s\-]{5,}$", "", text, flags=re.MULTILINE)

    # remove extra spaces/newlines
    text = re.sub(r"\s+", " ", text)

    # remove non-ascii characters
    text = re.sub(r"[^\x00-\x7F]+", " ", text)

    return text.strip()

def process_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for file in os.listdir(input_folder):
        input_path = os.path.join(input_folder, file)
        output_path = os.path.join(output_folder, file)

        if file.endswith(".txt"):
            with open(input_path, "r", encoding="utf-8") as f:
                text = f.read()

            cleaned = clean_text(text)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(cleaned)

def main():
    for category in ["aws_docs", "internal_docs"]:
        process_folder(
            os.path.join(RAW_PATH, category),
            os.path.join(CLEAN_PATH, category)
        )

if __name__ == "__main__":
    main()