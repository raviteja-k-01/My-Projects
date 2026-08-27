import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient


def main() -> None:
    load_dotenv()

    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    if not qdrant_url:
        raise ValueError("QDRANT_URL is missing")

    if not qdrant_api_key:
        raise ValueError("QDRANT_API_KEY is missing")

    client = QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key,
    )

    collections = client.get_collections()

    print("\nQDRANT CLOUD CONNECTION")
    print("Connection: PASS")
    print(
        f"Existing collections: "
        f"{len(collections.collections)}"
    )

    print("\nQDRANT CLOUD: PASS")

    client.close()


if __name__ == "__main__":
    main()