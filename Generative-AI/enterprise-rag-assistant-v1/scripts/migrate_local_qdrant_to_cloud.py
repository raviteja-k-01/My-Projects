import os
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

from src.indexing.embedding_client import EMBEDDING_DIMENSION
from src.indexing.qdrant_store import COLLECTION_NAME


LOCAL_QDRANT_PATH = Path("data/qdrant")
BATCH_SIZE = 100


def main() -> None:
    load_dotenv()

    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    if not qdrant_url:
        raise ValueError("QDRANT_URL is missing")

    if not qdrant_api_key:
        raise ValueError("QDRANT_API_KEY is missing")

    if not LOCAL_QDRANT_PATH.exists():
        raise ValueError(
            f"Local Qdrant path does not exist: "
            f"{LOCAL_QDRANT_PATH}"
        )

    # Connect directly to the OLD local database.
    local_client = QdrantClient(
        path=str(LOCAL_QDRANT_PATH)
    )

    # Connect to the NEW Qdrant Cloud database.
    cloud_client = QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key,
    )

    try:
        if not local_client.collection_exists(
            COLLECTION_NAME
        ):
            raise ValueError(
                "Local collection does not exist: "
                f"{COLLECTION_NAME}"
            )

        local_count = local_client.count(
            collection_name=COLLECTION_NAME,
            exact=True,
        ).count

        print("\nQDRANT MIGRATION")
        print(f"Local points found: {local_count}")

        # Create the cloud collection if needed.
        if not cloud_client.collection_exists(
            COLLECTION_NAME
        ):
            cloud_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=EMBEDDING_DIMENSION,
                    distance=models.Distance.COSINE,
                ),
            )

            print("Cloud collection created")

        else:
            print("Cloud collection already exists")

        next_offset = None
        migrated_count = 0

        while True:
            records, next_offset = local_client.scroll(
                collection_name=COLLECTION_NAME,
                limit=BATCH_SIZE,
                offset=next_offset,
                with_payload=True,
                with_vectors=True,
            )

            if not records:
                break

            points = []

            for record in records:
                if record.vector is None:
                    raise ValueError(
                        f"Point {record.id} has no vector"
                    )

                points.append(
                    models.PointStruct(
                        id=record.id,
                        vector=record.vector,
                        payload=record.payload,
                    )
                )

            cloud_client.upsert(
                collection_name=COLLECTION_NAME,
                points=points,
                wait=True,
            )

            migrated_count += len(points)

            print(
                f"Migrated "
                f"{migrated_count}/{local_count}"
            )

            if next_offset is None:
                break

        cloud_count = cloud_client.count(
            collection_name=COLLECTION_NAME,
            exact=True,
        ).count

        print("\nMIGRATION SUMMARY")
        print(f"Local points: {local_count}")
        print(f"Cloud points: {cloud_count}")

        if cloud_count != local_count:
            raise ValueError(
                "Migration count mismatch. "
                f"Local={local_count}, "
                f"Cloud={cloud_count}"
            )

        print(
            "\nLOCAL → QDRANT CLOUD MIGRATION: PASS"
        )

    finally:
        local_client.close()
        cloud_client.close()


if __name__ == "__main__":
    main()