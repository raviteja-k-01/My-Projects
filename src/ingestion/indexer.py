from typing import List, Dict
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct


def build_qdrant_client(db_path: str) -> QdrantClient:
    """
    Create local Qdrant client
    """
    print(f"[INDEX] Opening Qdrant at: {db_path}")
    return QdrantClient(path=db_path)


def create_collection(
    client: QdrantClient,
    collection_name: str,
    vector_size: int,
    recreate: bool = True
):
    """
    Create or recreate collection
    """
    if recreate:
        try:
            client.delete_collection(collection_name)
            print(f"[INDEX] Deleted existing collection '{collection_name}'")
        except Exception:
            pass

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE
        )
    )

    print(f"[INDEX] Created collection '{collection_name}' (dim={vector_size})")


def upsert_chunks(
    client: QdrantClient,
    collection_name: str,
    chunks: List[Dict],
    batch_size: int = 256
):
    """
    Insert chunks into Qdrant
    """

    total = len(chunks)

    for i in range(0, total, batch_size):
        batch = chunks[i:i + batch_size]

        points = []

        for idx, c in enumerate(batch):
            payload = {
                k: v for k, v in c.items()
                if k not in ["embedding", "chunk_id", "id"]
            }

            points.append(
                PointStruct(
                    id=i + idx,   # unique integer ID
                    vector=c["embedding"],
                    payload=payload
                )
            )

        client.upsert(
            collection_name=collection_name,
            points=points
        )

    print(f"[INDEX] {total} chunks indexed\n")