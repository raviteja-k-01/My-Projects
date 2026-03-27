from typing import List, Dict, Any
import gc
import time

from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def load_embedder(model_name: str, device: str):
    """
    Load embedding model
    """
    print(f"[EMBED] Loading model: {model_name}")
    model = SentenceTransformer(model_name, device=device)
    print("[EMBED] Model loaded\n")
    return model


def embed_chunks(chunks, model, batch_size=16):
    start = time.time()

    texts = [c["text"] for c in chunks]

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb.tolist()

    print(f"[EMBED] Done in {time.time()-start:.1f}s\n")
    return chunks


def unload_embedder(model):
    """
    Free memory after embedding
    """
    del model
    gc.collect()
    print("[EMBED] Model unloaded — RAM freed\n")