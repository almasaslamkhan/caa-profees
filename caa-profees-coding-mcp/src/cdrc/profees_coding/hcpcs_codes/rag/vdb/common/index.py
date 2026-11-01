from typing import Dict, List, Tuple

import faiss
import numpy as np
from openai import OpenAI


def build_faiss_index(
    encounter_summaries: List[Dict],
    embedding_model: str = "text-embedding-3-large",
) -> Tuple[faiss.IndexFlatIP, List[Dict]]:
    """
    Build a FAISS cosine-similarity index from encounter summaries.

    Parameters
    ----------
    encounter_summaries : List[Dict]
        Each dict must include:
          - encounter_id
          - encounter_source
          - gold_g2211
          - summary_compact

    embedding_model : str
        Embedding model name

    Returns
    -------
    index : faiss.IndexFlatIP
        FAISS index using cosine similarity

    metadata : List[Dict]
        Metadata aligned by FAISS index position
    """

    client = OpenAI()

    texts = []
    metadata = []

    for r in encounter_summaries:
        text = r.get("summary_compact")
        if not text:
            continue

        texts.append(text)
        metadata.append(
            {
                "encounter_id": r.get("encounter_id"),
                "encounter_source": r.get("encounter_source"),
                "gold_g2211": r.get("gold_g2211"),
            }
        )

    if not texts:
        raise ValueError("No valid summaries found to embed.")

    # ---- Embedding ----
    embeddings = []
    for t in texts:
        resp = client.embeddings.create(model=embedding_model, input=t)
        embeddings.append(resp.data[0].embedding)

    vecs = np.array(embeddings, dtype="float32")

    # ---- Normalize for cosine similarity ----
    faiss.normalize_L2(vecs)

    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product == cosine after normalization
    index.add(vecs)

    return index, metadata
