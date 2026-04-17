"""
Construction de l'index FAISS à partir des événements préprocessés.

Charge data/processed/events_clean.csv, génère un embedding par événement
via l'API Mistral (mistral-embed), puis construit et sérialise un index FAISS.

Prérequis : avoir exécuté fetch_events.py au préalable.
Résultat :
  - data/faiss_index/events.index   — index FAISS sérialisé
  - data/faiss_index/events_mapping.csv — correspondance position → uid/corpus
"""

import os
import time
import faiss
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langchain_mistralai import MistralAIEmbeddings

load_dotenv()

PROCESSED_PATH = "data/processed/events_clean.csv"
INDEX_PATH = "data/faiss_index/events.index"
MAPPING_PATH = "data/faiss_index/events_mapping.csv"
BATCH_SIZE = 50  # limite conservative pour éviter les erreurs 429


def load_embeddings_model() -> MistralAIEmbeddings:
    """Initialise le modèle d'embedding Mistral depuis la variable d'environnement."""
    return MistralAIEmbeddings(
        model="mistral-embed",
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
    )


def generate_embeddings(texts: list[str], model: MistralAIEmbeddings) -> list[list[float]]:
    """Génère les embeddings par batch avec pause pour respecter le rate limiting."""
    all_vectors = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        all_vectors.extend(model.embed_documents(batch))
        print(f"{len(all_vectors)}/{len(texts)}", end="\r")
        time.sleep(1)
    print()
    return all_vectors


def build_faiss_index(vectors: list[list[float]]) -> faiss.IndexFlatL2:
    """Construit un index FAISS à partir des vecteurs."""
    vectors_np = np.array(vectors, dtype=np.float32)  # FAISS attend du float32
    index = faiss.IndexFlatL2(vectors_np.shape[1])
    index.add(vectors_np)
    return index


def main():
    df = pd.read_csv(PROCESSED_PATH)
    print(f"{len(df)} événements chargés")

    model = load_embeddings_model()
    vectors = generate_embeddings(df["corpus"].tolist(), model)

    index = build_faiss_index(vectors)
    print(f"Index FAISS : {index.ntotal} vecteurs de dimension {index.d}")

    os.makedirs("data/faiss_index", exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    df[["uid", "corpus"]].to_csv(MAPPING_PATH, index=True)
    print(f"Index et mapping sauvegardés dans data/faiss_index/")


if __name__ == "__main__":
    main()
