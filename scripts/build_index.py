"""
Construction de l'index FAISS à partir des événements préprocessés.

Génère un embedding par événement via l'API Mistral (mistral-embed)
et construit un index FAISS via LangChain. 1 Document = 1 événement.

Prérequis : avoir exécuté fetch_events.py au préalable.
Résultat : data/faiss_index/ — index LangChain FAISS (index.faiss + index.pkl)
"""

import os
import time
import pandas as pd
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_mistralai import MistralAIEmbeddings

load_dotenv()

PROCESSED_PATH = "data/processed/events_clean.csv"
INDEX_DIR = "data/faiss_index"
BATCH_SIZE = 50  # limite conservative pour éviter les erreurs 429


def build_documents(df: pd.DataFrame) -> list[Document]:
    """Crée un Document LangChain par événement (corpus complet + métadonnées)."""
    documents = []
    for _, row in df.iterrows():
        metadata = {
            "uid": row["uid"],
            "title": row["title_fr"],
            "conditions": row["conditions_fr"],
            "date_start": row["firstdate_begin"],
            "date_end": row["lastdate_end"],
            "location": f"{row['location_name']}, {row['location_address']}",
            "url": row["canonicalurl"],
        }
        documents.append(Document(page_content=row["corpus"], metadata=metadata))
    return documents


def build_faiss_store(documents: list[Document], embed_model: MistralAIEmbeddings) -> FAISS:
    """Construit l'index FAISS par batch pour respecter le rate limiting."""
    faiss_store = None
    for i in range(0, len(documents), BATCH_SIZE):
        batch = documents[i:i + BATCH_SIZE]
        if faiss_store is None:
            faiss_store = FAISS.from_documents(batch, embed_model)
        else:
            faiss_store.add_documents(batch)
        print(f"{min(i + BATCH_SIZE, len(documents))}/{len(documents)}", end="\r")
        time.sleep(1)
    print()
    return faiss_store


def main():
    df = pd.read_csv(PROCESSED_PATH)
    df = df.dropna(subset=["longdescription_fr"])
    print(f"{len(df)} événements chargés")

    documents = build_documents(df)
    print(f"{len(documents)} chunks créés")

    embed_model = MistralAIEmbeddings(
        model="mistral-embed",
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
    )

    faiss_store = build_faiss_store(documents, embed_model)
    print(f"Index FAISS : {faiss_store.index.ntotal} vecteurs")

    os.makedirs(INDEX_DIR, exist_ok=True)
    faiss_store.save_local(INDEX_DIR)
    print(f"Index sauvegardé dans {INDEX_DIR}/")


if __name__ == "__main__":
    main()
