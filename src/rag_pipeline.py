"""
Pipeline RAG pour l'assistant culturel de Rennes.

Charge l'index FAISS et le mapping depuis data/faiss_index/,
puis expose une fonction ask() qui effectue les 3 étapes RAG :
retrieval (FAISS), augmentation (prompt), génération (Mistral).

Importé par api/main.py pour exposer le pipeline via FastAPI.
"""

import os
import faiss
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from langchain_mistralai import MistralAIEmbeddings

load_dotenv()

INDEX_PATH = "data/faiss_index/events.index"
MAPPING_PATH = "data/faiss_index/events_mapping.csv"

index = faiss.read_index(INDEX_PATH)
mapping = pd.read_csv(MAPPING_PATH, index_col=0)

embed_model = MistralAIEmbeddings(
    model="mistral-embed",
    mistral_api_key=os.getenv("MISTRAL_API_KEY"),
)

client = MistralClient(api_key=os.getenv("MISTRAL_API_KEY"))


def ask(question: str, k: int = 5) -> str:
    """Pipeline RAG complet : retrieval → augmentation → génération."""
    # 1. Retrieval
    question_vector = np.array([embed_model.embed_query(question)], dtype=np.float32)
    _, indices = index.search(question_vector, k)
    context_docs = [mapping.iloc[idx]["corpus"] for idx in indices[0]]
    context = "\n\n---\n\n".join(context_docs)

    # 2. Augmentation
    prompt = f"""Tu es un assistant spécialisé dans les événements culturels à Rennes.
Réponds à la question en te basant uniquement sur les événements fournis ci-dessous.
Si l'information n'est pas dans le contexte, dis-le clairement.

ÉVÉNEMENTS :
{context}

QUESTION : {question}

RÉPONSE :"""

    # 3. Génération
    response = client.chat(
        model="mistral-small-latest",
        messages=[ChatMessage(role="user", content=prompt)]
    )
    return response.choices[0].message.content
