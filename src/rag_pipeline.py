"""
Pipeline RAG pour l'assistant culturel de Rennes.

Charge l'index LangChain FAISS depuis data/faiss_index/,
puis expose une fonction ask() qui effectue les 3 étapes RAG :
retrieval (FAISS), augmentation (prompt), génération (Mistral).

Importé par api/main.py pour exposer le pipeline via FastAPI.
"""

import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_mistralai import MistralAIEmbeddings
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

load_dotenv()

INDEX_DIR = "data/faiss_index"

embed_model = MistralAIEmbeddings(
    model="mistral-embed",
    mistral_api_key=os.getenv("MISTRAL_API_KEY"),
)

faiss_store = FAISS.load_local(INDEX_DIR, embed_model, allow_dangerous_deserialization=True)
client = MistralClient(api_key=os.getenv("MISTRAL_API_KEY"))


def ask(question: str, k: int = 5) -> str:
    """Pipeline RAG complet : retrieval → augmentation → génération."""
    # 1. Retrieval — similarity_search retourne des Documents avec page_content + metadata
    docs = faiss_store.similarity_search(question, k=k)
    context_parts = []
    for doc in docs:
        m = doc.metadata
        context_parts.append(
            f"Titre : {m['title']}\n"
            f"Description : {doc.page_content}\n"
            f"Tarif : {m['conditions']}\n"
            f"Dates : {m['date_start']} → {m['date_end']}\n"
            f"Lieu : {m['location']}\n"
            f"Lien : {m['url']}"
        )
    context = "\n\n---\n\n".join(context_parts)

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
