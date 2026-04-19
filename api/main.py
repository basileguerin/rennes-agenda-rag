"""
API REST FastAPI pour l'assistant culturel de Rennes.

Endpoints :
  GET  /health  — état du système
  POST /ask     — pose une question, retourne la réponse RAG
  POST /rebuild — reconstruit l'index FAISS depuis les données fraîches
"""

import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, field_validator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.rag_pipeline import ask, reload_index
import scripts.fetch_events as fetch_events_module
import scripts.build_index as build_index_module

app = FastAPI(
    title="Rennes Agenda RAG",
    description="Assistant culturel pour les événements de Rennes — propulsé par Mistral + FAISS",
    version="1.0.0",
)


class AskRequest(BaseModel):
    question: str

    @field_validator("question")
    @classmethod
    def question_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("La question ne peut pas être vide.")
        return v.strip()


class AskResponse(BaseModel):
    answer: str
    contexts: list[str]


class HealthResponse(BaseModel):
    status: str
    vectors: int


class RebuildResponse(BaseModel):
    status: str
    vectors: int


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse, summary="État du système")
def health():
    """Vérifie que l'index FAISS est chargé et retourne le nombre de vecteurs."""
    from src.rag_pipeline import faiss_store
    return {"status": "ok", "vectors": faiss_store.index.ntotal}


@app.post("/ask", response_model=AskResponse, summary="Poser une question")
def ask_question(body: AskRequest):
    """
    Pose une question sur les événements culturels de Rennes.
    Retourne la réponse générée par Mistral et les documents récupérés par FAISS.
    """
    try:
        answer, contexts = ask(body.question)
        return {"answer": answer, "contexts": contexts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rebuild", response_model=RebuildResponse, summary="Reconstruire l'index")
def rebuild():
    """
    Reconstruit l'index FAISS depuis les données fraîches OpenAgenda.
    Attention : cette opération prend plusieurs minutes.
    """
    try:
        fetch_events_module.main()
        build_index_module.main()
        vectors = reload_index()
        return {"status": "ok", "vectors": vectors}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
