"""
Évaluation automatisée du pipeline RAG via RAGAS.

Calcule faithfulness, answer_relevancy, context_precision, context_recall
avec Mistral comme LLM-as-judge. Sort avec le code 1 si un seuil n'est pas atteint.

Prérequis : index FAISS construit (scripts/build_index.py) et MISTRAL_API_KEY dans .env
Usage    : python tests/evaluate_rag.py
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from datasets import Dataset
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_mistralai import MistralAIEmbeddings
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.rag_pipeline import ask

load_dotenv()

THRESHOLDS = {
    "faithfulness":      0.80,
    "answer_relevancy":  0.80,
    "context_precision": 0.80,
    "context_recall":    0.80,
}

TEST_QUESTIONS = [
    "Quels concerts sont prévus à Rennes ?",
    "Y a-t-il des expositions gratuites à Rennes ?",
    "Quels événements sont proposés pour les enfants à Rennes ?",
    "Y a-t-il des spectacles de danse à Rennes ?",
    "Quels événements gratuits peut-on trouver à Rennes ?",
    "Y a-t-il des conférences ou ateliers culturels à Rennes ?",
    "Y a-t-il des événements musicaux en plein air à Rennes ?",
]

GROUND_TRUTHS = [
    "Plusieurs concerts sont prévus à Rennes avec des informations sur les dates, les lieux et les tarifs.",
    "Oui, il existe des expositions gratuites à Rennes dans des musées et centres d'art de la ville.",
    "Des visites guidées pour enfants au musée (gratuit, sur inscription), des jeux de piste et des stages artistiques sont proposés à Rennes.",
    "Des spectacles de danse sont organisés à Rennes dans différentes salles culturelles.",
    "De nombreux événements gratuits sont disponibles à Rennes dans différents lieux culturels.",
    "Des conférences et ateliers culturels sont organisés à Rennes.",
    "Des événements musicaux en plein air sont organisés à Rennes, notamment des open air et la Fête de la Musique.",
]


def collect_rag_outputs() -> tuple[list[str], list[list[str]]]:
    """Lance le pipeline RAG sur chaque question et collecte réponses + contextes."""
    answers, contexts_list = [], []
    for i, question in enumerate(TEST_QUESTIONS):
        print(f"  [{i+1}/{len(TEST_QUESTIONS)}] {question}")
        answer, contexts = ask(question)
        answers.append(answer)
        contexts_list.append(contexts)
    return answers, contexts_list


def run_evaluation(answers: list[str], contexts_list: list[list[str]]) -> dict:
    """Construit le dataset RAGAS et calcule les 4 métriques."""
    dataset = Dataset.from_dict({
        "question":     TEST_QUESTIONS,
        "answer":       answers,
        "contexts":     contexts_list,
        "ground_truth": GROUND_TRUTHS,
    })
    ragas_llm = ChatMistralAI(
        model="mistral-small-latest",
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
    )
    ragas_embeddings = MistralAIEmbeddings(
        model="mistral-embed",
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
    )
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )
    return {
        "faithfulness":      result["faithfulness"],
        "answer_relevancy":  result["answer_relevancy"],
        "context_precision": result["context_precision"],
        "context_recall":    result["context_recall"],
    }


def print_report(scores: dict) -> bool:
    """Affiche les scores et retourne True si tous les seuils sont atteints."""
    print("\n=== Résultats RAGAS ===")
    all_pass = True
    for metric, score in scores.items():
        threshold = THRESHOLDS[metric]
        status = "✓" if score >= threshold else "✗"
        if score < threshold:
            all_pass = False
        print(f"  {status} {metric:<22} {score:.3f}  (seuil : {threshold})")
    return all_pass


def main():
    print("Collecte des réponses RAG...")
    answers, contexts_list = collect_rag_outputs()

    print("\nÉvaluation RAGAS en cours...")
    scores = run_evaluation(answers, contexts_list)

    all_pass = print_report(scores)

    if not all_pass:
        print("\nUn ou plusieurs seuils non atteints.")
        sys.exit(1)
    print("\nTous les seuils atteints.")


if __name__ == "__main__":
    main()
