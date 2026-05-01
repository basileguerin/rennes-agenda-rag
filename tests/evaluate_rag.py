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
    "context_precision": 0.75,
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
    "Quels spectacles de théâtre sont programmés à Rennes ?",
    "Y a-t-il des projections de cinéma à Rennes ?",
    "Quels festivals ont lieu à Rennes ?",
    "Y a-t-il des visites guidées à Rennes ?",
    "Quels événements de musique classique sont prévus à Rennes ?",
    "Y a-t-il des événements familiaux à Rennes ?",
    "Quels musées proposent des expositions à Rennes ?",
    "Y a-t-il des événements gratuits pour les enfants à Rennes ?",
    "Y a-t-il des ateliers créatifs ou artistiques à Rennes ?",
    "Quels événements de danse contemporaine sont programmés à Rennes ?",
    "Y a-t-il des lectures ou rencontres littéraires à Rennes ?",
    "Quels événements se déroulent au Théâtre National de Bretagne ?",
    "Y a-t-il des concerts gratuits à Rennes ?",
    "Quels événements de cirque ou arts de la rue sont programmés à Rennes ?",
    "Y a-t-il des expositions d'art contemporain à Rennes ?",
    "Quels événements se déroulent aux Champs Libres à Rennes ?",
    "Y a-t-il des événements de musique électronique à Rennes ?",
    "Y a-t-il des opéras ou spectacles lyriques à Rennes ?",
    "Quels événements nocturnes ou soirées culturelles sont prévus à Rennes ?",
    "Y a-t-il des concerts de jazz à Rennes ?",
    "Quels événements sont proposés pour les adolescents à Rennes ?",
    "Y a-t-il des marchés ou événements de rue à Rennes ?",
    "Quels événements liés à la gastronomie ou à la culture locale sont prévus à Rennes ?",
]

GROUND_TRUTHS = [
    "Plusieurs concerts sont prévus à Rennes avec des informations sur les dates, les lieux et les tarifs.",
    "Oui, il existe des expositions gratuites à Rennes dans des musées et centres d'art de la ville.",
    "Des activités pour enfants sont proposées à Rennes, notamment des ateliers, jeux et visites guidées.",
    "Des spectacles de danse sont organisés à Rennes dans différentes salles culturelles.",
    "De nombreux événements gratuits sont disponibles à Rennes dans différents lieux culturels.",
    "Des conférences et ateliers culturels sont organisés à Rennes.",
    "Des événements musicaux en plein air sont organisés à Rennes.",
    "Des spectacles de théâtre sont programmés à Rennes dans différentes salles.",
    "Des projections cinématographiques sont organisées à Rennes.",
    "Des festivals culturels ont lieu à Rennes sur différentes thématiques.",
    "Des visites guidées sont proposées à Rennes pour découvrir la ville et ses monuments.",
    "Des concerts de musique classique sont prévus à Rennes.",
    "Des événements familiaux sont proposés à Rennes, adaptés aux enfants et aux parents.",
    "Plusieurs musées de Rennes proposent des expositions temporaires et permanentes.",
    "Des événements gratuits pour les enfants sont proposés à Rennes.",
    "Des ateliers créatifs et artistiques sont proposés à Rennes pour différents publics.",
    "Des événements de danse contemporaine sont programmés à Rennes.",
    "Des lectures et rencontres avec des auteurs sont organisées à Rennes.",
    "Le Théâtre National de Bretagne accueille des spectacles et événements culturels à Rennes.",
    "Des concerts gratuits sont proposés à Rennes dans différents lieux.",
    "Des spectacles de cirque et d'arts de la rue sont programmés à Rennes.",
    "Des expositions d'art contemporain sont présentées dans les galeries et musées de Rennes.",
    "Les Champs Libres accueillent des événements culturels variés à Rennes.",
    "Des événements de musique électronique sont programmés à Rennes.",
    "Des opéras ou spectacles lyriques sont présentés à Rennes.",
    "Des soirées et événements nocturnes culturels sont proposés à Rennes.",
    "Des concerts de jazz sont programmés à Rennes.",
    "Des événements destinés aux adolescents sont proposés à Rennes.",
    "Des marchés et événements de rue sont organisés à Rennes.",
    "Des événements liés à la gastronomie et à la culture locale sont organisés à Rennes.",
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
