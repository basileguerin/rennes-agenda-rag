# Rennes Agenda RAG

Assistant culturel pour les événements de Rennes, propulsé par un pipeline RAG (Retrieval Augmented Generation) avec Mistral AI et FAISS.

## Architecture

```
Données OpenAgenda (API publique)
        ↓
[Collecte & Nettoyage]   → scripts/fetch_events.py
        ↓
[Embedding & Indexation] → scripts/build_index.py  (mistral-embed + FAISS)
        ↓
[Pipeline RAG]           → src/rag_pipeline.py     (LangChain + Mistral)
        ↓
[API REST]               → api/main.py             (FastAPI)
```

## Structure du projet

```
rennes-agenda-rag/
├── api/
│   └── main.py              # Endpoints FastAPI (/health, /ask, /rebuild)
├── scripts/
│   ├── fetch_events.py      # Collecte et nettoyage des données OpenAgenda
│   └── build_index.py       # Construction de l'index FAISS
├── src/
│   └── rag_pipeline.py      # Pipeline RAG (ask, reload_index)
├── tests/
│   ├── api_test.py          # Tests fonctionnels de l'API
│   └── evaluate_rag.py      # Évaluation RAGAS automatisée
├── notebooks/               # Exploration et expérimentations
├── data/
│   └── faiss_index/         # Index FAISS versionné (index.faiss + index.pkl)
├── docs/
│   └── rapport_technique.md # Rapport technique
├── .github/workflows/
│   ├── api_test.yml         # CI : tests API à chaque push develop
│   └── evaluate_rag.yml     # CI : évaluation RAGAS à chaque push develop
├── .env.example             # Template des variables d'environnement
├── Dockerfile
├── requirements.txt
└── README.md
```

## Stack technique

- **LangChain** — orchestration du pipeline RAG
- **Mistral AI** — embedding (`mistral-embed`) + génération (`mistral-small-latest`)
- **FAISS** — base vectorielle en mémoire
- **FastAPI + Uvicorn** — API REST
- **RAGAS** — évaluation automatisée des réponses
- **Docker** — conteneurisation

## Prérequis

- Python 3.12.3
- Clé API Mistral (variable d'environnement `MISTRAL_API_KEY`)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # puis renseigner MISTRAL_API_KEY
```

## Lancer l'API

**En local**
```bash
uvicorn api.main:app --reload
```

**Avec Docker**
```bash
docker build -t rennes-agenda-rag .
docker run --name rennes-rag -p 8000:8000 -e MISTRAL_API_KEY=ta_cle rennes-agenda-rag
```

L'API est disponible sur `http://localhost:8000`.  
La documentation Swagger est accessible sur `http://localhost:8000/docs`.

## Endpoints

| Méthode | Route | Description |
|---|---|---|
| `GET` | `/health` | État du système et nombre de vecteurs indexés |
| `POST` | `/ask` | Pose une question, retourne la réponse et les contextes récupérés |
| `POST` | `/rebuild` | Reconstruit l'index FAISS depuis les données fraîches OpenAgenda |

**Exemple `/ask`**
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Quels concerts sont prévus à Rennes ?"}'
```

## Reconstruire l'index

```bash
python scripts/fetch_events.py   # collecte les événements (~5 000 événements)
python scripts/build_index.py    # génère les embeddings et sauvegarde l'index
```

## Évaluation RAGAS

```bash
python tests/evaluate_rag.py
```

Calcule 4 métriques sur un jeu de 30 questions annotées : `faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`. Le script sort avec le code 1 si un seuil n'est pas atteint.

| Métrique | Score obtenu | Seuil |
|---|---|---|
| faithfulness | 0.919 | 0.80 |
| answer_relevancy | 0.887 | 0.80 |
| context_precision | 0.788 | 0.75 |
| context_recall | 1.000 | 0.80 |

## Tests

```bash
pytest tests/api_test.py -v --cov=api --cov=src --cov-report=term-missing
```

## CI/CD

Deux workflows GitHub Actions se déclenchent à chaque push sur `develop` :
- **API Tests** — tests fonctionnels de l'API (hors rebuild)
- **Evaluate RAG** — évaluation RAGAS avec seuils automatiques
