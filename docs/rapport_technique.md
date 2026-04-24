# Rapport technique — Assistant intelligent de recommandation d'événements culturels

---

## 1. Objectifs du projet

**Contexte**  
Puls-Events souhaite proposer à ses utilisateurs un assistant capable de répondre à des questions sur l'agenda culturel local. La solution doit interroger des données en temps réel et fournir des réponses précises, sourcées et contextualisées.

**Problématique**  
Un LLM seul ne connaît pas les événements locaux et récents — ses données d'entraînement sont figées dans le temps. Un système RAG (Retrieval Augmented Generation) résout ce problème en injectant dynamiquement des documents pertinents dans le contexte du modèle avant chaque génération, sans nécessiter de ré-entraînement.

**Objectif du POC**  
Démontrer la faisabilité technique d'un assistant culturel RAG : collecter des données événementielles réelles, les indexer, exposer le pipeline via une API REST, et évaluer automatiquement la qualité des réponses.

**Périmètre**  
- Zone géographique : ville de Rennes et sa métropole
- Période : événements des 12 derniers mois (glissant)
- Source : API publique OpenDataSoft — dataset OpenAgenda (~5 000 événements)

---

## 2. Architecture du système

**Schéma global**

```
┌─────────────────────────────────────────────────────────┐
│                    Données entrantes                     │
│         API OpenDataSoft (dataset OpenAgenda)            │
└─────────────────────┬───────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────┐
│              Prétraitement (fetch_events.py)             │
│  - Filtrage Rennes + fenêtre 12 mois                    │
│  - Suppression balises HTML (BeautifulSoup)             │
│  - Construction du corpus textuel                        │
└─────────────────────┬───────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────┐
│           Embedding & Indexation (build_index.py)        │
│  - 1 Document LangChain par événement                   │
│  - Embedding : mistral-embed (1024 dimensions)          │
│  - Index FAISS via LangChain (save_local)               │
└─────────────────────┬───────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────┐
│           Pipeline RAG LangChain (rag_pipeline.py)       │
│  1. Retrieval  — similarity_search FAISS (k=5)          │
│  2. Augmentation — prompt contextualisé                 │
│  3. Génération — mistral-small-latest                   │
└─────────────────────┬───────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────┐
│                API REST (FastAPI)                        │
│         /health   /ask   /rebuild                        │
└─────────────────────────────────────────────────────────┘
```

**Technologies utilisées**

| Composant | Technologie | Version |
|---|---|---|
| Orchestration RAG | LangChain | 0.1.20 |
| LLM & Embedding | Mistral AI API | mistralai 0.1.8 |
| Base vectorielle | FAISS (via LangChain) | faiss-cpu 1.8.0 |
| API REST | FastAPI + Uvicorn | 0.111.0 / 0.29.0 |
| Évaluation | RAGAS | 0.1.9 |
| Conteneurisation | Docker | — |
| Données | Pandas | 2.2.2 |

---

## 3. Préparation et vectorisation des données

**Source de données**  
API publique OpenDataSoft — dataset `evenements-publics-openagenda` :
- URL : `https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/records`
- Filtre ODSQL : `location_city="Rennes" AND lastdate_end >= "{date_limite_12_mois}"`
- Pagination par blocs de 100 enregistrements
- Champs collectés : uid, title_fr, longdescription_fr, conditions_fr, firstdate_begin, lastdate_end, location_name, location_address, canonicalurl

**Nettoyage**  
- Suppression des lignes sans titre ni description (`dropna`)
- Nettoyage des balises HTML dans `longdescription_fr` via BeautifulSoup
- Filtrage des descriptions vides après nettoyage HTML (descriptions composées uniquement de balises)
- Remplissage des tarifs manquants par une chaîne vide

**Chunking**  
Aucun découpage appliqué — **1 événement = 1 Document LangChain**.

Justification : les descriptions d'événements sont courtes (200–800 caractères en moyenne). Chaque événement est une unité sémantique atomique — le découper fragmenterait le sens sans apporter de bénéfice. Avec k=5 événements récupérés, le contexte total (~5 000 caractères) est très loin des limites du modèle (32k tokens). Des tests avec chunking ont dégradé la qualité du retrieval.

**Embedding**
- Modèle : `mistral-embed` (API Mistral AI)
- Dimensionnalité des vecteurs : 1024
- Batch size : 50 documents (limite conservative pour respecter le rate limiting Mistral)
- Pause d'1 seconde entre chaque batch
- Format : vecteurs float32 stockés dans FAISS

---

## 4. Choix du modèle NLP

**Modèle sélectionné**  
- Embedding : `mistral-embed`
- Génération : `mistral-small-latest`

**Pourquoi Mistral ?**  
- Modèle francophone performant, adapté aux descriptions d'événements culturels en français
- Cohérence : utiliser le même fournisseur pour l'embedding et la génération garantit l'alignement sémantique des espaces vectoriels
- Rapidité : mistral-small offre des temps de réponse plus faibles que les modèles plus puissants, adapté à un usage interactif
- Compatibilité native avec LangChain via `langchain-mistralai`

**Prompt**
```
Tu es un assistant spécialisé dans les événements culturels à Rennes.
Réponds à la question en te basant uniquement sur les événements fournis ci-dessous.
Si l'information n'est pas dans le contexte, dis-le clairement.

ÉVÉNEMENTS :
{contexte}

QUESTION : {question}

RÉPONSE :
```

La contrainte "uniquement sur les événements fournis" ancre la génération dans les données récupérées et limite les hallucinations.

**Limites**
- `mistral-small` peut halluciner sur des événements dont la description est très courte ou ne contient qu'une URL
- Les données OpenAgenda sont hétérogènes en qualité (certaines descriptions sont minimalistes)
- Pas de gestion du multilinguisme (certains événements ont des descriptions partiellement en anglais)

---

## 5. Construction de la base vectorielle

**FAISS via LangChain**  
Utilisation de `langchain_community.vectorstores.FAISS` qui encapsule l'index FAISS brut et gère le stockage des métadonnées associées à chaque vecteur.

Construction :
```python
faiss_store = FAISS.from_documents(batch, embed_model)  # premier batch
faiss_store.add_documents(batch)                          # batchs suivants
```

Recherche :
```python
docs = faiss_store.similarity_search(question, k=5)
# retourne les 5 Documents LangChain les plus proches sémantiquement
```

**Stratégie de persistance**
- Format : deux fichiers binaires — `index.faiss` (vecteurs) et `index.pkl` (métadonnées + mapping)
- Sauvegarde : `faiss_store.save_local("data/faiss_index")`
- Chargement : `FAISS.load_local("data/faiss_index", embed_model, allow_dangerous_deserialization=True)`
- L'index est versionné dans le dépôt Git (~20 Mo) pour éviter de reconstruire en CI

**Métadonnées associées à chaque document**

| Champ | Description |
|---|---|
| uid | Identifiant unique de l'événement |
| title | Titre de l'événement |
| conditions | Tarif (gratuit, payant, sur inscription…) |
| date_start | Date de début |
| date_end | Date de fin |
| location | Nom et adresse du lieu |
| url | Lien vers la page OpenAgenda |

---

## 6. API et endpoints exposés

**Framework** : FastAPI 0.111.0 + Uvicorn 0.29.0  
Documentation interactive : `http://localhost:8000/docs` (Swagger UI, générée automatiquement)

**Endpoints**

| Méthode | Route | Description |
|---|---|---|
| `GET` | `/health` | État du système, nombre de vecteurs indexés |
| `POST` | `/ask` | Question utilisateur → réponse RAG |
| `POST` | `/rebuild` | Reconstruit l'index depuis les données fraîches |
| `GET` | `/` | Redirige vers `/docs` |

**Format des requêtes/réponses**

`POST /ask`
```json
// Requête
{ "question": "Quels concerts sont prévus à Rennes ?" }

// Réponse
{
  "answer": "Voici les concerts prévus à Rennes...",
  "contexts": ["Titre : Concert Jelias\nDescription : ..."]
}
```

`GET /health`
```json
{ "status": "ok", "vectors": 5028 }
```

**Exemple d'appel**
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Y a-t-il des expositions gratuites à Rennes ?"}'
```

**Tests effectués**  
7 tests fonctionnels via `pytest` + `starlette.TestClient` (couverture 95%) :
- `/health` — statut et présence de vecteurs
- `/` — redirection vers `/docs`
- `/ask` avec question valide — réponse et contextes non vides
- `/ask` avec question vide, whitespace, champ manquant — erreur 422

**Gestion des erreurs**
- Question vide ou whitespace : validation Pydantic → HTTP 422
- Champ manquant dans le body : HTTP 422

---

## 7. Évaluation du système

**Jeu de test annoté**  
7 questions représentatives des cas d'usage cibles, avec réponses de référence (`ground_truth`) rédigées manuellement :

| # | Question |
|---|---|
| 1 | Quels concerts sont prévus à Rennes ? |
| 2 | Y a-t-il des expositions gratuites à Rennes ? |
| 3 | Quels événements sont proposés pour les enfants à Rennes ? |
| 4 | Y a-t-il des spectacles de danse à Rennes ? |
| 5 | Quels événements gratuits peut-on trouver à Rennes ? |
| 6 | Y a-t-il des conférences ou ateliers culturels à Rennes ? |
| 7 | Y a-t-il des événements musicaux en plein air à Rennes ? |

**Métriques RAGAS (LLM-as-judge avec Mistral)**

| Métrique | Description |
|---|---|
| `faithfulness` | La réponse est-elle fidèle aux documents récupérés ? |
| `answer_relevancy` | La réponse répond-elle bien à la question ? |
| `context_precision` | Les documents récupérés sont-ils tous pertinents ? |
| `context_recall` | Les bons documents ont-ils été retrouvés ? |

**Résultats obtenus**

| Métrique | Score |
|---|---|
| faithfulness | **0.91** |
| answer_relevancy | **0.87** |
| context_precision | **0.81** |
| context_recall | **1.00** |

**Analyse quantitative**  
`context_recall` à 1.00 confirme que FAISS retrouve systématiquement les bons documents. `faithfulness` à 0.91 indique que Mistral reste ancré dans les données fournies avec peu d'hallucinations. `context_precision` à 0.81 révèle que parmi les 5 documents récupérés, certains sont occasionnellement hors sujet.

**Analyse qualitative**  
Deux anomalies ont été investiguées manuellement :
- `context_precision = 0.00` sur la question "enfants" : faux négatif RAGAS — le retrieval était parfait mais le `ground_truth` trop générique a induit le LLM-as-judge en erreur
- `faithfulness = 0.50` sur la question "plein air" : faux négatif RAGAS — tous les détails fournis par le modèle étaient présents dans les contextes, le LLM-as-judge a mal évalué sur un document très dense

---

## 8. Recommandations et perspectives

**Ce qui fonctionne bien**
- Le retrieval sémantique FAISS est fiable (`context_recall = 1.00`)
- Mistral reste fidèle aux documents fournis (`faithfulness = 0.91`)
- Le pipeline complet (collecte → indexation → API) est automatisé et reproductible
- L'évaluation RAGAS est intégrée en CI pour détecter les régressions

**Limites du POC**
- L'index est statique : les nouveaux événements nécessitent un rebuild manuel via `/rebuild`
- Pas de filtrage par date dans le retrieval — le modèle peut retourner des événements passés
- Qualité hétérogène des descriptions OpenAgenda (certaines ne contiennent qu'une URL)
- `mistral-small` comme LLM-as-judge RAGAS produit des faux négatifs sur des contextes très longs

**Améliorations possibles**
- Ajout d'un filtre temporel dans le retrieval (métadonnées `date_start` / `date_end`)
- Planification automatique du rebuild (cron job hebdomadaire)
- Passage à `mistral-large` comme LLM-as-judge pour une évaluation plus fiable
- Passage en production via un déploiement Docker sur un VPS ou un service cloud (Railway, Render…)

---

## 9. Organisation du dépôt GitHub

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
├── notebooks/
│   ├── 01_fetch_events.ipynb
│   ├── 02_process_events.ipynb
│   ├── 03_build_index.ipynb
│   ├── 04_rag_pipeline.ipynb
│   └── 05_evaluate_rag.ipynb
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

---

## 10. Annexes

### Prompt système
```
Tu es un assistant spécialisé dans les événements culturels à Rennes.
Réponds à la question en te basant uniquement sur les événements fournis ci-dessous.
Si l'information n'est pas dans le contexte, dis-le clairement.

ÉVÉNEMENTS :
Titre : {titre}
Description : {description}
Tarif : {conditions}
Dates : {date_start} → {date_end}
Lieu : {location}
Lien : {url}

QUESTION : {question}

RÉPONSE :
```

### Exemple de réponse JSON `/ask`
```json
{
  "answer": "Voici les concerts prévus à Rennes selon les informations fournies :\n\n1. **Le concert du mardi - Élisa chante Jeanne**\n   - **Date** : 22 juillet 2025 (16h-18h)\n   - **Lieu** : À proximité de la place Eugène Aulnette (1er set à 18h devant la Maison du Parc, 2e set à 19h place Eugène Aulnette)\n   - **Tarif** : Entrée libre\n   - **Description** : Concert swing en hommage à Jeanne Moreau, avec des chansons de Serge Rezvani, Norge et Jeanne Moreau.\n\n2. **Concert Jelias**\n   - **Date** : 7 novembre 2025 (17h30-18h15)\n   - **Lieu** : Hors les murs, Les Champs Libres (10 Cours des Alliés)\n   - **Tarif** : Gratuit\n   - **Description** : Groupe pop/rock/jazz/funk de la ville jumelle Erlangen (Allemagne), interprétant les chansons de Julius.\n\n3. **Le Grand Concert de Noël**\n   - **Date** : 14 décembre 2025 (18h-19h)\n   - **Lieu** : Salle polyvalente, Association du Bourg L'Evêque (16 rue Papu)\n   - **Tarif** : Entrée libre\n   - **Description** : Concert de 100 choristes.\n\n4. **Le concert du mardi - Arôme(s)**\n   - **Date** : 8 juillet 2025 (16h-17h30)\n   - **Lieu** : Pump track du parc de Quincé (rue Aurélie Nemours)\n   - **Tarif** : Entrée libre\n   - **Description** : DJ-set de Franco Nery et Pierre Normand mêlant musiques électroniques africaines, house, afrobeat, funk, jazz et hip hop.\n\n5. **Keren Ann en concert**\n   - **Date** : 7 mai 2026 (18h30-20h)\n   - **Lieu** : Théâtre National de Bretagne (1 rue Saint-Hélier)\n   - **Tarif** : Non précisé\n   - **Description** : Concert de Keren Ann dans le cadre de sa tournée \"Paris Amour\".",
  "contexts": [
    "Titre : Le concert du mardi - 𝗘́𝗹𝗶𝘀𝗮 𝗰𝗵𝗮𝗻𝘁𝗲 𝗝𝗲𝗮𝗻𝗻𝗲\nDescription : Élisa Chante Jeanne “Élisa Chante Jeanne” est un quartet swing en hommage à Jeanne Moreau. Des chansons écrites par Serge Rezvani, alias Cyrus Bassiak, le poète Norge et Jeanne Moreau elle-même. C'est avec beaucoup d'émotion que vous baladeront les chansons écrites par ou pour Jeanne et chantées par Élisa. Vous connaissez sans doute les célèbres Le Tourbillon De La Vie et J’ai La Mémoire Qui Flanche , qui sauront vous faire chanter en chœur. Vous découvrirez aussi des pépites cachées de la chanson française. Distribution : Elisa Robin - Chant Olivier Roth - Guitare Bérenger Heurtaux - Guitare Lyn Aubert - Contrebasse 1er set : 18h devant la Maison du Parc (av. André Malraux, Rennes) 2e set : 19h place Eugène Aulnette\nTarif : Entrée libre\nDate de début : 2025-07-22T16:00:00+00:00\nDate de fin : 2025-07-22T18:00:00+00:00\nLieu : À proximité de la place Eugène Aulnette, Place Eugène Aulnette, 35000, RENNES\nLien : https://openagenda.com/ete-rennes/events/le-concert-du-mardi-9337325",
    "Titre : Concert Jelias\nDescription : Basé à Erlangen, ville jumelée avec Rennes, Jelias est un tout jeune groupe de quatre musiciens navigant entre pop, rock, jazz et funk. Le groupe interprète les chansons du chanteur et auteur-compositeur Julius. Il a remporté le prix du public au festival des nouveaux talents de l'E-Werk d'Erlangen. Dans le cadre du jumelage entre Rennes et Erlangen (Allemagne).\nTarif : Gratuit\nDate de début : 2025-11-07T17:30:00+00:00\nDate de fin : 2025-11-07T18:15:00+00:00\nLieu : Hors les murs, Les Champs Libres - 10 Cours des Alliés  - 35000 RENNES\nLien : https://openagenda.com/sortir-rennesmetropole/events/concert-jelias",
    "Titre : Le Grand Concert de Noël\nDescription : Le Grand Concert de Noël - 100 choristes sur scène ! 14 décembre 2025 à 19h00 Salle polyvalente Entrée libre\nTarif : \nDate de début : 2025-12-14T18:00:00+00:00\nDate de fin : 2025-12-14T19:00:00+00:00\nLieu : Association du Bourg L'Evêque, 16 rue papu 35000 RENNES\nLien : https://openagenda.com/sortir-rennesmetropole/events/le-grand-concert-de-noel",
    "Titre : Le concert du mardi - 𝗔𝗿𝗼̂𝗺𝗲(𝘀)\nDescription : Le concert du mardi - 𝗔𝗿𝗼𝗺𝗲(𝘀) Pour ce nouveau projet, Franco Nery s’associe au percussionniste Pierre Normand pour proposer un DJ-set de 1h30. Les arômes des musiques électroniques africaines, de la house, de l’afrobeat, du funk, du jazz ou encore du hip hop se réunissent pour reproduire la saveur dansante du Groove. Celles et ceux qui voudront goûter à ce flow sont invité·es à se laisser guider par le rythme des instruments et des boucles du set ! Arôme[s] est avant tout un moment de partage où chacun·e peut ajouter son grain de sel à la danse. Distribution : Franco : DJ / MC Pierre Normand : percussions\nTarif : Entrée libre\nDate de début : 2025-07-08T16:00:00+00:00\nDate de fin : 2025-07-08T17:30:00+00:00\nLieu : Pump track du parc de Quincé, rue Aurélie Nemours, 35000, RENNES\nLien : https://openagenda.com/ete-rennes/events/le-concert-du-mardi",
    "Titre : KEREN ANN EN CONCERT\nDescription : L’auteure, compositrice, interprète, Keren Ann revient au TNB dans le cadre de sa tournée \"Paris Amour\", son 10e album studio porté par son écriture lumineuse et poétique et ses arrangements rock et éclectiques. Artiste associée, Keren Ann revient régulièrement au TNB, en solo ou accompagnée par les musiciens de l’ONB pour une version inédite et orchestrale de son album \"Bleue\" en 2019, avec le Quatuor Debussy en 2022, ou encore à l’Opéra de Rennes pour la création de \"Red Waters\", opéra composé avec Bardi Jóhannsson avec lequel elle forme le duo Lady & Bird.\nTarif : \nDate de début : 2026-05-07T18:30:00+00:00\nDate de fin : 2026-05-07T20:00:00+00:00\nLieu : Théâtre National de Bretagne, 1, rue saint-hélier\nLien : https://openagenda.com/sortir-rennesmetropole/events/keren-ann-en-concert"
  ]
}
```