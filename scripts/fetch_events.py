"""
Collecte et preprocessing des événements culturels à Rennes.

Interroge l'API publique OpenDataSoft (dataset OpenAgenda) sans clé API,
filtre les événements sur la ville de Rennes et la fenêtre des 12 derniers mois,
nettoie les données (HTML, valeurs nulles) et construit le corpus textuel.

Résultat : data/processed/events_clean.csv — jeu de données structuré,
prêt à être indexé par build_index.py.
"""

import os
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
from bs4 import BeautifulSoup

BASE_URL = "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/records"
FIELDS = "uid,title_fr,longdescription_fr,conditions_fr,firstdate_begin,lastdate_end,location_name,location_address,canonicalurl"
RAW_PATH = "data/raw/events.csv"
PROCESSED_PATH = "data/processed/events_clean.csv"


def fetch_events() -> list[dict]:
    """Récupère tous les événements de Rennes des 12 derniers mois via pagination."""
    date_limit = (datetime.now(timezone.utc) - timedelta(days=365)).strftime("%Y-%m-%dT%H:%M:%SZ")
    where = f'location_city="Rennes" AND lastdate_end >= "{date_limit}"'

    total = requests.get(BASE_URL, params={"where": where, "limit": 1}).json()["total_count"]

    records = []
    offset = 0
    while offset < total:
        response = requests.get(BASE_URL, params={
            "where": where,
            "select": FIELDS,
            "limit": 100,
            "offset": offset,
        })
        response.raise_for_status()
        records.extend(response.json()["results"])
        offset += 100
        print(f"{len(records)}/{total}", end="\r")

    print(f"\n{len(records)} événements récupérés")
    return records


def strip_html(text: str) -> str:
    """Supprime les balises HTML et retourne le texte brut."""
    return BeautifulSoup(text, "html.parser").get_text(separator=" ", strip=True)


def build_corpus(row: pd.Series) -> str:
    """Assemble les champs d'un événement en un texte structuré pour l'embedding."""
    return (
        f"Titre : {row['title_fr']}\n"
        f"Description : {row['longdescription_fr']}\n"
        f"Tarif : {row['conditions_fr']}\n"
        f"Date de début : {row['firstdate_begin']}\n"
        f"Date de fin : {row['lastdate_end']}\n"
        f"Lieu : {row['location_name']}, {row['location_address']}\n"
        f"Lien : {row['canonicalurl']}"
    )


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie le DataFrame et construit la colonne corpus."""
    df = df.dropna(subset=["longdescription_fr", "title_fr"]).copy()
    df["conditions_fr"] = df["conditions_fr"].fillna("")
    df["longdescription_fr"] = df["longdescription_fr"].apply(strip_html)
    # filtre les descriptions vides après nettoyage HTML (balises seules → chaîne vide)
    df = df[df["longdescription_fr"].str.strip() != ""]
    df["corpus"] = df.apply(build_corpus, axis=1)
    return df.reset_index(drop=True)


def main():
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    records = fetch_events()
    df_raw = pd.DataFrame(records)
    df_raw.to_csv(RAW_PATH, index=False, encoding="utf-8")

    df_clean = preprocess(df_raw)
    df_clean.to_csv(PROCESSED_PATH, index=False, encoding="utf-8")
    print(f"{len(df_clean)} événements sauvegardés dans {PROCESSED_PATH}")


if __name__ == "__main__":
    main()
