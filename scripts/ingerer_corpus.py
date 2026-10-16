"""Ingestion réelle du corpus (data/clean/*.txt) dans pgvector.

Lit les textes déjà nettoyés (cf. clean_legifrance_pdf.py), les fait passer
par preparer_morceaux/ingerer_document (chunking.py, embeddings.py,
vector_store.py) et les range en base. Sans contextualisation par défaut
(aucun appel Groq) : ajouter --contextualiser pour l'activer (1 appel Groq
par document, cf. contextualizer.py).

Lancement :
    python scripts/ingerer_corpus.py loi_06_07_1989.txt
    python scripts/ingerer_corpus.py            # data/clean/ sauf TEXTES_NON_CONSOLIDES (pipeline.py)
    python scripts/ingerer_corpus.py --contextualiser loi_06_07_1989.txt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))

from app.config import Settings
from app.rag.contextualizer import get_context_generator
from app.rag.embeddings import get_embedding_provider
from app.rag.pipeline import fichiers_du_corpus, ingerer_document
from app.rag.vector_store import VectorStore

DOSSIER = RACINE / "data" / "clean"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fichiers", nargs="*", help="noms de fichiers dans data/clean/ (défaut : tous)")
    parser.add_argument("--contextualiser", action="store_true")
    args = parser.parse_args()

    noms = args.fichiers or fichiers_du_corpus(DOSSIER)

    settings = Settings()
    embeddings = get_embedding_provider(settings)
    vector_store = VectorStore(settings.database_url, settings.embeddings_dimension)
    contextualizer = get_context_generator(settings) if args.contextualiser else None

    for nom in noms:
        chemin = DOSSIER / nom
        doc_id = chemin.stem
        texte = chemin.read_text(encoding="utf-8")
        print(f"Ingestion de {nom}…", flush=True)
        rapport = ingerer_document(
            texte, doc_id, embeddings=embeddings, vector_store=vector_store,
            contextualizer=contextualizer,
        )
        print(
            f"  -> {rapport.articles_indexes} articles indexés "
            f"({rapport.articles_abroges} abrogés), contextualisé={rapport.contextualise}"
        )


if __name__ == "__main__":
    main()
