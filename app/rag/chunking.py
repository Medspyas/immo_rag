"""
Découpage d'un texte de loi nettoyé (data/clean/*.txt) en chunks alignés sur
les articles.

Analogie : plutôt que de découper le texte tous les 500 mots comme on
trancherait une baguette à intervalles réguliers (une fenêtre glissante
générique), on découpe le long des articulations naturelles du texte — un
article de loi = un chunk. Comme découper un livre de recettes recette par
recette plutôt que page par page : on ne coupe jamais une recette en deux.

Le nettoyage fait par scripts/clean_legifrance_pdf.py a déjà repéré les
articles abrogés et les a entourés de deux marqueurs (MARQUEUR_DEBUT_ABROGE /
MARQUEUR_FIN_ABROGE). Ce module s'appuie sur ces mêmes marqueurs pour
regrouper correctement un article abrogé avec son marqueur de début, qui se
trouve juste avant la ligne "Article X" dans le texte.
"""
from __future__ import annotations

import re

from pydantic import BaseModel, Field

# Identique à DEBUT_ARTICLE dans scripts/clean_legifrance_pdf.py : "Article 1",
# "Article L. 111-1", "Article 8-2-1", etc. Le groupe capture le numéro pour
# pouvoir citer précisément l'article dans une réponse.
DEBUT_ARTICLE = re.compile(r"^Article\s+([\w.\s-]+?)\s*$")

MARQUEUR_DEBUT_ABROGE = "=== ARTICLE ABROGÉ (n'est plus en vigueur — ne pas citer comme droit actuel) ==="
MARQUEUR_FIN_ABROGE = "=== FIN ARTICLE ABROGÉ ==="

# Un export Légifrance d'une "loi" (pas d'un code) se termine par l'historique
# parlementaire (débats, votes) — ni un article, ni du bruit d'en-tête/pied de
# page (donc pas retiré par clean_legifrance_pdf.py), mais pas non plus du
# droit à citer. Ce repère marque la fin du contenu normatif du document.
# Pas ancré en début de ligne : dans les PDF réels la mention apparaît parfois
# au milieu d'une ligne (ex. "Loi n° 65-557, TRAVAUX PREPARATOIRES"), et sans
# accent selon le document — d'où [ÉE] plutôt que juste "é".
DEBUT_TRAVAUX_PREPARATOIRES = re.compile(r"TRAVAUX PR[ÉE]PARATOIRES", re.IGNORECASE)


class RawChunk(BaseModel):
    """Un chunk avant contextualisation/embedding — juste le texte d'un article."""

    doc_id: str = Field(min_length=1)
    article: str = Field(min_length=1)
    content: str = Field(min_length=1)
    abroge: bool = False


def decouper_en_articles(texte: str, doc_id: str) -> list[RawChunk]:
    """Découpe `texte` (un fichier de data/clean/) en un RawChunk par article.

    Le texte avant le premier "Article X" (titre du document, bannière...) et
    après le dernier (travaux préparatoires) n'est pas rattaché à un article
    précis : il ne peut pas être cité comme source d'une réponse, donc on ne
    le garde pas comme chunk.
    """
    lignes = texte.split("\n")
    chunks: list[RawChunk] = []
    bloc: list[str] = []

    def cloturer_bloc() -> None:
        if not bloc:
            return
        contenu_brut = "\n".join(bloc).strip()
        if not contenu_brut:
            return
        premiere_ligne = contenu_brut.splitlines()[0].strip()
        correspondance = DEBUT_ARTICLE.match(premiere_ligne)
        if correspondance is None:
            # Pas un article (préambule avant le premier "Article X", ou
            # travaux préparatoires après le dernier) : pas de source citable.
            return
        abroge = MARQUEUR_FIN_ABROGE in contenu_brut
        contenu = (
            contenu_brut.replace(MARQUEUR_DEBUT_ABROGE, "")
            .replace(MARQUEUR_FIN_ABROGE, "")
            .strip()
        )
        chunks.append(
            RawChunk(
                doc_id=doc_id,
                article=correspondance.group(1).strip(),
                content=contenu,
                abroge=abroge,
            )
        )

    for ligne in lignes:
        ligne_nettoyee = ligne.strip()
        if ligne_nettoyee == MARQUEUR_DEBUT_ABROGE:
            # Précède "Article X" dans le texte nettoyé : clôt le bloc en
            # cours (l'article précédent), le marqueur ouvre le suivant.
            cloturer_bloc()
            bloc = [ligne]
        elif ligne_nettoyee == MARQUEUR_FIN_ABROGE:
            # Clôt tout de suite l'article abrogé, sans attendre le prochain
            # "Article X" — sinon bloc[0] resterait le marqueur de DÉBUT même
            # une fois rendu à l'article suivant, qui n'est pourtant pas abrogé.
            bloc.append(ligne)
            cloturer_bloc()
            bloc = []
        elif DEBUT_ARTICLE.match(ligne_nettoyee) or DEBUT_TRAVAUX_PREPARATOIRES.search(ligne_nettoyee):
            cloturer_bloc()
            bloc = [ligne]
        else:
            bloc.append(ligne)
    cloturer_bloc()

    return chunks
