"""Découpage d'un texte de loi en morceaux ("chunks").

Un texte de loi nettoyé (data/clean/*.txt) est fourni en entrée ; une liste de
morceaux est produite en sortie, à raison d'un morceau par article. La coupure
suit les articles plutôt qu'un nombre de mots fixe, ce qui évite de séparer un
article en deux. Le contenu situé avant le premier article (titre) et après le
dernier (historique parlementaire) est écarté : il ne constitue pas une source
citable.

Le nettoyage en amont a entouré chaque article abrogé de deux marqueurs. Ce
module s'en sert pour écarter ces morceaux : un article abrogé n'est plus le
droit en vigueur (cf. le marqueur source lui-même, « ne pas citer comme droit
actuel ») et ne doit pas être indexé comme une source citable. C'est aussi ce
qui évite une collision d'identifiant : Légifrance liste parfois plusieurs
versions successives d'un même numéro d'article (l'ancienne abrogée, puis
l'actuelle) — sans ce filtre, les deux généreraient le même identifiant
(doc_id + numéro, cf. pipeline.py) et l'une écraserait l'autre en base, au
hasard de l'ordre du fichier source.

Contenu du fichier :
- RawChunk : un morceau avant traitement (texte d'un article et repères :
  document, numéro d'article, abrogé ou non).
- decouper_en_articles : réalise le découpage.
"""
from __future__ import annotations

import re

from pydantic import BaseModel, Field

# Détecte le début d'un article : "Article 1", "Article L. 111-1"...
# Le numéro est capturé afin de pouvoir citer l'article dans une réponse.
DEBUT_ARTICLE = re.compile(r"^Article\s+([\w.\s-]+?)\s*$")

MARQUEUR_DEBUT_ABROGE = "=== ARTICLE ABROGÉ (n'est plus en vigueur — ne pas citer comme droit actuel) ==="
MARQUEUR_FIN_ABROGE = "=== FIN ARTICLE ABROGÉ ==="

# Une loi se termine par l'historique parlementaire (débats, votes), qui ne se
# cite pas. Ce repère marque la fin du contenu utile. La recherche n'est pas
# ancrée en début de ligne (la mention se trouve parfois au milieu) et accepte
# É ou E (l'accent varie selon le document).
DEBUT_TRAVAUX_PREPARATOIRES = re.compile(r"TRAVAUX PR[ÉE]PARATOIRES", re.IGNORECASE)


class RawChunk(BaseModel):
    """Un morceau de texte avant traitement : le contenu d'un article.

    abroge vaut toujours False ici : les articles abrogés sont écartés avant
    la création du morceau (cf. docstring du module). Le champ reste présent
    pour la compatibilité du schéma avec la suite du pipeline (vector_store.py,
    generator.py), qui le lit depuis les métadonnées d'un morceau rangé.
    """

    doc_id: str = Field(min_length=1)
    article: str = Field(min_length=1)
    content: str = Field(min_length=1)
    abroge: bool = False


def decouper_en_articles(texte: str, doc_id: str) -> list[RawChunk]:
    """Découpe un texte de loi en une liste de morceaux, un par article.

    Le début (titre, bannière) et la fin (historique parlementaire) ne font
    partie d'aucun article et sont ignorés.
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
            # Ni un article (préambule ou historique) : aucun morceau n'est créé.
            return
        if MARQUEUR_FIN_ABROGE in contenu_brut:
            # Article abrogé : écarté, pas indexé (cf. docstring du module).
            return
        chunks.append(
            RawChunk(
                doc_id=doc_id,
                article=correspondance.group(1).strip(),
                content=contenu_brut,
            )
        )

    for ligne in lignes:
        ligne_nettoyee = ligne.strip()
        if ligne_nettoyee == MARQUEUR_DEBUT_ABROGE:
            # Ce marqueur précède un article : le bloc précédent est clôturé.
            cloturer_bloc()
            bloc = [ligne]
        elif ligne_nettoyee == MARQUEUR_FIN_ABROGE:
            # L'article abrogé est clôturé immédiatement. Attendre l'article
            # suivant lui appliquerait à tort l'étiquette "abrogé".
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
