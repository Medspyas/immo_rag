"""Développement des sigles dans la question, avant la recherche.

Les professionnels écrivent « IFI », « DPE » ou « AG » ; les textes de loi
écrivent « impôt sur la fortune immobilière », « diagnostic de performance
énergétique », « assemblée générale ». Ni les embeddings, ni l'index BM25, ni
le reclasseur ne font le lien : mesuré le 2026-10-13 sur l'article 964 du code
général des impôts, le reclasseur note 0,038 la question « … doit-on payer
l'IFI ? » et 0,979 la même question avec « l'impôt sur la fortune
immobilière ». Sous le seuil du garde-fou étage 1 (0,3), la première était
refusée alors que la base contient la réponse.

La forme développée est ajoutée entre parenthèses après le sigle, qui reste en
place : rien n'est retiré de la question, et rien n'est inventé (contrairement
à une reformulation par LLM, écartée le 2026-10-03, cf. retriever.py). Seuls
figurent ici des sigles sans ambiguïté dans le domaine, dont la forme
développée apparaît telle quelle dans le corpus.

Contenu du fichier :
- SIGLES : sigle → forme développée, telle qu'écrite dans les textes.
- developper_sigles : ajoute la forme développée après chaque sigle reconnu.
"""
from __future__ import annotations

import re

SIGLES = {
    "IFI": "impôt sur la fortune immobilière",
    "DPE": "diagnostic de performance énergétique",
    "VEFA": "vente en l'état futur d'achèvement",
    "SCI": "société civile immobilière",
    "AG": "assemblée générale",
    "PTZ": "prêt ne portant pas intérêt",
    "PLU": "plan local d'urbanisme",
    "TVA": "taxe sur la valeur ajoutée",
    "EDL": "état des lieux",
    "CSG": "contribution sociale généralisée",
}

# Sigle écrit en majuscules, en mot entier (« l'IFI », « AG. »), jamais à
# l'intérieur d'un mot (« AGENT »).
_SIGLE = re.compile(r"\b(" + "|".join(SIGLES) + r")\b")


def developper_sigles(question: str) -> str:
    """La question, chaque sigle reconnu suivi de sa forme développée.

    Un sigle déjà accompagné de sa forme développée dans la question n'est pas
    développé une seconde fois.
    """

    def remplacer(correspondance: re.Match) -> str:
        sigle = correspondance.group(1)
        forme = SIGLES[sigle]
        if forme.lower() in question.lower():
            return sigle
        return f"{sigle} ({forme})"

    return _SIGLE.sub(remplacer, question)
