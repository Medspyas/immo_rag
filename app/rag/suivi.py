"""Questions de suivi : réécriture en question autonome avant la recherche.

Dans une conversation, « Et pour un meublé ? » ou « applique le calcul à
580 000 € » n'a de sens qu'avec l'échange précédent. Cherchée seule, une telle
question ne retrouve aucun article et l'assistant refuse. Un modèle de langage
la réécrit donc en question complète (« Quel est le préavis du locataire d'un
logement meublé ? »), qui suit ensuite le chemin habituel : recherche,
réponse citée, garde-fous.

La conversation ne sert qu'à comprendre la question, jamais de source : la
réponse reste fondée et vérifiée sur les seuls textes de loi retrouvés pour la
question réécrite. Une question déjà autonome est renvoyée telle quelle, et
sans historique aucun appel n'est fait. En cas d'échec (réponse illisible,
quota épuisé), la question d'origine est gardée : la conversation se dégrade
en questions indépendantes, comme avant, sans jamais bloquer la réponse.

Contenu du fichier :
- Echange : une question précédente et la réponse affichée.
- reformuler_question : la question rendue autonome.
"""
from __future__ import annotations

import json
import logging
from typing import Protocol

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Échanges transmis au modèle : les plus récents suffisent à résoudre « ce
# délai », « et pour… » ; au-delà, le coût en tokens croît sans gain.
ECHANGES_MAX = 3
# Une réponse précédente est tronquée : son début dit de quoi il était question.
REPONSE_MAX_CARACTERES = 800
# Une question réécrite plus longue que cela n'est plus une question.
QUESTION_MAX_CARACTERES = 1000

CONSIGNE_SUIVI = """Tu prépares une recherche dans des textes de loi français \
(immobilier, copropriété, baux, fiscalité immobilière).

On te donne les derniers échanges d'une conversation, puis une nouvelle question. \
Si la nouvelle question se comprend seule, sans la conversation, recopie-la telle \
quelle. Sinon, réécris-la en une seule question autonome :
- remplace chaque pronom ou renvoi (« le », « il », « ce délai », « cette taxe », \
« la première réponse », « le calcul ») par ce qu'il désigne, nommé en toutes \
lettres (« le syndic de copropriété », « la taxe sur les plus-values \
immobilières élevées ») ;
- n'ajoute jamais de numéro d'article ni de nom de loi ou de code ;
- reprends les faits et montants de l'utilisateur avec ses propres mots, sans \
les requalifier : un prix de vente reste un prix de vente (ce n'est pas une \
plus-value), un commerce reste un commerce (ce n'est pas un logement) ;
- supprime les formules de politesse et le tutoiement.
Une personne qui lit seulement ta question doit savoir exactement de quoi il s'agit.

Exemples :
- après une question sur les missions du syndic, « Qui le désigne ? » devient \
« Qui désigne le syndic de copropriété ? » ;
- après une réponse sur le préavis du locataire, « et si je pars dans \
2 semaines ? » devient « Un locataire peut-il quitter son logement 2 semaines \
après avoir donné congé, compte tenu du délai de préavis ? » ;
- « Dans quel délai le dépôt de garantie doit-il être restitué ? » se comprend \
seule : elle est recopiée telle quelle.

Ne réponds pas à la question. N'ajoute aucune information absente de la \
conversation. Écris en français.

Réponds UNIQUEMENT par ce JSON : {"question": "<la question autonome>"}"""


class Echange(BaseModel):
    """Une question précédente de la conversation et la réponse affichée (vide
    en cas de refus)."""

    question: str = Field(min_length=1, max_length=QUESTION_MAX_CARACTERES)
    reponse: str = Field(default="", max_length=20_000)


class _LLM(Protocol):
    def generer(self, consigne: str, question: str) -> str: ...


def _message(question: str, historique: list[Echange]) -> str:
    lignes = ["Conversation :"]
    for echange in historique[-ECHANGES_MAX:]:
        reponse = echange.reponse.strip() or "(pas de réponse : l'assistant a refusé)"
        if len(reponse) > REPONSE_MAX_CARACTERES:
            reponse = reponse[:REPONSE_MAX_CARACTERES] + " […]"
        lignes += [f"Utilisateur : {echange.question}", f"Assistant : {reponse}"]
    lignes += ["", f"Nouvelle question : {question}"]
    return "\n".join(lignes)


def reformuler_question(question: str, historique: list[Echange], llm: _LLM) -> str:
    """La question rendue autonome grâce aux derniers échanges ; la question
    d'origine sans historique ou si la réécriture échoue."""
    if not historique:
        return question
    try:
        brut = llm.generer(CONSIGNE_SUIVI, _message(question, historique))
        reecrite = json.loads(brut).get("question", "")
    except Exception:  # repli : la conversation ne doit jamais bloquer la réponse
        logger.warning("Réécriture de la question de suivi impossible", exc_info=True)
        return question
    if not isinstance(reecrite, str) or not reecrite.strip():
        return question
    reecrite = reecrite.strip()
    return reecrite if len(reecrite) <= QUESTION_MAX_CARACTERES else question
