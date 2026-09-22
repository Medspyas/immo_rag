"""Ajout d'une phrase de contexte devant chaque morceau, avant l'embedding.

Un article isolé perd ses repères : une formule comme "le présent article"
n'indique plus de quelle loi ni de quel chapitre il s'agit, ce qui dégrade la
recherche par le sens. Un modèle de langage rédige une ou deux phrases de
situation ("Cet extrait vient de la loi du 10 juillet 1965 sur la
copropriété.") ; ces phrases sont placées devant chaque morceau, suivies du
numéro de l'article, avant le calcul de l'embedding.

Un seul appel de modèle par DOCUMENT, pas par morceau : Groq limite le volume
de tokens par requête sur son offre gratuite (8000 tokens/minute pour le
modèle retenu), très en dessous de la taille des textes de loi complets (de
~15 000 à plus d'1 000 000 de tokens selon le document). Envoyer le document
entier à chaque morceau dépasse donc systématiquement cette limite — vérifié
en conditions réelles le 2026-09-18 (document de ~43 000 tokens, requête
refusée : « Request too large »). Le résumé, lui, ne dépend pas du morceau :
un seul appel par document suffit, réutilisé tel quel pour tous ses articles.
Seul le DÉBUT du document est transmis pour ce résumé (LONGUEUR_MAX_RESUME_
CARACTERES caractères) : le titre et l'objet d'une loi se lisent en général
dans ses premières lignes, et la limite de tokens interdit de toute façon
d'envoyer un document long en entier.

Le fournisseur passe par le contrat ContextGenerator et se règle dans .env.
Seule l'offre gratuite de Groq est proposée.

Contenu du fichier :
- ContextGenerator : contrat commun aux fournisseurs de modèle.
- GroqContextGenerator : implémentation Groq (gratuite, par défaut).
- get_context_generator : sélectionne l'implémentation selon .env.
- contextualiser_chunks : applique le résumé du document à ses morceaux.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from app.config import Settings
from app.rag.chunking import RawChunk

# Longueur, en caractères, du début de document transmis pour le résumé
# (~1/4 de ce nombre en tokens). Choisie pour rester largement sous la limite
# Groq du tier gratuit (8000 tokens/minute) une fois le gabarit du prompt et
# la réponse comptés.
LONGUEUR_MAX_RESUME_CARACTERES = 12_000

PROMPT_RESUME = """Voici le début d'un document juridique (le document est long, seul le début est fourni) :
<document>
{debut_document}
</document>

Rédige une à deux phrases courtes, en français, qui identifient ce document : de quel texte il s'agit (loi, code...) et son objet général. Réponds uniquement avec ces phrases, rien d'autre."""


class ContextGenerator(ABC):
    """Contrat commun aux fournisseurs de modèle de langage."""

    @abstractmethod
    def resumer_document(self, debut_document: str) -> str:
        """Renvoie une ou deux phrases identifiant le document."""


class GroqContextGenerator(ContextGenerator):
    """Implémentation Groq : gratuite, retenue par défaut."""

    def __init__(self, api_key: str, model: str):
        from groq import Groq

        self._client = Groq(api_key=api_key)
        self._model = model

    def resumer_document(self, debut_document: str) -> str:
        reponse = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {
                    "role": "user",
                    "content": PROMPT_RESUME.format(debut_document=debut_document),
                }
            ],
            temperature=0.0,
            max_tokens=400,
            # openai/gpt-oss-20b (modèle par défaut) raisonne avant de répondre ;
            # ce raisonnement consomme max_tokens en priorité. Sans ce réglage,
            # un raisonnement long épuise le budget avant d'atteindre la
            # réponse : contenu vide, finish_reason="length" — vérifié en
            # conditions réelles le 2026-09-18. "low" borne le raisonnement
            # pour une tâche aussi simple qu'un résumé en une phrase.
            reasoning_effort="low",
        )
        return reponse.choices[0].message.content.strip()


def get_context_generator(settings: Settings) -> ContextGenerator:
    """Sélectionne l'implémentation selon le réglage LLM_PROVIDER de .env."""
    if settings.llm_provider == "groq":
        if not settings.groq_api_key:
            raise ValueError("GROQ_API_KEY manquant dans .env")
        return GroqContextGenerator(settings.groq_api_key, settings.llm_model)
    raise ValueError(f"Fournisseur LLM inconnu : {settings.llm_provider}")


def contextualiser_chunks(
    chunks: list[RawChunk], document_complet: str, generator: ContextGenerator
) -> list[RawChunk]:
    """Ajoute le résumé du document et le numéro d'article devant chaque morceau.

    Un seul appel au modèle par document (resumer_document), quel que soit le
    nombre de morceaux : le résumé ne dépend pas du morceau et est réutilisé
    tel quel pour chacun. Renvoie une nouvelle liste ; les morceaux d'origine
    ne sont pas modifiés.
    """
    if not chunks:
        return []

    resume = generator.resumer_document(document_complet[:LONGUEUR_MAX_RESUME_CARACTERES])

    resultat: list[RawChunk] = []
    for chunk in chunks:
        contenu_enrichi = f"{resume} Article {chunk.article}.\n\n{chunk.content}"
        resultat.append(chunk.model_copy(update={"content": contenu_enrichi}))
    return resultat
