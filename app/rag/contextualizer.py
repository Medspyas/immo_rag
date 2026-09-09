"""Ajout d'une phrase de contexte devant chaque morceau, avant l'embedding.

Un article isolé perd ses repères : une formule comme "le présent article"
n'indique plus de quelle loi ni de quel chapitre il s'agit, ce qui dégrade la
recherche par le sens. Un modèle de langage, qui reçoit le document entier,
rédige une ou deux phrases de situation ("Cet extrait vient de la loi du
10 juillet 1965 sur la copropriété, article 6.") ; ces phrases sont placées
devant le morceau avant le calcul de l'embedding.

Le document entier est renvoyé à l'identique à chaque appel. Ce texte répété
est mis en cache par Groq, ce qui limite le volume traité.

Le fournisseur passe par le contrat ContextGenerator et se règle dans .env.
Seule l'offre gratuite de Groq est proposée.

Contenu du fichier :
- ContextGenerator : contrat commun aux fournisseurs de modèle.
- GroqContextGenerator : implémentation Groq (gratuite, par défaut).
- get_context_generator : sélectionne l'implémentation selon .env.
- contextualiser_chunks : applique le contexte à une liste de morceaux.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from app.config import Settings
from app.rag.chunking import RawChunk

PROMPT_CONTEXTE = """Voici un document juridique complet :
<document>
{document}
</document>

Voici un extrait précis de ce document, qu'on souhaite situer dans l'ensemble :
<extrait>
{chunk}
</extrait>

Rédige une à deux phrases courtes, en français, qui situent cet extrait dans le document (de quel texte il vient, quel chapitre/titre le cas échéant). Réponds uniquement avec ces phrases, rien d'autre."""


class ContextGenerator(ABC):
    """Contrat commun aux fournisseurs de modèle de langage."""

    @abstractmethod
    def generer_contexte(self, document: str, chunk: str) -> str:
        """Renvoie une ou deux phrases situant l'extrait dans le document."""


class GroqContextGenerator(ContextGenerator):
    """Implémentation Groq : gratuite, retenue par défaut."""

    def __init__(self, api_key: str, model: str):
        from groq import Groq

        self._client = Groq(api_key=api_key)
        self._model = model

    def generer_contexte(self, document: str, chunk: str) -> str:
        reponse = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {
                    "role": "user",
                    "content": PROMPT_CONTEXTE.format(document=document, chunk=chunk),
                }
            ],
            temperature=0.0,
            max_tokens=150,
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
    """Ajoute sa phrase de contexte devant chaque morceau.

    Renvoie une nouvelle liste ; les morceaux d'origine ne sont pas modifiés.
    """
    resultat: list[RawChunk] = []
    for chunk in chunks:
        contexte = generator.generer_contexte(document_complet, chunk.content)
        contenu_enrichi = f"{contexte}\n\n{chunk.content}"
        resultat.append(chunk.model_copy(update={"content": contenu_enrichi}))
    return resultat
