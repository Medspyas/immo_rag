"""
Contextual retrieval : ajoute, devant chaque chunk, une courte phrase de
contexte générée par un LLM à partir du document entier, avant l'embedding.

Le problème que ça résout : un article de loi isolé perd parfois son contexte
("le présent article" — mais l'article de *quelle* loi, dans *quel*
chapitre ?). Une recherche par sens (embedding) sur un chunk trop nu peut donc
mal le retrouver. On demande à un LLM de lire le document complet et de
rédiger 1-2 phrases qui replacent le chunk dedans (ex. "Cet extrait provient
de la loi du 10 juillet 1965 sur le statut de la copropriété, chapitre I,
article 6."), collées devant le chunk avant de l'embedder.

Analogie : comme ajouter un post-it de contexte sur une fiche avant de la
ranger, pour qu'elle reste compréhensible même sortie du classeur.

Le document complet est envoyé identique à chaque appel (un par chunk) — ce
préfixe stable et répété est justement ce qui permet au prompt caching
automatique de Groq de s'activer (cf. .env.example, LLM_MODEL) : les tokens du
document ne sont traités en entier qu'une fois, puis réutilisés en cache pour
tous les chunks suivants du même document.
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
    """Contrat commun à tous les fournisseurs LLM utilisés pour le contextual retrieval."""

    @abstractmethod
    def generer_contexte(self, document: str, chunk: str) -> str:
        """Renvoie une courte phrase de contexte situant `chunk` dans `document`."""


class GroqContextGenerator(ContextGenerator):
    """Groq (gratuit) — cf. CLAUDE.md, contrainte budget : pas d'API payante par défaut."""

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


class AnthropicContextGenerator(ContextGenerator):
    """Anthropic (payant) — dispo en secours via .env, jamais le défaut (cf. CLAUDE.md)."""

    def __init__(self, api_key: str, model: str):
        import anthropic

        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def generer_contexte(self, document: str, chunk: str) -> str:
        reponse = self._client.messages.create(
            model=self._model,
            max_tokens=150,
            messages=[
                {
                    "role": "user",
                    "content": PROMPT_CONTEXTE.format(document=document, chunk=chunk),
                }
            ],
        )
        return reponse.content[0].text.strip()


def get_context_generator(settings: Settings) -> ContextGenerator:
    """Fabrique : choisit l'implémentation selon LLM_PROVIDER dans .env."""
    if settings.llm_provider == "groq":
        if not settings.groq_api_key:
            raise ValueError("GROQ_API_KEY manquant dans .env")
        return GroqContextGenerator(settings.groq_api_key, settings.llm_model)
    if settings.llm_provider == "anthropic":
        if not settings.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY manquant dans .env")
        return AnthropicContextGenerator(settings.anthropic_api_key, settings.llm_model)
    raise ValueError(f"Fournisseur LLM inconnu : {settings.llm_provider}")


def contextualiser_chunks(
    chunks: list[RawChunk], document_complet: str, generator: ContextGenerator
) -> list[RawChunk]:
    """Préfixe chaque chunk avec sa phrase de contexte générée par le LLM.

    Renvoie une nouvelle liste (les RawChunk d'origine, hors contenu, ne sont
    pas modifiés en place) : `content` devient "<contexte>\\n\\n<article>".
    """
    resultat: list[RawChunk] = []
    for chunk in chunks:
        contexte = generator.generer_contexte(document_complet, chunk.content)
        contenu_enrichi = f"{contexte}\n\n{chunk.content}"
        resultat.append(chunk.model_copy(update={"content": contenu_enrichi}))
    return resultat
