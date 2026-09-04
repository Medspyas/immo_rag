"""
Interface commune pour transformer du texte en vecteurs (embeddings).

Analogie : un embedding donne des coordonnées à une phrase, mais dans un espace
de "sens" plutôt que géographique — deux phrases qui parlent de la même chose
(même avec des mots différents) auront des coordonnées proches. Toute la
recherche par similarité (retriever.py, vector_store.py) repose sur cette
propriété : chercher les voisins les plus proches d'une question revient à
chercher les chunks qui "veulent dire" la même chose.

Le projet reste agnostique au fournisseur (cf. CLAUDE.md, règles de travail) :
chaque fournisseur implémente `EmbeddingProvider`, et `get_embedding_provider`
choisit lequel utiliser selon `EMBEDDINGS_PROVIDER` dans .env.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from app.config import Settings


class EmbeddingProvider(ABC):
    """Contrat commun à tous les fournisseurs d'embeddings."""

    dimension: int

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embeddings pour des chunks à indexer (ingestion)."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embedding pour une question utilisateur (requête).

        Séparé de embed_documents car certains fournisseurs (dont Voyage)
        utilisent des instructions internes différentes selon qu'on encode un
        document ou une requête (input_type), ce qui améliore la pertinence.
        """


class LocalEmbeddings(EmbeddingProvider):
    """Modèle open-source exécuté localement (sentence-transformers) — gratuit
    et sans limite de tokens, au prix d'un calcul fait sur ta machine plutôt
    qu'une API payante. Bon défaut pour développer/tester avant de comparer
    avec un fournisseur payant via le checkpoint mini-ablation (cf. CLAUDE.md).

    Par défaut : `intfloat/multilingual-e5-base` — architecture transformers
    standard (pas de `trust_remote_code`, donc pas de code tiers exécuté sur
    la machine, contrairement à certains modèles récents comme gte-multilingual).
    Les modèles E5 attendent des préfixes "query: "/"passage: " sur le texte
    pour bien fonctionner (convention propre à cette famille de modèles) — à
    retirer si tu changes pour un modèle local qui n'utilise pas cette convention.

    Le modèle n'est chargé qu'au premier appel (`_get_model`), pas à la
    construction : instancier `LocalEmbeddings` reste instantané (utile pour
    les tests), le téléchargement/chargement ne se fait qu'à l'usage réel.
    """

    def __init__(self, model_name: str, dimension: int):
        self._model_name = model_name
        self.dimension = dimension
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name, truncate_dim=self.dimension)
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        prefixed = [f"passage: {t}" for t in texts]
        return self._get_model().encode(prefixed, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        prefixed = f"query: {text}"
        return self._get_model().encode([prefixed], normalize_embeddings=True).tolist()[0]


class VoyageEmbeddings(EmbeddingProvider):
    """Voyage AI (voyage-4 par défaut) — recommandé pour ce corpus (cf. CLAUDE.md).

    TODO : vérifier dans la doc Voyage actuelle le nom exact du paramètre de
    troncature Matryoshka (`output_dimension` au moment de l'écriture) avant le
    premier appel réel — l'API évolue vite sur ce point.
    """

    def __init__(self, api_key: str, model: str, dimension: int):
        import voyageai

        self._client = voyageai.Client(api_key=api_key)
        self._model = model
        self.dimension = dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        result = self._client.embed(
            texts,
            model=self._model,
            input_type="document",
            output_dimension=self.dimension,
        )
        return result.embeddings

    def embed_query(self, text: str) -> list[float]:
        result = self._client.embed(
            [text],
            model=self._model,
            input_type="query",
            output_dimension=self.dimension,
        )
        return result.embeddings[0]


class OpenAIEmbeddings(EmbeddingProvider):
    """OpenAI (text-embedding-3-small/large) — alternative si Voyage n'est pas retenu."""

    def __init__(self, api_key: str, model: str, dimension: int):
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key)
        self._model = model
        self.dimension = dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(
            input=texts, model=self._model, dimensions=self.dimension
        )
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def get_embedding_provider(settings: Settings) -> EmbeddingProvider:
    """Fabrique : choisit l'implémentation selon EMBEDDINGS_PROVIDER dans .env."""
    if settings.embeddings_provider == "local":
        return LocalEmbeddings(settings.embeddings_model, settings.embeddings_dimension)
    if settings.embeddings_provider == "voyage":
        if not settings.voyage_api_key:
            raise ValueError("VOYAGE_API_KEY manquant dans .env")
        return VoyageEmbeddings(
            settings.voyage_api_key, settings.embeddings_model, settings.embeddings_dimension
        )
    if settings.embeddings_provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY manquant dans .env")
        return OpenAIEmbeddings(
            settings.openai_api_key, settings.embeddings_model, settings.embeddings_dimension
        )
    raise ValueError(f"Fournisseur d'embeddings inconnu : {settings.embeddings_provider}")
