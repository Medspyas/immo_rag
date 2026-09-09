"""Transformation de texte en vecteurs ("embeddings").

Un embedding est une position attribuée à un texte dans un espace de sens :
deux formulations d'une même idée, même avec des mots différents, obtiennent
des positions proches. La recherche par similarité s'appuie entièrement sur
cette propriété.

Le fournisseur passe par le contrat EmbeddingProvider et se règle dans .env.
Seule une implémentation locale et gratuite est proposée ; un autre modèle
local se branche en changeant EMBEDDINGS_MODEL, sans nouvelle classe.

Contenu du fichier :
- EmbeddingProvider : contrat commun.
- LocalEmbeddings : modèle libre exécuté sur la machine, gratuit, retenu par défaut.
- get_embedding_provider : sélectionne l'implémentation selon .env.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from app.config import Settings


class EmbeddingProvider(ABC):
    """Contrat commun à tous les fournisseurs d'embeddings."""

    dimension: int

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Vecteurs pour des morceaux à ranger dans la base."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Vecteur pour une question.

        Séparé de embed_documents : certains fournisseurs traitent différemment
        une question et un texte à ranger, ce qui améliore la pertinence.
        """


class LocalEmbeddings(EmbeddingProvider):
    """Modèle libre exécuté sur la machine : gratuit et sans quota, au prix du
    calcul local. Implémentation retenue par défaut.

    Modèle retenu : intfloat/multilingual-e5-base. Cette famille attend un
    préfixe "query: " ou "passage: " devant le texte ; ce préfixe est à retirer
    pour un modèle qui ne suit pas cette convention.

    Le modèle n'est chargé qu'au premier appel : la création de l'objet reste
    instantanée et les tests n'ont pas à le charger.
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


def get_embedding_provider(settings: Settings) -> EmbeddingProvider:
    """Sélectionne l'implémentation selon le réglage EMBEDDINGS_PROVIDER de .env."""
    if settings.embeddings_provider == "local":
        return LocalEmbeddings(settings.embeddings_model, settings.embeddings_dimension)
    raise ValueError(f"Fournisseur d'embeddings inconnu : {settings.embeddings_provider}")
