"""Reclassement des résultats de recherche par un modèle de pertinence.

La recherche hybride (retriever.py) présélectionne un lot de morceaux à partir
de deux classements fusionnés. Cette fusion ne compare que des rangs : aucun
modèle n'y relit la question et le texte ensemble.

Le reclassement comble ce manque. Un modèle dit « cross-encoder » reçoit la
paire (question, morceau) et produit un score de pertinence après avoir traité
les deux textes conjointement. L'opération est coûteuse — un passage de modèle
par morceau — donc réservée aux quelques candidats déjà retenus par la
recherche hybride, jamais appliquée au corpus entier.

Le fournisseur passe par le contrat RerankProvider et se règle dans .env, comme
pour les embeddings. Seules des implémentations gratuites sont proposées :
LocalReranker couvre le besoin ; une autre implémentation locale se brancherait
en changeant RERANK_MODEL, sans nouvelle classe.

Contenu du fichier :
- RerankProvider : contrat commun.
- LocalReranker : modèle libre exécuté sur la machine (BGE-m3), gratuit, retenu par défaut.
- get_rerank_provider : sélectionne l'implémentation selon .env.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod

from app.config import Settings


def _sigmoide(x: float) -> float:
    """Ramène un score sans borne dans l'intervalle 0–1.

    Les cross-encoders produisent un score brut (logit) dont seule la valeur
    relative compte pour le classement. La sigmoïde le convertit en une valeur
    entre 0 et 1, interprétable comme un seuil pour le garde-fou de retrieval.
    La forme utilisée ici évite un dépassement de capacité sur les valeurs très
    négatives.
    """
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


class RerankProvider(ABC):
    """Contrat commun à tous les fournisseurs de reclassement."""

    @abstractmethod
    def rerank(self, query: str, documents: list[str]) -> list[tuple[int, float]]:
        """Réordonne des morceaux selon leur pertinence pour la question.

        documents est la liste des textes à reclasser. La valeur renvoyée est
        une liste de couples (indice d'origine, score), triée du plus au moins
        pertinent ; l'indice renvoie à la position du texte dans documents. Le
        score est ramené dans l'intervalle 0–1.
        """


class LocalReranker(RerankProvider):
    """Modèle libre exécuté sur la machine : gratuit et sans quota, au prix du
    calcul local. Implémentation retenue par défaut.

    Modèle retenu : BAAI/bge-reranker-v2-m3, cross-encoder multilingue de la
    famille BGE-m3, adapté au français.

    Le modèle n'est chargé qu'au premier appel : la création de l'objet reste
    instantanée et les tests n'ont pas à le charger.
    """

    def __init__(self, model_name: str):
        self._model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self._model_name)
        return self._model

    def rerank(self, query: str, documents: list[str]) -> list[tuple[int, float]]:
        if not documents:
            return []
        paires = [(query, texte) for texte in documents]
        scores = self._get_model().predict(paires)
        ordre = sorted(
            range(len(documents)),
            key=lambda i: scores[i],
            reverse=True,
        )
        return [(i, _sigmoide(float(scores[i]))) for i in ordre]


def get_rerank_provider(settings: Settings) -> RerankProvider:
    """Sélectionne l'implémentation selon le réglage RERANK_PROVIDER de .env."""
    if settings.rerank_provider == "local":
        return LocalReranker(settings.rerank_model)
    raise ValueError(f"Fournisseur de reclassement inconnu : {settings.rerank_provider}")
