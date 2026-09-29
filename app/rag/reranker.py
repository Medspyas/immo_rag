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
pour les embeddings. Seules des implémentations gratuites sont proposées. Un
autre modèle local se branche en changeant RERANK_MODEL, sans nouvelle classe.

Contenu du fichier :
- RerankProvider : contrat commun.
- LocalReranker : modèle libre exécuté sur la machine (BGE-m3) via
  sentence-transformers. Retenu par défaut, fonctionne partout.
- OnnxReranker : le même modèle, exporté en ONNX, exécuté sur la carte
  graphique quand elle est disponible.
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

    def __init__(self, model_name: str, max_length: int = 512):
        self._model_name = model_name
        self._max_length = max_length
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            # max_length tronque chaque paire (question, morceau) : sans borne,
            # le modèle traite les articles longs en entier et une passe CPU
            # prend plusieurs dizaines de secondes.
            self._model = CrossEncoder(self._model_name, max_length=self._max_length)
        return self._model

    def rerank(self, query: str, documents: list[str]) -> list[tuple[int, float]]:
        if not documents:
            return []
        paires = [(query, texte) for texte in documents]
        # Ce modèle applique une sigmoïde en interne par défaut. On la neutralise
        # (fonction identité) pour récupérer le score brut, puis _sigmoide est
        # appliquée une seule fois plus bas — sinon les scores se tassent tous
        # entre 0,5 et 0,73 et le seuil du garde-fou devient illisible.
        scores = self._get_model().predict(paires, activation_fn=lambda x: x)
        ordre = sorted(
            range(len(documents)),
            key=lambda i: scores[i],
            reverse=True,
        )
        return [(i, _sigmoide(float(scores[i]))) for i in ordre]


class OnnxReranker(RerankProvider):
    """Le même modèle que LocalReranker, exporté au format ONNX et exécuté
    par ONNX Runtime : sur la carte graphique via DirectML, sur le processeur
    à défaut.

    Le reclassement représente l'essentiel du temps d'une recherche. Mesuré
    avec eval/latence.py (2026-09-25, 461 articles) : médiane de 22,6 s par
    requête sur processeur avec LocalReranker, 0,6 s avec cette classe sur
    carte graphique.

    Le modèle exporté (~2,2 Go) est produit par
    scripts/exporter_reclasseur_onnx.py dans rerank_onnx_dir ; il n'est pas
    versionné. Classe utilisée si RERANK_PROVIDER=onnx.
    """

    def __init__(self, model_dir: str, max_length: int = 512):
        self._model_dir = model_dir
        self._max_length = max_length
        self._session = None
        self._tokenizer = None

    def _charger(self):
        if self._session is None:
            import onnxruntime as ort
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self._model_dir)
            self._session = ort.InferenceSession(
                f"{self._model_dir}/model.onnx",
                providers=_fournisseurs_onnx(ort.get_available_providers()),
            )
        return self._session, self._tokenizer

    def rerank(self, query: str, documents: list[str]) -> list[tuple[int, float]]:
        if not documents:
            return []
        session, tokenizer = self._charger()
        entrees = tokenizer(
            [query] * len(documents),
            documents,
            padding=True,
            truncation=True,
            max_length=self._max_length,
            return_tensors="np",
        )
        logits = session.run(
            ["logits"],
            {"input_ids": entrees["input_ids"], "attention_mask": entrees["attention_mask"]},
        )[0]
        scores = logits[:, 0]
        ordre = sorted(range(len(documents)), key=lambda i: scores[i], reverse=True)
        return [(i, _sigmoide(float(scores[i]))) for i in ordre]


def _fournisseurs_onnx(disponibles: list[str]) -> list[str]:
    """Moteurs d'exécution à demander à ONNX Runtime, du plus rapide au plus
    lent, parmi ceux réellement installés sur la machine.

    DirectML (carte graphique sous Windows) n'existe pas partout, par exemple
    sur un serveur Linux sans carte graphique : ne demander que ce qui est
    disponible évite un avertissement, et le processeur reste toujours en
    dernier recours.
    """
    preferes = ["DmlExecutionProvider", "CPUExecutionProvider"]
    choisis = [f for f in preferes if f in disponibles]
    return choisis or ["CPUExecutionProvider"]


def get_rerank_provider(settings: Settings) -> RerankProvider:
    """Sélectionne l'implémentation selon le réglage RERANK_PROVIDER de .env."""
    if settings.rerank_provider == "local":
        return LocalReranker(settings.rerank_model, settings.rerank_max_length)
    if settings.rerank_provider == "onnx":
        return OnnxReranker(settings.rerank_onnx_dir, settings.rerank_max_length)
    raise ValueError(f"Fournisseur de reclassement inconnu : {settings.rerank_provider}")
