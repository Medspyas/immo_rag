"""Vérification d'implication logique (NLI) entre une source et une affirmation.

Le garde-fou étage 2 (cf. CLAUDE.md, generator.py) doit s'assurer qu'une
affirmation rédigée par le LLM découle réellement du texte source cité, et
n'est pas une extrapolation plausible mais non fondée. Un modèle NLI (Natural
Language Inference) répond directement à cette question : recevant une
prémisse (le texte source) et une hypothèse (l'affirmation), il évalue si la
seconde est impliquée par la première (entailment), la contredit
(contradiction), ou n'a pas de rapport avec elle (neutral).

Le fournisseur passe par le contrat NLIProvider et se règle dans .env, comme
pour les embeddings et le reclassement. Seule une implémentation locale et
gratuite est proposée : un modèle NLI multilingue ou français (mDeBERTa-v3-
base-mnli-xnli par défaut), pas un modèle anglais uniquement — celui-ci
échouerait silencieusement sur des affirmations en français (cf. CLAUDE.md,
section garde-fou).

Contenu du fichier :
- ScoreNLI : scores d'implication et de contradiction d'un couple.
- NLIProvider : contrat commun.
- LocalNLI : modèle libre exécuté sur la machine, retenu par défaut.
- get_nli_provider : sélectionne l'implémentation selon .env.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.config import Settings


def _trouver_indice(id2label: dict, prefixe: str) -> int:
    """Retrouve l'indice d'une classe ("entailment", "contradiction") dans la
    configuration d'un modèle NLI, d'après le début de son étiquette.

    L'ordre des classes (entailment/neutral/contradiction) dépend du modèle :
    mieux vaut le lire dans sa configuration que le figer en dur, ce qui
    casserait silencieusement avec un autre modèle NLI.
    """
    for indice, etiquette in id2label.items():
        if str(etiquette).lower().startswith(prefixe):
            return int(indice)
    raise ValueError(
        f"Aucune étiquette « {prefixe}… » trouvée dans la configuration du modèle."
    )


@dataclass(frozen=True)
class ScoreNLI:
    """Probabilités (0 à 1) qu'un texte source implique une affirmation, ou
    la contredise.

    Les deux sont utiles au garde-fou : un extrait court peut sembler
    impliquer une affirmation fausse par simple proximité des mots, alors
    qu'un autre extrait du même article la contredit nettement.
    """

    implication: float
    contradiction: float


class NLIProvider(ABC):
    """Contrat commun à tous les fournisseurs de vérification NLI."""

    @abstractmethod
    def evaluer(self, premisses: list[str], hypothese: str) -> list[ScoreNLI]:
        """Scores de chaque prémisse face à une même hypothèse, dans l'ordre.

        Les prémisses sont des extraits du texte source (l'article cité),
        l'hypothèse l'affirmation générée à vérifier. Traiter les extraits
        ensemble permet un seul passage du modèle (un lot) au lieu d'un
        passage par extrait.
        """


class LocalNLI(NLIProvider):
    """Modèle libre exécuté sur la machine : gratuit et sans quota, au prix du
    calcul local. Implémentation retenue par défaut.

    Modèle retenu : MoritzLaurer/mDeBERTa-v3-base-mnli-xnli, entraîné sur des
    données multilingues (dont le français) — un point important, un modèle
    NLI anglais-only ne détecterait pas les affirmations non fondées en
    français, il donnerait juste un score sans rapport avec la réalité.

    Le modèle n'est chargé qu'au premier appel : la création de l'objet reste
    instantanée et les tests n'ont pas à le charger.
    """

    def __init__(self, model_name: str, max_length: int = 512):
        self._model_name = model_name
        self._max_length = max_length
        self._tokenizer = None
        self._model = None
        self._indice_entailment: int | None = None
        self._indice_contradiction: int | None = None

    def _charger(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        # float32 imposé : sans précision explicite, transformers reprend celle
        # enregistrée avec le modèle (float16 pour mDeBERTa), un format sans
        # accélération sur processeur — environ 12 fois plus lent, mesuré.
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self._model_name, dtype=torch.float32
        )
        self._model.eval()
        etiquettes = self._model.config.id2label
        self._indice_entailment = _trouver_indice(etiquettes, "entail")
        self._indice_contradiction = _trouver_indice(etiquettes, "contradict")

    def evaluer(self, premisses: list[str], hypothese: str) -> list[ScoreNLI]:
        import torch

        if not premisses:
            return []
        self._charger()
        # truncation="only_first" : si un couple dépasse la limite, seule la
        # prémisse est raccourcie, jamais l'affirmation à vérifier.
        entrees = self._tokenizer(
            premisses,
            [hypothese] * len(premisses),
            return_tensors="pt",
            padding=True,
            truncation="only_first",
            max_length=self._max_length,
        )
        with torch.no_grad():
            probabilites = torch.softmax(self._model(**entrees).logits, dim=-1)
        return [
            ScoreNLI(
                implication=float(ligne[self._indice_entailment]),
                contradiction=float(ligne[self._indice_contradiction]),
            )
            for ligne in probabilites
        ]


def get_nli_provider(settings: Settings) -> NLIProvider:
    """Sélectionne l'implémentation selon le réglage NLI_PROVIDER de .env."""
    if settings.nli_provider == "local":
        return LocalNLI(settings.nli_model, settings.nli_max_length)
    raise ValueError(f"Fournisseur NLI inconnu : {settings.nli_provider}")
