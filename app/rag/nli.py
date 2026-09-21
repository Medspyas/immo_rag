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
- NLIProvider : contrat commun.
- LocalNLI : modèle libre exécuté sur la machine, retenu par défaut.
- get_nli_provider : sélectionne l'implémentation selon .env.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from app.config import Settings


def _trouver_indice_entailment(id2label: dict) -> int:
    """Retrouve l'indice de la classe "entailment" dans la configuration d'un
    modèle NLI.

    L'ordre des classes (entailment/neutral/contradiction) dépend du modèle :
    mieux vaut le lire dans sa configuration que le figer en dur, ce qui
    casserait silencieusement avec un autre modèle NLI.
    """
    for indice, etiquette in id2label.items():
        if str(etiquette).lower().startswith("entail"):
            return int(indice)
    raise ValueError(
        'Aucune étiquette « entailment » trouvée dans la configuration du modèle.'
    )


class NLIProvider(ABC):
    """Contrat commun à tous les fournisseurs de vérification NLI."""

    @abstractmethod
    def score_implication(self, premisse: str, hypothese: str) -> float:
        """Probabilité (0 à 1) que hypothese soit impliquée par premisse.

        premisse est le texte source (l'article cité), hypothese l'affirmation
        générée à vérifier. Une valeur proche de 1 signifie que le texte source
        fonde bien l'affirmation ; proche de 0, qu'il ne la fonde pas
        (contradiction ou absence de rapport — les deux comptent comme un
        défaut de fondement pour le garde-fou).
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

    def _charger(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self._model_name)
        self._model.eval()
        self._indice_entailment = _trouver_indice_entailment(self._model.config.id2label)

    def score_implication(self, premisse: str, hypothese: str) -> float:
        import torch

        self._charger()
        entrees = self._tokenizer(
            premisse,
            hypothese,
            return_tensors="pt",
            truncation=True,
            max_length=self._max_length,
        )
        with torch.no_grad():
            logits = self._model(**entrees).logits[0]
        probabilites = torch.softmax(logits, dim=-1)
        return float(probabilites[self._indice_entailment])


def get_nli_provider(settings: Settings) -> NLIProvider:
    """Sélectionne l'implémentation selon le réglage NLI_PROVIDER de .env."""
    if settings.nli_provider == "local":
        return LocalNLI(settings.nli_model, settings.nli_max_length)
    raise ValueError(f"Fournisseur NLI inconnu : {settings.nli_provider}")
