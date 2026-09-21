import pytest
import torch

from app.config import Settings
from app.rag.nli import LocalNLI, _trouver_indice_entailment, get_nli_provider


def _settings(**overrides) -> Settings:
    base = {"database_url": "postgresql://x"}
    base.update(overrides)
    return Settings(**base)


# --- _trouver_indice_entailment ---------------------------------------------


def test_trouve_lindice_quel_que_soit_son_rang():
    assert _trouver_indice_entailment({0: "contradiction", 1: "neutral", 2: "entailment"}) == 2
    assert _trouver_indice_entailment({0: "entailment", 1: "neutral", 2: "contradiction"}) == 0


def test_insensible_a_la_casse():
    assert _trouver_indice_entailment({0: "ENTAILMENT", 1: "neutral"}) == 0


def test_etiquette_absente_leve_une_erreur():
    with pytest.raises(ValueError):
        _trouver_indice_entailment({0: "positive", 1: "negative"})


# --- LocalNLI ----------------------------------------------------------------


class _TokenizerFake:
    def __call__(self, premisse, hypothese, return_tensors=None, truncation=None, max_length=None):
        return {}


class _ModeleFake:
    """Remplace le modèle de classification : renvoie des logits fixés, sans
    rien charger."""

    def __init__(self, logits: list[float]):
        self._logits = logits

    def __call__(self, **kwargs):
        return type("Sortie", (), {"logits": torch.tensor([self._logits])})()


def test_local_ne_charge_pas_le_modele_a_la_construction():
    nli = LocalNLI("modele-test")
    assert nli._model is None


def test_local_borne_la_longueur_a_512_par_defaut():
    assert LocalNLI("modele-test")._max_length == 512


def test_local_transmet_la_longueur_max_du_reglage():
    provider = get_nli_provider(_settings(nli_provider="local", nli_max_length=256))
    assert isinstance(provider, LocalNLI)
    assert provider._max_length == 256


def test_local_renvoie_la_probabilite_dentailment():
    nli = LocalNLI("modele-test")
    nli._tokenizer = _TokenizerFake()
    nli._model = _ModeleFake([5.0, 0.0, -5.0])
    nli._indice_entailment = 2
    attendu = torch.softmax(torch.tensor([5.0, 0.0, -5.0]), dim=-1)[2].item()
    assert nli.score_implication("texte source", "affirmation") == pytest.approx(attendu)


def test_local_score_bas_quand_lentailment_nest_pas_lindice_favorise():
    nli = LocalNLI("modele-test")
    nli._tokenizer = _TokenizerFake()
    nli._model = _ModeleFake([5.0, 0.0, -5.0])
    nli._indice_entailment = 0  # "entailment" est ici l'indice dominant
    assert nli.score_implication("texte", "affirmation") > 0.9


# --- get_nli_provider --------------------------------------------------------


def test_local_est_selectionne_sans_aucune_cle_api():
    provider = get_nli_provider(_settings(nli_provider="local"))
    assert isinstance(provider, LocalNLI)


def test_fournisseur_inconnu_est_refuse():
    with pytest.raises(ValueError):
        _settings(nli_provider="autre")
