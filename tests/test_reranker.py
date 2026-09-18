import pytest

from app.config import Settings
from app.rag.reranker import LocalReranker, _sigmoide, get_rerank_provider


def _settings(**overrides) -> Settings:
    base = {"database_url": "postgresql://x"}
    base.update(overrides)
    return Settings(**base)


class _ModeleFake:
    """Remplace le cross-encoder : renvoie les scores fournis, dans l'ordre des
    paires reçues, sans rien charger."""

    def __init__(self, scores: list[float]):
        self._scores = scores
        self.paires_recues: list[tuple[str, str]] | None = None
        self.activation_recue = "absente"

    def predict(self, paires, activation_fn=None):
        self.paires_recues = list(paires)
        self.activation_recue = activation_fn
        return self._scores


# --- _sigmoide -------------------------------------------------------------


def test_sigmoide_centre_sur_zero():
    assert _sigmoide(0.0) == pytest.approx(0.5)


def test_sigmoide_reste_dans_zero_un_sur_valeurs_extremes():
    # Sur une valeur très négative, une sigmoïde naïve déborde ; celle-ci non.
    assert 0.0 < _sigmoide(-40.0) < 0.01
    assert 0.99 < _sigmoide(40.0) <= 1.0


# --- LocalReranker --------------------------------------------------------


def test_local_ne_charge_pas_le_modele_a_la_construction():
    reranker = LocalReranker("BAAI/bge-reranker-v2-m3")
    assert reranker._model is None


def test_local_borne_la_longueur_a_512_par_defaut():
    assert LocalReranker("modele-test")._max_length == 512


def test_local_transmet_la_longueur_max_du_reglage():
    provider = get_rerank_provider(_settings(rerank_provider="local", rerank_max_length=256))
    assert isinstance(provider, LocalReranker)
    assert provider._max_length == 256


def test_local_trie_du_plus_pertinent_au_moins_pertinent():
    reranker = LocalReranker("modele-test")
    reranker._model = _ModeleFake([0.1, 2.0, -1.0])
    ordre = reranker.rerank("question", ["a", "b", "c"])
    assert [indice for indice, _ in ordre] == [1, 0, 2]


def test_local_ramene_les_scores_entre_zero_et_un():
    reranker = LocalReranker("modele-test")
    reranker._model = _ModeleFake([0.0, 5.0])
    scores = {indice: score for indice, score in reranker.rerank("q", ["a", "b"])}
    assert scores[0] == pytest.approx(0.5)
    assert 0.99 < scores[1] < 1.0


def test_local_passe_la_question_avec_chaque_morceau():
    reranker = LocalReranker("modele-test")
    fake = _ModeleFake([1.0, 1.0])
    reranker._model = fake
    reranker.rerank("préavis", ["texte 1", "texte 2"])
    assert fake.paires_recues == [("préavis", "texte 1"), ("préavis", "texte 2")]


def test_local_liste_vide_renvoie_liste_vide():
    assert LocalReranker("modele-test").rerank("q", []) == []


def test_local_neutralise_la_sigmoide_interne_du_modele():
    # Le modèle applique une sigmoïde par défaut ; LocalReranker doit demander
    # le score brut pour ne pas l'appliquer deux fois.
    reranker = LocalReranker("modele-test")
    fake = _ModeleFake([0.0])
    reranker._model = fake
    reranker.rerank("q", ["a"])
    assert callable(fake.activation_recue)
    assert fake.activation_recue(3.7) == 3.7  # fonction identité


# --- get_rerank_provider ------------------------------------------------


def test_local_est_selectionne_sans_aucune_cle_api():
    provider = get_rerank_provider(_settings(rerank_provider="local"))
    assert isinstance(provider, LocalReranker)


def test_fournisseur_inconnu_est_refuse():
    with pytest.raises(ValueError):
        _settings(rerank_provider="jina")
