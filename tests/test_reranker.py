import numpy as np
import pytest

from app.config import Settings
from app.rag.reranker import (
    LocalReranker,
    OnnxReranker,
    _fournisseurs_onnx,
    _sigmoide,
    get_rerank_provider,
)


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


# --- OnnxReranker -----------------------------------------------------------


class _TokenizerFake:
    """Remplace le tokenizer : garde les arguments reçus et renvoie des tableaux
    de la bonne forme (une ligne par paire), sans rien charger."""

    def __init__(self):
        self.appel: dict | None = None

    def __call__(self, questions, textes, **options):
        self.appel = {"questions": list(questions), "textes": list(textes), **options}
        n = len(textes)
        return {
            "input_ids": np.ones((n, 4), dtype=np.int64),
            "attention_mask": np.ones((n, 4), dtype=np.int64),
            "token_type_ids": np.zeros((n, 4), dtype=np.int64),
        }


class _SessionFake:
    """Remplace la session ONNX Runtime : renvoie les logits fournis, au format
    du modèle exporté (une colonne par paire)."""

    def __init__(self, scores: list[float]):
        self._scores = scores
        self.sorties_demandees = None
        self.entrees_recues: dict | None = None

    def run(self, sorties, entrees):
        self.sorties_demandees = sorties
        self.entrees_recues = entrees
        return [np.array([[s] for s in self._scores], dtype=np.float32)]


def _onnx_fake(scores: list[float]) -> tuple[OnnxReranker, _SessionFake, _TokenizerFake]:
    reranker = OnnxReranker("dossier-test")
    session, tokenizer = _SessionFake(scores), _TokenizerFake()
    reranker._session, reranker._tokenizer = session, tokenizer
    return reranker, session, tokenizer


def test_onnx_ne_charge_pas_le_modele_a_la_construction():
    reranker = OnnxReranker("dossier-test")
    assert reranker._session is None
    assert reranker._tokenizer is None


def test_onnx_est_selectionne_avec_dossier_et_longueur_du_reglage():
    provider = get_rerank_provider(
        _settings(rerank_provider="onnx", rerank_onnx_dir="mon/dossier", rerank_max_length=256)
    )
    assert isinstance(provider, OnnxReranker)
    assert provider._model_dir == "mon/dossier"
    assert provider._max_length == 256


def test_onnx_trie_du_plus_pertinent_au_moins_pertinent():
    reranker, _, _ = _onnx_fake([0.1, 2.0, -1.0])
    ordre = reranker.rerank("question", ["a", "b", "c"])
    assert [indice for indice, _ in ordre] == [1, 0, 2]


def test_onnx_ramene_les_scores_entre_zero_et_un():
    # Le modèle exporté renvoie des logits bruts : une seule sigmoïde, ici.
    reranker, _, _ = _onnx_fake([0.0, 5.0])
    scores = {indice: score for indice, score in reranker.rerank("q", ["a", "b"])}
    assert scores[0] == pytest.approx(0.5)
    assert 0.99 < scores[1] < 1.0


def test_onnx_donne_les_memes_scores_que_local_pour_les_memes_logits():
    # Deux moteurs, un même modèle : à logits égaux, classement et scores
    # doivent être identiques, sinon le seuil du garde-fou changerait de sens
    # selon le moteur.
    logits = [1.5, -0.3, 0.8]
    onnx, _, _ = _onnx_fake(logits)
    local = LocalReranker("modele-test")
    local._model = _ModeleFake(logits)
    textes = ["a", "b", "c"]
    resultat_onnx, resultat_local = onnx.rerank("q", textes), local.rerank("q", textes)
    assert [i for i, _ in resultat_onnx] == [i for i, _ in resultat_local]
    # ONNX calcule en float32 : écart infime, sans effet sur le seuil.
    assert [s for _, s in resultat_onnx] == pytest.approx([s for _, s in resultat_local])


def test_onnx_passe_la_question_avec_chaque_morceau_tronque():
    reranker, _, tokenizer = _onnx_fake([1.0, 1.0])
    reranker.rerank("préavis", ["texte 1", "texte 2"])
    assert tokenizer.appel["questions"] == ["préavis", "préavis"]
    assert tokenizer.appel["textes"] == ["texte 1", "texte 2"]
    assert tokenizer.appel["truncation"] is True
    assert tokenizer.appel["max_length"] == 512


def test_onnx_transmet_seulement_les_entrees_du_modele():
    # Le modèle exporté n'attend que input_ids et attention_mask : toute autre
    # sortie du tokenizer (token_type_ids) ferait échouer ONNX Runtime.
    reranker, session, _ = _onnx_fake([1.0])
    reranker.rerank("q", ["a"])
    assert set(session.entrees_recues) == {"input_ids", "attention_mask"}
    assert session.sorties_demandees == ["logits"]


def test_onnx_liste_vide_renvoie_liste_vide_sans_charger():
    reranker = OnnxReranker("dossier-inexistant")
    assert reranker.rerank("q", []) == []
    assert reranker._session is None


# --- _fournisseurs_onnx -----------------------------------------------------


def test_onnx_prefere_la_carte_graphique_quand_elle_est_disponible():
    disponibles = ["CPUExecutionProvider", "DmlExecutionProvider"]
    assert _fournisseurs_onnx(disponibles) == ["DmlExecutionProvider", "CPUExecutionProvider"]


def test_onnx_sans_carte_graphique_utilise_le_processeur():
    # Cas d'un serveur Linux : DirectML n'est pas installé.
    assert _fournisseurs_onnx(["CPUExecutionProvider"]) == ["CPUExecutionProvider"]


def test_onnx_ignore_les_moteurs_inconnus_et_garde_le_processeur():
    assert _fournisseurs_onnx(["AzureExecutionProvider"]) == ["CPUExecutionProvider"]
