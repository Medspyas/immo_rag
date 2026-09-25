import pytest
import torch

from app.config import Settings
from app.rag.nli import LocalNLI, ScoreNLI, _trouver_indice, get_nli_provider


def _settings(**overrides) -> Settings:
    base = {"database_url": "postgresql://x"}
    base.update(overrides)
    return Settings(**base)


# --- _trouver_indice ---------------------------------------------------------


def test_trouve_lindice_quel_que_soit_son_rang():
    assert _trouver_indice({0: "contradiction", 1: "neutral", 2: "entailment"}, "entail") == 2
    assert _trouver_indice({0: "entailment", 1: "neutral", 2: "contradiction"}, "entail") == 0
    assert _trouver_indice({0: "entailment", 1: "neutral", 2: "contradiction"}, "contradict") == 2


def test_insensible_a_la_casse():
    assert _trouver_indice({0: "ENTAILMENT", 1: "neutral"}, "entail") == 0


def test_etiquette_absente_leve_une_erreur():
    with pytest.raises(ValueError):
        _trouver_indice({0: "positive", 1: "negative"}, "entail")


# --- LocalNLI ----------------------------------------------------------------


class _TokenizerFake:
    """Retient les options reçues ; le modèle factice n'utilise pas les entrées."""

    def __init__(self):
        self.options: dict = {}

    def __call__(self, premisses, hypotheses, **options):
        self.options = options
        return {}


class _ModeleFake:
    """Remplace le modèle de classification : renvoie des logits fixés (une
    ligne par prémisse), sans rien charger."""

    def __init__(self, logits: list[float] | list[list[float]]):
        self._logits = logits if isinstance(logits[0], list) else [logits]

    def __call__(self, **kwargs):
        return type("Sortie", (), {"logits": torch.tensor(self._logits)})()


def test_local_ne_charge_pas_le_modele_a_la_construction():
    nli = LocalNLI("modele-test")
    assert nli._model is None


def test_local_borne_la_longueur_a_512_par_defaut():
    assert LocalNLI("modele-test")._max_length == 512


def test_local_charge_le_modele_en_float32(monkeypatch):
    import transformers

    options_recues = {}

    class _ModeleCharge(_ModeleFake):
        config = type("Config", (), {"id2label": {0: "entailment", 1: "contradiction"}})()

        def eval(self):
            return self

    def _from_pretrained(nom, **options):
        options_recues.update(options)
        return _ModeleCharge([0.0, 0.0])

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda nom: _TokenizerFake())
    monkeypatch.setattr(transformers.AutoModelForSequenceClassification, "from_pretrained", _from_pretrained)
    LocalNLI("modele-test")._charger()
    assert options_recues.get("dtype") is torch.float32


def test_local_transmet_la_longueur_max_du_reglage():
    provider = get_nli_provider(_settings(nli_provider="local", nli_max_length=256))
    assert isinstance(provider, LocalNLI)
    assert provider._max_length == 256


def _nli_fake(logits) -> LocalNLI:
    nli = LocalNLI("modele-test")
    nli._tokenizer = _TokenizerFake()
    nli._model = _ModeleFake(logits)
    nli._indice_entailment = 0
    nli._indice_contradiction = 2
    return nli


def test_local_renvoie_implication_et_contradiction():
    nli = _nli_fake([5.0, 0.0, -5.0])
    attendu = torch.softmax(torch.tensor([5.0, 0.0, -5.0]), dim=-1)
    [score] = nli.evaluer(["texte source"], "affirmation")
    assert score.implication == pytest.approx(attendu[0].item())
    assert score.contradiction == pytest.approx(attendu[2].item())


def test_local_un_score_par_premisse_dans_lordre():
    nli = _nli_fake([[5.0, 0.0, -5.0], [-5.0, 0.0, 5.0]])
    scores = nli.evaluer(["extrait qui implique", "extrait qui contredit"], "affirmation")
    assert scores[0].implication > 0.9
    assert scores[1].contradiction > 0.9


def test_local_ne_tronque_que_la_premisse():
    nli = _nli_fake([0.0, 0.0, 0.0])
    nli.evaluer(["texte"], "affirmation")
    assert nli._tokenizer.options["truncation"] == "only_first"


def test_local_sans_premisse_ne_sollicite_pas_le_modele():
    assert LocalNLI("modele-test").evaluer([], "affirmation") == []


def test_score_nli_est_immuable():
    score = ScoreNLI(implication=0.9, contradiction=0.1)
    with pytest.raises(AttributeError):
        score.implication = 0.0


# --- get_nli_provider --------------------------------------------------------


def test_local_est_selectionne_sans_aucune_cle_api():
    provider = get_nli_provider(_settings(nli_provider="local"))
    assert isinstance(provider, LocalNLI)


def test_fournisseur_inconnu_est_refuse():
    with pytest.raises(ValueError):
        _settings(nli_provider="autre")
