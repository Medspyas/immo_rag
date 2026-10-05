import pytest

from eval.latence import Chronometre, percentile, tableau


class _Objet:
    attribut = "conservé"

    def calcul(self, x):
        return x * 2


def _ligne(texte: str, debut: str) -> str:
    return next(ligne for ligne in texte.splitlines() if ligne.startswith(debut))


# --- percentile -------------------------------------------------------------


def test_percentile_mediane_nombre_impair():
    assert percentile([3.0, 1.0, 2.0], 50) == 2.0


def test_percentile_interpole_entre_deux_mesures():
    assert percentile([0.0, 10.0], 50) == pytest.approx(5.0)


def test_percentile_extremes():
    assert percentile([4.0, 1.0, 9.0], 0) == 1.0
    assert percentile([4.0, 1.0, 9.0], 100) == 9.0


def test_percentile_une_seule_mesure():
    assert percentile([7.0], 95) == 7.0


def test_percentile_sans_mesure_est_refuse():
    with pytest.raises(ValueError):
        percentile([], 50)


# --- Chronometre ------------------------------------------------------------


def test_chronometre_ne_change_pas_le_resultat():
    chrono = Chronometre()
    chrono.nouvelle_requete()
    assert chrono.envelopper(_Objet(), "calcul", "étape").calcul(21) == 42


def test_chronometre_additionne_les_appels_d_une_meme_requete():
    chrono = Chronometre()
    enveloppe = chrono.envelopper(_Objet(), "calcul", "étape")
    chrono.nouvelle_requete()
    enveloppe.calcul(1)
    enveloppe.calcul(2)
    chrono.nouvelle_requete()
    enveloppe.calcul(3)
    assert len(chrono.requetes) == 2
    assert set(chrono.requetes[0]) == {"étape"}
    assert chrono.requetes[0]["étape"] >= 0


def test_chronometre_nomme_autrement_les_appels_suivants():
    # Le LLM rédige d'abord la réponse, puis sert de juge : même méthode,
    # deux étapes distinctes.
    chrono = Chronometre()
    llm = chrono.envelopper(_Objet(), "calcul", "génération", "juge LLM")
    chrono.nouvelle_requete()
    llm.calcul(1)
    llm.calcul(2)
    chrono.nouvelle_requete()
    llm.calcul(3)
    assert set(chrono.requetes[0]) == {"génération", "juge LLM"}
    assert set(chrono.requetes[1]) == {"génération"}


def test_chronometre_conserve_les_autres_attributs():
    enveloppe = Chronometre().envelopper(_Objet(), "calcul", "étape")
    assert enveloppe.attribut == "conservé"


# --- tableau ----------------------------------------------------------------


def test_tableau_ecarte_la_premiere_requete():
    # La première requête (chargement des modèles, 1000 ms) ne doit pas peser.
    texte = tableau([{"sens": 1000.0}, {"sens": 10.0}, {"sens": 10.0}])
    assert "10ms" in _ligne(texte, "sens")
    assert "1000" not in texte


def test_tableau_additionne_les_etapes_sans_total_mesure():
    texte = tableau([{}, {"sens": 10.0, "mots-clés": 5.0}])
    assert "15ms" in _ligne(texte, "total")


def test_tableau_prefere_le_total_mesure_de_bout_en_bout():
    texte = tableau([{}, {"sens": 10.0, "total": 40.0}])
    assert "40ms" in _ligne(texte, "total")


def test_tableau_compte_une_etape_absente_pour_zero_et_indique_sa_frequence():
    # Juge LLM sollicité sur 1 requête sur 2 : médiane à mi-chemin, « 1/2 ».
    texte = tableau([{}, {"juge LLM": 100.0}, {}])
    ligne = _ligne(texte, "juge LLM")
    assert "50ms" in ligne
    assert "1/2" in ligne


def test_tableau_omet_une_etape_jamais_executee():
    texte = tableau([{}, {"sens": 10.0}])
    assert "reclassement" not in texte


def test_tableau_sans_requete_retenue_est_refuse():
    with pytest.raises(ValueError):
        tableau([{"sens": 10.0}])
