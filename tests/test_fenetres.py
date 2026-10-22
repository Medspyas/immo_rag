from app.rag.fenetres import (
    LONGUEUR_ARTICLE_LONG,
    PAS_FENETRE,
    TAILLE_FENETRE,
    fenetres_article,
)


def test_article_court_sans_fenetre():
    assert fenetres_article("a " * (LONGUEUR_ARTICLE_LONG // 2)) == []


def test_article_long_couvert_jusqu_a_la_fin():
    texte = " ".join(f"mot{i}" for i in range(1000))
    fenetres = fenetres_article(texte)
    assert fenetres
    assert all(len(f) <= TAILLE_FENETRE for f in fenetres)
    assert texte.endswith(fenetres[-1])
    assert "mot999" in fenetres[-1]


def test_fenetres_commencent_au_debut_d_un_mot():
    texte = " ".join(f"mot{i}" for i in range(1000))
    for f in fenetres_article(texte):
        assert f.startswith("mot")


def test_premiere_fenetre_apres_le_debut_de_l_article():
    texte = " ".join(f"mot{i}" for i in range(1000))
    premiere = fenetres_article(texte)[0]
    assert texte.index(premiere) <= PAS_FENETRE
    assert not premiere.startswith("mot0 ")
