from app.rag.chunking import decouper_en_articles

TEXTE_EXEMPLE = """Loi n° 65-557 du 10 juillet 1965 fixant le statut de la
copropriété des immeubles bâtis
EN VIGUEUR AU 01/09/2026

Article 1
La présente loi régit tout immeuble bâti ou groupe d'immeubles bâtis dont la
propriété est répartie entre plusieurs personnes.

=== ARTICLE ABROGÉ (n'est plus en vigueur — ne pas citer comme droit actuel) ===
Article 6-1
Abrogé par LOI n°2018-1021 du 23 novembre 2018 - art. 206 (V)
=== FIN ARTICLE ABROGÉ ===

Article 7
Chaque copropriétaire dispose d'un nombre de voix correspondant à sa
quote-part dans les parties communes.

Loi n° 65-557, TRAVAUX PREPARATOIRES
Assemblée nationale, projet de loi n° 1498 ;
Discussion et adoption le 26 juin 1965."""


def test_un_chunk_par_article():
    chunks = decouper_en_articles(TEXTE_EXEMPLE, doc_id="loi-1965")
    assert [c.article for c in chunks] == ["1", "6-1", "7"]


def test_preambule_et_travaux_preparatoires_exclus():
    chunks = decouper_en_articles(TEXTE_EXEMPLE, doc_id="loi-1965")
    contenu_total = "\n".join(c.content for c in chunks)
    assert "EN VIGUEUR AU" not in contenu_total
    assert "TRAVAUX PREPARATOIRES" not in contenu_total


def test_article_abroge_est_marque_et_nettoye():
    chunks = decouper_en_articles(TEXTE_EXEMPLE, doc_id="loi-1965")
    article_6_1 = next(c for c in chunks if c.article == "6-1")
    assert article_6_1.abroge is True
    assert "ARTICLE ABROGÉ" not in article_6_1.content
    assert "FIN ARTICLE ABROGÉ" not in article_6_1.content
    assert "Abrogé par LOI n°2018-1021" in article_6_1.content


def test_article_normal_non_marque_abroge():
    chunks = decouper_en_articles(TEXTE_EXEMPLE, doc_id="loi-1965")
    article_1 = next(c for c in chunks if c.article == "1")
    assert article_1.abroge is False


def test_doc_id_propage_a_chaque_chunk():
    chunks = decouper_en_articles(TEXTE_EXEMPLE, doc_id="loi-1965")
    assert all(c.doc_id == "loi-1965" for c in chunks)


def test_texte_sans_article_ne_produit_aucun_chunk():
    assert decouper_en_articles("Juste un texte sans structure.", doc_id="x") == []
