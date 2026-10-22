from app.rag.renvois import articles_cites, famille_document


def test_renvoi_vers_le_code_civil():
    texte = "Les deux derniers alinéas de l'article 1724 du code civil sont applicables."
    assert articles_cites(texte, "loi_06_07_1989") == [("code_civil", "1724")]


def test_liste_d_articles_de_la_presente_loi():
    texte = "les articles 17, 17-1, 17-2 et 18 de la présente loi"
    assert articles_cites(texte, "loi_06_07_1989") == [
        ("loi_06_07_1989", "17"), ("loi_06_07_1989", "17-1"),
        ("loi_06_07_1989", "17-2"), ("loi_06_07_1989", "18"),
    ]


def test_renvoi_vers_une_loi_du_corpus_par_son_numero_avec_subdivisions():
    texte = "dans les conditions de l'article 24, I, 1° de la loi n° 65-557 du 10 juillet 1965"
    assert articles_cites(texte, "loi_06_07_1989") == [
        ("loi-n°65-557_du_10_juillet_1965", "24")
    ]


def test_loi_hors_corpus_ignoree():
    assert articles_cites("l'article 41 ter de la loi n° 86-1290", "loi_06_07_1989") == []


def test_numero_du_code_des_impots_avec_lettres_et_suffixe():
    texte = "l'article 150 VC du code général des impôts et l'article 235 ter ZE bis du présent code"
    assert articles_cites(texte, "code_impot2") == [
        ("code_impot", "150 VC"), ("code_impot", "235 ter ZE bis")
    ]


def test_article_sans_texte_vise_non_suivi():
    # « article 6 » seul peut viser un autre texte cité plus haut.
    assert articles_cites("conformément à l'article 6. Les parties", "loi_06_07_1989") == []


def test_retour_a_la_ligne_dans_le_renvoi_et_doublon():
    texte = "l'article 15 de la\nprésente loi ; l'article 15 de la présente loi"
    assert articles_cites(texte, "loi_06_07_1989") == [("loi_06_07_1989", "15")]


def test_famille_du_code_des_impots():
    assert famille_document("code_impot3") == "code_impot"
    assert famille_document("code_civil") == "code_civil"
