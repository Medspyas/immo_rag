from app.rag.chunking import decouper_en_articles, titre_document

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
    assert [c.article for c in chunks] == ["1", "7"]


def test_preambule_et_travaux_preparatoires_exclus():
    chunks = decouper_en_articles(TEXTE_EXEMPLE, doc_id="loi-1965")
    contenu_total = "\n".join(c.content for c in chunks)
    assert "EN VIGUEUR AU" not in contenu_total
    assert "TRAVAUX PREPARATOIRES" not in contenu_total


def test_article_abroge_est_ecarte():
    """Un article abrogé n'est pas indexé : ni comme source citable (le
    marqueur source dit explicitement de ne pas le citer comme droit actuel),
    ni comme source d'une collision d'identifiant avec une version plus
    récente du même numéro d'article (cf. docstring du module)."""
    chunks = decouper_en_articles(TEXTE_EXEMPLE, doc_id="loi-1965")
    assert "6-1" not in [c.article for c in chunks]
    contenu_total = "\n".join(c.content for c in chunks)
    assert "Abrogé par LOI n°2018-1021" not in contenu_total
    assert "ARTICLE ABROGÉ" not in contenu_total


def test_article_normal_non_marque_abroge():
    chunks = decouper_en_articles(TEXTE_EXEMPLE, doc_id="loi-1965")
    article_1 = next(c for c in chunks if c.article == "1")
    assert article_1.abroge is False


def test_doc_id_propage_a_chaque_chunk():
    chunks = decouper_en_articles(TEXTE_EXEMPLE, doc_id="loi-1965")
    assert all(c.doc_id == "loi-1965" for c in chunks)


def test_texte_sans_article_ne_produit_aucun_chunk():
    assert decouper_en_articles("Juste un texte sans structure.", doc_id="x") == []


TEXTE_AVEC_INTITULES = """Titre Ier : Des rapports entre bailleurs et locataires (Articles 1 à 15)
Chapitre II : De la durée du contrat de location. (Articles 10 à 15)
Article 15
Le délai de préavis applicable au congé est de trois mois.
Titre Ier bis : Des rapports entre bailleurs et locataires dans les logements meublés résidence
principale (Articles 25-3 à 25-11)
Article 25-8
Le locataire peut résilier le contrat à tout moment, sous réserve d'un préavis d'un mois."""


def test_chaque_article_recoit_ses_intitules():
    chunks = decouper_en_articles(TEXTE_AVEC_INTITULES, doc_id="loi-1989")
    sections = {c.article: c.section for c in chunks}
    assert sections["15"] == (
        "Titre Ier : Des rapports entre bailleurs et locataires"
        " > Chapitre II : De la durée du contrat de location"
    )


def test_intitule_coupe_par_la_mise_en_page_est_recolle_et_efface_le_chapitre():
    chunks = decouper_en_articles(TEXTE_AVEC_INTITULES, doc_id="loi-1989")
    article = next(c for c in chunks if c.article == "25-8")
    assert article.section == (
        "Titre Ier bis : Des rapports entre bailleurs et locataires"
        " dans les logements meublés résidence principale"
    )


def test_texte_sans_intitule_donne_une_section_vide():
    chunks = decouper_en_articles(TEXTE_EXEMPLE, doc_id="loi-1965")
    assert all(c.section == "" for c in chunks)


_EN_TETE_LEGIFRANCE = """Loi n° 70-9 du 2 janvier 1970 réglementant les conditions d'exercice des
activités relatives à certaines opérations portant sur les immeubles
EN VIGUEUR AU 30/08/2026
Intitulé(s) non officiel(s)
•
loi Hoguet 1
•
loi Hoguet 1 [1970]
Titre Ier : De l'exercice des activités (Articles 1 à 8-3)
Article 1
Les dispositions de la présente loi s'appliquent.
"""


def test_titre_lu_en_tete_avec_son_nom_dusage():
    assert titre_document(_EN_TETE_LEGIFRANCE) == (
        "Loi n° 70-9 du 2 janvier 1970 réglementant les conditions d'exercice des "
        "activités relatives à certaines opérations portant sur les immeubles (loi Hoguet)"
    )


def test_titre_journal_officiel_sans_rubrique_ni_appel_de_note():
    texte = "LOIS\nLOI no 2014-366 du 24 mars 2014 pour l’accès\nau logement (1)\nNOR: ETLX\nArticle 1\nTexte."
    assert titre_document(texte) == "LOI no 2014-366 du 24 mars 2014 pour l’accès au logement"


def test_titre_arrete_avant_un_pictogramme_de_mise_a_jour():
    texte = "Loi n° 89-462 du 6 juillet 1989\n\uf05a Dernière mise à jour : 2023\nArticle 1\nTexte."
    assert titre_document(texte) == "Loi n° 89-462 du 6 juillet 1989"


def test_chaque_article_porte_le_titre_de_sa_loi():
    chunks = decouper_en_articles(_EN_TETE_LEGIFRANCE, "Loi_hoguet")
    assert chunks[0].titre.endswith("(loi Hoguet)")
