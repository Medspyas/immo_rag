from app.rag.retriever import fusion_rrf


def test_document_premier_partout_reste_premier():
    dense = ["a", "b", "c"]
    bm25 = ["a", "c", "b"]
    assert fusion_rrf([dense, bm25])[0] == "a"


def test_document_absent_dun_classement_est_quand_meme_inclus():
    dense = ["a", "b"]
    bm25 = ["c"]
    resultat = fusion_rrf([dense, bm25])
    assert set(resultat) == {"a", "b", "c"}


def test_bien_classe_dans_les_deux_bat_bien_classe_dans_un_seul():
    # "b" est 2e dans les deux classements ; "a" est 1er mais seulement dans le premier
    dense = ["a", "b"]
    bm25 = ["b", "c"]
    resultat = fusion_rrf([dense, bm25])
    assert resultat[0] == "b"


def test_classements_vides():
    assert fusion_rrf([[], []]) == []


def test_un_seul_classement_garde_son_ordre():
    assert fusion_rrf([["x", "y", "z"]]) == ["x", "y", "z"]
