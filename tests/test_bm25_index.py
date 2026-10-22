from uuid import uuid4

from app.rag.bm25_index import BM25Document, BM25Index, tokeniser_fr

# --- découpage du texte en mots -------------------------------------------

def test_reference_legale_preservee_en_un_token():
    tokens = tokeniser_fr("La carte est régie par l'article L. 111-1 du code")
    assert "l.111-1" in tokens
    # la référence n'a pas été coupée en morceaux
    assert "111" not in tokens


def test_numero_de_loi_preserve():
    assert "89-462" in tokeniser_fr("prévu par la loi 89-462")


def test_accents_et_casse_normalises():
    assert tokeniser_fr("PRÉAVIS") == tokeniser_fr("preavis")


def test_stopwords_retires():
    assert tokeniser_fr("le la les de des et ou dans sur pour") == []


def test_singulier_et_pluriel_meme_racine():
    assert tokeniser_fr("logements")[0] == tokeniser_fr("logement")[0]


def test_plus_value_un_seul_mot_quelle_que_soit_la_graphie():
    attendu = tokeniser_fr("plus-values")
    assert attendu == ["plusvalu"]
    for graphie in ("Plus-value", "plus value", "plus valus"):
        assert tokeniser_fr(graphie) == attendu
    assert tokeniser_fr("moins-values") != attendu
    # « plus » seul reste un mot-outil
    assert tokeniser_fr("plus de valeur") == tokeniser_fr("valeur")


def test_verbe_et_nom_meme_racine():
    # Snowball sépare "revoqu" / "revoc" et "resil" / "resili"
    assert tokeniser_fr("révoque") == tokeniser_fr("révocation")
    assert tokeniser_fr("convoquer") == tokeniser_fr("convocation")
    assert tokeniser_fr("résilie") == tokeniser_fr("résiliation")


# --- l'index et ses recherches -------------------------------------------

DOCS = [
    BM25Document(
        id=uuid4(), doc_id="loi-89",
        content="Le délai de préavis du locataire qui donne congé est de trois mois.",
    ),
    BM25Document(
        id=uuid4(), doc_id="loi-65",
        content="Les charges de copropriété sont réparties entre tous les copropriétaires.",
    ),
    BM25Document(
        id=uuid4(), doc_id="hoguet",
        content="La carte professionnelle d'agent immobilier est régie par l'article L. 111-1.",
    ),
]


def test_search_trouve_le_chunk_pertinent():
    resultats = BM25Index(DOCS).search("préavis du locataire", k=1)
    assert resultats[0].doc_id == "loi-89"


def test_search_par_reference_legale():
    resultats = BM25Index(DOCS).search("que dit l'article L. 111-1 ?", k=1)
    assert resultats[0].doc_id == "hoguet"


def test_search_sans_mot_commun_renvoie_vide():
    assert BM25Index(DOCS).search("gastronomie bordelaise et cyclisme") == []


def test_search_ignore_les_scores_nuls():
    # "copropriété" ne concerne qu'un seul morceau : 3 demandés, 1 renvoyé
    resultats = BM25Index(DOCS).search("copropriété", k=3)
    assert [r.doc_id for r in resultats] == ["loi-65"]


def test_complement_indexe_mais_jamais_renvoye():
    doc = BM25Document(
        id=uuid4(), doc_id="hoguet", content="Le mandat comporte une limitation dans le temps.",
        complement="Un mandat peut-il être à durée indéterminée ?",
    )
    resultats = BM25Index([doc, *DOCS]).search("durée indéterminée")
    assert [r.doc_id for r in resultats] == ["hoguet"]
    assert "indéterminée" not in resultats[0].content


def test_index_vide():
    assert BM25Index([]).search("préavis") == []


def test_requete_vide_apres_tokenisation():
    assert BM25Index(DOCS).search("le la les de") == []


def test_article_retrouve_par_son_numero_et_sa_famille():
    doc = BM25Document(
        id=uuid4(), doc_id="code_impot2", content="Article 150 VC", metadata={"article": "150 VC"}
    )
    index = BM25Index([doc])
    assert index.article("code_impot", "150 VC") == doc
    assert index.article("code_impot", "150 VD") is None
