from uuid import uuid4

from app.rag.bm25_index import BM25Result
from app.rag.retriever import HybridRetriever, fusion_rrf
from app.rag.vector_store import SearchResult


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


# --- HybridRetriever : les deux recherches et leur fusion --------------------


class _EmbeddingsFake:
    """Renvoie toujours le même vecteur : le contenu du texte est sans effet
    ici, la recherche par le sens étant simulée par _RechercheDenseFake."""

    def embed_query(self, text: str) -> list[float]:
        return [0.0]


class _RechercheDenseFake:
    def __init__(self, resultats: list[SearchResult]):
        self._resultats = resultats

    def search(self, query_embedding: list[float], k: int) -> list[SearchResult]:
        return self._resultats[:k]


class _RechercheSparseFake:
    def __init__(self, resultats: list[BM25Result]):
        self._resultats = resultats

    def search(self, query: str, k: int) -> list[BM25Result]:
        return self._resultats[:k]


def _dense(chunk_id, doc_id="loi-89-462"):
    return SearchResult(
        id=chunk_id, doc_id=doc_id, content=f"contenu {chunk_id}", metadata={}, score=0.9
    )


def _sparse(chunk_id, doc_id="loi-89-462"):
    return BM25Result(
        id=chunk_id, doc_id=doc_id, content=f"contenu {chunk_id}", metadata={}, score=4.2
    )


def _retriever(denses, sparses, **kwargs):
    return HybridRetriever(
        _EmbeddingsFake(),
        _RechercheDenseFake(denses),
        _RechercheSparseFake(sparses),
        **kwargs,
    )


def test_hybride_chunk_bien_place_dans_les_deux_remonte_en_tete():
    a, b, c = uuid4(), uuid4(), uuid4()
    # "b" est 2e par le sens mais 1er par mots-clés ; "a" n'est 1er que d'un côté.
    resultats = _retriever([_dense(a), _dense(b)], [_sparse(b), _sparse(c)]).search("q")
    assert resultats[0].id == b


def test_hybride_inclut_un_chunk_remonte_par_une_seule_recherche():
    a, b = uuid4(), uuid4()
    resultats = _retriever([_dense(a)], [_sparse(b)]).search("q")
    assert {r.id for r in resultats} == {a, b}


def test_hybride_expose_les_rangs_de_chaque_recherche():
    a, b = uuid4(), uuid4()
    par_id = {r.id: r for r in _retriever([_dense(a), _dense(b)], [_sparse(b)]).search("q")}
    assert (par_id[a].rang_dense, par_id[a].rang_bm25) == (1, None)
    assert (par_id[b].rang_dense, par_id[b].rang_bm25) == (2, 1)


def test_hybride_respecte_k():
    ids = [uuid4() for _ in range(5)]
    resultats = _retriever([_dense(i) for i in ids], []).search("q", k=3)
    assert len(resultats) == 3


def test_hybride_sans_resultat_sparse_garde_lordre_dense():
    ids = [uuid4() for _ in range(3)]
    resultats = _retriever([_dense(i) for i in ids], []).search("q")
    assert [r.id for r in resultats] == ids


def test_hybride_aucun_resultat_renvoie_liste_vide():
    assert _retriever([], []).search("q") == []
