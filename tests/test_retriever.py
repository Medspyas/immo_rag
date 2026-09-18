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


class _RerankerFake:
    """Reclasseur de substitution : renvoie les candidats dans l'ordre inverse
    de celui reçu, et retient les textes reçus pour vérification."""

    def __init__(self):
        self.textes_recus: list[str] | None = None

    def rerank(self, query: str, documents: list[str]) -> list[tuple[int, float]]:
        self.textes_recus = list(documents)
        n = len(documents)
        return [(i, round((i + 1) / n, 3)) for i in reversed(range(n))]


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


# --- HybridRetriever : reclassement facultatif ------------------------------


def test_sans_reranker_score_rerank_reste_none():
    a = uuid4()
    resultats = _retriever([_dense(a)], []).search("q")
    assert resultats[0].score_rerank is None


def test_reranker_impose_son_ordre_sur_celui_de_la_fusion():
    a, b, c = uuid4(), uuid4(), uuid4()
    # Fusion : a, b, c (a 1er des deux côtés). Le reclasseur inverse : c, b, a.
    resultats = _retriever(
        [_dense(a), _dense(b), _dense(c)],
        [_sparse(a), _sparse(b), _sparse(c)],
        reranker=_RerankerFake(),
    ).search("q")
    assert [r.id for r in resultats] == [c, b, a]


def test_reranker_renseigne_le_score_rerank():
    a = uuid4()
    resultats = _retriever([_dense(a)], [], reranker=_RerankerFake()).search("q")
    assert resultats[0].score_rerank == 1.0


def test_reranker_ne_recoit_que_les_meilleurs_candidats():
    ids = [uuid4() for _ in range(10)]
    fake = _RerankerFake()
    _retriever([_dense(i) for i in ids], [], reranker=fake, k_avant_rerank=3).search("q")
    assert fake.textes_recus is not None
    assert len(fake.textes_recus) == 3


def test_reranker_respecte_k():
    ids = [uuid4() for _ in range(5)]
    resultats = _retriever(
        [_dense(i) for i in ids], [], reranker=_RerankerFake()
    ).search("q", k=2)
    assert len(resultats) == 2


def test_reranker_sans_candidat_renvoie_liste_vide():
    assert _retriever([], [], reranker=_RerankerFake()).search("q") == []


# --- Garde-fou étage 1 : seuil sur le score de reclassement ------------------


class _RerankerScores:
    """Reclasseur de substitution qui renvoie des scores fixés, dans l'ordre
    reçu, pour tester le filtrage par seuil."""

    def __init__(self, scores: list[float]):
        self._scores = scores

    def rerank(self, query: str, documents: list[str]) -> list[tuple[int, float]]:
        paires = sorted(enumerate(self._scores[: len(documents)]), key=lambda p: -p[1])
        return paires


def test_seuil_ecarte_les_candidats_sous_le_score():
    a, b, c = uuid4(), uuid4(), uuid4()
    resultats = _retriever(
        [_dense(a), _dense(b), _dense(c)],
        [],
        reranker=_RerankerScores([0.9, 0.2, 0.05]),
        seuil_rerank=0.3,
    ).search("q")
    assert [r.id for r in resultats] == [a]


def test_seuil_tous_sous_le_score_renvoie_liste_vide():
    a = uuid4()
    resultats = _retriever(
        [_dense(a)], [], reranker=_RerankerScores([0.1]), seuil_rerank=0.3
    ).search("q")
    assert resultats == []


def test_seuil_ignore_sans_reclasseur():
    a = uuid4()
    # seuil_rerank n'a de sens qu'avec un score de reclassement à comparer.
    resultats = _retriever([_dense(a)], [], seuil_rerank=0.3).search("q")
    assert [r.id for r in resultats] == [a]


def test_seuil_none_ne_filtre_rien():
    a, b = uuid4(), uuid4()
    resultats = _retriever(
        [_dense(a), _dense(b)], [], reranker=_RerankerScores([0.9, 0.01])
    ).search("q")
    assert {r.id for r in resultats} == {a, b}
