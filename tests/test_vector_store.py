"""Tests nécessitant la base réelle (docker compose up -d db).

Ignorés automatiquement si la base n'est pas joignable.
"""
import uuid

import psycopg
import pytest

from app.rag.vector_store import Chunk, VectorStore

DSN = "postgresql://immo:immo@localhost:5433/immo_rag"
DIMENSION = 768


def _db_disponible() -> bool:
    try:
        psycopg.connect(DSN, connect_timeout=1).close()
        return True
    except psycopg.OperationalError:
        return False


pytestmark = pytest.mark.skipif(
    not _db_disponible(), reason="Postgres/pgvector non démarré (docker compose up -d db)"
)


def _vecteur(direction: int) -> list[float]:
    v = [0.0] * DIMENSION
    v[direction] = 1.0
    return v


def test_search_retrouve_le_chunk_le_plus_proche():
    store = VectorStore(DSN, DIMENSION)
    doc_id = f"doc-test-{uuid.uuid4()}"
    proche = Chunk(
        id=uuid.uuid4(), doc_id=doc_id, content="préavis de trois mois",
        embedding=_vecteur(0), metadata={},
    )
    loin = Chunk(
        id=uuid.uuid4(), doc_id=doc_id, content="charges de copropriété",
        embedding=_vecteur(1), metadata={},
    )
    store.upsert([proche, loin])

    resultats = store.search(_vecteur(0), k=1)

    assert resultats[0].id == proche.id
    assert resultats[0].score == pytest.approx(1.0, abs=1e-6)

    store.delete_by_doc(doc_id)


def test_dimension_incoherente_leve_une_erreur_a_la_construction():
    with pytest.raises(ValueError):
        VectorStore(DSN, dimension=42)


def test_upsert_rejette_un_embedding_de_mauvaise_longueur():
    store = VectorStore(DSN, DIMENSION)
    chunk = Chunk(id=uuid.uuid4(), doc_id="doc-x", content="texte", embedding=[0.0, 1.0], metadata={})
    with pytest.raises(ValueError):
        store.upsert([chunk])


def test_tous_renvoie_les_morceaux_ranges_sans_leur_vecteur():
    store = VectorStore(DSN, DIMENSION)
    doc_id = f"doc-test-{uuid.uuid4()}"
    chunk = Chunk(
        id=uuid.uuid4(), doc_id=doc_id, content="préavis de trois mois",
        embedding=_vecteur(0), metadata={"article": "15"},
    )
    store.upsert([chunk])

    morceaux = {m.id: m for m in store.tous()}

    assert morceaux[chunk.id].doc_id == doc_id
    assert morceaux[chunk.id].content == "préavis de trois mois"
    assert morceaux[chunk.id].metadata == {"article": "15"}

    store.delete_by_doc(doc_id)


def test_documents_resume_chaque_document_indexe():
    store = VectorStore(DSN, DIMENSION)
    doc_id = f"doc-test-{uuid.uuid4()}"
    store.upsert([
        Chunk(
            id=uuid.uuid4(), doc_id=doc_id, content=f"article {i}",
            embedding=_vecteur(i), metadata={"titre": "Loi test"},
        )
        for i in range(2)
    ])

    documents = {d.doc_id: d for d in store.documents()}

    assert documents[doc_id].morceaux == 2
    assert documents[doc_id].titre == "Loi test"

    store.delete_by_doc(doc_id)
    assert doc_id not in {d.doc_id for d in store.documents()}


def test_search_retrouve_un_article_par_sa_fenetre():
    store = VectorStore(DSN, DIMENSION)
    doc_id = f"doc-test-{uuid.uuid4()}"
    long = Chunk(
        id=uuid.uuid4(), doc_id=doc_id, content="article long",
        embedding=_vecteur(1), metadata={}, fenetres=[_vecteur(2), _vecteur(0)],
    )
    store.upsert([long])

    resultats = store.search(_vecteur(0), k=1)

    assert resultats[0].id == long.id
    assert resultats[0].score == pytest.approx(1.0, abs=1e-6)

    store.delete_by_doc(doc_id)


def test_upsert_remplace_les_fenetres_d_un_article():
    store = VectorStore(DSN, DIMENSION)
    doc_id = f"doc-test-{uuid.uuid4()}"
    identifiant = uuid.uuid4()
    store.upsert([Chunk(id=identifiant, doc_id=doc_id, content="v1", embedding=_vecteur(1),
                        fenetres=[_vecteur(0)])])
    store.upsert([Chunk(id=identifiant, doc_id=doc_id, content="v2", embedding=_vecteur(1))])

    # L'ancienne fenêtre (identique à la requête) ne doit plus le faire remonter.
    resultats = store.search(_vecteur(0), k=5)

    assert all(r.score < 0.5 for r in resultats if r.id == identifiant)

    store.delete_by_doc(doc_id)
