"""
Stockage et recherche des chunks dans pgvector.

Analogie : pgvector ajoute à Postgres la capacité de chercher "les vecteurs les
plus proches" d'un vecteur donné — comme chercher les adresses les plus proches
sur une carte, mais dans l'espace des embeddings plutôt que géographique. C'est
la moitié "dense" de la recherche hybride (l'autre moitié est bm25_index.py) ;
les deux résultats sont ensuite combinés par fusion RRF dans retriever.py.
"""
from __future__ import annotations

import json
from uuid import UUID

import psycopg
from pgvector.psycopg import register_vector
from pydantic import BaseModel, Field, field_validator


class Chunk(BaseModel):
    id: UUID
    doc_id: str = Field(min_length=1)
    content: str = Field(min_length=1)
    embedding: list[float]
    metadata: dict = Field(default_factory=dict)

    @field_validator("embedding")
    @classmethod
    def _embedding_non_vide(cls, v: list[float]) -> list[float]:
        if not v:
            raise ValueError("embedding ne peut pas être vide")
        return v


class SearchResult(BaseModel):
    id: UUID
    doc_id: str
    content: str
    metadata: dict
    score: float  # similarité cosinus : 1.0 = identique, 0.0 = aucun rapport


class VectorStore:
    def __init__(self, dsn: str, dimension: int):
        self._dsn = dsn
        self._dimension = dimension
        self._verifier_dimension()

    def _connect(self) -> psycopg.Connection:
        conn = psycopg.connect(self._dsn)
        register_vector(conn)
        return conn

    def _verifier_dimension(self) -> None:
        """Vérifie au démarrage que la dimension configurée (EMBEDDINGS_DIMENSION
        dans .env) correspond bien à la colonne `embedding` déjà créée en base par
        init.sql — évite une incohérence silencieuse entre les deux, détectée sinon
        seulement au moment (tardif et confus) du premier insert."""
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT atttypmod FROM pg_attribute
                WHERE attrelid = 'chunks'::regclass AND attname = 'embedding'
                """
            )
            row = cur.fetchone()
        if row is not None and row[0] != self._dimension:
            raise ValueError(
                f"Dimension configurée ({self._dimension}) différente de la "
                f"colonne 'embedding' en base ({row[0]}) — vérifie "
                f"EMBEDDINGS_DIMENSION dans .env par rapport à init.sql."
            )

    def _verifier_longueur(self, embedding: list[float]) -> None:
        if len(embedding) != self._dimension:
            raise ValueError(
                f"Embedding de longueur {len(embedding)}, attendu {self._dimension}."
            )

    def upsert(self, chunks: list[Chunk]) -> None:
        """Insère ou remplace des chunks (par id) — un chunk ré-ingéré écrase l'ancien."""
        if not chunks:
            return
        for c in chunks:
            self._verifier_longueur(c.embedding)
        with self._connect() as conn, conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO chunks (id, doc_id, content, embedding, metadata)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    content = EXCLUDED.content,
                    embedding = EXCLUDED.embedding,
                    metadata = EXCLUDED.metadata
                """,
                [
                    (c.id, c.doc_id, c.content, c.embedding, json.dumps(c.metadata))
                    for c in chunks
                ],
            )
            conn.commit()

    def delete_by_doc(self, doc_id: str) -> None:
        """Supprime tous les chunks d'un document — nécessaire avant de ré-ingérer
        un PDF mis à jour dont le nombre de chunks a changé (sinon les anciens
        chunks en trop restent orphelins en base indéfiniment)."""
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM chunks WHERE doc_id = %s", (doc_id,))
            conn.commit()

    def search(self, query_embedding: list[float], k: int = 10) -> list[SearchResult]:
        """kNN par similarité cosinus.

        `<=>` est l'opérateur de distance cosinus de pgvector (0 = identique,
        2 = opposé) ; on renvoie 1 - distance pour avoir un score de similarité
        plus intuitif (1 = identique, proche de 0 = sans rapport).

        `::vector` est nécessaire : psycopg envoie une liste Python comme un
        tableau Postgres générique (`double precision[]`), que `<=>` ne sait
        pas comparer sans cast explicite (contrairement à un INSERT, où le
        type de la colonne force la conversion implicitement).
        """
        self._verifier_longueur(query_embedding)
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, doc_id, content, metadata, 1 - (embedding <=> %s::vector) AS score
                FROM chunks
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (query_embedding, query_embedding, k),
            )
            rows = cur.fetchall()
        return [
            SearchResult(id=r[0], doc_id=r[1], content=r[2], metadata=r[3], score=r[4])
            for r in rows
        ]
