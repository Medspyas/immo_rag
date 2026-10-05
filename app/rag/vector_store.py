"""Rangement et recherche des morceaux par le sens (base pgvector).

L'extension pgvector ajoute à PostgreSQL la recherche des vecteurs les plus
proches d'un vecteur donné, dans l'espace de sens produit par les embeddings.
Ce module fournit la composante "sens" de la recherche ; la composante
"mots-clés" est dans bm25_index.py, et retriever.py combine les deux.

Contenu du fichier :
- Chunk : un morceau prêt à être rangé (texte, vecteur, métadonnées).
- SearchResult : un morceau retrouvé, avec son score de proximité.
- MorceauStocke : un morceau rangé, sans son vecteur (cf. VectorStore.tous).
- DocumentIndexe : résumé d'un document en base (cf. VectorStore.documents).
- VectorStore : ajout, remplacement, suppression et recherche des morceaux,
  liste des documents.
"""
from __future__ import annotations

import json
from datetime import datetime
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
    score: float  # proximité de sens : 1.0 = identique, 0.0 = aucun rapport


class MorceauStocke(BaseModel):
    """Un morceau rangé en base, sans son vecteur : sert à reconstruire l'index
    BM25 (bm25_index.py), qui n'a besoin que du texte."""

    id: UUID
    doc_id: str
    content: str
    metadata: dict


class DocumentIndexe(BaseModel):
    """Résumé d'un document présent en base : ce que la page « documents » du
    front affiche, sans charger le texte des morceaux."""

    doc_id: str
    titre: str | None
    morceaux: int
    mis_a_jour: datetime


class VectorStore:
    def __init__(self, dsn: str, dimension: int, connect_timeout: int = 5):
        self._dsn = dsn
        self._dimension = dimension
        # Sans borne, une base injoignable (port fermé, réseau filtré) bloque
        # l'appelant plusieurs dizaines de secondes avant d'échouer : trop long
        # pour une route web censée dégrader vite (cf. app/main.py).
        self._connect_timeout = connect_timeout
        self._verifier_dimension()

    def _connect(self) -> psycopg.Connection:
        conn = psycopg.connect(self._dsn, connect_timeout=self._connect_timeout)
        register_vector(conn)
        return conn

    def _verifier_dimension(self) -> None:
        """Vérifie au démarrage que la dimension configurée dans .env
        correspond à celle de la colonne en base. Un écart non détecté ici
        n'apparaîtrait qu'au premier ajout, avec un message peu clair."""
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
                f"colonne 'embedding' en base ({row[0]}) — vérifier "
                f"EMBEDDINGS_DIMENSION dans .env par rapport à init.sql."
            )

    def _verifier_longueur(self, embedding: list[float]) -> None:
        if len(embedding) != self._dimension:
            raise ValueError(
                f"Embedding de longueur {len(embedding)}, attendu {self._dimension}."
            )

    def upsert(self, chunks: list[Chunk]) -> None:
        """Ajoute des morceaux, ou les remplace si l'identifiant existe déjà."""
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

    def delete_by_doc(self, doc_id: str) -> int:
        """Supprime tous les morceaux d'un document et renvoie leur nombre (0 :
        document inconnu). Opération à effectuer avant la réingestion d'une
        version à jour, faute de quoi d'anciens morceaux subsistent en base."""
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM chunks WHERE doc_id = %s", (doc_id,))
            supprimes = cur.rowcount
            conn.commit()
        return supprimes

    def search(self, query_embedding: list[float], k: int = 10) -> list[SearchResult]:
        """Renvoie les k morceaux les plus proches du vecteur donné.

        Le score renvoyé vaut 1 - distance, plus lisible : 1 = identique,
        0 = aucun rapport.

        La conversion `::vector` dans la requête est nécessaire : sans elle,
        PostgreSQL reçoit une simple liste de nombres et ne peut pas la
        comparer.
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

    def tous(self) -> list[MorceauStocke]:
        """Renvoie tous les morceaux rangés, sans leur vecteur.

        Sert à reconstruire l'index BM25 (en mémoire, cf. bm25_index.py) au
        démarrage de l'application : la recherche par mots-clés n'a pas besoin
        du vecteur, seulement du texte.
        """
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT id, doc_id, content, metadata FROM chunks")
            rows = cur.fetchall()
        return [
            MorceauStocke(id=r[0], doc_id=r[1], content=r[2], metadata=r[3]) for r in rows
        ]

    def documents(self) -> list[DocumentIndexe]:
        """Liste les documents indexés, avec leur nombre de morceaux et la date
        de la dernière ingestion. Le titre vient des métadonnées posées au
        découpage (chunking.titre_document), absent pour un texte sans titre."""
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT doc_id, max(metadata->>'titre'), count(*), max(updated_at)
                FROM chunks GROUP BY doc_id ORDER BY doc_id
                """
            )
            rows = cur.fetchall()
        return [
            DocumentIndexe(doc_id=r[0], titre=r[1], morceaux=r[2], mis_a_jour=r[3])
            for r in rows
        ]
