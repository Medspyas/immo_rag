CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    id UUID PRIMARY KEY,
    doc_id TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding VECTOR(768) NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Recherche/suppression par document (ex. ré-ingestion d'un PDF mis à jour).
CREATE INDEX IF NOT EXISTS chunks_doc_id_idx ON chunks (doc_id);

-- HNSW : bon compromis vitesse/précision pour la recherche par similarité cosinus.
-- La dimension (768, celle du modèle local par défaut) doit correspondre à
-- EMBEDDINGS_DIMENSION dans .env — un
-- écart est de toute façon détecté au démarrage par VectorStore (voir
-- vector_store.py, _verifier_dimension) plutôt que de corrompre silencieusement
-- les données.
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);

-- updated_at suit les ré-ingestions (ON CONFLICT DO UPDATE dans vector_store.py).
CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS chunks_set_updated_at ON chunks;
CREATE TRIGGER chunks_set_updated_at
    BEFORE UPDATE ON chunks
    FOR EACH ROW
    EXECUTE FUNCTION set_updated_at();
