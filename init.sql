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

-- Index pour rechercher et supprimer les chunks par document.
CREATE INDEX IF NOT EXISTS chunks_doc_id_idx ON chunks (doc_id);

-- Index HNSW : bon compromis vitesse/précision pour la similarité cosinus.
-- La dimension (768, celle du modèle local par défaut) doit correspondre à
-- EMBEDDINGS_DIMENSION dans .env ; un écart est détecté au démarrage par
-- VectorStore (vector_store.py, _verifier_dimension).
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);

-- updated_at est rafraîchi à chaque ré-ingestion d'un chunk.
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
