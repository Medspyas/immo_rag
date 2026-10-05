"""Petite interface web pour tester la recherche hybride en local.

Aucune base de données n'est nécessaire : les vecteurs des articles sont gardés
en mémoire (recherche par le sens = produit scalaire numpy) et l'index BM25 est
reconstruit au démarrage. Le but est de juger « à l'œil » les résultats du
retriever avant de brancher pgvector et l'évaluation chiffrée.

Le contextual retrieval est volontairement sauté ici (il demande une clé Groq) :
on indexe le texte brut des articles.

Deux classements sont renvoyés pour comparaison :
  - fusion RRF seule (recherche hybride) ;
  - fusion RRF puis reclassement par le cross-encoder BGE-m3.
La page affiche, pour chaque résultat reclassé, sa position d'origine dans la
fusion, ce qui rend visible l'effet du reclassement.

Lancement :
    python scripts/demo_search.py
puis ouvrir http://127.0.0.1:8001

Le premier lancement calcule les embeddings (quelques minutes sur CPU) ; ils
sont ensuite mis en cache dans .cache/ et les lancements suivants sont
immédiats. La première recherche avec reclassement télécharge le modèle
BAAI/bge-reranker-v2-m3 (~2,3 Go), mis en cache par la suite.
"""
from __future__ import annotations

import hashlib
import sys
import uuid
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))

from app.rag.bm25_index import BM25Document, BM25Index
from app.rag.chunking import decouper_en_articles
from app.rag.embeddings import LocalEmbeddings
from app.rag.reranker import LocalReranker
from app.rag.retriever import HybridRetriever

DOSSIER = RACINE / "data" / "clean"
CACHE = RACINE / ".cache"

# Corpus par défaut : les lois immo « cœur », sans le code des impôts (~5000
# articles, trop lourd pour un premier test). Ajouter des noms de fichiers de
# data/clean/ ici au besoin.
FICHIERS = [
    "loi_06_07_1989.txt",
    "Loi_hoguet.txt",
    "loi-n°65-557_du_10_juillet_1965.txt",
]

# Espace de noms fixe : chaque article reçoit un identifiant stable, dérivé de
# son document et de son numéro, réutilisé par les deux recherches.
NS = uuid.UUID("00000000-0000-0000-0000-000000000001")


class _ResultatDense:
    """Forme minimale d'un résultat attendue par HybridRetriever."""

    def __init__(self, id_: uuid.UUID, doc_id: str, content: str, metadata: dict):
        self.id = id_
        self.doc_id = doc_id
        self.content = content
        self.metadata = metadata


class RechercheDenseMemoire:
    """Recherche par le sens sans base : le vecteur de la question est multiplié
    à la matrice des vecteurs d'articles (déjà normalisés), les meilleurs scores
    sont renvoyés."""

    def __init__(self, chunks: list[dict], matrice: np.ndarray):
        self._chunks = chunks
        self._matrice = matrice

    def search(self, query_embedding: list[float], k: int) -> list[_ResultatDense]:
        q = np.asarray(query_embedding, dtype=np.float32)
        scores = self._matrice @ q
        meilleurs = np.argsort(scores)[::-1][:k]
        return [
            _ResultatDense(
                c["id"], c["doc_id"], c["content"], c["metadata"]
            )
            for c in (self._chunks[i] for i in meilleurs)
        ]


def _charger_chunks() -> list[dict]:
    chunks: list[dict] = []
    for nom in FICHIERS:
        chemin = DOSSIER / nom
        texte = chemin.read_text(encoding="utf-8")
        for rc in decouper_en_articles(texte, chemin.stem):
            chunks.append(
                {
                    "id": uuid.uuid5(NS, f"{rc.doc_id}:{rc.article}"),
                    "doc_id": rc.doc_id,
                    "content": rc.content,
                    "metadata": {"article": rc.article, "abroge": rc.abroge},
                }
            )
    return chunks


def _matrice_embeddings(chunks: list[dict], modele: LocalEmbeddings) -> np.ndarray:
    CACHE.mkdir(exist_ok=True)
    empreinte = hashlib.sha1(
        "".join(c["content"] for c in chunks).encode("utf-8")
    ).hexdigest()[:16]
    fichier = CACHE / f"embeddings_{empreinte}.npy"
    if fichier.exists():
        return np.load(fichier)
    print(f"Calcul des embeddings pour {len(chunks)} articles (une seule fois)…")
    vecteurs = np.asarray(
        modele.embed_documents([c["content"] for c in chunks]), dtype=np.float32
    )
    np.save(fichier, vecteurs)
    return vecteurs


print("Préparation du corpus…")
_chunks = _charger_chunks()
_modele = LocalEmbeddings("intfloat/multilingual-e5-base", 768)
_matrice = _matrice_embeddings(_chunks, _modele)

_dense = RechercheDenseMemoire(_chunks, _matrice)
_bm25 = BM25Index(
    [
        BM25Document(
            id=c["id"], doc_id=c["doc_id"], content=c["content"], metadata=c["metadata"]
        )
        for c in _chunks
    ]
)

# Modèle de reclassement chargé au premier appel seulement (téléchargement
# ~2,3 Go la première fois). Le démarrage du serveur n'attend pas ce chargement.
_reranker = LocalReranker("BAAI/bge-reranker-v2-m3")

# Deux retrievers sur les mêmes recherches : l'un s'arrête à la fusion, l'autre
# reclasse ses vingt premiers candidats.
_retriever_rrf = HybridRetriever(_modele, _dense, _bm25)
_retriever_rerank = HybridRetriever(
    _modele, _dense, _bm25, reranker=_reranker, k_avant_rerank=20
)
print(f"Prêt : {len(_chunks)} articles indexés sur {len(FICHIERS)} fichiers.")


app = FastAPI(title="demo retriever — immo-rag")


@app.get("/api/search")
def rechercher(q: str, k: int = 8, rerank: bool = False):
    # Classement de la fusion seule : sert de référence pour situer chaque
    # résultat reclassé (position d'origine).
    base = _retriever_rrf.search(q, k=20)
    rang_rrf = {r.id: i for i, r in enumerate(base, start=1)}

    resultats = _retriever_rerank.search(q, k=k) if rerank else base[:k]

    return JSONResponse(
        [
            {
                "article": r.metadata.get("article"),
                "doc_id": r.doc_id,
                "abroge": r.metadata.get("abroge", False),
                "score_rrf": round(r.score_rrf, 4),
                "score_rerank": round(r.score_rerank, 4) if r.score_rerank is not None else None,
                "rang_rrf": rang_rrf.get(r.id),
                "rang_dense": r.rang_dense,
                "rang_bm25": r.rang_bm25,
                "content": r.content,
            }
            for r in resultats
        ]
    )


PAGE = """<!doctype html>
<html lang="fr">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>demo retriever — immo-rag</title>
<style>
  :root { color-scheme: light dark; }
  body { font: 15px/1.5 system-ui, sans-serif; margin: 0; background: #f6f6f4; color: #1a1a1a; }
  @media (prefers-color-scheme: dark) { body { background: #17181a; color: #e6e6e6; } }
  header { padding: 20px 24px; border-bottom: 1px solid #0002; }
  h1 { font-size: 16px; margin: 0 0 12px; font-weight: 600; }
  form { display: flex; gap: 8px; max-width: 760px; }
  input[type=text] { flex: 1; padding: 9px 12px; border: 1px solid #0003; border-radius: 8px; background: inherit; color: inherit; }
  input[type=number] { width: 60px; padding: 9px; border: 1px solid #0003; border-radius: 8px; background: inherit; color: inherit; }
  button { padding: 9px 16px; border: 0; border-radius: 8px; background: #2f6f4f; color: #fff; cursor: pointer; }
  main { padding: 16px 24px 48px; max-width: 900px; }
  .vide { opacity: .6; }
  .carte { border: 1px solid #0002; border-radius: 10px; padding: 12px 14px; margin: 10px 0; background: #fff; }
  @media (prefers-color-scheme: dark) { .carte { background: #202224; } }
  .entete { display: flex; flex-wrap: wrap; gap: 8px; align-items: baseline; margin-bottom: 8px; }
  .art { font-weight: 600; }
  .doc { opacity: .6; font-size: 13px; }
  .badge { font-size: 12px; padding: 1px 7px; border-radius: 20px; border: 1px solid #0003; opacity: .85; }
  .abroge { color: #b23; border-color: #b234; }
  .monte { color: #2f6f4f; border-color: #2f6f4f66; }
  .descend { color: #b26b23; border-color: #b26b2366; }
  .texte { white-space: pre-wrap; max-height: 220px; overflow: auto; font-size: 14px; padding: 8px; background: #0001; border-radius: 6px; }
  .options { display: flex; gap: 6px; align-items: center; margin-top: 8px; font-size: 13px; opacity: .8; }
</style>
<header>
  <h1>demo retriever — recherche hybride (sens + mots-clés)</h1>
  <form id="f">
    <input type="text" id="q" placeholder="ex. délai de préavis du locataire" autofocus>
    <input type="number" id="k" value="8" min="1" max="30" title="nombre de résultats">
    <button>Chercher</button>
  </form>
  <label class="options">
    <input type="checkbox" id="rerank"> reclassement BGE-m3 (1re fois : téléchargement ~2,3 Go)
  </label>
</header>
<main id="out"><p class="vide">Pose une question métier pour voir les articles remontés.</p></main>
<script>
const f = document.getElementById('f'), out = document.getElementById('out');
f.addEventListener('submit', async (e) => {
  e.preventDefault();
  const q = document.getElementById('q').value.trim();
  const k = document.getElementById('k').value || 8;
  const rerank = document.getElementById('rerank').checked;
  if (!q) return;
  out.innerHTML = '<p class="vide">Recherche…' + (rerank ? ' (reclassement — le 1er appel télécharge le modèle)' : '') + '</p>';
  const r = await fetch(`/api/search?q=${encodeURIComponent(q)}&k=${k}&rerank=${rerank}`);
  const data = await r.json();
  if (!data.length) { out.innerHTML = '<p class="vide">Aucun résultat.</p>'; return; }
  out.innerHTML = data.map((d, i) => {
    let deplacement = '';
    if (d.score_rerank !== null && d.rang_rrf) {
      const delta = d.rang_rrf - (i + 1);
      const cls = delta > 0 ? 'monte' : (delta < 0 ? 'descend' : '');
      const fleche = delta > 0 ? '▲' : (delta < 0 ? '▼' : '=');
      deplacement = `<span class="badge ${cls}">RRF #${d.rang_rrf} → #${i + 1} ${fleche}</span>`;
    }
    const scoreRerank = d.score_rerank !== null
      ? `<span class="badge">rerank ${d.score_rerank}</span>` : '';
    return `
    <div class="carte">
      <div class="entete">
        <span class="art">${i + 1}. Article ${d.article ?? '?'}</span>
        <span class="doc">${d.doc_id}</span>
        ${d.abroge ? '<span class="badge abroge">abrogé</span>' : ''}
        ${scoreRerank}
        ${deplacement}
        <span class="badge">RRF ${d.score_rrf}</span>
        <span class="badge">sens : ${d.rang_dense ?? '—'}</span>
        <span class="badge">mots-clés : ${d.rang_bm25 ?? '—'}</span>
      </div>
      <div class="texte">${d.content.replace(/</g, '&lt;')}</div>
    </div>`;
  }).join('');
});
</script>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def page() -> str:
    return PAGE


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
