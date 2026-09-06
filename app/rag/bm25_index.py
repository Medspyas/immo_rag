"""
Index sparse BM25 : la moitié "mots-clés" de la recherche hybride.

Analogie : l'embedding (vector_store.py) cherche par *sens* — il retrouve un
chunk qui parle de la même chose même avec d'autres mots. BM25 cherche par
*mots exacts* — il retrouve un chunk parce qu'il contient littéralement
"préavis" ou "L. 111-1". Les deux se complètent : le sens rattrape les
reformulations, les mots-clés rattrapent les termes techniques et les
références d'articles qu'un embedding a tendance à "lisser". retriever.py
lance les deux et fusionne les classements (RRF).

BM25, en une phrase : un score de pertinence qui monte quand la question et le
chunk partagent des mots rares, et qui sature (un mot répété 10 fois ne compte
pas 10 fois) en tenant compte de la longueur du chunk.

## Tokenizer français

Un index par mots-clés ne vaut que par sa façon de découper le texte en
"mots". Ici (cf. CLAUDE.md, ligne Sparse) :

- **accents** : "préavis" et "preavis" doivent matcher → on retire les accents.
- **casse** : tout en minuscules.
- **stopwords** : "le", "de", "et"… sont partout, ils n'aident pas à
  discriminer un chunk → retirés.
- **stemming** : "logement" / "logements" / (et racines proches) ramenés à une
  forme commune, pour qu'une question au singulier retrouve un texte au
  pluriel (stemmer Snowball français).
- **références légales** : "L. 111-1", "R. 123-4", "89-462"… sont *préservées
  comme un seul token* (`l.111-1`). Sans ça, le point, l'espace et le tiret les
  casseraient en `l` / `111` / `1` — or c'est exactement la clé de recherche
  qu'un professionnel tape telle quelle.

Option évoquée dans CLAUDE.md : utiliser plutôt le Full-Text Search natif de
Postgres avec la configuration `french` (`to_tsvector('french', …)`), qui gère
déjà accents/stopwords/stemming. On garde rank-bm25 en mémoire pour l'instant :
plus simple à faire évoluer et à mesurer dans l'étude d'ablation, et l'algo
BM25 y est explicite plutôt que caché derrière le classement `ts_rank`.
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from uuid import UUID

from pydantic import BaseModel, Field

# Une référence légale française : lettre de partie optionnelle (L/R/D/A pour
# Législatif, Réglementaire décret, Décret, Arrêté) suivie d'un numéro
# éventuellement composé de plusieurs tranches séparées par des tirets
# ("L. 111-1", "R123-4", "D. 3-1"), OU un numéro composé seul ("89-462" pour un
# numéro de loi, "6-1" pour un article). On capture tout le motif d'un bloc
# pour le sortir du texte avant le découpage normal.
_REF_LEGALE = re.compile(
    r"\b(?:[LRDA]\.?\s*\d+(?:-\d+)*|\d+-\d+(?:-\d+)*)\b",
    re.IGNORECASE,
)

_MOT = re.compile(r"[a-z0-9]+")

# Mots-outils français les plus fréquents (déjà sans accents : la comparaison se
# fait après normalisation). Volontairement court — une liste trop agressive
# retire des mots parfois utiles ("plus", "sans", "contre" ont un sens juridique).
_STOPWORDS_FR = {
    "le", "la", "les", "un", "une", "des", "du", "de", "d", "l", "au", "aux",
    "et", "ou", "ni", "mais", "donc", "or", "car", "que", "qui", "quoi", "dont",
    "ce", "ces", "cette", "cet", "se", "sa", "son", "ses", "leur", "leurs",
    "il", "elle", "ils", "elles", "on", "nous", "vous", "je", "tu", "lui",
    "est", "sont", "etre", "ete", "a", "ai", "as", "ont", "avoir",
    "dans", "sur", "pour", "par", "en", "y", "avec", "comme", "si",
    "ne", "pas", "plus", "moins", "tres", "trop", "aussi", "meme",
    "quel", "quelle", "quels", "quelles", "tout", "tous", "toute", "toutes",
    "ceci", "cela", "ca",
}


def _sans_accents(texte: str) -> str:
    """"éàïôç" → "eaioc". Décompose chaque caractère accentué (NFKD) puis retire
    les diacritiques (catégorie Unicode "Mn", marque non espaçante)."""
    decompose = unicodedata.normalize("NFKD", texte)
    return "".join(c for c in decompose if not unicodedata.combining(c))


@lru_cache(maxsize=1)
def _stemmer():
    """Stemmer Snowball français, chargé une seule fois (léger, pur Python)."""
    import snowballstemmer

    return snowballstemmer.stemmer("french")


def tokeniser_fr(texte: str) -> list[str]:
    """Découpe un texte français en tokens comparables pour BM25.

    Les références légales ("L. 111-1") sont renvoyées en tête, normalisées en
    un seul token (`l.111-1`), avant les mots courants (sans accents, sans
    stopwords, ramenés à leur racine).
    """
    refs = [
        re.sub(r"\s+", "", correspondance.group(0)).lower()
        for correspondance in _REF_LEGALE.finditer(texte)
    ]

    # On efface les références du texte avant le découpage normal : sinon "111"
    # et "1" de "L. 111-1" reviendraient aussi comme tokens numériques parasites.
    texte_sans_refs = _REF_LEGALE.sub(" ", texte)
    mots = _MOT.findall(_sans_accents(texte_sans_refs).lower())

    stemmer = _stemmer()
    tokens = [
        stemmer.stemWord(mot)
        for mot in mots
        if len(mot) > 1 and mot not in _STOPWORDS_FR
    ]
    return refs + tokens


class BM25Document(BaseModel):
    """Un chunk tel qu'indexé par BM25 — pas besoin de l'embedding ici,
    seulement du texte et de quoi retrouver le chunk ensuite."""

    id: UUID
    doc_id: str = Field(min_length=1)
    content: str = Field(min_length=1)
    metadata: dict = Field(default_factory=dict)


class BM25Result(BaseModel):
    id: UUID
    doc_id: str
    content: str
    metadata: dict
    score: float  # score BM25 brut (≥ 0, non borné) — sert au classement, pas de seuil absolu


class BM25Index:
    """Index BM25 en mémoire, reconstruit à partir des chunks au démarrage.

    rank-bm25 fige le corpus à la construction (pas d'ajout incrémental) : pour
    ré-indexer après une ingestion, on recrée un BM25Index. Acceptable tant que
    le corpus tient en mémoire (réglementation publique = quelques Mo de texte) ;
    à revoir si le volume explose (bascule vers Postgres FTS, cf. module).
    """

    def __init__(self, documents: list[BM25Document]):
        from rank_bm25 import BM25Okapi

        self._documents = list(documents)
        corpus_tokens = [tokeniser_fr(d.content) for d in self._documents]
        # BM25Okapi refuse un corpus vide ; on garde None et search renvoie [].
        self._bm25 = BM25Okapi(corpus_tokens) if corpus_tokens else None

    def search(self, query: str, k: int = 10) -> list[BM25Result]:
        """Les k chunks au meilleur score BM25 pour `query`.

        Les chunks à score nul (aucun mot de la question en commun) sont exclus :
        les renvoyer ferait croire à une pertinence inexistante et fausserait la
        fusion RRF en aval.
        """
        if self._bm25 is None:
            return []
        tokens = tokeniser_fr(query)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)
        classement = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        resultats: list[BM25Result] = []
        for i in classement[:k]:
            if scores[i] <= 0:
                break
            doc = self._documents[i]
            resultats.append(
                BM25Result(
                    id=doc.id,
                    doc_id=doc.doc_id,
                    content=doc.content,
                    metadata=doc.metadata,
                    score=float(scores[i]),
                )
            )
        return resultats
