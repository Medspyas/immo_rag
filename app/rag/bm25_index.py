"""Recherche par mots-clés, fondée sur l'algorithme BM25.

Le projet combine deux recherches. La recherche par le sens (vector_store.py)
retrouve un morceau qui traite du même sujet avec d'autres mots. La recherche
par mots-clés retrouve un morceau qui contient exactement les termes de la
question, par exemple « préavis » ou « L. 111-1 ». Le module retriever.py
interroge les deux et réunit leurs résultats.

Le texte est d'abord ramené à une liste de mots comparables : accents
supprimés, passage en minuscules, mots-outils retirés, pluriels réduits au
singulier. Les références légales (« L. 111-1 ») sont conservées d'un seul
tenant : elles forment une clé de recherche à part entière.

L'index est conservé en mémoire vive. Pour un corpus plus volumineux que la
mémoire disponible, le Full-Text Search de PostgreSQL est une alternative.

Contenu du fichier :
- tokeniser_fr : découpe un texte français en mots pour l'index.
- BM25Document : un morceau à indexer (texte et repères, sans vecteur).
- BM25Result : un morceau retrouvé, avec son score.
- BM25Index : construit l'index et répond aux recherches.
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from uuid import UUID

from pydantic import BaseModel, Field

# Une référence légale : lettre optionnelle (L/R/D/A) puis un numéro à tranches
# ("L. 111-1", "R123-4"), ou un numéro composé seul ("89-462", "6-1"). Repérée
# d'un bloc pour être mise de côté avant le découpage normal.
_REF_LEGALE = re.compile(
    r"\b(?:[LRDA]\.?\s*\d+(?:-\d+)*|\d+-\d+(?:-\d+)*)\b",
    re.IGNORECASE,
)

_MOT = re.compile(r"[a-z0-9]+")

# « plus-value » et « moins-value » forment un seul mot. Découpés, « plus » et
# « moins » partaient avec les mots-outils et « plus-value » se réduisait à
# « value ». Toutes les graphies (« plus value », « plus-valus ») donnent la
# même forme, fautes de frappe comprises (constaté le 2026-10-14 : « les règles
# des taxes plus valus » ne retrouvait aucun article).
_PLUS_MOINS_VALUE = re.compile(r"\b(plus|moins)[\s-]*valu[a-z]*")

# Mots-outils français courants (sans accents : la comparaison se fait après
# normalisation). Liste volontairement courte : trop large, elle retirerait des
# mots utiles en droit ("plus", "sans", "contre").
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
    """Retire les accents : "éàïôç" devient "eaioc"."""
    decompose = unicodedata.normalize("NFKD", texte)
    return "".join(c for c in decompose if not unicodedata.combining(c))


@lru_cache(maxsize=1)
def _stemmer():
    """Ramène les mots français à leur racine ; chargé une seule fois."""
    import snowballstemmer

    return snowballstemmer.stemmer("french")


def tokeniser_fr(texte: str) -> list[str]:
    """Découpe un texte français en mots comparables pour l'index.

    Les références légales ("L. 111-1") viennent d'abord, d'un bloc, puis les
    mots courants sans accents, sans mots-outils et ramenés à leur racine.
    """
    refs = [
        re.sub(r"\s+", "", correspondance.group(0)).lower()
        for correspondance in _REF_LEGALE.finditer(texte)
    ]

    # Références retirées avant le découpage normal : sinon le "111" et le "1"
    # de "L. 111-1" reviendraient comme des nombres isolés parasites.
    texte_sans_refs = _REF_LEGALE.sub(" ", texte)
    texte_normalise = _PLUS_MOINS_VALUE.sub(
        lambda m: m.group(1) + "value", _sans_accents(texte_sans_refs).lower()
    )
    mots = _MOT.findall(texte_normalise)

    stemmer = _stemmer()
    tokens = [
        _racine_commune(stemmer.stemWord(mot))
        for mot in mots
        if len(mot) > 1 and mot not in _STOPWORDS_FR
    ]
    return refs + tokens


def _racine_commune(racine: str) -> str:
    """Rapproche les racines d'un verbe et de son nom, que Snowball sépare.

    « révoque » donne "revoqu", « révocation » donne "revoc" : la question
    (verbe) ne retrouvait pas l'article (nom). Même écart pour « convoque » /
    « convocation » et « résilie » / « résiliation » ("resil" / "resili").
    """
    if racine.endswith("qu"):
        return racine[:-2] + "c"
    if racine.endswith("ili"):
        return racine[:-1]
    return racine


class BM25Document(BaseModel):
    """Un morceau à indexer : le texte et de quoi le retrouver ensuite, sans
    vecteur."""

    id: UUID
    doc_id: str = Field(min_length=1)
    content: str = Field(min_length=1)
    metadata: dict = Field(default_factory=dict)
    # Texte indexé en plus du contenu, jamais renvoyé : les questions auxquelles
    # l'article répond (doc2query, cf. questions_articles.py). Il aide à
    # retrouver l'article, mais n'est pas du texte de loi : il ne doit atteindre
    # ni le reclasseur, ni le LLM, ni la vérification des citations.
    complement: str = ""


class BM25Result(BaseModel):
    id: UUID
    doc_id: str
    content: str
    metadata: dict
    score: float  # score brut (≥ 0, sans maximum) : sert au classement, pas de seuil


class BM25Index:
    """Index construit en mémoire à partir des morceaux.

    Le corpus est figé à la construction : après une nouvelle ingestion, un
    nouvel index doit être créé. Convient tant que le texte tient en mémoire
    (quelques Mo) ; au-delà, une bascule vers PostgreSQL s'impose (voir en-tête
    du fichier).
    """

    def __init__(self, documents: list[BM25Document]):
        from rank_bm25 import BM25Okapi

        self._documents = list(documents)
        corpus_tokens = [tokeniser_fr(d.content + "\n" + d.complement) for d in self._documents]
        # rank-bm25 refuse un corpus vide : il reste à None et search renvoie
        # une liste vide.
        self._bm25 = BM25Okapi(corpus_tokens) if corpus_tokens else None

    def search(self, query: str, k: int = 10) -> list[BM25Result]:
        """Les k morceaux au meilleur score pour la question.

        Les morceaux sans aucun mot commun avec la question (score nul) sont
        écartés : les garder ferait croire à une pertinence inexistante.
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
