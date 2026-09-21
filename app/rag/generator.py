"""Génération de la réponse à partir des articles récupérés.

La recherche (retriever.py) fournit une liste d'articles jugés pertinents pour
la question. Ce module demande à un modèle de langage de rédiger une réponse
en français, structurée, fondée uniquement sur ces articles, et de citer pour
chaque affirmation le passage exact qui la soutient.

Trois garde-fous encadrent cette génération (cf. CLAUDE.md) :
  - étage 1 (retrieval, dans retriever.py) : hors du périmètre de ce fichier ;
  - le modèle doit lui-même refuser si les articles ne contiennent pas de quoi
    répondre, plutôt que de compléter avec des connaissances extérieures ;
  - étage 2 (verifier_citations) : chaque passage cité est vérifié contre le
    texte réel de l'article — une citation qu'on ne retrouve pas dans une
    source réellement récupérée invalide la réponse, qui bascule en refus ;
  - étage 2, suite (verifier_fidelite) : même une citation authentique ne
    garantit pas que le texte rédigé par le LLM colle à ce qu'elle dit (le
    LLM peut broder à côté). La réponse est décomposée en affirmations, et
    chacune est confrontée aux articles cités par un modèle NLI, avec repli
    sur un juge LLM pour les cas où le score NLI seul ne tranche pas.

Contenu du fichier :
- Citation : un passage d'article invoqué à l'appui de la réponse.
- ReponseGeneree : la réponse rédigée, ou un refus motivé, avec ses citations.
- LLM : contrat commun aux fournisseurs de modèle de langage.
- GroqLLM : implémentation Groq (gratuite, par défaut).
- get_llm : sélectionne l'implémentation selon .env.
- generer_reponse : construit la consigne, appelle le modèle, lit sa réponse.
- verifier_citations : confronte chaque passage cité au texte des articles.
- verifier_fidelite : confronte chaque affirmation de la réponse aux articles
  cités, via un modèle NLI (garde-fou étage 2).
"""
from __future__ import annotations

import json
import re
import unicodedata
from abc import ABC, abstractmethod
from difflib import SequenceMatcher
from typing import Protocol

from pydantic import BaseModel, Field, ValidationError

from app.config import Settings

CONSIGNE = """Tu es un assistant documentaire pour des professionnels de l'immobilier \
(agents, gestionnaires, syndics). Tu réponds à des questions de réglementation.

Règles impératives :
- Réponds UNIQUEMENT à partir des articles fournis ci-dessous. N'utilise aucune \
connaissance extérieure, même si tu penses la connaître.
- Si les articles ne permettent pas de répondre, refuse : il vaut mieux ne pas \
répondre qu'inventer, dans un domaine réglementé.
- Un article signalé « ABROGÉ » ne doit pas être présenté comme le droit en vigueur.
- Réponds en français, dans une langue claire, en structurant la réponse quand \
c'est utile (cas, étapes, conditions).
- Pour chaque affirmation, cite le passage EXACT qui la fonde : recopie mot pour \
mot une phrase ou une portion de phrase de l'article, pas l'article entier.

Réponds au format JSON, sans texte autour :
{"refus": false, "reponse": "<ta réponse en français>", "citations": [{"doc_id": \
"<identifiant du document>", "article": "<numéro>", "passage": "<phrase exacte \
recopiée de l'article>"}]}

En cas de refus :
{"refus": true, "motif": "<raison courte>", "reponse": "", "citations": []}"""


class Citation(BaseModel):
    """Un passage d'article cité à l'appui de la réponse.

    passage est censé être recopié mot pour mot depuis l'article ; l'étape de
    vérification confirmera ou non cette correspondance.
    """

    doc_id: str
    article: str
    passage: str


class ReponseGeneree(BaseModel):
    """Le résultat de la génération.

    refus vaut True lorsque le modèle estime que les articles ne permettent pas
    de répondre ; motif en donne alors la raison et reponse reste vide. Sinon,
    reponse contient le texte rédigé et citations la liste des passages
    invoqués.
    """

    refus: bool
    motif: str | None = None
    reponse: str = ""
    citations: list[Citation] = Field(default_factory=list)


class _ResultatRecherche(Protocol):
    """Partie d'un résultat de recherche utilisée ici : le texte de l'article et
    ses repères."""

    doc_id: str
    content: str
    metadata: dict


class LLM(ABC):
    """Contrat commun aux fournisseurs de modèle de langage."""

    @abstractmethod
    def generer(self, consigne: str, question: str) -> str:
        """Renvoie la réponse brute du modèle (chaîne JSON attendue)."""


class GroqLLM(LLM):
    """Implémentation Groq : gratuite, retenue par défaut."""

    def __init__(self, api_key: str, model: str):
        from groq import Groq

        self._client = Groq(api_key=api_key)
        self._model = model

    def generer(self, consigne: str, question: str) -> str:
        reponse = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": consigne},
                {"role": "user", "content": question},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        return reponse.choices[0].message.content


def get_llm(settings: Settings) -> LLM:
    """Sélectionne l'implémentation selon le réglage LLM_PROVIDER de .env."""
    if settings.llm_provider == "groq":
        if not settings.groq_api_key:
            raise ValueError("GROQ_API_KEY manquant dans .env")
        return GroqLLM(settings.groq_api_key, settings.llm_model)
    raise ValueError(f"Fournisseur LLM inconnu : {settings.llm_provider}")


def _bloc_articles(resultats: list[_ResultatRecherche]) -> str:
    """Met en forme les articles récupérés pour la consigne."""
    blocs: list[str] = []
    for i, r in enumerate(resultats, start=1):
        article = r.metadata.get("article", "?")
        marque = " — ABROGÉ" if r.metadata.get("abroge") else ""
        blocs.append(
            f"[{i}] Document : {r.doc_id} — Article {article}{marque}\n{r.content}"
        )
    return "\n\n".join(blocs)


def _lire_reponse(brut: str) -> ReponseGeneree:
    """Transforme la réponse JSON du modèle en ReponseGeneree.

    Toute réponse illisible (JSON invalide, champs manquants) est traitée comme
    un refus : une sortie non conforme ne doit pas être présentée à
    l'utilisateur comme une réponse fondée.
    """
    try:
        donnees = json.loads(brut)
        return ReponseGeneree.model_validate(donnees)
    except (json.JSONDecodeError, ValidationError, TypeError):
        return ReponseGeneree(
            refus=True, motif="réponse du modèle illisible ou non conforme"
        )


def generer_reponse(
    question: str, resultats: list[_ResultatRecherche], llm: LLM
) -> ReponseGeneree:
    """Rédige une réponse à la question à partir des articles récupérés.

    Renvoie un refus immédiat si aucun article n'a été récupéré : il n'y a alors
    aucune source sur laquelle fonder une réponse.
    """
    if not resultats:
        return ReponseGeneree(refus=True, motif="aucun passage pertinent trouvé")

    message = f"Question : {question}\n\nArticles disponibles :\n\n{_bloc_articles(resultats)}"
    brut = llm.generer(CONSIGNE, message)
    return _lire_reponse(brut)


# --- Vérification des citations --------------------------------------------

# Part minimale du passage cité qui doit se retrouver telle quelle dans
# l'article pour qu'un rapprochement approché soit accepté (léger écart de
# ponctuation ou de mot toléré, invention non).
SEUIL_CITATION = 0.85

# En dessous de cette longueur (après normalisation), seul un rapprochement
# exact est retenu : un fragment court se retrouverait trop facilement par
# hasard dans un long article.
_LONGUEUR_MIN_APPROX = 15


def _normaliser(texte: str) -> str:
    """Ramène un texte à une forme comparable : sans accents, en minuscules,
    ponctuation et espaces multiples réduits à une espace simple."""
    sans_accents = "".join(
        c for c in unicodedata.normalize("NFKD", texte) if not unicodedata.combining(c)
    )
    return re.sub(r"[^a-z0-9]+", " ", sans_accents.lower()).strip()


def _passage_present(passage: str, contenu: str, seuil: float) -> bool:
    """Indique si le passage figure dans le contenu de l'article.

    Un passage présent tel quel (après normalisation) est accepté. Sinon, pour
    un passage assez long, un écart mineur est toléré tant que la plus longue
    portion commune couvre au moins `seuil` du passage.
    """
    p = _normaliser(passage)
    c = _normaliser(contenu)
    if not p:
        return False
    if p in c:
        return True
    if len(p) < _LONGUEUR_MIN_APPROX:
        return False
    bloc = SequenceMatcher(None, p, c, autojunk=False).find_longest_match(
        0, len(p), 0, len(c)
    )
    return bloc.size / len(p) >= seuil


def _articles_candidats(
    citation: Citation, resultats: list[_ResultatRecherche]
) -> list[_ResultatRecherche]:
    """Articles où chercher le passage : celui annoncé par la citation en
    priorité, sinon tous les articles récupérés (le modèle a pu mal reporter le
    numéro ou le document)."""
    exacts = [
        r
        for r in resultats
        if r.doc_id == citation.doc_id
        and str(r.metadata.get("article")) == str(citation.article)
    ]
    return exacts or list(resultats)


def verifier_citations(
    reponse: ReponseGeneree,
    resultats: list[_ResultatRecherche],
    seuil: float = SEUIL_CITATION,
) -> ReponseGeneree:
    """Confronte chaque passage cité au texte des articles réellement récupérés.

    Un refus est renvoyé tel quel. Une réponse sans citation, ou dont au moins
    une citation ne se retrouve dans aucune source, bascule en refus : dans un
    domaine réglementé, une citation inventée invalide la réponse entière.
    """
    if reponse.refus:
        return reponse

    if not reponse.citations:
        return ReponseGeneree(refus=True, motif="réponse sans citation")

    introuvables = [
        citation
        for citation in reponse.citations
        if not any(
            _passage_present(citation.passage, r.content, seuil)
            for r in _articles_candidats(citation, resultats)
        )
    ]
    if introuvables:
        exemple = introuvables[0].passage
        return ReponseGeneree(
            refus=True,
            motif=f"citation introuvable dans les sources : « {exemple} »",
        )

    return reponse


# --- Garde-fou étage 2 : fidélité de la réponse aux articles cités ----------

# Découpe une réponse en phrases (approximation d'« affirmations atomiques") :
# un point, un point d'exclamation ou d'interrogation suivi d'une espace.
_SEPARATEUR_PHRASE = re.compile(r"(?<=[.!?])\s+")

CONSIGNE_JUGE_FIDELITE = """Tu vérifies si une affirmation est fondée sur un texte source, dans le \
cadre d'un assistant juridique qui ne doit rien avancer que le texte ne dise \
vraiment.

Réponds UNIQUEMENT par ce JSON, sans texte autour :
{"fondee": true} si le texte source implique l'affirmation (dit la même chose, \
même reformulée) ; {"fondee": false} sinon (l'affirmation va au-delà du texte, \
le contredit, ou n'a pas de rapport avec lui)."""


class _NLI(Protocol):
    """Partie d'un fournisseur NLI utilisée ici : le score d'implication entre
    un texte source et une affirmation."""

    def score_implication(self, premisse: str, hypothese: str) -> float: ...


def _decouper_en_affirmations(texte: str) -> list[str]:
    """Découpe une réponse en affirmations, une par phrase.

    Une décomposition par phrase reste une approximation d'une décomposition
    en affirmations atomiques (une phrase peut porter plusieurs faits, cf.
    RAGAS faithfulness) — suffisante pour un garde-fou rapide en ligne, sans
    appel de modèle supplémentaire pour la décomposition elle-même.
    """
    return [p.strip() for p in _SEPARATEUR_PHRASE.split(texte) if p.strip()]


def _contextes_cites(
    reponse: ReponseGeneree, resultats: list[_ResultatRecherche]
) -> list[str]:
    """Textes des articles cités par la réponse, à confronter aux affirmations.

    Restreint aux articles réellement cités : c'est leur contenu, et non celui
    de tous les articles récupérés, que la réponse est censée refléter. Repli
    sur tous les articles récupérés si aucune citation ne correspond (ne
    devrait pas arriver après verifier_citations, mais garde la fonction
    utilisable seule).
    """
    cites = {(c.doc_id, str(c.article)) for c in reponse.citations}
    contextes = [
        r.content
        for r in resultats
        if (r.doc_id, str(r.metadata.get("article"))) in cites
    ]
    return contextes or [r.content for r in resultats]


def _juger_fidelite(affirmation: str, contexte: str, llm: LLM) -> bool:
    """Repli pour les cas ambigus : demande à un LLM si le texte source fonde
    l'affirmation. Une réponse illisible est traitée comme non fondée, par
    prudence dans un domaine réglementé."""
    message = f"Texte source :\n{contexte}\n\nAffirmation à vérifier :\n{affirmation}"
    brut = llm.generer(CONSIGNE_JUGE_FIDELITE, message)
    try:
        return bool(json.loads(brut).get("fondee", False))
    except (json.JSONDecodeError, AttributeError, TypeError):
        return False


def verifier_fidelite(
    reponse: ReponseGeneree,
    resultats: list[_ResultatRecherche],
    nli: _NLI,
    seuil_haut: float,
    seuil_bas: float,
    llm: LLM | None = None,
) -> ReponseGeneree:
    """Garde-fou étage 2 (suite) : la réponse rédigée colle-t-elle aux articles
    qu'elle cite ?

    Une bonne citation (vérifiée par verifier_citations) ne garantit pas que le
    texte rédigé autour reste fidèle à ce qu'elle dit : le LLM peut broder à
    côté même avec la bonne source sous les yeux. La réponse est découpée en
    affirmations, chacune confrontée par un modèle NLI aux articles cités.

    Pour chaque affirmation, le score d'implication (0 à 1) tranche : au-dessus
    de seuil_haut, fondée ; en dessous de seuil_bas, non fondée, refus direct.
    Entre les deux (score ambigu, le NLI seul ne suffit pas), llm sert de juge
    de repli si fourni ; sans llm, un score ambigu est traité comme non fondé
    (prudence par défaut, domaine réglementé).

    Un refus est transmis tel quel. Une réponse sans texte (cas dégénéré déjà
    exclu par ailleurs) n'a aucune affirmation à vérifier et passe telle quelle.
    """
    if reponse.refus:
        return reponse

    affirmations = _decouper_en_affirmations(reponse.reponse)
    if not affirmations:
        return reponse

    contextes = _contextes_cites(reponse, resultats)

    for affirmation in affirmations:
        score = max((nli.score_implication(c, affirmation) for c in contextes), default=0.0)
        if score >= seuil_haut:
            continue
        if score <= seuil_bas:
            fondee = False
        else:
            fondee = llm is not None and any(
                _juger_fidelite(affirmation, c, llm) for c in contextes
            )
        if not fondee:
            return ReponseGeneree(
                refus=True,
                motif=f"affirmation non fondée sur les sources : « {affirmation} »",
            )

    return reponse
