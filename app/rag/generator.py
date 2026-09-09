"""Génération de la réponse à partir des articles récupérés.

La recherche (retriever.py) fournit une liste d'articles jugés pertinents pour
la question. Ce module demande à un modèle de langage de rédiger une réponse
en français, structurée, fondée uniquement sur ces articles, et de citer pour
chaque affirmation le passage exact qui la soutient.

Deux garde-fous encadrent cette génération (cf. CLAUDE.md) :
  - le modèle doit refuser si les articles ne contiennent pas de quoi répondre,
    plutôt que de compléter avec des connaissances extérieures ;
  - chaque passage cité est ensuite vérifié contre le texte réel de l'article
    (verifier_citations) : une citation qu'on ne retrouve pas dans une source
    réellement récupérée invalide la réponse, qui bascule alors en refus.

Contenu du fichier :
- Citation : un passage d'article invoqué à l'appui de la réponse.
- ReponseGeneree : la réponse rédigée, ou un refus motivé, avec ses citations.
- LLM : contrat commun aux fournisseurs de modèle de langage.
- GroqLLM : implémentation Groq (gratuite, par défaut).
- get_llm : sélectionne l'implémentation selon .env.
- generer_reponse : construit la consigne, appelle le modèle, lit sa réponse.
- verifier_citations : confronte chaque passage cité au texte des articles.
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
