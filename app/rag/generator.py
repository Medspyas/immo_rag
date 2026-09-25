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
from app.rag.bm25_index import tokeniser_fr
from app.rag.nli import ScoreNLI

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


# Nombre de nouvelles tentatives du client Groq sur erreur temporaire (limite
# de débit 429, surcharge 5xx). Partagé avec contextualizer.py.
GROQ_MAX_ESSAIS = 6


class LLM(ABC):
    """Contrat commun aux fournisseurs de modèle de langage."""

    @abstractmethod
    def generer(self, consigne: str, question: str) -> str:
        """Renvoie la réponse brute du modèle (chaîne JSON attendue)."""


class GroqLLM(LLM):
    """Implémentation Groq : gratuite, retenue par défaut."""

    def __init__(self, api_key: str, model: str):
        from groq import Groq

        # Le forfait gratuit Groq plafonne les tokens par minute : deux appels
        # rapprochés (rédaction puis juge de repli) peuvent le dépasser
        # (429, constaté le 2026-09-24). Le client réessaie alors après
        # l'attente indiquée par Groq ; 2 essais par défaut ne suffisaient pas.
        self._client = Groq(api_key=api_key, max_retries=GROQ_MAX_ESSAIS)
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
            max_tokens=1024,
            # openai/gpt-oss-20b (modèle par défaut) raisonne avant de répondre ;
            # ce raisonnement consomme max_tokens en priorité. Sans ce réglage,
            # un raisonnement long épuise le budget avant d'atteindre la
            # réponse JSON, qui arrive vide (constaté en conditions réelles,
            # 2026-09-21, sur le juge de repli de verifier_fidelite — même
            # cause que dans contextualizer.py, corrigée là le 2026-09-18).
            reasoning_effort="low",
        )
        return reponse.choices[0].message.content


def get_llm(settings: Settings) -> LLM:
    """Sélectionne l'implémentation selon le réglage LLM_PROVIDER de .env."""
    if settings.llm_provider == "groq":
        if not settings.groq_api_key:
            raise ValueError("GROQ_API_KEY manquant dans .env")
        return GroqLLM(settings.groq_api_key, settings.llm_model)
    raise ValueError(f"Fournisseur LLM inconnu : {settings.llm_provider}")


# Budget de caractères pour le bloc d'articles envoyé au LLM. Le forfait
# gratuit Groq plafonne openai/gpt-oss-20b à 8000 tokens/minute — une seule
# requête dépassant déjà ce plafond échoue, peu importe le débit accumulé.
# Constaté en conditions réelles (2026-09-22, eval/run_eval.py) : k=10
# articles complets peut demander plus de 10 000 tokens pour une seule
# question (10120 tokens demandés pour ~39 000 caractères de consigne +
# message, soit environ 4,3 caractères par token). Le budget laisse la place
# à la consigne, à la question et à max_tokens=1024 de réponse, avec marge.
LONGUEUR_MAX_CONTEXTE_CARACTERES = 24_000


def _bloc_articles(resultats: list[_ResultatRecherche]) -> str:
    """Met en forme les articles récupérés pour la consigne.

    Les articles arrivent déjà triés par pertinence (retriever.py) : au-delà
    du budget de caractères, les suivants sont simplement omis plutôt que
    tronqués au milieu — une troncature casserait la vérification des
    citations, faite contre le texte complet de l'article dans `resultats`.
    Le premier article est toujours inclus, même s'il dépasse le budget à
    lui seul (cas limite non rencontré dans le corpus actuel).
    """
    blocs: list[str] = []
    longueur = 0
    for i, r in enumerate(resultats, start=1):
        article = r.metadata.get("article", "?")
        marque = " — ABROGÉ" if r.metadata.get("abroge") else ""
        bloc = f"[{i}] Document : {r.doc_id} — Article {article}{marque}\n{r.content}"
        if blocs and longueur + len(bloc) > LONGUEUR_MAX_CONTEXTE_CARACTERES:
            break
        blocs.append(bloc)
        longueur += len(bloc)
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

# Découpe une ligne de réponse en phrases : un point, un point d'exclamation ou
# d'interrogation suivi d'une espace.
_SEPARATEUR_PHRASE = re.compile(r"(?<=[.!?])\s+")
# Début d'un élément de liste : « 1. », « 2) », « - », « • », « * ».
_PUCE = re.compile(r"^\s*(?:\d+[.)]|[-•*])\s+")
# « Oui, » / « Non. » en tête de réponse : une réaction à la question, pas une
# affirmation que le texte source pourrait impliquer.
_OUI_NON = re.compile(r"^(?:oui|non)\s*[,.:]\s*", re.IGNORECASE)
# Découpe d'un article en phrases : ponctuation forte, point-virgule ou
# deux-points (qui séparent les éléments d'une énumération légale).
_SEPARATEUR_PHRASE_SOURCE = re.compile(r"(?<=[.!?;:])\s+")

# Un extrait regroupe quelques phrases consécutives de l'article : assez pour
# garder le contexte immédiat d'une phrase, assez court pour le modèle NLI,
# entraîné sur des prémisses de la taille d'une phrase (un article entier lui
# fait rejeter même des paraphrases quasi littérales — mesuré le 2026-09-24).
_PHRASES_PAR_EXTRAIT = 3
# Nombre d'extraits soumis au NLI pour chaque affirmation, retenus parmi ceux
# qui partagent le plus de mots avec elle : évite de passer tout l'article au
# modèle, ce qui coûterait un passage NLI par extrait.
_EXTRAITS_PAR_AFFIRMATION = 8

CONSIGNE_JUGE_FIDELITE = """Tu vérifies si une affirmation est fondée sur un texte source, dans le \
cadre d'un assistant juridique qui ne doit rien avancer que le texte ne dise \
vraiment.

Réponds UNIQUEMENT par ce JSON, sans texte autour :
{"fondee": true} si le texte source implique l'affirmation (dit la même chose, \
même reformulée) ; {"fondee": false} sinon (l'affirmation va au-delà du texte, \
le contredit, ou n'a pas de rapport avec lui)."""


class _NLI(Protocol):
    """Partie d'un fournisseur NLI utilisée ici : les scores d'implication et
    de contradiction d'extraits de texte source face à une affirmation."""

    def evaluer(self, premisses: list[str], hypothese: str) -> list[ScoreNLI]: ...


def _decouper_en_affirmations(texte: str) -> list[str]:
    """Découpe une réponse en affirmations, une par phrase ou par élément de
    liste.

    Une décomposition par phrase reste une approximation d'une décomposition
    en affirmations atomiques (une phrase peut porter plusieurs faits, cf.
    RAGAS faithfulness) — suffisante pour un garde-fou rapide en ligne, sans
    appel de modèle supplémentaire pour la décomposition elle-même.

    Les listes demandent un traitement à part : « Les missions du syndic
    sont : » n'affirme rien seule, et « Conserver l'immeuble » n'a pas de
    sujet. Chaque élément de liste est donc rattaché à la phrase qui
    l'introduit (celle qui se termine par « : »), pour former une affirmation
    complète que le NLI peut vérifier.
    """
    affirmations: list[str] = []
    introduction: str | None = None
    for ligne in texte.splitlines():
        if not ligne.strip():
            continue
        if _PUCE.match(ligne):
            element = _PUCE.sub("", ligne).strip()
            affirmations.append(f"{introduction} {element}" if introduction else element)
            continue
        for phrase in _SEPARATEUR_PHRASE.split(ligne.strip()):
            phrase = phrase.strip()
            if not phrase:
                continue
            if phrase.endswith(":"):
                introduction = phrase[:-1].strip()
                continue
            introduction = None
            affirmations.append(phrase)
    affirmations = [_OUI_NON.sub("", a).strip() for a in affirmations]
    return [a for a in affirmations if a]


def _extraits(contextes: list[str]) -> list[str]:
    """Découpe les articles en extraits de quelques phrases consécutives, qui
    se chevauchent (fenêtre glissante d'une phrase) pour qu'aucune phrase ne
    soit séparée de ses voisines immédiates."""
    extraits: list[str] = []
    for contexte in contextes:
        phrases = [
            p.strip()
            for ligne in contexte.splitlines()
            for p in _SEPARATEUR_PHRASE_SOURCE.split(ligne)
            if p.strip()
        ]
        n = _PHRASES_PAR_EXTRAIT
        if len(phrases) <= n:
            extraits.append(" ".join(phrases))
        else:
            extraits.extend(" ".join(phrases[i : i + n]) for i in range(len(phrases) - n + 1))
    return [e for e in extraits if e]


def _preselectionner(extraits: list[str], affirmation: str) -> list[str]:
    """Extraits partageant le plus de mots avec l'affirmation (même
    tokenisation française que l'index BM25 : accents, racines, mots vides).

    Un tri rapide qui ne coûte aucun passage de modèle : sur un article de
    150 extraits, seuls les plus proches sont confiés au NLI."""
    mots = set(tokeniser_fr(affirmation))
    return sorted(extraits, key=lambda e: -len(mots & set(tokeniser_fr(e))))[
        :_EXTRAITS_PAR_AFFIRMATION
    ]


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
    seuil_contradiction: float,
    llm: LLM | None = None,
) -> ReponseGeneree:
    """Garde-fou étage 2 (suite) : la réponse rédigée colle-t-elle aux articles
    qu'elle cite ?

    Une bonne citation (vérifiée par verifier_citations) ne garantit pas que le
    texte rédigé autour reste fidèle à ce qu'elle dit : le LLM peut broder à
    côté même avec la bonne source sous les yeux. La réponse est découpée en
    affirmations, chacune confrontée par un modèle NLI aux extraits les plus
    proches des articles cités.

    Pour chaque affirmation, on retient le meilleur score d'implication et le
    plus fort score de contradiction parmi ces extraits :
    - implication au-dessus de seuil_haut ET aucune contradiction au-dessus de
      seuil_contradiction : fondée ;
    - implication en dessous de seuil_bas : non fondée, refus direct ;
    - sinon (score ambigu, ou signaux opposés) : le NLI seul ne suffit pas,
      llm sert de juge de repli sur les articles entiers s'il est fourni ;
      sans llm, traité comme non fondé (prudence, domaine réglementé).

    Le contrôle de contradiction protège d'un piège des extraits courts : un
    extrait qui partage les mots de l'affirmation (« dépôt de garantie »,
    « deux mois ») peut sembler l'impliquer alors qu'un autre extrait du même
    article la contredit (« un mois » et non « deux mois »).

    Un refus est transmis tel quel. Une réponse sans texte (cas dégénéré déjà
    exclu par ailleurs) n'a aucune affirmation à vérifier et passe telle quelle.
    """
    if reponse.refus:
        return reponse

    affirmations = _decouper_en_affirmations(reponse.reponse)
    if not affirmations:
        return reponse

    contextes = _contextes_cites(reponse, resultats)
    extraits = _extraits(contextes)

    for affirmation in affirmations:
        scores = nli.evaluer(_preselectionner(extraits, affirmation), affirmation)
        implication = max((s.implication for s in scores), default=0.0)
        contradiction = max((s.contradiction for s in scores), default=0.0)
        if implication >= seuil_haut and contradiction < seuil_contradiction:
            continue
        if implication <= seuil_bas:
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
