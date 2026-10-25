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
- ModeleIndisponible : erreur d'un modèle saturé ou momentanément en panne.
- LLMEnCascade : passe au modèle suivant quand le premier est saturé.
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
- Réponds à la question posée, pas à une question voisine. Si elle demande une \
information précise (une durée, un délai, un montant, un taux, un seuil, une \
majorité) et qu'aucun article ne la donne, refuse, même si les articles \
traitent du même sujet.
- Un article signalé « ABROGÉ » ne doit pas être présenté comme le droit en vigueur.
- La ligne « Situé dans » indique le régime auquel un article s'applique \
(par exemple les logements meublés, ou le bail mobilité). Une règle propre à \
un régime particulier ne doit jamais être présentée comme la règle générale : \
précise toujours le régime concerné, et donne d'abord la règle générale \
lorsqu'un article la fournit.
- N'affirme pas qu'une règle ou une exception est la seule (« uniquement », \
« seulement ») si l'article ne le dit pas : d'autres textes peuvent en prévoir.
- Ne transforme pas une règle de calcul en égalité (« est égale à ») : \
garde le verbe de l'article (« établie d'après », « sous déduction de »).
- Réponds en français, dans une langue claire, en structurant la réponse quand \
c'est utile (cas, étapes, conditions).
- Pour chaque affirmation, cite le passage EXACT qui la fonde : recopie mot pour \
mot une phrase ou une portion de phrase de l'article, pas l'article entier.
- Une citation contient UNE seule phrase ou UN seul élément de liste, sans rien \
retirer au milieu. Pour appuyer une liste, fais une citation par élément, ou \
cite seulement la phrase qui introduit la liste.

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

    question_comprise : la question réellement traitée, quand une question de
    suivi a été réécrite à partir de la conversation (suivi.py) ; None sinon.
    Affichée à l'utilisateur, qui voit ainsi comment sa question a été lue.
    """

    refus: bool
    motif: str | None = None
    reponse: str = ""
    citations: list[Citation] = Field(default_factory=list)
    question_comprise: str | None = None


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

    def __init__(self, api_key: str, model: str, max_essais: int = GROQ_MAX_ESSAIS):
        from groq import Groq

        # Le forfait gratuit Groq plafonne les tokens par minute : deux appels
        # rapprochés (rédaction puis juge de repli) peuvent le dépasser
        # (429, constaté le 2026-09-24). Le client réessaie alors après
        # l'attente indiquée par Groq ; 2 essais par défaut ne suffisaient pas.
        # Dans une cascade (LLMEnCascade), max_essais=0 : un modèle saturé
        # passe la main au suivant au lieu d'attendre.
        self._client = Groq(api_key=api_key, max_retries=max_essais)
        self._model = model
        self.nom = model

    def generer(self, consigne: str, question: str) -> str:
        from groq import (
            APIConnectionError,
            APIStatusError,
            InternalServerError,
            RateLimitError,
        )

        try:
            return self._appeler(consigne, question)
        except RateLimitError as exc:
            # Groq signale le plafond journalier (« tokens per day (TPD) »,
            # « requests per day ») par le même code 429 que le plafond par
            # minute : seul le message les distingue.
            if "per day" in str(exc):
                raise QuotaJournalierEpuise(f"{self._model} : {exc}") from exc
            raise ModeleIndisponible(f"{self._model} : {exc}") from exc
        except (InternalServerError, APIConnectionError) as exc:
            raise ModeleIndisponible(f"{self._model} : {exc}") from exc
        except APIStatusError as exc:
            # 413 : une seule requête dépasse le plafond de tokens par minute du
            # modèle. Un autre modèle de la cascade peut l'accepter ; sinon
            # l'API répond 503 plutôt que de planter.
            if exc.status_code == 413:
                raise ModeleIndisponible(f"{self._model} : {exc}") from exc
            # 400 json_validate_failed : le modèle n'a pas produit de JSON
            # valide (sortie vide, constaté sur le juge, 2026-10-06). Un autre
            # modèle de la cascade peut réussir ; sinon l'API répond 503.
            if exc.status_code == 400 and "json_validate_failed" in str(exc):
                raise ModeleIndisponible(f"{self._model} : {exc}") from exc
            raise

    def _appeler(self, consigne: str, question: str) -> str:
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


class ModeleIndisponible(Exception):
    """Le modèle ne peut pas répondre maintenant : quota atteint, surcharge ou
    coupure réseau. Un autre modèle peut prendre le relais."""


class QuotaJournalierEpuise(ModeleIndisponible):
    """Le quota du jour du modèle est consommé : attendre quelques secondes
    n'y change rien, il ne répondra plus avant des heures."""


class LLMEnCascade(LLM):
    """Répartit les appels sur plusieurs modèles pour contourner le quota.

    Le forfait gratuit Groq plafonne chaque modèle séparément (8000 tokens par
    minute chacun, vérifié le 2026-09-29 sur les compteurs renvoyés par Groq).
    Un modèle saturé renvoie aussitôt ModeleIndisponible et l'appel passe au
    suivant, sans attente. Si tous sont saturés, un modèle de
    `derniers_recours` (même modèle, mais qui attend la fin de la limite puis
    réessaie) répond : celui du premier modèle dont le quota du jour n'est pas
    épuisé. Attendre un modèle épuisé pour la journée bloquait l'appel puis
    échouait, alors qu'un autre modèle pouvait répondre (constaté le
    2026-10-03 : quota journalier du juge gpt-oss-120b consommé).
    """

    def __init__(self, modeles: list[LLM], derniers_recours: list[LLM]):
        if len(modeles) != len(derniers_recours):
            raise ValueError("un dernier recours par modèle de la cascade")
        self._modeles = modeles
        self._derniers_recours = derniers_recours
        # Modèles qui ont effectivement répondu, dans l'ordre des appels : les
        # modèles ne jugent pas tous pareil, l'évaluation doit savoir lequel a
        # tranché (run_eval.py).
        self.historique: list[str] = []

    def generer(self, consigne: str, question: str) -> str:
        epuises: set[int] = set()
        for rang, modele in enumerate(self._modeles):
            try:
                brut = modele.generer(consigne, question)
                self.historique.append(getattr(modele, "nom", "?"))
                return brut
            except QuotaJournalierEpuise:
                epuises.add(rang)
            except ModeleIndisponible:
                continue
        for rang, recours in enumerate(self._derniers_recours):
            if rang not in epuises:
                brut = recours.generer(consigne, question)
                self.historique.append(f"{getattr(recours, 'nom', '?')} (attente)")
                return brut
        raise QuotaJournalierEpuise("quota du jour épuisé pour tous les modèles")


def get_llm(settings: Settings, premier: str | None = None) -> LLM:
    """Sélectionne l'implémentation selon le réglage LLM_PROVIDER de .env.

    Avec des modèles de relais (llm_modeles_relais), les appels passent par
    une cascade : llm_model d'abord, puis les relais dans l'ordre. premier
    place un autre modèle en tête (cf. get_llm_juge). Quand le quota du jour
    est épuisé partout, l'appel échoue (QuotaJournalierEpuise) : l'ancien
    secours Mistral (ministral-14b) a été retiré le 2026-10-23, ses réponses
    n'étaient acceptées qu'à 48 % contre 87 % pour gpt-oss-20b (traces du
    golden) — mieux vaut pas de réponse qu'une réponse dégradée.
    """
    if settings.llm_provider == "groq":
        if not settings.groq_api_key:
            raise ValueError("GROQ_API_KEY manquant dans .env")
        cle = settings.groq_api_key
        tete = premier or settings.llm_model
        noms = list(dict.fromkeys([tete, settings.llm_model, *settings.llm_modeles_relais]))
        modeles: list[LLM] = [GroqLLM(cle, nom, max_essais=0) for nom in noms]
        recours: list[LLM] = [GroqLLM(cle, nom) for nom in noms]
        if len(modeles) == 1:
            return GroqLLM(cle, tete)
        return LLMEnCascade(modeles, derniers_recours=recours)
    raise ValueError(f"Fournisseur LLM inconnu : {settings.llm_provider}")


def get_llm_juge(settings: Settings) -> LLM:
    """LLM du juge de repli de verifier_fidelite : llm_modele_juge en tête,
    les modèles de génération en relais."""
    return get_llm(settings, premier=settings.llm_modele_juge)


# Budget de caractères pour le bloc d'articles envoyé au LLM. Le forfait
# gratuit Groq plafonne openai/gpt-oss-20b à 8000 tokens/minute — une seule
# requête dépassant déjà ce plafond échoue, peu importe le débit accumulé.
# Constaté en conditions réelles (2026-09-22, eval/run_eval.py) : k=10
# articles complets peut demander plus de 10 000 tokens pour une seule
# question (10120 tokens demandés pour ~39 000 caractères de consigne +
# message, soit environ 4,3 caractères par token). Le budget laisse la place
# à la consigne, à la question et à max_tokens=1024 de réponse, avec marge.
LONGUEUR_MAX_CONTEXTE_CARACTERES = 24_000


def _bloc_articles(
    resultats: list[_ResultatRecherche], budget: int = LONGUEUR_MAX_CONTEXTE_CARACTERES
) -> str:
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
        section = r.metadata.get("section")
        # L'intitulé (« Titre Ier bis : … logements meublés … ») dit à quel
        # régime l'article s'applique, ce que son texte ne répète pas toujours.
        situation = f"\nSitué dans : {section}" if section else ""
        bloc = (
            f"[{i}] Document : {r.doc_id} — Article {article}{marque}{situation}\n{r.content}"
        )
        if blocs and longueur + len(bloc) > budget:
            break
        blocs.append(bloc)
        longueur += len(bloc)
    return "\n\n".join(blocs)


# Ligne qui ouvre un exemple rédigé par le modèle : « Exemple : », « **Exemple
# concret** : », « ### Exemple de calcul », en tête de ligne ou de puce.
_LIGNE_EXEMPLE = re.compile(r"^\s*(?:#{1,6}\s*|[-*•]\s+)?(?:\*\*|__)?\s*exemples?\b", re.IGNORECASE)


def retirer_exemples(texte: str) -> str:
    """Retire de la réponse les exemples que le rédacteur invente pour
    illustrer (« Exemple concret : si le dépôt de 1 000 €… »), avec la liste
    qui les détaille.

    Un exemple chiffré ne vient d'aucun article : il n'a pas de source à
    citer, et le garde-fou refusait la réponse entière à cause de lui
    (montant absent des articles, ou juge). Mesuré le 2026-10-24 sur les
    traces : 18 réponses sur 696 contenaient un exemple, 13 refusées (72 %).
    Le retirer, sans rien réécrire, garde la règle sourcée et seulement elle.
    Un « par exemple » au fil d'une phrase n'est pas touché."""
    lignes = texte.splitlines()
    gardees: list[str] = []
    i = 0
    while i < len(lignes):
        if not _LIGNE_EXEMPLE.match(lignes[i]):
            gardees.append(lignes[i])
            i += 1
            continue
        # La ligne d'exemple, puis ses détails : les lignes plus en retrait
        # qu'elle, et, si elle annonce une liste (« Exemple de calcul : »), les
        # puces qui suivent. Un élément numéroté au même niveau est une autre
        # règle et reste (q13 : « 2. En cas de vente par lots… » après un
        # sous-élément « - Exemple : … », retiré à tort sinon).
        retrait = len(lignes[i]) - len(lignes[i].lstrip())
        annonce_liste = _GRAS.sub("", lignes[i]).rstrip().endswith(":")
        i += 1
        while i < len(lignes):
            ligne = lignes[i]
            if not ligne.strip():
                suite = next((x for x in lignes[i + 1 :] if x.strip()), "")
                if not (annonce_liste and re.match(r"\s*[-*•]\s", suite)):
                    break
            elif len(ligne) - len(ligne.lstrip()) <= retrait and not (
                annonce_liste and re.match(r"\s*[-*•]\s", ligne)
            ):
                break
            i += 1
    return re.sub(r"\n{3,}", "\n\n", "\n".join(gardees)).strip()


def _lire_reponse(brut: str) -> ReponseGeneree:
    """Transforme la réponse JSON du modèle en ReponseGeneree.

    Toute réponse illisible (JSON invalide, champs manquants) est traitée comme
    un refus : une sortie non conforme ne doit pas être présentée à
    l'utilisateur comme une réponse fondée.
    """
    try:
        donnees = json.loads(brut)
        reponse = ReponseGeneree.model_validate(donnees)
    except (json.JSONDecodeError, ValidationError, TypeError):
        return ReponseGeneree(
            refus=True, motif="réponse du modèle illisible ou non conforme"
        )
    return sans_exemples(reponse)


def sans_exemples(reponse: ReponseGeneree) -> ReponseGeneree:
    """La réponse privée de ses exemples (retirer_exemples). Une réponse qui
    n'était qu'un exemple reste entière : vide, elle n'aurait plus aucune
    affirmation à vérifier et passerait le garde-fou."""
    if reponse.refus:
        return reponse
    texte = retirer_exemples(reponse.reponse)
    if not re.search(r"\w", texte):
        return reponse
    return reponse.model_copy(update={"reponse": texte})


def generer_reponse(
    question: str,
    resultats: list[_ResultatRecherche],
    llm: LLM,
    omission: str | None = None,
) -> ReponseGeneree:
    """Rédige une réponse à la question à partir des articles récupérés.

    Renvoie un refus immédiat si aucun article n'a été récupéré : il n'y a alors
    aucune source sur laquelle fonder une réponse.

    omission : élément qu'une première réponse taisait (verifier_completude),
    signalé au LLM pour qu'il l'intègre en rédigeant de nouveau.
    """
    if not resultats:
        return ReponseGeneree(refus=True, motif="aucun passage pertinent trouvé")

    message = f"Question : {question}\n\nArticles disponibles :\n\n{_bloc_articles(resultats)}"
    if omission:
        message += (
            "\n\nUne première réponse taisait l'élément suivant, qui change la réponse "
            f"à la question : {omission}\nIntègre-le, avec sa citation."
        )
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
# Tolérance aux fautes de frappe dans une citation : longueur minimale d'un
# mot pour l'admettre, similarité minimale avec le mot d'origine, et marge
# (en caractères) autour de la zone de l'article où le passage est ancré.
_BLOC_MIN_APPROX = 5
_SIMILARITE_FAUTE_DE_FRAPPE = 0.8
_MARGE_APPROX = 20

# Points de suspension par lesquels le modèle abrège une citation : « ... »,
# « … » ou « [...] ». Une citation qui recolle l'introduction d'une
# énumération et l'un de ses éléments (« … à peine de nullité : 2° Le montant
# … »), ou plusieurs éléments sur plusieurs lignes, est coupée de même : chaque
# morceau doit figurer dans l'article, dans l'ordre. Ces citations exactes par
# morceaux étaient refusées (q17, q14, q22, traces de 2026-10-03 à 2026-10-06).
_ELLIPSE = re.compile(r"\[?(?:\.{3}|…)\]?|\n+|(?<=:)\s+(?=\d+\s*°|[a-z]\)|[-–•])")


def _normaliser(texte: str) -> str:
    """Ramène un texte à une forme comparable : sans accents, en minuscules,
    ponctuation et espaces multiples réduits à une espace simple."""
    # La ligature « œ » ne se décompose pas : « mise en œuvre » cité contre
    # « mise en oeuvre » dans l'article faisait refuser une citation exacte
    # (q25, 2026-10-06).
    texte = texte.replace("œ", "oe").replace("Œ", "OE").replace("æ", "ae").replace("Æ", "AE")
    sans_accents = "".join(
        c for c in unicodedata.normalize("NFKD", texte) if not unicodedata.combining(c)
    )
    return re.sub(r"[^a-z0-9]+", " ", sans_accents.lower()).strip()


def _passage_present(passage: str, contenu: str, seuil: float) -> bool:
    """Indique si le passage figure dans le contenu de l'article.

    Un passage abrégé par des points de suspension est découpé en fragments :
    chacun doit figurer dans l'article, dans l'ordre de la citation.
    """
    c = _normaliser(contenu)
    fragments = [f for f in (_normaliser(m) for m in _ELLIPSE.split(passage)) if f]
    if not fragments:
        return False
    debut = 0
    for fragment in fragments:
        position = c.find(fragment, debut)
        if position >= 0:
            debut = position + len(fragment)
        elif fragment.replace(" ", "") in c[debut:].replace(" ", ""):
            # Mots collés par l'extraction du PDF (« suivantsdu ») : même texte,
            # seules les espaces diffèrent.
            continue
        elif not _fragment_approche(fragment, c[debut:], seuil):
            return False
    return True


def _fragment_approche(p: str, c: str, seuil: float) -> bool:
    """Rapprochement approché d'un fragment déjà normalisé : pour un fragment
    assez long, un écart mineur est toléré tant que la plus longue portion
    commune couvre au moins `seuil` du fragment.

    Une faute de frappe au milieu d'une longue citation (« alinéau ») coupe
    cette portion commune en deux : le fragment est alors comparé mot à mot à
    la zone de l'article où il s'ancre (_fautes_de_frappe_seulement)."""
    if len(p) < _LONGUEUR_MIN_APPROX:
        return False
    ancre = SequenceMatcher(None, p, c, autojunk=False).find_longest_match(
        0, len(p), 0, len(c)
    )
    if ancre.size / len(p) >= seuil:
        return True
    debut = max(ancre.b - ancre.a - _MARGE_APPROX, 0)
    zone = c[debut : ancre.b - ancre.a + len(p) + _MARGE_APPROX]
    return _fautes_de_frappe_seulement(p.split(), zone.split())


def _fautes_de_frappe_seulement(mots: list[str], zone: list[str]) -> bool:
    """Indique si les mots du fragment se retrouvent à la suite dans la zone,
    chacun à l'identique ou à une faute de frappe près.

    Aucun mot ajouté, retiré ou remplacé n'est toléré : « six mois » au lieu
    de « trois mois » change le sens. Une faute de frappe n'est admise que sur
    un mot long et sans chiffre, très proche de l'original."""
    operations = SequenceMatcher(None, mots, zone, autojunk=False).get_opcodes()
    # Seules comptent les opérations qui touchent le fragment, entre son
    # premier et son dernier mot : la zone déborde du passage de part et d'autre.
    internes = [op for op in operations if op[1] < op[2] or 0 < op[1] < len(mots)]
    for operation, i1, i2, j1, j2 in internes:
        if operation == "equal":
            continue
        if operation != "replace" or i2 - i1 != j2 - j1:
            return False
        if not all(_faute_de_frappe(a, b) for a, b in zip(mots[i1:i2], zone[j1:j2])):
            return False
    return True


def _faute_de_frappe(mot: str, original: str) -> bool:
    """Un mot long, sans chiffre, qui ne diffère de l'original que d'une ou
    deux lettres (« alineau » pour « alinea »)."""
    if min(len(mot), len(original)) < _BLOC_MIN_APPROX:
        return False
    if any(ch.isdigit() for ch in mot + original):
        return False
    return SequenceMatcher(None, mot, original).ratio() >= _SIMILARITE_FAUTE_DE_FRAPPE


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


def citations_introuvables(
    reponse: ReponseGeneree,
    resultats: list[_ResultatRecherche],
    seuil: float = SEUIL_CITATION,
) -> list[Citation]:
    """Citations de la réponse qui ne se retrouvent dans aucun article récupéré."""
    return [
        citation
        for citation in reponse.citations
        if not any(
            _passage_present(citation.passage, r.content, seuil)
            for r in _articles_candidats(citation, resultats)
        )
    ]


def _phrase_la_plus_proche(citation: Citation, resultats: list[_ResultatRecherche]) -> str:
    """Phrase des articles candidats la plus ressemblante au passage cité,
    proposée au modèle comme modèle exact à recopier."""
    cible = _normaliser(citation.passage)
    phrases = [
        texte
        for r in _articles_candidats(citation, resultats)
        for _, texte in _phrases_source(r.content)
    ]
    return max(
        phrases,
        key=lambda p: SequenceMatcher(None, cible, _normaliser(p), autojunk=False).ratio(),
        default="",
    )


# Une citation tronquée n'est remplacée par la phrase entière de l'article que
# si elle en garde l'essentiel : assez de mots, et au moins cette part de la
# phrase d'origine.
_MOTS_MIN_TRONQUEE = 8
_PART_MIN_TRONQUEE = 0.6


def _sous_suite(mots: list[str], phrase: list[str]) -> bool:
    """Les mots de la citation figurent-ils dans la phrase, dans le même ordre,
    seuls des mots ayant été retirés ?"""
    reste = iter(phrase)
    return all(mot in reste for mot in mots)


def _phrase_entiere(citation: Citation, resultats: list[_ResultatRecherche]) -> str | None:
    """Phrase de l'article dont la citation n'est qu'une version amputée de
    quelques mots, ou None."""
    mots = _normaliser(citation.passage).split()
    if len(mots) < _MOTS_MIN_TRONQUEE:
        return None
    for r in _articles_candidats(citation, resultats):
        for _, phrase in _phrases_source(r.content):
            cible = _normaliser(phrase).split()
            if len(mots) >= _PART_MIN_TRONQUEE * len(cible) and _sous_suite(mots, cible):
                # Numéro de page du PDF collé en tête (« 34 Par dérogation… »).
                return re.sub(r"^\d+\s+(?=[A-ZÀ-Ý])", "", phrase)
    return None


def reparer_citations(
    reponse: ReponseGeneree,
    resultats: list[_ResultatRecherche],
    seuil: float = SEUIL_CITATION,
) -> ReponseGeneree:
    """Remplace une citation amputée de quelques mots par la phrase entière de
    l'article, sans appel de modèle.

    Le rédacteur retire parfois une incise au milieu d'une phrase citée
    (« les décisions, à l'exclusion de celles relatives au vote du budget
    prévisionnel [et à l'approbation des comptes], peuvent… », q20, refusée
    entière le 2026-10-22). Afficher la phrase complète montre à l'utilisateur
    le vrai texte de loi ; les affirmations de la réponse restent vérifiées
    ensuite (NLI, juge) contre les articles. Une citation qui ajoute ou change
    des mots n'est pas réparée ici."""
    if reponse.refus:
        return reponse
    introuvables = citations_introuvables(reponse, resultats, seuil)
    if not introuvables:
        return reponse
    citations = []
    for c in reponse.citations:
        entiere = _phrase_entiere(c, resultats) if c in introuvables else None
        citations.append(c.model_copy(update={"passage": entiere}) if entiere else c)
    return reponse.model_copy(update={"citations": citations})


def corriger_citations(
    question: str,
    resultats: list[_ResultatRecherche],
    llm: LLM,
    reponse: ReponseGeneree,
    seuil: float = SEUIL_CITATION,
) -> ReponseGeneree:
    """Seconde chance, unique, pour une réponse dont des citations sont
    introuvables.

    Un LLM reformule parfois légèrement le passage qu'il recopie (« Lorsque le
    congé émane » pour « Lorsqu'il émane ») : la vérification refuse à juste
    titre, mais la réponse pouvait être juste. Le modèle reçoit alors les
    citations fautives et doit renvoyer sa réponse avec des citations
    recopiées mot pour mot. La vérification n'est pas assouplie : la nouvelle
    réponse repasse ensuite par verifier_citations comme la première.

    Renvoie la réponse telle quelle si elle est un refus ou si toutes ses
    citations sont retrouvées (aucun appel supplémentaire).
    """
    if reponse.refus:
        return reponse
    reponse = reparer_citations(reponse, resultats, seuil)
    introuvables = citations_introuvables(reponse, resultats, seuil)
    if not introuvables:
        return reponse
    fautives = "\n".join(
        f"- Article {c.article} ({c.doc_id}) : « {c.passage} »\n"
        f"  phrase exacte la plus proche dans l'article : « {_phrase_la_plus_proche(c, resultats)} »"
        for c in introuvables
    )
    suite = (
        f"\n\nTa réponse précédente :\n{reponse.model_dump_json()}"
        f"\n\nCes citations ne figurent pas mot pour mot dans les articles :\n{fautives}"
        "\n\nRenvoie ta réponse complète, au même format JSON, en recopiant chaque "
        "citation exactement comme dans l'article (mêmes mots, même ordre). Une "
        "citation qui recolle plusieurs phrases ou éléments de liste doit être "
        "découpée en plusieurs citations, une par phrase ou élément, recopiés en "
        "entier. Si un passage exact ne fonde pas une affirmation, retire cette "
        "affirmation."
    )
    # La réponse précédente et les citations fautives s'ajoutent aux articles :
    # le bloc d'articles leur cède la place, sans quoi la requête dépassait le
    # plafond de Groq (413, 8412 tokens pour 8000, q02 le 2026-10-05).
    articles = _bloc_articles(resultats, LONGUEUR_MAX_CONTEXTE_CARACTERES - len(suite))
    message = f"Question : {question}\n\nArticles disponibles :\n\n{articles}{suite}"
    return _lire_reponse(llm.generer(CONSIGNE, message))


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

    introuvables = citations_introuvables(reponse, resultats, seuil)
    if introuvables:
        exemple = introuvables[0].passage
        return ReponseGeneree(
            refus=True,
            motif=f"citation introuvable dans les sources : « {exemple} »",
        )

    return reponse


# --- Garde-fou étage 2 : fidélité de la réponse aux articles cités ----------

# Découpe une ligne de réponse en phrases : un point, un point d'exclamation ou
# d'interrogation suivi d'une espace. Le point d'une référence légale (« L. 126-26 »,
# « R. 111-1 », « art. 3 ») ne termine pas la phrase : couper là produisait
# des fragments sans contenu (« 126-26 du code de la construction ») refusés
# par le juge (mesuré sur q03, q14, q22, q28 le 2026-10-03). De même pour
# « ex. » et « cf. » : « (ex. emploi, santé) » coupé laissait une affirmation
# finissant par « (ex. », refusée par le juge (q02 juste refusée, 2026-10-22).
# Une phrase citée entre guillemets finit avant le guillemet fermant : couper
# là laissait « » (article 1724 du code civil). » seul, refusé par le juge
# (q55 juste refusée, 2026-10-23) ; la coupe se fait après lui quand une
# nouvelle phrase commence (« … louée. » De plus, … »).
_SEPARATEUR_PHRASE = re.compile(
    r"(?<=[.!?])(?<!\b[LRDA]\.)(?<!\bart\.)(?<!\bex\.)(?<!\bcf\.)\s+(?![»”\"])"
    r"|(?<=[.!?]\s[»”\"])\s+(?=[«\"A-ZÀ-Ý])"
)
# Début d'un élément de liste : « 1. », « 2) », « a. », « b) », « - », « • », « * ».
# Sans les lettres, « a. Le locataire … » était coupé en deux phrases et « a. »
# devenait une affirmation refusée par le juge (q03, 2026-10-14).
_PUCE = re.compile(r"^\s*(?:\d+[.)]|[a-z][.)]|[-•*])\s+")
# Élément de liste numérotée écrit à la suite du précédent, sur la même ligne
# (« 1. Assurer … communes. 2. Mettre à jour … ») : remis à la ligne pour être
# vérifié seul.
_PUCE_EN_LIGNE = re.compile(r"(?<=[.;:])[ \t]+(?=\d{1,2}[.)][ \t])")
# Mise en forme markdown (gras, italique, titres) : le NLI la lit comme du
# texte, et « **Règles générales :** » ne finit plus par « : », donc passait
# pour une affirmation (réponses de Mistral surtout, constaté le 2026-10-11).
_GRAS = re.compile(r"\*\*|__")
_ITALIQUE = re.compile(r"(?<![\w*])\*(?=\S)([^*\n]+?)(?<=\S)\*(?![\w*])")
# Un titre (« ## Règle ») n'affirme rien : la ligne est retirée entière.
_TITRE = re.compile(r"^[ \t]*#{1,6}[ \t].*$", re.MULTILINE)
# Intertitre écrit en gras sur sa propre ligne, numéroté ou non
# (« **2. Interdiction de recevoir des fonds** ») : un titre aussi, retiré
# avant le gras. Vérifié comme une affirmation, il faisait refuser une réponse
# juste (q54, rédigée par Kimi, 2026-10-23). Un intertitre qui finit par « : »
# introduit une liste et reste en place.
_TITRE_GRAS = re.compile(r"^[ \t]*(?:\d+[.)][ \t]*)?\*\*[^*\n]*[^*:\s]\*\*[ \t]*$", re.MULTILINE)
# « Oui, » / « Non. » en tête de réponse : une réaction à la question, pas une
# affirmation que le texte source pourrait impliquer.
_OUI_NON = re.compile(r"^(?:oui|non)\s*[,.:]\s*", re.IGNORECASE)
# « Par ailleurs, l'article 19 précise que … » : une formule d'attribution que
# le texte de l'article ne contient pas, et qui suffit à faire échouer le NLI
# alors que le fond est fidèle. L'attribution elle-même relève de
# verifier_citations ; seul le contenu rapporté est confronté aux sources.
_ATTRIBUTION = re.compile(
    r"^(?:(?:par ailleurs|en effet|de plus|en outre|ainsi|enfin)\s*,\s*)?"
    r"l['’]article\s+[^,«»]{1,60}?\s+"
    r"(?:précise|dispose|prévoit|indique|énonce|ajoute|stipule)\s+qu(?:e\s+|['’])",
    re.IGNORECASE,
)
# Même raison pour une référence entre parenthèses (« (article 1724 du code
# civil) », « (art. 6-2) », « (selon l'article 3 de la loi Hoguet) ») ou en tête de
# phrase (« Selon l'article 3 de la loi Hoguet, ») : mesuré le 2026-10-24, la
# phrase exacte de l'article 1724 est impliquée à 1,00, la même suivie de
# « (article 1724 du code civil) » à 0,00 — le NLI exige que l'extrait dise
# aussi « code civil ». Sur le golden du 2026-10-23, ces références envoyaient
# au juge LLM des affirmations recopiées de l'article.
_REFERENCE_ENTRE_PARENTHESES = re.compile(
    r"\s*\((?:[\w-]+\s*,\s*|(?:selon|cf\.|voir)\s+l?['’]?)?(?:articles?|art\.)\s+[^()]{1,80}\)",
    re.IGNORECASE
)
_SELON_ARTICLE = re.compile(
    r"^(?:selon|d['’]après|aux termes de|conformément à)\s+l['’]article\s+[^,«»]{1,80},\s*",
    re.IGNORECASE,
)
# Fragments qui n'affirment rien et que la découpe en phrases isole : un
# numéro d'étape écrit dans le texte (« article 20. 1. Le notaire… » donne
# « 1. », ou « 3 bis. » dans la numérotation légale), ou un renvoi seul
# après « etc. » (« (article 18-1 AA). »), éventuellement précédé de
# l'identifiant du document (« (loi_06_07_1989, article 24, I‑1) », q17 refusée
# à tort le 2026-10-16), ou désignant une plage d'éléments (« (Article 15, I,
# 1°‑5°) », q03 refusée à tort le 2026-10-17). La référence elle-même relève
# de verifier_citations.
_SANS_CONTENU = re.compile(
    r"^\(?\s*(?:\d+°?(?:\s*(?:bis|ter|quater))?|(?:\w+\s*,\s*)?(?:articles?|art\.)\s+[\w\s.,°‑-]{1,40})\s*\)?\s*[.;:]?$",
    re.IGNORECASE,
)
# Découpe d'un article en phrases : ponctuation forte, point-virgule ou
# deux-points (qui séparent les éléments d'une énumération légale).
_SEPARATEUR_PHRASE_SOURCE = re.compile(r"(?<=[.!?;:])\s+")
# Début d'un élément d'énumération légale : « a) », « 2° », « 3o » (PDF).
_ELEMENT_ENUMERATION = re.compile(r"^«?\s*(?:[a-z]\)|\d+\s*(?:°|o\b))")

# Un extrait regroupe quelques phrases consécutives de l'article : assez pour
# garder le contexte immédiat d'une phrase, assez court pour le modèle NLI,
# entraîné sur des prémisses de la taille d'une phrase (un article entier lui
# fait rejeter même des paraphrases quasi littérales — mesuré le 2026-09-24).
_PHRASES_PAR_EXTRAIT = 3
# Longueur maximale (en caractères) d'une énumération soumise d'un bloc au
# NLI : au-delà, le modèle tronque le texte et perd la fin de la liste.
_LONGUEUR_MAX_ENUMERATION = 1500
# Intitulé d'un régime à part : « Dispositions particulières aux syndicats dont
# le nombre de voix est réparti entre deux copropriétaires », logements meublés,
# bail mobilité, résidences-services. Le NLI ne lit que le texte de l'article,
# pas cet intitulé : il validait une règle propre aux copropriétés à deux
# présentée comme générale (q21, art. 41-15, mesuré le 2026-10-05).
_REGIME_PARTICULIER = re.compile(r"particuli|meublé|mobilité|résidences-services", re.IGNORECASE)
# Extrait qui déroge à une règle posée ailleurs (« La limite mentionnée au
# deuxième alinéa est rehaussée… ») : vraie mot à mot, une affirmation tirée
# de lui seul peut taire la règle générale. Le NLI fondait ainsi « jusqu'à
# 21 400 € » sans le plafond de 10 700 € (q58, art. 156, 2026-10-20).
_DEROGATION = re.compile(r"par dérogation|par exception|rehauss", re.IGNORECASE)
# Nombre d'extraits soumis au NLI pour chaque affirmation, retenus parmi ceux
# qui partagent le plus de mots avec elle : évite de passer tout l'article au
# modèle, ce qui coûterait un passage NLI par extrait.
_EXTRAITS_PAR_AFFIRMATION = 8

CONSIGNE_JUGE_FIDELITE = """Tu vérifies si des affirmations sont fondées sur un texte source, dans le \
cadre d'un assistant juridique qui ne doit rien avancer que le texte ne dise \
vraiment.

Les affirmations sont numérotées. Pour chacune, réponds true si le texte source \
l'implique (dit la même chose, même reformulée, y compris le régime indiqué par \
l'intitulé « Situé dans ») ; false sinon (elle va au-delà du texte, le contredit, \
ou n'a pas de rapport avec lui). En droit, le présent de l'indicatif exprime une \
obligation : « un état des lieux est établi » signifie qu'il doit l'être. Si le \
texte réserve la règle à une condition ou à un cas particulier (« quand », \
« lorsque », « toutefois », un régime précis) et que l'affirmation la présente \
comme la règle générale sans cette condition, réponds false.

La réponse complète est fournie seulement pour comprendre les renvois d'une \
affirmation (« ce délai », « il ») : elle n'est pas une source, seul le texte \
source fonde une affirmation.

Réponds UNIQUEMENT par ce JSON, sans texte autour, une valeur par affirmation \
dans l'ordre :
{"fondees": [true, false, ...]}"""


CONSIGNE_COMPLETUDE = """Tu relis la réponse d'un assistant juridique destiné à des \
professionnels de l'immobilier. Chaque phrase de la réponse a déjà été vérifiée : \
elle est vraie d'après les articles qu'elle cite. Tu cherches un seul défaut : la réponse \
présente comme la règle commune ce qui n'est qu'un régime d'exception.

Test : un professionnel applique la réponse au cas ordinaire visé par la question \
(location vide, copropriété ordinaire, contribuable sans situation particulière). \
Obtient-il un résultat faux, parce que la réponse :
- donne seulement le chiffre, le délai ou la règle d'une exception, d'une \
dérogation, d'un plafond majoré ou d'un régime particulier (meublé, zone, \
travaux…), sans la règle générale qui s'applique au cas ordinaire ;
- ou donne une règle qui, d'après l'article, ne vaut que sous une condition que \
le cas ordinaire ne remplit pas ?
Si oui, la réponse est trompeuse.

Ne sont PAS des défauts : une réponse qui donne la règle générale sans ses \
exceptions ; une liste ou une énumération incomplète ; une modalité de procédure, \
un délai annexe, une sanction ou une précision non demandés par la question. Si \
la question porte elle-même sur l'exception (« dans quels cas… », « en meublé… »), \
la réponse est complète. En cas de doute, la réponse est complète.

Si la réponse est trompeuse, recopie mot pour mot, depuis les articles récupérés, la \
phrase qui pose la règle générale omise : sans elle, ton verdict est ignoré.

Réponds UNIQUEMENT par ce JSON, sans texte autour :
{"complete": true, "regle_generale": "", "manque": ""}
ou, si elle est trompeuse :
{"complete": false, "regle_generale": "<phrase recopiée mot pour mot>", \
"manque": "<la règle générale omise, en une phrase, avec son article>"}"""


class _NLI(Protocol):
    """Partie d'un fournisseur NLI utilisée ici : les scores d'implication et
    de contradiction d'extraits de texte source face à une affirmation."""

    def evaluer(self, premisses: list[str], hypothese: str) -> list[ScoreNLI]: ...


def _rattacher(introduction: str | None, element: str) -> str:
    """Élément de liste précédé de l'introduction qui lui donne son sujet.

    « notamment » n'est pas répété quand l'introduction le porte déjà
    (« Elle permet notamment notamment : », vu dans les traces).

    Un intitulé sans sujet (« Cas particuliers », « Précautions ») reste
    rattaché, bien qu'il fasse chuter le NLI sur une phrase exacte (1,00 →
    0,00) : essayé le 2026-10-24, l'élément seul a été validé par le NLI
    sans juge alors qu'il était faux (q56 : préavis du locataire « 3 ou 6
    mois », le délai de 6 mois est celui du bailleur) ou sans sujet (q28 :
    « Gestion immobilière. »). Mieux vaut un appel au juge de trop."""
    if not introduction:
        return element
    if re.search(r"\bnotamment$", introduction, re.IGNORECASE):
        return f"{introduction} : {element}"
    return f"{introduction} notamment : {element}"


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

    Une liste écrite sur une seule ligne est d'abord remise à raison d'un
    élément par ligne : vérifiée d'un bloc, elle passait grâce à ses éléments
    exacts même quand l'un d'eux était faux (mesuré sur q22 le 2026-10-03 :
    implication 0,99 pour trois missions dont une prêtée à tort au syndic).
    """
    texte = _TITRE.sub("", _ITALIQUE.sub(r"\1", _GRAS.sub("", _TITRE_GRAS.sub("", texte))))
    affirmations: list[str] = []
    introduction: str | None = None
    # Introduction de la liste principale, rétablie après une sous-liste.
    introduction_principale: str | None = None
    for ligne in _PUCE_EN_LIGNE.sub("\n", texte).splitlines():
        if not ligne.strip():
            continue
        if _PUCE.match(ligne):
            if not ligne[:1].isspace():
                introduction = introduction_principale
            # Un élément de plusieurs phrases donne une affirmation par phrase ;
            # seule la première porte l'introduction de la liste.
            element = _PUCE.sub("", ligne).strip()
            sous_introduction: str | None = None
            for rang, phrase in enumerate(p.strip() for p in _SEPARATEUR_PHRASE.split(element)):
                if not phrase:
                    continue
                if phrase.endswith(":"):
                    # « 3. Saisine du juge : … Le juge peut : » puis une
                    # sous-liste : la phrase introduit les éléments suivants
                    # et n'affirme rien seule (q16, refus à tort le 2026-10-05).
                    sous_introduction = phrase[:-1].strip()
                    continue
                affirmations.append(_rattacher(introduction, phrase) if rang == 0 else phrase)
            if sous_introduction:
                introduction = sous_introduction
            continue
        for phrase in _SEPARATEUR_PHRASE.split(ligne.strip()):
            phrase = phrase.strip()
            if not phrase:
                continue
            if phrase.endswith(":"):
                introduction = introduction_principale = phrase[:-1].strip()
                continue
            introduction = introduction_principale = None
            affirmations.append(phrase)
    affirmations = [
        _REFERENCE_ENTRE_PARENTHESES.sub(
            "", _SELON_ARTICLE.sub("", _ATTRIBUTION.sub("", _OUI_NON.sub("", a)))
        ).strip()
        for a in affirmations
    ]
    # Une ligne faite seulement de ponctuation (un guillemet fermant « » »)
    # n'affirme rien, une question non plus (« Qui peut encaisser l'acompte ? »,
    # q54, 2026-10-23).
    return [
        a
        for a in affirmations
        if re.search(r"\w", a) and not _SANS_CONTENU.match(a) and not a.endswith("?")
    ]


def _phrases_source(contexte: str) -> list[tuple[str | None, str]]:
    """Découpe un article en phrases, chacune avec l'introduction de
    l'énumération à laquelle elle appartient (ou None).

    Les textes issus de PDF gardent les retours à la ligne de la mise en page,
    y compris au milieu d'une phrase : les lignes sont donc recollées avant la
    découpe, seule la ponctuation marque une fin de phrase.

    Dans une énumération légale (« … doivent préciser : » puis des éléments
    terminés par « ; »), un élément seul n'a pas de sujet. L'introduction reste
    attachée aux éléments jusqu'au point qui clôt l'énumération. Un élément
    peut lui-même contenir un point (« a) … à l'article 24. Lorsque … ; ») :
    l'élément suivant (« b) », « 2° ») reprend alors la dernière introduction.
    """
    texte = " ".join(ligne.strip() for ligne in contexte.splitlines() if ligne.strip())
    phrases: list[tuple[str | None, str]] = []
    introduction: str | None = None
    derniere_introduction: str | None = None
    for phrase in _SEPARATEUR_PHRASE_SOURCE.split(texte):
        phrase = phrase.strip()
        if not phrase:
            continue
        if introduction is None and _ELEMENT_ENUMERATION.match(phrase):
            introduction = derniere_introduction
        phrases.append((introduction, phrase))
        if phrase.endswith(":"):
            introduction = derniere_introduction = phrase
        elif phrase.endswith((".", "!", "?")):
            introduction = None
    return phrases


def _extraits(contextes: list[str]) -> list[str]:
    """Découpe les articles en extraits de quelques phrases consécutives, qui
    se chevauchent (fenêtre glissante d'une phrase) pour qu'aucune phrase ne
    soit séparée de ses voisines immédiates.

    Un extrait qui commence au milieu d'une énumération reprend en tête
    l'introduction de celle-ci, pour que le NLI sache de quoi parlent les
    éléments.

    Chaque énumération assez courte donne en plus un extrait complet
    (introduction et tous ses éléments) : une affirmation qui résume toute la
    liste ne peut être impliquée par une fenêtre de trois phrases."""
    extraits: list[str] = []
    for contexte in contextes:
        phrases = _phrases_source(contexte)
        n = _PHRASES_PAR_EXTRAIT
        debuts = range(max(len(phrases) - n + 1, 1)) if phrases else range(0)
        for i in debuts:
            fenetre = phrases[i : i + n]
            introduction = fenetre[0][0]
            textes = [texte for _, texte in fenetre]
            if introduction:
                textes.insert(0, introduction)
            extraits.append(" ".join(textes))
        elements: dict[str, list[str]] = {}
        for introduction, texte in phrases:
            if introduction:
                elements.setdefault(introduction, []).append(texte)
        for introduction, textes in elements.items():
            complet = " ".join([introduction, *textes])
            if len(textes) > n - 1 and len(complet) <= _LONGUEUR_MAX_ENUMERATION:
                extraits.append(complet)
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


def _articles_cites(
    reponse: ReponseGeneree, resultats: list[_ResultatRecherche]
) -> list[_ResultatRecherche]:
    """Articles cités par la réponse, à confronter aux affirmations.

    Restreint aux articles réellement cités : c'est leur contenu, et non celui
    de tous les articles récupérés, que la réponse est censée refléter. Repli
    sur tous les articles récupérés si aucune citation ne correspond (ne
    devrait pas arriver après verifier_citations, mais garde la fonction
    utilisable seule).
    """
    cites = {(c.doc_id, str(c.article)) for c in reponse.citations}
    articles = [
        r for r in resultats if (r.doc_id, str(r.metadata.get("article"))) in cites
    ]
    return articles or list(resultats)


def _contextes_cites(
    reponse: ReponseGeneree, resultats: list[_ResultatRecherche]
) -> list[str]:
    """Textes des articles cités par la réponse (cf. _articles_cites)."""
    return [r.content for r in _articles_cites(reponse, resultats)]


def _juger_fidelite(
    affirmations: list[str], contexte: str, llm: LLM, reponse_complete: str = ""
) -> list[bool]:
    """Repli pour les affirmations que le NLI n'a pas validées : demande à un
    LLM si le texte source fonde chacune d'elles.

    Toutes les affirmations douteuses d'une réponse partent dans un seul
    appel : le quota gratuit de Groq ne permet pas un appel par phrase. Une
    réponse illisible, ou qui ne donne pas une décision par affirmation, est
    traitée comme non fondée, par prudence dans un domaine réglementé.

    La réponse complète accompagne les affirmations : une phrase découpée
    (« Ce délai est réduit à un mois… ») ne dit pas seule de quoi elle parle,
    et le juge la rejetait à tort (mesuré sur q02 le 2026-10-03)."""
    liste = "\n".join(f"{i}. {a}" for i, a in enumerate(affirmations, 1))
    message = f"Texte source :\n{contexte}\n\n"
    if reponse_complete:
        message += f"Réponse complète (pour comprendre les renvois) :\n{reponse_complete}\n\n"
    message += f"Affirmations à vérifier :\n{liste}"
    brut = llm.generer(CONSIGNE_JUGE_FIDELITE, message)
    try:
        fondees = json.loads(brut).get("fondees")
    except (json.JSONDecodeError, AttributeError, TypeError):
        return [False] * len(affirmations)
    if not isinstance(fondees, list) or len(fondees) != len(affirmations):
        return [False] * len(affirmations)
    return [f is True for f in fondees]


# Montant de la réponse : nombre (chiffres groupés par trois avec une espace,
# décimales après la virgule) suivi d'une unité monétaire ou de pourcentage.
_ESPACE_CHIFFRES = r"[   ]"
_MONTANT = re.compile(
    rf"(?<![\d,])(\d{{1,3}}(?:{_ESPACE_CHIFFRES}\d{{3}})+|\d+)(?:,(\d+))?{_ESPACE_CHIFFRES}*"
    r"(?:€|euros?\b|%|pour\s*cent)",
    re.IGNORECASE,
)
# Nombre d'un article : les groupes de chiffres peuvent y être séparés par un
# retour à la ligne de la mise en page PDF (« 1 300\n000 € »).
_NOMBRE_SOURCE = re.compile(r"(?<![\d,])(\d{1,3}(?:\s\d{3})+|\d+)(?:,(\d+))?")


def _valeur(entier: str, decimales: str | None) -> str:
    """Forme comparable d'un nombre : chiffres sans espaces, décimales sans
    zéros finaux (« 0,50 » et « 0,5 » se valent)."""
    decimales = (decimales or "").rstrip("0")
    return re.sub(r"\D", "", entier) + (f",{decimales}" if decimales else "")


def _montant_inconnu(reponse: str, contextes: list[str]) -> str | None:
    """Premier montant (€, %) de la réponse qu'aucun article cité ne contient.

    Le NLI et le juge validaient des montants déformés : « supérieure à
    1 300 € » pour « 1 300 000 € » dans le barème de l'IFI (q48, réponse
    fausse acceptée le 2026-10-23). Une comparaison exacte des chiffres ne
    laisse pas passer ce genre d'erreur. Mesuré hors ligne sur 570 réponses
    acceptées des traces : 6 signalées, l'erreur de q48 et des montants
    calculés par le rédacteur (« 6 % × 5 = 30 % »), qu'aucun article ne fonde.
    Un montant égal à un nombre de l'article plus un est admis : « de
    800 001 € à… » reformule « supérieure à 800 000 € ».

    Ce contrôle tient lieu de consigne au rédacteur : la règle « recopie
    chaque montant, n'en calcule aucun » le faisait refuser des questions
    à taux (q42, 3 refus sur 3 avec la règle, 3 réponses sur 3 sans elle,
    2026-10-24)."""
    source: set[str] = set()
    for contexte in contextes:
        for m in _NOMBRE_SOURCE.finditer(contexte):
            valeur = _valeur(m.group(1), m.group(2))
            source.add(valeur)
            if "," not in valeur:
                source.add(str(int(valeur) + 1))
    for m in _MONTANT.finditer(reponse):
        if _valeur(m.group(1), m.group(2)) not in source:
            return m.group(0).strip()
    return None


def verifier_fidelite(
    reponse: ReponseGeneree,
    resultats: list[_ResultatRecherche],
    nli: _NLI,
    seuil_haut: float,
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
      seuil_contradiction, tirée d'un article de régime général : fondée ;
    - sinon : le NLI seul ne suffit pas, llm sert de juge de repli sur les
      articles entiers s'il est fourni ; sans llm, traité comme non fondé
      (prudence, domaine réglementé).

    Un score NLI bas n'entraîne pas de refus direct : mesuré le 2026-09-30, le
    NLI note bas des affirmations justes (régime déduit de l'intitulé,
    abréviations, phrases de renvoi). Le NLI sert donc de voie rapide
    d'acceptation, et le juge tranche le reste, en un seul appel par réponse.

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

    cites = _articles_cites(reponse, resultats)
    inconnu = _montant_inconnu(reponse.reponse, [r.content for r in cites])
    if inconnu:
        return ReponseGeneree(
            refus=True, motif=f"montant absent des articles cités : « {inconnu} »"
        )
    extraits = _extraits([r.content for r in cites])
    # Extraits d'articles d'un régime particulier : une affirmation que le NLI
    # fonde sur l'un d'eux part au juge, seul à voir l'intitulé « Situé dans ».
    particuliers = set(
        _extraits(
            [r.content for r in cites if _REGIME_PARTICULIER.search(r.metadata.get("section") or "")]
        )
    )
    # Le juge reçoit tous les articles cités ensemble, avec leur intitulé :
    # une affirmation de synthèse (« trois mois en location vide, un mois en
    # meublé ») s'appuie sur plusieurs articles à la fois, et seul l'intitulé
    # dit qu'un article concerne les logements meublés.
    # Les passages cités viennent en tête : dans plus de 15 000 caractères
    # d'articles, le juge passait à côté de la phrase utile (mesuré sur q02
    # le 2026-10-02 : rejet à tort, accepté une fois les passages en tête).
    passages = "\n".join(f"- Article {c.article} : « {c.passage} »" for c in reponse.citations)
    texte_cite = f"Passages cités par la réponse :\n{passages}\n\nArticles complets :\n{_bloc_articles(cites)}"

    douteuses: list[str] = []
    for affirmation in affirmations:
        selection = _preselectionner(extraits, affirmation)
        scores = nli.evaluer(selection, affirmation)
        implication = max((s.implication for s in scores), default=0.0)
        contradiction = max((s.contradiction for s in scores), default=0.0)
        regime = any(
            (e in particuliers or _DEROGATION.search(e)) and s.implication == implication
            for e, s in zip(selection, scores)
        )
        if implication < seuil_haut or contradiction >= seuil_contradiction or regime:
            douteuses.append(affirmation)

    if not douteuses:
        return reponse
    if llm is None:
        fondees = [False] * len(douteuses)
    else:
        fondees = _juger_fidelite(douteuses, texte_cite, llm, reponse.reponse)
    for affirmation, fondee in zip(douteuses, fondees):
        if not fondee:
            # Pas de seconde rédaction sans l'affirmation rejetée : essayée le
            # 2026-10-23 sur q42, q54, q56, elle a rendu deux réponses fausses
            # acceptées (q54 : acompte permis après la vente ; q56 : délai du
            # congé pour reprise appliqué à la vente). Un refus vaut mieux.
            return ReponseGeneree(
                refus=True,
                motif=f"affirmation non fondée sur les sources : « {affirmation} »",
            )

    return reponse


# --- Complétude ---------------------------------------------------------------


def completude_a_verifier(reponse: ReponseGeneree, resultats: list[_ResultatRecherche]) -> bool:
    """La réponse s'appuie-t-elle sur un passage qui peut cacher une règle
    générale : un passage cité de dérogation (« est rehaussée », q58) ou un
    article de régime particulier (meublé, q02) ?

    Le contrôle de complétude appelé sur chaque réponse épuisait le quota
    gratuit du juge : le golden s'arrêtait à q17 au lieu de passer en entier
    (2026-10-22). Les omissions trouvées jusqu'ici viennent toutes de ces deux
    cas ; les autres réponses ne paient plus l'appel."""
    if reponse.refus:
        return False
    if any(_DEROGATION.search(c.passage) for c in reponse.citations):
        return True
    return any(
        _REGIME_PARTICULIER.search(r.metadata.get("section") or "")
        for r in _articles_cites(reponse, resultats)
    )


def verifier_completude(
    question: str,
    reponse: ReponseGeneree,
    resultats: list[_ResultatRecherche],
    llm: LLM,
) -> str | None:
    """Élément des articles récupérés que la réponse tait alors qu'il change la
    réponse à la question ; None si elle est complète.

    verifier_fidelite contrôle que chaque phrase est vraie, pas que rien
    d'essentiel ne manque : « jusqu'à 21 400 € » est vrai mot à mot (plafond
    majoré pour travaux de rénovation énergétique), mais trompeur sans la
    règle générale de 10 700 € (q58, art. 156 du CGI, 2026-10-20). Un LLM
    relit donc la réponse en partant des articles cités.

    Le juge lit tous les articles récupérés, pas seulement les articles
    cités : la règle générale est souvent dans un article que la réponse n'a
    pas cité (q02 : seul l'article des meublés est cité, la règle des trois
    mois est à l'article 15). Il doit recopier la phrase qui pose la règle
    omise, retrouvée ensuite dans ces articles comme une citation : sans cette preuve, il
    signalait des omissions sans appui, articles inventés compris (mesuré sur
    le golden le 2026-10-21 : q16, q23, q56).

    Un refus n'a rien à compléter. Une réponse illisible du LLM vaut
    « complète » : ce contrôle s'ajoute à un garde-fou déjà passé, et un
    échec technique ne doit pas transformer une réponse vérifiée en refus.
    """
    if reponse.refus:
        return None
    message = f"Question : {question}\n\nRéponse :\n{reponse.reponse}\n\n"
    # Un passage cité qui énonce une dérogation est signalé au juge : sans ce
    # signal, il laissait passer « jusqu'à 21 400 € » seul une fois sur trois
    # (q58, 2026-10-22), alors que le passage dit « rehaussée ».
    derogations = [c.passage for c in reponse.citations if _DEROGATION.search(c.passage)]
    if derogations:
        liste = "\n".join(f"- « {p} »" for p in derogations)
        message += (
            "Attention : ces passages cités énoncent une dérogation ; vérifie que la "
            f"réponse donne aussi la règle qu'ils modifient.\n{liste}\n\n"
        )
    message += f"Articles récupérés :\n{_bloc_articles(resultats)}"
    try:
        verdict = json.loads(llm.generer(CONSIGNE_COMPLETUDE, message))
    except json.JSONDecodeError:
        return None
    if not isinstance(verdict, dict) or verdict.get("complete") is not False:
        return None
    regle = str(verdict.get("regle_generale") or "").strip()
    manque = str(verdict.get("manque") or "").strip()
    if not regle or not manque:
        return None
    if not any(_passage_present(regle, r.content, SEUIL_CITATION) for r in resultats):
        return None
    return f"{manque} (« {regle} »)"
