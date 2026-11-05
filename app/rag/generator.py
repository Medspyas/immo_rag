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

# Écartées le 2026-10-29 (jeu de test figé, 19 questions) : deux règles contre
# les refus à tort (« un article qui interdit répond à « peut-on ? » », « ne
# refuse pas pour une différence de vocabulaire »). Le rédacteur répondait
# davantage, mais faux : t14 (procurations données et non reçues) et t15
# (copropriétaire majoritaire « décide seul ») acceptées à tort.
CONSIGNE ="""Tu es un assistant documentaire pour des professionnels de l'immobilier \
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
- Pour nommer la loi ou le code, reprends la ligne « Texte » de l'article, \
jamais un autre nom.
- Un article signalé « ABROGÉ » ne doit pas être présenté comme le droit en vigueur.
- La ligne « Situé dans » indique le régime auquel un article s'applique \
(par exemple les logements meublés, ou le bail mobilité). Une règle propre à \
un régime particulier ne doit jamais être présentée comme la règle générale : \
précise toujours le régime concerné, et donne d'abord la règle générale \
lorsqu'un article la fournit.
- La ligne « Champ d'application » borne un article à l'application d'un autre \
texte (par exemple une déduction fiscale) : ne présente pas sa règle comme celle \
d'un autre sujet, et nomme ce cadre dans la réponse.
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
    resultats: list[_ResultatRecherche],
    budget: int = LONGUEUR_MAX_CONTEXTE_CARACTERES,
    requete: str = "",
) -> str:
    """Met en forme les articles récupérés pour la consigne.

    Les articles arrivent déjà triés par pertinence (retriever.py) : au-delà
    du budget de caractères, les suivants sont simplement omis plutôt que
    tronqués au milieu — une troncature casserait la vérification des
    citations, faite contre le texte complet de l'article dans `resultats`.
    Le premier article est toujours inclus. S'il dépasse le budget à lui
    seul (art. 31 du CGI, 51 800 caractères depuis les textes PISTE : 413
    de Groq sur t34, 2026-10-30), seuls ses alinéas les plus proches de
    `requete` sont montrés (_raccourcir) ; la citation reste vérifiée contre
    l'article complet.
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
        # Titre de la loi ou du code : sans lui, le modèle devinait le texte
        # d'après l'identifiant et le nommait mal (« code de la construction »
        # pour la loi de 1989, « code civil » pour son article 6).
        titre = r.metadata.get("titre")
        texte = f"\nTexte : {titre}" if titre else ""
        # Article qui ne vaut que pour l'application d'un autre article
        # (« Pour l'application du o du 1° du I de l'article 31 du CGI : »).
        champ = champ_application(r.content)
        portee = f"\nChamp d'application : {champ}" if champ else ""
        entete = (
            f"[{i}] Document : {r.doc_id} — Article {article}{marque}{texte}{situation}{portee}\n"
        )
        contenu = r.content
        if not blocs and len(entete) + len(contenu) > budget:
            contenu = _raccourcir(contenu, requete, budget - len(entete))
        bloc = entete + contenu
        if blocs and longueur + len(bloc) > budget:
            break
        blocs.append(bloc)
        longueur += len(bloc)
    return "\n\n".join(blocs)


def _raccourcir(contenu: str, requete: str, place: int) -> str:
    """Alinéas d'un article trop long qui partagent le plus de mots avec la
    requête, dans l'ordre du texte, jusqu'à `place` caractères ; « […] »
    marque chaque coupure. La première ligne (« Article 31 ») est gardée."""
    lignes = contenu.split("\n")
    mots = {m for m in re.findall(r"\w+", requete.lower()) if len(m) > 3}
    rangs = sorted(
        range(1, len(lignes)),
        key=lambda i: -len(mots & set(re.findall(r"\w+", lignes[i].lower()))),
    )
    gardees = {0}
    longueur = len(lignes[0])
    for i in rangs:
        if longueur + len(lignes[i]) + 1 > place:
            continue
        gardees.add(i)
        longueur += len(lignes[i]) + 1
    morceaux: list[str] = []
    for i, ligne in enumerate(lignes):
        if i in gardees:
            morceaux.append(ligne)
        elif morceaux[-1] != "[…]":
            morceaux.append("[…]")
    return "\n".join(morceaux)


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

    message = f"Question : {question}\n\nArticles disponibles :\n\n{_bloc_articles(resultats, requete=question)}"
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


def _passage_mot_a_mot(passage: str, contenu: str) -> bool:
    """Variante stricte de _passage_present : chaque fragment doit figurer dans
    l'article mot pour mot, à une faute de frappe ou une terminaison près
    (« justifier » pour « justifie »), sans le seuil de portion commune qui
    laissait passer une phrase réécrite partageant 85 % de ses lettres avec
    l'article."""
    mots_article = _normaliser(contenu).split()
    fragments = [f.split() for f in (_normaliser(m) for m in _ELLIPSE.split(passage)) if f]
    if not fragments:
        return False
    debut = 0
    for fragment in fragments:
        n = len(fragment)
        position = next(
            (
                i for i in range(debut, len(mots_article) - n + 1)
                if all(_meme_mot(a, b) for a, b in zip(fragment, mots_article[i : i + n]))
            ),
            None,
        )
        if position is None:
            return False
        debut = position + n
    return True


def _meme_mot(mot: str, original: str) -> bool:
    """Même mot, à une terminaison (« établie » / « établies ») ou une faute de
    frappe près. Un nombre doit être identique."""
    if mot == original:
        return True
    if any(ch.isdigit() for ch in mot + original) or min(len(mot), len(original)) < 4:
        return False
    court, long = sorted((mot, original), key=len)
    return (long.startswith(court) and len(long) - len(court) <= 3) or _faute_de_frappe(mot, original)


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
    articles = _bloc_articles(resultats, LONGUEUR_MAX_CONTEXTE_CARACTERES - len(suite), question)
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
# Passage cité qui ne vaut que dans le cas posé juste avant dans l'article
# (« Dans ce cas, le vendeur doit… ») : recopié seul, il fait d'une règle
# conditionnelle une obligation générale (v065, art. 8-1 de la loi de 1965 :
# notification au syndic seulement si le règlement prévoit un droit de
# priorité, 2026-11-03).
_CAS_PRECEDENT = re.compile(
    r"^\W*(?:dans ce cas|en ce cas|dans cette hypothèse|dans ces cas|en pareil cas)\b",
    re.IGNORECASE,
)
# Article dont la première phrase borne la portée à l'application d'un autre
# texte : l'art. 2 terdecies G de l'annexe III du CGI fixe des plafonds de
# ressources « pour l'application du o du 1° du I de l'article 31 » (déduction
# fiscale du bailleur) ; le juge l'acceptait comme les plafonds pour obtenir un
# logement social (vh13, hors corpus, 2026-11-05).
_CHAMP = re.compile(r"\A[^\n]*\n\s*(Pour l['’]application d[^:\n]{1,300}?)\s*:")


def champ_application(contenu: str) -> str | None:
    """Phrase qui borne la portée d'un article (« Pour l'application du … de
    l'article 31 du code général des impôts »), ou None."""
    m = _CHAMP.search(contenu)
    return m.group(1).strip() if m else None


# Acteurs juridiques (qui détient le droit, qui supporte l'obligation). Une
# affirmation qui nomme un acteur absent des articles cités part au juge : le
# NLI fondait « la commune dispose de six mois » sur l'art. 1904 du CGI, où ce
# délai est celui de l'administration fiscale (vh05, 2026-11-04). Un synonyme
# (« preneur » pour « locataire ») ne coûte qu'un appel au juge, pas un refus.
_ACTEURS = re.compile(
    r"\b(commune|maire|préfet|bailleur|locataire|preneur|syndic|syndicat|copropriétaire"
    r"|vendeur|acquéreur|acheteur|notaire|administration|trésor|indivisaire"
    r"|usufruitier|nu-propriétaire|mandataire|caution|organisme)s?\b",
    re.IGNORECASE,
)


def _acteurs(texte: str) -> set[str]:
    """Acteurs juridiques nommés dans un texte, au singulier et en minuscules."""
    return {m.group(1).lower() for m in _ACTEURS.finditer(texte)}


# Exception posée par une affirmation de la réponse (« sauf si… »).
_EXCEPTION = re.compile(r"\bsauf\b|à l'exception|excepté|hormis", re.IGNORECASE)
# Condition, délai ou terme posés par la suite d'une réponse : la réponse
# directe négative qui les précède est relue (CONSIGNE_JUGE_REPONSE_DIRECTE).
# Restreint aux réponses négatives (« ne reste pas tenu », n05) : sans cette
# restriction, 21 réponses acceptées sur ~90 auraient demandé un appel de plus.
_CONDITION = re.compile(
    r"\blorsqu|\bquand\b|\b(?:à|au) défaut\b|\bau plus tard\b|\bjusqu|\btant qu"
    r"|\bsous réserve\b|\bà condition\b|\bà compter\b|\bdans un délai\b",
    re.IGNORECASE,
)
_NEGATION = re.compile(r"\bne\b|\bn'|\bn’|\bnon\b", re.IGNORECASE)
# Nombre d'extraits soumis au NLI pour chaque affirmation, retenus parmi ceux
# qui partagent le plus de mots avec elle : évite de passer tout l'article au
# modèle, ce qui coûterait un passage NLI par extrait.
_EXTRAITS_PAR_AFFIRMATION = 8

# Exception d'un article greffée sur la règle d'un autre : t05 (2026-10-30),
# « sauf étudiant ou apprenti » (art. 22-1, bailleur assuré) rattaché à
# l'interdiction de la cosignature des parents (art. 22-2), accepté.
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
comme la règle générale sans cette condition, réponds false. De même si \
l'affirmation rattache à une règle une exception (« sauf… ») que le texte attache \
à une autre règle : chaque exception vaut seulement pour la phrase qui la porte. Un article doté \
d'une ligne « Champ d'application » ne vaut que dans ce cadre : une affirmation \
qui applique sa règle à un autre sujet (celui de la question, par exemple) est \
false.

La réponse complète est fournie seulement pour comprendre les renvois d'une \
affirmation (« ce délai », « il ») : elle n'est pas une source, seul le texte \
source fonde une affirmation.
La question de l'utilisateur, si elle est fournie, décrit sa situation (dates, \
type de bien, décision déjà prise par sa commune…). Une affirmation qui applique \
la règle du texte aux faits de cette situation est fondée si la règle l'est : ces \
faits n'ont pas à figurer dans le texte source. La question ne fonde en revanche \
jamais une règle de droit, une durée, un montant, une condition légale, ni qui \
détient un droit ou supporte une obligation : si le texte donne un délai à \
l'administration, il ne le donne pas à la commune.
Réponds UNIQUEMENT par ce JSON, sans texte autour, une valeur par affirmation \
dans l'ordre :
{"fondees": [true, false, ...]}"""


# Seconde lecture des seules affirmations à exception acceptées par le juge
# groupé : il acceptait une fois sur trois l'exception de l'art. 22-1 greffée
# sur la règle de l'art. 22-2 (t05, 2026-10-31). Recopier la phrase qui porte
# l'exception l'oblige à la lire avant de trancher.
CONSIGNE_JUGE_EXCEPTION = """Tu vérifies des affirmations d'un assistant juridique \
qui posent une règle et une exception (« sauf… », « à l'exception de… »).

Pour chaque affirmation :
1. recopie mot pour mot, depuis le texte source, la phrase qui porte l'exception ;
2. écris la règle que cette phrase soumet à l'exception, avec qui elle vise et ses \
conditions (« le cautionnement ne peut être demandé par un bailleur qui a souscrit \
une assurance ») ;
3. réponds true seulement si la règle de l'affirmation est cette règle-là, avec les \
mêmes personnes et les mêmes conditions ; false si l'affirmation rattache \
l'exception à une autre règle, en omet une condition, ou si aucune phrase du texte \
ne porte cette exception.

Réponds UNIQUEMENT par ce JSON, sans texte autour, un objet par affirmation dans \
l'ordre :
{"verdicts": [{"phrase": "<phrase recopiée>", "regle": "<règle soumise à \
l'exception>", "fondee": true}, ...]}"""


# Seconde lecture de la réponse directe (1re affirmation) quand la suite de la
# réponse pose des conditions ou un délai : n05 (2026-10-31), « ne reste pas
# tenu de payer la part des autres » en tête, puis la solidarité qui dure
# jusqu'au nouveau colocataire ou six mois. Le juge groupé l'acceptait 6/6,
# même averti qu'une nuance plus loin ne rattrape pas une phrase fausse.
CONSIGNE_JUGE_REPONSE_DIRECTE = """Tu vérifies la phrase qui donne la réponse \
directe d'un assistant juridique (la première phrase de sa réponse).

Pour chaque affirmation :
1. recopie mot pour mot, depuis le texte source, la phrase qui fonde cette réponse ;
2. écris la règle de cette phrase, avec ses conditions et ses délais (« la \
solidarité prend fin lorsqu'un nouveau colocataire figure au bail, au plus tard six \
mois après le congé ») ;
3. réponds true si l'affirmation, lue seule, reste vraie au regard de cette règle ; \
false si elle affirme sans réserve ce que la règle ne prévoit que sous une condition, \
après un délai ou jusqu'à un terme, ou si elle la contredit. Les phrases suivantes \
de la réponse ne corrigent pas une première phrase fausse.

Réponds UNIQUEMENT par ce JSON, sans texte autour, un objet par affirmation dans \
l'ordre :
{"verdicts": [{"phrase": "<phrase recopiée>", "regle": "<règle avec ses \
conditions>", "fondee": true}, ...]}"""


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
le cas ordinaire ne remplit pas ;
- ou tait la règle qui répond directement à la question, posée par un article \
récupéré, au profit d'une règle voisine ou de procédure ?
Si oui, la réponse est trompeuse.

Ne sont PAS des défauts : une réponse qui donne la règle générale sans ses \
exceptions ; une liste ou une énumération incomplète ; une modalité de procédure, \
un délai annexe, une sanction ou une précision non demandés par la question. Si \
la question porte elle-même sur l'exception (« dans quels cas… », « en meublé… »), \
la réponse est complète. En cas de doute, la réponse est complète.

Si la réponse est trompeuse, recopie mot pour mot, depuis les articles récupérés, la \
phrase qui pose la règle générale omise ; si elle n'y figure pas, recopie l'intitulé \
du régime particulier signalé plus haut. Sans cette preuve, ton verdict est ignoré.

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


def _forme_affirmation(phrase: str) -> str:
    """Phrase débarrassée de ce qui n'est pas à vérifier : « Oui, » en tête,
    formule d'attribution, référence d'article (cf. les expressions plus haut)."""
    return _REFERENCE_ENTRE_PARENTHESES.sub(
        "", _SELON_ARTICLE.sub("", _ATTRIBUTION.sub("", _OUI_NON.sub("", phrase)))
    ).strip()


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
    affirmations = [_forme_affirmation(a) for a in affirmations]
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
    affirmations: list[str],
    contexte: str,
    llm: LLM,
    reponse_complete: str = "",
    question: str = "",
) -> list[bool]:
    """Repli pour les affirmations que le NLI n'a pas validées : demande à un
    LLM si le texte source fonde chacune d'elles.

    Toutes les affirmations douteuses d'une réponse partent dans un seul
    appel : le quota gratuit de Groq ne permet pas un appel par phrase. Une
    réponse illisible, ou qui ne donne pas une décision par affirmation, est
    traitée comme non fondée, par prudence dans un domaine réglementé.

    La réponse complète accompagne les affirmations : une phrase découpée
    (« Ce délai est réduit à un mois… ») ne dit pas seule de quoi elle parle,
    et le juge la rejetait à tort (mesuré sur q02 le 2026-10-03).

    La question accompagne aussi les affirmations : une réponse qui reprend la
    situation de l'utilisateur (« logements de plus de 40 ans », v089 ; « si
    ma commune a voté », v102) était rejetée, ces faits ne figurant pas dans
    le texte source (2026-11-04)."""
    liste = "\n".join(f"{i}. {a}" for i, a in enumerate(affirmations, 1))
    message = ""
    if question:
        message += f"Question de l'utilisateur (sa situation) :\n{question}\n\n"
    message += f"Texte source :\n{contexte}\n\n"
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


def _juger_exceptions(
    affirmations: list[str],
    contextes: list[str],
    texte_cite: str,
    llm: LLM,
    consigne: str = CONSIGNE_JUGE_EXCEPTION,
) -> list[bool]:
    """Seconde lecture des affirmations acceptées par le juge groupé :
    affirmations à exception (CONSIGNE_JUGE_EXCEPTION) ou réponse directe
    suivie de conditions (CONSIGNE_JUGE_REPONSE_DIRECTE).

    Le verdict compte seulement si la phrase recopiée se retrouve dans un
    article cité, comme une citation. Sans preuve, ou réponse illisible,
    l'affirmation garde le verdict du juge groupé : ce contrôle s'ajoute à
    lui et ne doit pas refuser sur un échec technique."""
    liste = "\n".join(f"{i}. {a}" for i, a in enumerate(affirmations, 1))
    message = f"Texte source :\n{texte_cite}\n\nAffirmations à vérifier :\n{liste}"
    try:
        verdicts = json.loads(llm.generer(consigne, message)).get("verdicts")
    except (json.JSONDecodeError, AttributeError, TypeError):
        return [True] * len(affirmations)
    if not isinstance(verdicts, list) or len(verdicts) != len(affirmations):
        return [True] * len(affirmations)
    fondees = []
    for v in verdicts:
        phrase = str(v.get("phrase") or "").strip() if isinstance(v, dict) else ""
        prouve = bool(phrase) and any(
            _passage_present(phrase, c, SEUIL_CITATION) for c in contextes
        )
        fondees.append(not (prouve and v.get("fondee") is False))
    return fondees


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


# Passage mis entre guillemets dans le texte de la réponse (« … », “ … ” ou
# " … "). En dessous de MOTS_MIN_GUILLEMETS mots, c'est un terme repris
# (« délégation de vote ») plutôt qu'une citation.
_GUILLEMETS = re.compile(r"«\s*([^«»]+?)\s*»|“([^“”]+?)”|\"([^\"]+?)\"")
MOTS_MIN_GUILLEMETS = 5


def _guillemets_inconnus(reponse: str, contextes: list[str]) -> str | None:
    """Premier passage entre guillemets du texte de la réponse qu'aucun article
    cité ne contient.

    Seules les citations de la liste `citations` étaient vérifiées : le texte
    rédigé pouvait présenter entre guillemets une phrase réécrite comme si
    elle venait de la loi. Sur t02 (2026-10-29), « si le bailleur manifeste sa
    volonté … est réputé avoir renoncé » inversait l'article 17-1 (« À défaut
    de manifester sa volonté … ») et la réponse était acceptée. Le passage est
    comparé aux articles mot à mot (_passage_mot_a_mot) : la comparaison des
    citations, qui tolère 15 % d'écart, acceptait l'inversion de t02.
    Rejoué sur 130 réponses acceptées des traces : t02 seule signalée."""
    for m in _GUILLEMETS.finditer(reponse):
        # Un ajout entre crochets (« jusqu'à [la fin du bail] ») est une
        # précision du rédacteur : traité comme une coupure.
        passage = re.sub(r"\[[^\]]*\]", " … ", next(g for g in m.groups() if g))
        if len(_normaliser(passage).split()) < MOTS_MIN_GUILLEMETS:
            continue
        if not any(_passage_mot_a_mot(passage, c) for c in contextes):
            return passage
    return None


def _retirer_affirmations(texte: str, rejetees: list[str]) -> str | None:
    """Texte de la réponse sans les phrases dont l'affirmation a été rejetée.

    Chaque phrase du texte est ramenée à la forme vérifiée
    (_forme_affirmation) ; un élément de liste a été vérifié précédé de son
    introduction (_rattacher), d'où la comparaison sur la fin. Une ligne vidée
    disparaît, de même qu'une introduction de liste (« … : ») dont tous les
    éléments ont été retirés. Renvoie None si une affirmation rejetée n'a pas
    été retrouvée dans le texte : la réponse est alors refusée, par prudence.
    """
    restantes = list(rejetees)
    lignes: list[str] = []
    for ligne in _PUCE_EN_LIGNE.sub("\n", texte).splitlines():
        puce = _PUCE.match(ligne)
        tete = puce.group(0) if puce else ""
        corps = ligne[len(tete):].strip()
        if not corps:
            lignes.append(ligne)
            continue
        gardees: list[str] = []
        for phrase in _SEPARATEUR_PHRASE.split(corps):
            forme = _forme_affirmation(_ITALIQUE.sub(r"\1", _GRAS.sub("", phrase)).strip())
            cible = next(
                (r for r in restantes if forme and (r == forme or r.endswith(f" : {forme}"))), None
            )
            if cible is None:
                gardees.append(phrase)
            else:
                restantes.remove(cible)
        if gardees:
            lignes.append(tete + " ".join(gardees))
        elif puce is None:
            # Une ligne de texte vidée laisse une ligne vide, pas un élément.
            lignes.append("")
    if restantes:
        return None
    # Introduction de liste restée sans élément : retirée.
    resultat: list[str] = []
    for i, ligne in enumerate(lignes):
        suivante = next((s for s in lignes[i + 1 :] if s.strip()), "")
        if ligne.rstrip().endswith(":") and not _PUCE.match(suivante):
            continue
        resultat.append(ligne)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(resultat)).strip()


def _tronquer(reponse: ReponseGeneree, affirmations: list[str], rejetees: list[str]) -> ReponseGeneree:
    """La réponse privée de ses affirmations rejetées, ou un refus.

    Refus si la première affirmation est rejetée : elle porte la réponse
    directe (« Oui, mais seulement à l'échéance du bail », q56), et la retirer
    laisserait un « Oui, » sans objet ou une réponse sans conclusion. Refus
    aussi s'il ne reste rien, ou si une phrase rejetée n'est pas retrouvée.
    """
    motif = f"affirmation non fondée sur les sources : « {rejetees[0]} »"
    if affirmations[0] in rejetees or len(rejetees) == len(affirmations):
        return ReponseGeneree(refus=True, motif=motif)
    texte = _retirer_affirmations(reponse.reponse, rejetees)
    if not texte:
        return ReponseGeneree(refus=True, motif=motif)
    return reponse.model_copy(update={"reponse": texte, "motif": f"réponse tronquée — {motif}"})


def verifier_fidelite(
    reponse: ReponseGeneree,
    resultats: list[_ResultatRecherche],
    nli: _NLI,
    seuil_haut: float,
    seuil_contradiction: float,
    llm: LLM | None = None,
    question: str = "",
) -> ReponseGeneree:
    """Garde-fou étage 2 (suite) : la réponse rédigée colle-t-elle aux articles
    qu'elle cite ?

    `question` sert au contrôle des montants (un montant repris de la question,
    « une hausse de 15 % », v008, n'est pas tiré des sources) et est montrée
    au juge comme situation de l'utilisateur (v089, v102).

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
    inconnu = _montant_inconnu(reponse.reponse, [r.content for r in cites] + [question])
    if inconnu:
        # L'affirmation qui porte le montant est retirée comme une affirmation
        # rejetée (cf. _tronquer) : « Ces taux, cumulés, représentent 100 % »
        # (calcul du rédacteur, q42) faisait refuser une réponse juste.
        porteuses = [a for a in affirmations if inconnu in a] or [affirmations[0]]
        tronquee = _tronquer(reponse, affirmations, porteuses)
        if tronquee.refus:
            return ReponseGeneree(
                refus=True, motif=f"montant absent des articles cités : « {inconnu} »"
            )
        return verifier_fidelite(
            tronquee, resultats, nli, seuil_haut, seuil_contradiction, llm, question
        )
    fausse = _guillemets_inconnus(reponse.reponse, [r.content for r in cites])
    if fausse:
        # Même traitement qu'un montant inconnu : l'affirmation qui porte la
        # fausse citation est retirée, refus si c'est la réponse directe.
        porteuses = [a for a in affirmations if _normaliser(fausse) in _normaliser(a)]
        tronquee = _tronquer(reponse, affirmations, porteuses or [affirmations[0]])
        if tronquee.refus:
            return ReponseGeneree(
                refus=True, motif=f"citation absente des articles cités : « {fausse} »"
            )
        return verifier_fidelite(
            tronquee, resultats, nli, seuil_haut, seuil_contradiction, llm, question
        )
    extraits = _extraits([r.content for r in cites])
    # Extraits d'articles d'un régime particulier : une affirmation que le NLI
    # fonde sur l'un d'eux part au juge, seul à voir l'intitulé « Situé dans ».
    particuliers = set(
        _extraits(
            [
                r.content
                for r in cites
                if _REGIME_PARTICULIER.search(r.metadata.get("section") or "")
                or champ_application(r.content)
            ]
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
    texte_cite = f"Passages cités par la réponse :\n{passages}\n\nArticles complets :\n{_bloc_articles(cites, requete=reponse.reponse)}"

    acteurs_cites = _acteurs(" ".join(r.content for r in cites))
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
        # Une affirmation qui pose une exception part au juge : le NLI fondait
        # « sauf si le logement est loué à un étudiant » (art. 22-1) rattaché
        # à la cosignature des parents (art. 22-2), t05 le 2026-10-30.
        exception = _EXCEPTION.search(affirmation) is not None
        acteur = not _acteurs(affirmation) <= acteurs_cites
        if (
            implication < seuil_haut
            or contradiction >= seuil_contradiction
            or regime
            or exception
            or acteur
        ):
            douteuses.append(affirmation)

    if not douteuses:
        return reponse
    if llm is None:
        fondees = [False] * len(douteuses)
    else:
        fondees = _juger_fidelite(douteuses, texte_cite, llm, reponse.reponse, question)
        exceptions = [a for a, f in zip(douteuses, fondees) if f and _EXCEPTION.search(a)]
        if exceptions:
            relues = dict(
                zip(exceptions, _juger_exceptions(exceptions, [r.content for r in cites], texte_cite, llm))
            )
            fondees = [f and relues.get(a, True) for a, f in zip(douteuses, fondees)]
        # Réponse directe négative acceptée par le juge groupé alors que la
        # suite pose des conditions : relue seule (n05). Celles qui portent une exception
        # ont déjà eu leur seconde lecture.
        directe = affirmations[0]
        suite = " ".join(affirmations[1:])
        if (
            directe in douteuses
            and fondees[douteuses.index(directe)]
            and not _EXCEPTION.search(directe)
            and _NEGATION.search(directe)
            and _CONDITION.search(suite)
        ):
            relue = _juger_exceptions(
                [directe], [r.content for r in cites], texte_cite, llm,
                consigne=CONSIGNE_JUGE_REPONSE_DIRECTE,
            )[0]
            fondees[douteuses.index(directe)] = relue
    rejetees = [a for a, fondee in zip(douteuses, fondees) if not fondee]
    if not rejetees:
        return reponse
    # Les affirmations rejetées sont retirées du texte, le reste est gardé tel
    # quel (cf. _tronquer) : jusqu'au 2026-10-28, une seule phrase rejetée
    # faisait refuser toute la réponse. Rejoué le 2026-10-28 sur les 13
    # réponses refusées des traces récentes (réponses figées, verdicts du juge
    # enregistrés) : 5 récupérées et justes (q42, q46, q54, n08 ; q25 juste
    # sur l'essentiel, avec un détail inexact que le juge acceptait déjà),
    # aucune nouvelle réponse fausse ; q32, q58 et q56 (deux fois) restent
    # refusées. 4 des 13 étaient acceptées au simple rejeu du juge, sans
    # troncature : le juge ne rend pas toujours le même verdict.
    # Pas de seconde rédaction sans l'affirmation rejetée : essayée le
    # 2026-10-23 sur q42, q54, q56, elle a rendu deux réponses fausses
    # acceptées (q54 : acompte permis après la vente ; q56 : délai du congé
    # pour reprise appliqué à la vente). Retirer ne réécrit rien.
    return _tronquer(reponse, affirmations, rejetees)


# --- Complétude ---------------------------------------------------------------


def completude_a_verifier(reponse: ReponseGeneree, resultats: list[_ResultatRecherche]) -> bool:
    """La réponse s'appuie-t-elle sur un passage qui peut cacher une règle
    générale : un passage cité de dérogation (« est rehaussée », q58), un
    article de régime particulier (meublé, q02), un article cité trop long
    (t34), ou un premier article de la recherche laissé de côté (t37) ?

    Le contrôle de complétude appelé sur chaque réponse épuisait le quota
    gratuit du juge : le golden s'arrêtait à q17 au lieu de passer en entier
    (2026-10-22). Les omissions trouvées jusqu'ici relèvent toutes de ces
    cas ; les autres réponses ne paient plus l'appel."""
    if reponse.refus:
        return False
    if any(_DEROGATION.search(c.passage) or _CAS_PRECEDENT.search(c.passage) for c in reponse.citations):
        return True
    # Premier article de la recherche non cité : la règle qui répond le plus
    # directement peut y être. Couvre les omissions connues q02, q58 et t37
    # (taxe foncière : art. 1415, « faits existants au 1er janvier », tu au
    # profit de la mutation cadastrale) ; environ une réponse sur cinq
    # (traces du 2026-10-28 et du 2026-10-30).
    if _premier_non_cite(reponse, resultats) is not None:
        return True
    # Un article cité plus long que le budget mêle règle générale et régimes
    # optionnels : t34 (2026-10-30) donnait l'amortissement de l'agrandissement
    # « lorsque l'option est exercée » (art. 31 du CGI) pour la règle, qui
    # l'exclut (b). Cas rare : le quota du juge n'en souffre pas.
    return any(
        _REGIME_PARTICULIER.search(r.metadata.get("section") or "")
        or champ_application(r.content)
        or len(r.content) > LONGUEUR_MAX_CONTEXTE_CARACTERES
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
    conditionnels = [c.passage for c in reponse.citations if _CAS_PRECEDENT.search(c.passage)]
    if conditionnels:
        liste = "\n".join(f"- « {p} »" for p in conditionnels)
        message += (
            "Attention : ces passages cités ne s'appliquent que dans le cas posé juste "
            "avant eux dans l'article ; vérifie que la réponse donne cette condition.\n"
            f"{liste}\n\n"
        )
    # Même signal pour un article cité de régime particulier, avec son
    # intitulé : t15 (2026-10-30) présentait l'art. 41-16 (syndicats à deux
    # copropriétaires) comme la règle de toute copropriété, et la règle
    # générale (art. 22) n'était pas parmi les articles récupérés.
    regimes = _intitules_particuliers(reponse, resultats)
    if regimes:
        liste = "\n".join(f"- « {s} »" for s in regimes)
        message += (
            "Attention : ces articles cités relèvent d'un régime particulier. Une "
            "réponse qui applique leur règle sans nommer ce régime (par exemple « en "
            "meublé », « dans un syndicat de deux copropriétaires ») est trompeuse, "
            f"sauf si la question le nomme elle-même.\n{liste}\n\n"
        )
    premier = _premier_non_cite(reponse, resultats)
    if premier is not None:
        message += (
            f"Attention : l'article {premier.metadata.get('article', '?')}, classé premier "
            "par la recherche, n'est pas cité par la réponse ; vérifie qu'il ne pose pas "
            "la règle qui répond directement à la question.\n\n"
        )
    message += f"Articles récupérés :\n{_bloc_articles(resultats, requete=question)}"
    try:
        verdict = json.loads(llm.generer(CONSIGNE_COMPLETUDE, message))
    except json.JSONDecodeError:
        return None
    if not isinstance(verdict, dict) or verdict.get("complete") is not False:
        return None
    regle = str(verdict.get("regle_generale") or "").strip()
    manque = str(verdict.get("manque") or "").strip()
    # Preuve : une phrase d'un article récupéré, ou l'intitulé qui borne le
    # régime particulier quand la règle générale n'a pas été récupérée.
    if regle and manque and (
        any(_passage_present(regle, r.content, SEUIL_CITATION) for r in resultats)
        or any(_passage_present(regle, s, SEUIL_CITATION) for s in regimes)
    ):
        return f"{manque} (« {regle} »)"
    # Omission signalée sans preuve retrouvée : sur un article de régime
    # particulier, son intitulé suffit — t15 (2026-10-30), le juge voyait
    # l'omission mais recopiait une règle générale absente des articles
    # récupérés (art. 22 non retrouvé).
    return precision_regimes(reponse, resultats)


def precision_regimes(
    reponse: ReponseGeneree, resultats: list[_ResultatRecherche]
) -> str | None:
    """Phrase qui rappelle le régime particulier des articles cités (« L'article
    41-16 relève des dispositions : « Section 2 : … deux copropriétaires ». »),
    tirée de leur intitulé ; None si aucun article cité n'en relève."""
    phrases = []
    for r in _articles_cites(reponse, resultats):
        section = r.metadata.get("section") or ""
        if _REGIME_PARTICULIER.search(section):
            dernier = section.split(" > ")[-1].strip()
            phrases.append(f"L'article {r.metadata.get('article', '?')} relève de : « {dernier} ».")
        champ = champ_application(r.content)
        if champ:
            phrases.append(f"L'article {r.metadata.get('article', '?')} vaut « {champ} ».")
    return " ".join(dict.fromkeys(phrases)) or None


def ajouter_precision(
    reponse: ReponseGeneree, resultats: list[_ResultatRecherche]
) -> ReponseGeneree:
    """Réponse complétée du rappel de régime particulier (precision_regimes).

    Quand une omission est signalée et que la seconde rédaction échoue aux
    garde-fous, la première est gardée : le rappel, tiré tel quel des
    intitulés, l'empêche de passer pour la règle commune (t15)."""
    precision = precision_regimes(reponse, resultats)
    if reponse.refus or precision is None:
        return reponse
    return reponse.model_copy(update={"reponse": f"{reponse.reponse}\n\nPrécision : {precision}"})


# Article qui borne le champ (« … de l'article 31 du code … » → « 31 »).
_ARTICLE_VISE = re.compile(r"\barticle\s+(\d[\w-]*)")


def rappeler_champ(
    reponse: ReponseGeneree, resultats: list[_ResultatRecherche]
) -> ReponseGeneree:
    """Réponse complétée du champ d'application d'un article cité, quand elle
    ne nomme pas l'article qui borne ce champ.

    vh13 (2026-11-05) : les plafonds de l'art. 2 terdecies G (annexe III du
    CGI), qui ne valent que pour la déduction de l'article 31, étaient donnés
    comme ceux du logement social ; le juge acceptait, malgré la ligne
    « Champ d'application » et sa consigne. Le rappel est tiré tel quel de
    l'article : il ne coûte aucun appel et ne peut rien inventer."""
    if reponse.refus:
        return reponse
    texte = _normaliser(reponse.reponse)
    phrases = []
    for r in _articles_cites(reponse, resultats):
        champ = champ_application(r.content)
        vise = _ARTICLE_VISE.search(champ or "")
        if champ is None or (vise and _normaliser(f"article {vise.group(1)}") in texte):
            continue
        phrases.append(f"L'article {r.metadata.get('article', '?')} vaut seulement « {champ} ».")
    if not phrases:
        return reponse
    precision = " ".join(dict.fromkeys(phrases))
    return reponse.model_copy(update={"reponse": f"{reponse.reponse}\n\nPrécision : {precision}"})


def _premier_non_cite(
    reponse: ReponseGeneree, resultats: list[_ResultatRecherche]
) -> _ResultatRecherche | None:
    """Premier article récupéré, s'il n'est pas cité par la réponse."""
    if not resultats or not reponse.citations:
        return None
    premier = resultats[0]
    cites = {(c.doc_id, str(c.article)) for c in reponse.citations}
    if (premier.doc_id, str(premier.metadata.get("article"))) in cites:
        return None
    return premier


def _intitules_particuliers(
    reponse: ReponseGeneree, resultats: list[_ResultatRecherche]
) -> list[str]:
    """Dernier niveau d'intitulé des articles cités qui relèvent d'un régime
    particulier (« Section 2 : Dispositions particulières aux syndicats dont
    le nombre de voix est réparti entre deux copropriétaires »), sans doublon."""
    intitules: list[str] = []
    for r in _articles_cites(reponse, resultats):
        section = r.metadata.get("section") or ""
        if not _REGIME_PARTICULIER.search(section):
            continue
        dernier = section.split(" > ")[-1].strip()
        if dernier not in intitules:
            intitules.append(dernier)
    return intitules
