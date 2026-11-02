"""Textes consolidés de Légifrance, lus via l'API PISTE plutôt qu'en PDF.

Un PDF de loi est une mise en page : l'extraction du texte colle des mots
(« suivantsdu »), fond un en-tête d'article dans sa mention de version
(« Article 14Modifié par… ») et laisse en fin d'article l'intitulé de la
section suivante. L'API Légifrance (piste.gouv.fr, gratuite) renvoie au
contraire le texte déjà structuré : chaque article séparé, avec son numéro,
son état (en vigueur ou abrogé), la section qui le contient, et la version en
vigueur à la date demandée.

Le téléchargement se fait hors ligne (scripts/telecharger_legifrance.py) : la
réponse brute de l'API est enregistrée dans data/clean/<doc_id>.json, puis
relue à l'ingestion et par l'évaluation. Ni l'application ni les tests
n'appellent l'API, et une ingestion se rejoue à l'identique.

Les doc_id sont ceux des anciens fichiers texte (loi_06_07_1989, code_civil…) :
le golden dataset et les questions doc2query (questions_articles.jsonl)
désignent les articles par (doc_id, numéro), ils restent valables.

Contenu du fichier :
- TEXTES_LEGIFRANCE : textes du corpus, doc_id → identifiant Légifrance.
- ClientLegifrance : jeton OAuth PISTE puis lecture d'un texte consolidé.
- html_en_texte : texte brut d'un article (l'API le renvoie en HTML).
- articles_legifrance : un RawChunk par article en vigueur d'un texte.
"""
from __future__ import annotations

import re
import time
from html.parser import HTMLParser

import httpx

from app.rag.chunking import NOM_USAGE, RawChunk

URL_JETON = "https://oauth.piste.gouv.fr/api/oauth/token"
URL_API = "https://api.piste.gouv.fr/dila/legifrance/lf-engine-app"

# Identifiants LEGITEXT (texte consolidé), trouvés par /search et /list/code.
# La loi ALUR n'y figure pas : loi modificative, lue à travers les textes
# qu'elle modifie (cf. pipeline.TEXTES_NON_CONSOLIDES).
TEXTES_LEGIFRANCE = {
    "loi_06_07_1989": "LEGITEXT000006069108",
    "loi-n°65-557_du_10_juillet_1965": "LEGITEXT000006068256",
    "Loi_hoguet": "LEGITEXT000006068387",
    "code_civil": "LEGITEXT000006070721",
    "code_impot": "LEGITEXT000006069577",
    "code_impot1": "LEGITEXT000006069568",
    "code_impot2": "LEGITEXT000006069569",
    "code_impot3": "LEGITEXT000006069574",
    "code_impot4": "LEGITEXT000006069576",
}


class ClientLegifrance:
    """Accès à l'API Légifrance. PISTE délivre un jeton valable une heure
    (OAuth « client credentials ») en échange des identifiants de
    l'application ; il est demandé au premier appel puis réutilisé."""

    def __init__(
        self, client_id: str, client_secret: str, http: httpx.Client | None = None
    ):
        if not client_id or not client_secret:
            raise ValueError("PISTE_CLIENT_ID et PISTE_CLIENT_SECRET sont requis (.env).")
        self._client_id = client_id
        self._client_secret = client_secret
        self._http = http or httpx.Client(timeout=120)
        self._jeton: str | None = None
        self._expiration = 0.0

    def _jeton_valide(self) -> str:
        # Une minute de marge : un jeton qui expire pendant l'appel serait refusé.
        if self._jeton is None or time.monotonic() > self._expiration - 60:
            reponse = self._http.post(
                URL_JETON,
                data={
                    "grant_type": "client_credentials",
                    "client_id": self._client_id,
                    "client_secret": self._client_secret,
                    "scope": "openid",
                },
            )
            reponse.raise_for_status()
            corps = reponse.json()
            self._jeton = corps["access_token"]
            self._expiration = time.monotonic() + float(corps.get("expires_in", 3600))
        return self._jeton

    def texte_consolide(self, text_id: str, date: str) -> dict:
        """Texte entier (sections, articles et leur contenu) dans sa version en
        vigueur à la date donnée (AAAA-MM-JJ). /consult/legiPart sert aussi
        bien les lois que les codes ; /consult/code renvoie une erreur 500
        pour le Code civil (essayé le 2026-10-28)."""
        reponse = self._http.post(
            f"{URL_API}/consult/legiPart",
            json={"textId": text_id, "date": date},
            headers={"Authorization": f"Bearer {self._jeton_valide()}"},
        )
        reponse.raise_for_status()
        return reponse.json()


# Balises qui ouvrent une nouvelle ligne : paragraphes, sauts de ligne, lignes
# de tableau.
_BALISES_BLOC = {"p", "br", "div", "tr", "table", "center"}


class _ExtracteurTexte(HTMLParser):
    """Parcourt le HTML d'un article et en garde le texte, une ligne par
    paragraphe. Les cellules d'une ligne de tableau (barème de l'IFI, art.
    977 du CGI) sont séparées par « | » pour rester lisibles côte à côte."""

    def __init__(self) -> None:
        super().__init__()
        self.lignes: list[str] = [""]
        self._cellules_dans_ligne = 0
        self._dans_cellule = False

    def _coupure(self, tag: str) -> None:
        # Dans une cellule, un paragraphe ne coupe pas la ligne du tableau.
        if self._dans_cellule and tag not in ("tr", "table"):
            self.lignes[-1] += " "
        else:
            self.lignes.append("")

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in _BALISES_BLOC:
            self._coupure(tag)
            if tag == "tr":
                self._cellules_dans_ligne = 0
        elif tag in ("td", "th"):
            if self._cellules_dans_ligne:
                self.lignes[-1] += " | "
            self._cellules_dans_ligne += 1
            self._dans_cellule = True

    def handle_endtag(self, tag: str) -> None:
        if tag in ("td", "th"):
            self._dans_cellule = False
        elif tag in _BALISES_BLOC:
            self._coupure(tag)

    def handle_data(self, data: str) -> None:
        self.lignes[-1] += data


def html_en_texte(html: str) -> str:
    """Texte brut d'un contenu HTML d'article : une ligne par paragraphe,
    espaces multiples réduits, lignes vides retirées."""
    extracteur = _ExtracteurTexte()
    extracteur.feed(html)
    extracteur.close()
    lignes = (re.sub(r"\s+", " ", ligne).strip() for ligne in extracteur.lignes)
    return "\n".join(ligne for ligne in lignes if ligne and ligne != "|")


def _titre(texte: dict) -> str:
    """Titre du texte suivi de son nom d'usage s'il en a un (« … (loi
    Hoguet) »), comme chunking.titre_document pour un fichier texte."""
    titre = (texte.get("title") or "").strip()
    appellations = texte.get("appellations") or []
    if appellations:
        nom = NOM_USAGE.match(appellations[0].strip()).group(1)
        titre = f"{titre} ({nom})" if titre else nom
    return titre


def articles_legifrance(texte: dict, doc_id: str) -> list[RawChunk]:
    """Un morceau par article en vigueur d'une réponse de /consult/legiPart.

    Sections et articles sont parcourus dans l'ordre du texte (intOrdre).
    Une section abrogée est sautée avec tout son contenu : l'API y laisse
    parfois des articles marqués « en vigueur », copies de ceux qui ont
    rejoint une autre section (Code civil, art. 2062 à 2064, aussi listés
    sous l'ancien titre « Du nantissement »). Un même numéro vu deux fois
    n'est gardé qu'une fois : deux articles d'un même numéro (CGI, art. 1613
    quater) auraient le même identifiant en base (cf. pipeline.identifiant_chunk).

    Le contenu commence par « Article <numéro> », comme celui découpé dans un
    fichier texte : citations, recherche et reclasseur le lisent ainsi.
    """
    titre = _titre(texte)
    morceaux: list[RawChunk] = []
    vus: set[str] = set()

    def parcourir(noeud: dict, chemin: list[str]) -> None:
        elements = [("article", a) for a in noeud.get("articles") or []] + [
            ("section", s) for s in noeud.get("sections") or []
        ]
        elements.sort(key=lambda e: e[1].get("intOrdre") or 0)
        for genre, element in elements:
            if element.get("etat") != "VIGUEUR":
                continue
            if genre == "section":
                intitule = (element.get("title") or "").strip().rstrip(" .")
                parcourir(element, chemin + [intitule] if intitule else chemin)
                continue
            numero = (element.get("num") or "").strip()
            contenu = html_en_texte(element.get("content") or "")
            if not numero or not contenu or numero in vus:
                continue
            vus.add(numero)
            morceaux.append(
                RawChunk(
                    doc_id=doc_id,
                    article=numero,
                    content=f"Article {numero}\n{contenu}",
                    section=" > ".join(chemin),
                    titre=titre,
                )
            )

    parcourir(texte, [])
    return morceaux
