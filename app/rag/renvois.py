"""Renvois d'un article vers d'autres articles, pour les ajouter à la recherche.

Un texte de loi en cite souvent un autre au lieu d'en recopier la règle.
L'article 7 de la loi de 1989 dit seulement que « les deux derniers alinéas de
l'article 1724 du code civil sont applicables à ces travaux » ; la règle (loyer
diminué au-delà de vingt et un jours de travaux) n'est écrite que dans
l'article 1724. Une question sur ces travaux retrouve l'article 7, jamais
l'article 1724 (q55). La recherche ajoute donc aux candidats les articles cités
par les mieux classés, et le reclasseur juge ensuite chacun sur son texte
(cf. retriever.py).

Seuls les renvois qui désignent clairement leur texte sont suivis : « du code
civil », « du code général des impôts », « du présent code », « de la présente
loi », « de la loi n° 65-557 ». Un « article 6 » seul peut viser un autre
texte cité plus haut dans la phrase : le suivre ajoutait quatre fois plus
d'articles sans rien gagner (mesuré le 2026-10-19).

Contenu du fichier :
- famille_document : identifiant commun aux fichiers d'un même code.
- articles_cites : les articles qu'un texte cite, sous forme (document, numéro).
"""
from __future__ import annotations

import re

# Lois du corpus, par numéro (« loi n° 89-462 du 6 juillet 1989 »).
LOIS = {
    "89-462": "loi_06_07_1989",
    "65-557": "loi-n°65-557_du_10_juillet_1965",
    "70-9": "Loi_hoguet",
}
CODE_CIVIL = "code_civil"
# Le code général des impôts est découpé en plusieurs fichiers (code_impot,
# code_impot1…) : un renvoi désigne le code entier, pas l'un d'eux.
CODE_IMPOTS = "code_impot"

_SUFFIXE = r"(?:bis|ter|quater|quinquies|sexies|septies|octies|nonies|decies|undecies|duodecies)"
# Numéro d'article : « 1724 », « 17-1 », « 41 ter », « 150 VC », « 46 AZA
# undecies ». Les lettres majuscules ne sont prises que seules (« VC »), pas
# au début d'un mot (« Les »).
_NUMERO = rf"\d+(?:-\d+)*(?:\s{_SUFFIXE})?(?:\s[A-Z]{{1,4}}(?![a-zéè]))?(?:\s{_SUFFIXE})?"
_LISTE = rf"{_NUMERO}(?:\s*(?:,|\bet\b)\s*{_NUMERO})*"
# Subdivisions éventuelles entre le numéro et le texte visé (« article 24, I,
# 1° de la présente loi »).
_SUBDIVISIONS = r"(?:\s*,?\s*(?:alinéa\s+\w+|[IVX]+|\d+°)\s*,?)*"
_TEXTE = (
    r"(?P<civil>du\s+code\s+civil)"
    r"|(?P<impots>du\s+code\s+général\s+des\s+impôts)"
    r"|(?P<present_code>du\s+présent\s+code)"
    r"|(?P<presente_loi>de\s+la\s+présente\s+loi)"
    r"|de\s+la\s+loi\s+n°\s*(?P<loi>\d\d-\d+)"
)
_RENVOI = re.compile(rf"\b[Aa]rticles?\s+(?P<liste>{_LISTE}){_SUBDIVISIONS}\s*(?:{_TEXTE})")
_SEPARATEUR = re.compile(r"\s*(?:,|\bet\b)\s*")


def famille_document(doc_id: str) -> str:
    """Identifiant sous lequel chercher un article cité : le même pour tous
    les fichiers du code général des impôts, le doc_id tel quel sinon."""
    return CODE_IMPOTS if doc_id.startswith(CODE_IMPOTS) else doc_id


def articles_cites(texte: str, doc_id: str) -> list[tuple[str, str]]:
    """Articles cités par texte, dans l'ordre, sans doublon : (famille du
    document visé, numéro). doc_id est le document du texte, visé par « la
    présente loi » ou « le présent code ». Les lois hors corpus sont
    ignorées."""
    cites: list[tuple[str, str]] = []
    for m in _RENVOI.finditer(texte):
        if m["civil"]:
            cible = CODE_CIVIL
        elif m["impots"]:
            cible = CODE_IMPOTS
        elif m["present_code"] or m["presente_loi"]:
            cible = famille_document(doc_id)
        else:
            cible = LOIS.get(m["loi"])
            if cible is None:
                continue
        for numero in _SEPARATEUR.split(m["liste"]):
            cle = (cible, " ".join(numero.split()))
            if cle not in cites:
                cites.append(cle)
    return cites
