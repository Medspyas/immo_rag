"""Découpage d'un texte de loi en morceaux ("chunks").

Un texte de loi nettoyé (data/clean/*.txt) est fourni en entrée ; une liste de
morceaux est produite en sortie, à raison d'un morceau par article. La coupure
suit les articles plutôt qu'un nombre de mots fixe, ce qui évite de séparer un
article en deux. Le contenu situé avant le premier article (titre) et après le
dernier (historique parlementaire) est écarté : il ne constitue pas une source
citable.

Le nettoyage en amont a entouré chaque article abrogé de deux marqueurs. Ce
module s'en sert pour écarter ces morceaux : un article abrogé n'est plus le
droit en vigueur (cf. le marqueur source lui-même, « ne pas citer comme droit
actuel ») et ne doit pas être indexé comme une source citable. C'est aussi ce
qui évite une collision d'identifiant : Légifrance liste parfois plusieurs
versions successives d'un même numéro d'article (l'ancienne abrogée, puis
l'actuelle) — sans ce filtre, les deux généreraient le même identifiant
(doc_id + numéro, cf. pipeline.py) et l'une écraserait l'autre en base, au
hasard de l'ordre du fichier source.

Contenu du fichier :
- RawChunk : un morceau avant traitement (texte d'un article et repères :
  document, numéro d'article, abrogé ou non).
- decouper_en_articles : réalise le découpage.
"""
from __future__ import annotations

import re

from pydantic import BaseModel, Field

# Détecte le début d'un article : "Article 1", "Article L. 111-1"...
# Le numéro est capturé afin de pouvoir citer l'article dans une réponse.
DEBUT_ARTICLE = re.compile(r"^Article\s+([\w.\s-]+?)\s*$")

MARQUEUR_DEBUT_ABROGE = "=== ARTICLE ABROGÉ (n'est plus en vigueur — ne pas citer comme droit actuel) ==="
MARQUEUR_FIN_ABROGE = "=== FIN ARTICLE ABROGÉ ==="

# Une loi se termine par l'historique parlementaire (débats, votes), qui ne se
# cite pas. Ce repère marque la fin du contenu utile. La recherche n'est pas
# ancrée en début de ligne (la mention se trouve parfois au milieu) et accepte
# É ou E (l'accent varie selon le document).
DEBUT_TRAVAUX_PREPARATOIRES = re.compile(r"TRAVAUX PR[ÉE]PARATOIRES", re.IGNORECASE)

# Intitulé de subdivision : « Titre Ier bis : Des rapports … », « Section 1 :
# Dispositions générales ». Le niveau (Titre, Chapitre…) est capturé pour
# ranger l'intitulé dans la hiérarchie du texte.
INTITULE = re.compile(
    r"^(Partie|Livre|Titre|Chapitre|Section|Sous-section|Paragraphe)\s+[^:]{1,30}?\s:\s+\S"
)
NIVEAUX = ["Partie", "Livre", "Titre", "Chapitre", "Section", "Sous-section", "Paragraphe"]
# Renvoi « (Articles 25-3 à 25-11) » en fin d'intitulé : sans intérêt pour le sens.
RENVOI_ARTICLES = re.compile(r"\s*\(Articles?\b.*$")


class RawChunk(BaseModel):
    """Un morceau de texte avant traitement : le contenu d'un article.

    abroge vaut toujours False ici : les articles abrogés sont écartés avant
    la création du morceau (cf. docstring du module). Le champ reste présent
    pour la compatibilité du schéma avec la suite du pipeline (vector_store.py,
    generator.py), qui le lit depuis les métadonnées d'un morceau rangé.
    """

    doc_id: str = Field(min_length=1)
    article: str = Field(min_length=1)
    content: str = Field(min_length=1)
    abroge: bool = False
    # Subdivisions du texte où se trouve l'article (« Titre Ier bis : … meublés
    # … > Chapitre II : … »). Le texte d'un article ne dit pas toujours à quel
    # régime il s'applique : le préavis d'un mois de l'article 25-8 ne vaut que
    # pour les logements meublés, ce que seul l'intitulé du titre indique.
    section: str = ""
    # Titre de la loi dont l'article fait partie (cf. titre_document).
    titre: str = ""


# Lignes qui terminent le titre d'un document en tête de fichier : mentions de
# version Légifrance, numéro NOR, formule de promulgation, premier intitulé ou
# premier article. Un pictogramme peut précéder la mention (PDF Légifrance).
FIN_TITRE = re.compile(
    r"^\W*(EN VIGUEUR|Derni[èe]re\s+mise|Version en vigueur|NOR\b|Intitul[ée]"
    r"|L[’']Assembl[ée]e|Partie|Livre|Titre|TITRE|Chapitre|CHAPITRE|Article)"
)
# Appel de note en fin de titre au Journal officiel (« … urbanisme rénové (1) »).
APPEL_DE_NOTE = re.compile(r"\s*\(\d+\)$")
# Intitulé non officiel Légifrance (« loi Hoguet 1 », « loi Hoguet 1 [1970] ») :
# le nom d'usage, sans le numéro de version ni l'année.
NOM_USAGE = re.compile(r"^(.+?)(?:\s+\d+)?(?:\s+\[\d{4}\])?$")


def titre_document(texte: str) -> str:
    """Titre d'un texte de loi, lu dans ses premières lignes, suivi de son nom
    d'usage s'il en a un (« Loi n° 70-9 du 2 janvier 1970 … (loi Hoguet) »).

    Le texte d'un article ne dit presque jamais de quelle loi il fait partie :
    l'article 1er de la loi Hoguet ne contient ni « Hoguet » ni « agent
    immobilier ». Ce titre sert à le dire au reclasseur (cf. retriever.py).
    Une ligne seule en majuscules avant le titre (« LOIS », rubrique du
    Journal officiel) est ignorée.
    """
    lignes = [ligne.strip() for ligne in texte.splitlines() if ligne.strip()]
    titre: list[str] = []
    for ligne in lignes:
        if FIN_TITRE.match(ligne):
            break
        if not titre and ligne.isupper() and " " not in ligne:
            continue
        titre.append(ligne)
    resultat = APPEL_DE_NOTE.sub("", " ".join(titre))

    if "Intitulé(s) non officiel(s)" in lignes:
        suite = lignes[lignes.index("Intitulé(s) non officiel(s)") + 1 :]
        noms = [ligne for ligne in suite[:4] if ligne not in ("•", "-")]
        if noms:
            nom = NOM_USAGE.match(noms[0]).group(1)
            resultat = f"{resultat} ({nom})" if resultat else nom
    return resultat


def _hierarchie_texte(hierarchie: dict[int, str]) -> str:
    """Intitulés du plus haut niveau au plus bas, sans leurs renvois d'articles."""
    return " > ".join(
        RENVOI_ARTICLES.sub("", hierarchie[n]).rstrip(" .") for n in sorted(hierarchie)
    )


def decouper_en_articles(texte: str, doc_id: str) -> list[RawChunk]:
    """Découpe un texte de loi en une liste de morceaux, un par article.

    Le début (titre, bannière) et la fin (historique parlementaire) ne font
    partie d'aucun article et sont ignorés. Les intitulés de subdivisions
    (titres, chapitres, sections) sont suivis au fil du texte et rattachés à
    chaque article qu'ils couvrent.
    """
    lignes = texte.split("\n")
    chunks: list[RawChunk] = []
    bloc: list[str] = []
    # Intitulés en vigueur, par niveau (indice dans NIVEAUX) ; un intitulé
    # efface ceux des niveaux inférieurs (un nouveau titre clôt les chapitres
    # du précédent).
    hierarchie: dict[int, str] = {}
    niveau_en_cours: int | None = None
    section_bloc = ""
    titre = titre_document(texte)

    def cloturer_bloc() -> None:
        if not bloc:
            return
        contenu_brut = "\n".join(bloc).strip()
        if not contenu_brut:
            return
        premiere_ligne = contenu_brut.splitlines()[0].strip()
        correspondance = DEBUT_ARTICLE.match(premiere_ligne)
        if correspondance is None:
            # Ni un article (préambule ou historique) : aucun morceau n'est créé.
            return
        if MARQUEUR_FIN_ABROGE in contenu_brut:
            # Article abrogé : écarté, pas indexé (cf. docstring du module).
            return
        chunks.append(
            RawChunk(
                doc_id=doc_id,
                article=correspondance.group(1).strip(),
                content=contenu_brut,
                section=section_bloc,
                titre=titre,
            )
        )

    for ligne in lignes:
        ligne_nettoyee = ligne.strip()
        # Suivi des intitulés : ils restent aussi dans le contenu du bloc en
        # cours, seul leur rattachement aux articles suivants est ajouté.
        intitule = INTITULE.match(ligne_nettoyee)
        if intitule:
            niveau_en_cours = NIVEAUX.index(intitule.group(1))
            hierarchie = {n: t for n, t in hierarchie.items() if n < niveau_en_cours}
            hierarchie[niveau_en_cours] = ligne_nettoyee
        elif niveau_en_cours is not None and (
            ligne_nettoyee[:1].islower() or ligne_nettoyee.startswith("(")
        ):
            # Suite d'un intitulé coupé par la mise en page (« … résidence »
            # puis « principale (Articles 25-3 à 25-11) »).
            hierarchie[niveau_en_cours] += " " + ligne_nettoyee
        else:
            niveau_en_cours = None
        if ligne_nettoyee == MARQUEUR_DEBUT_ABROGE:
            # Ce marqueur précède un article : le bloc précédent est clôturé.
            cloturer_bloc()
            bloc = [ligne]
        elif ligne_nettoyee == MARQUEUR_FIN_ABROGE:
            # L'article abrogé est clôturé immédiatement. Attendre l'article
            # suivant lui appliquerait à tort l'étiquette "abrogé".
            bloc.append(ligne)
            cloturer_bloc()
            bloc = []
        elif DEBUT_ARTICLE.match(ligne_nettoyee) or DEBUT_TRAVAUX_PREPARATOIRES.search(ligne_nettoyee):
            cloturer_bloc()
            bloc = [ligne]
            # Après la clôture : le bloc précédent garde sa propre section.
            section_bloc = _hierarchie_texte(hierarchie)
        else:
            bloc.append(ligne)
    cloturer_bloc()

    return chunks
