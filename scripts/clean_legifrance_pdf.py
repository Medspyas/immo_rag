"""Nettoyage des PDF juridiques du dossier data/ (Légifrance, Journal officiel).

Ces PDF proviennent d'une impression navigateur ou d'un export officiel :
chaque page répète un en-tête et un pied de page (URL, numéro de page, date)
étrangers au texte de loi. Conservés, ils dégradent les morceaux puis la
recherche.

Le dossier contient trois formats, chacun avec un bruit spécifique : impression
navigateur d'une page "loi", export "code consolidé", scan du Journal officiel.
Les trois sont traités ligne par ligne, sans détection préalable du format.

Déroulé : lecture du texte page par page (pdfplumber), suppression des lignes
de bruit, encadrement des articles abrogés par des marqueurs afin qu'ils ne
soient pas pris pour du droit en vigueur.

Usage :
    python scripts/clean_legifrance_pdf.py data/Loi_hoguet.pdf   -> data/clean/Loi_hoguet.txt
    python scripts/clean_legifrance_pdf.py data/                 -> nettoie tous les .pdf
"""

import argparse
import itertools
import re
from pathlib import Path

import pdfplumber

# Format 1 (impression navigateur) : une URL Légifrance quelque part dans la
# ligne, seule en pied de page ou à la fin d'un titre coupé.
LIGNE_URL_LEGIFRANCE = re.compile(r"https://www\.legifrance\.gouv\.fr\S*")

# Format 1 variante Loi_hoguet.pdf : "3 sur 28    30/08/2026, 12:28"
LIGNE_PIED_PAGE_NUMERO_DATE = re.compile(r"^\d+\s+sur\s+\d+\s+\d{2}/\d{2}/\d{4},?\s*\d{1,2}[:h]\d{2}$")

# Format 1 variante loi_06_07_1989.pdf : en-tête "29/03/2023 17:22 <titre>..."
LIGNE_ENTETE_HORODATEE = re.compile(r"^\d{2}/\d{2}/\d{4}\s+\d{1,2}:\d{2}\s")

# Bannière Légifrance affichée en haut de la première page (logo + slogan)
LIGNES_BANNIERE = {
    "RÉPUBLIQUE FRANÇAISE",
    "Liberté",
    "Égalité",
    "Fraternité",
    "Legifrance.gouv.fr",
    "Le service public de la diffusion du droit",
}

# Format 2 (export "code consolidé") : pied de page répété sur chaque page,
# ex. "Code civil - Dernière modification le 29 juillet 2026 - Document
# généré le 25 août 2026"
LIGNE_PIED_PAGE_CODE = re.compile(r"Dernière modification le .+Document généré le")

# Format 3 (scan Journal officiel) : en-tête répété, ex. "26 mars 2014
# JOURNAL OFFICIEL DE LA RÉPUBLIQUE FRANÇAISE Texte 1 sur 136"
LIGNE_ENTETE_JOURNAL_OFFICIEL = re.compile(r"JOURNAL OFFICIEL DE LA RÉPUBLIQUE FRANÇAISE")

# Certaines pages portent une annotation ("Modifié par LOI...") dans la marge
# droite, à la hauteur d'une ligne du texte. pdfplumber les fusionne. La
# séparation se fait sur l'écart horizontal : environ 2pt entre deux mots,
# plus de 8pt entre le texte et la marge.
SEUIL_GAP_COLONNE = 8

# Début d'un article : "Article 1", "Article 8-2-1", etc.
DEBUT_ARTICLE = re.compile(r"^Article\s+[\w-]+\s*$")

# "VERSION EN VIGUEUR DU ... AU ..." ne signifie pas abrogé : souvent une
# modification future d'un article encore actif. Le seul signal fiable est la
# mention "Abrogé par ..." dans l'historique.
ARTICLE_ABROGE = re.compile(r"^Abrogé par ", re.MULTILINE)

MARQUEUR_DEBUT_ABROGE = "=== ARTICLE ABROGÉ (n'est plus en vigueur — ne pas citer comme droit actuel) ==="
MARQUEUR_FIN_ABROGE = "=== FIN ARTICLE ABROGÉ ==="


def ligne_est_du_bruit(ligne: str) -> bool:
    """Vrai si la ligne est un en-tête, un pied de page ou une bannière."""
    ligne_nettoyee = ligne.strip()
    if not ligne_nettoyee:
        return False
    if LIGNE_URL_LEGIFRANCE.search(ligne_nettoyee):
        return True
    if LIGNE_PIED_PAGE_NUMERO_DATE.match(ligne_nettoyee):
        return True
    if LIGNE_ENTETE_HORODATEE.match(ligne_nettoyee):
        return True
    if LIGNE_PIED_PAGE_CODE.search(ligne_nettoyee):
        return True
    if LIGNE_ENTETE_JOURNAL_OFFICIEL.search(ligne_nettoyee):
        return True
    return ligne_nettoyee in LIGNES_BANNIERE


def regrouper_par_ligne_visuelle(mots: list, tolerance: float = 2.0) -> list:
    """Regroupe les mots par ligne d'après leur position verticale, pour
    pouvoir ensuite mesurer les écarts horizontaux sur chaque ligne."""
    mots_tries = sorted(mots, key=lambda m: m["top"])
    lignes, ligne_courante, top_reference = [], [], None
    for mot in mots_tries:
        if top_reference is None or abs(mot["top"] - top_reference) <= tolerance:
            ligne_courante.append(mot)
            top_reference = mot["top"] if top_reference is None else top_reference
        else:
            lignes.append(sorted(ligne_courante, key=lambda m: m["x0"]))
            ligne_courante, top_reference = [mot], mot["top"]
    if ligne_courante:
        lignes.append(sorted(ligne_courante, key=lambda m: m["x0"]))
    return lignes


def separer_corps_et_annotations(mots_ligne: list) -> list:
    """Coupe la ligne à chaque écart horizontal anormal (une annotation de
    marge collée au texte). Sans écart, renvoie la ligne entière."""
    segments = [[mots_ligne[0]]]
    for precedent, courant in itertools.pairwise(mots_ligne):
        ecart = courant["x0"] - precedent["x1"]
        if ecart > SEUIL_GAP_COLONNE:
            segments.append([])
        segments[-1].append(courant)
    return [" ".join(mot["text"] for mot in segment) for segment in segments]


def extraire_texte_brut(chemin_pdf: Path) -> str:
    """Extrait le texte page par page en filtrant le bruit et en séparant les
    annotations de marge du corps du texte."""
    lignes_sortie = []
    with pdfplumber.open(chemin_pdf) as pdf:
        for page in pdf.pages:
            mots = page.extract_words()
            for mots_ligne in regrouper_par_ligne_visuelle(mots):
                ligne_brute = " ".join(mot["text"] for mot in mots_ligne)
                if ligne_est_du_bruit(ligne_brute):
                    continue
                lignes_sortie.extend(separer_corps_et_annotations(mots_ligne))
            # pdfplumber conserve chaque page en mémoire tant que le document
            # est ouvert : sur un PDF de milliers de pages, chaque page est
            # libérée après usage pour éviter que la mémoire croisse sans limite.
            page.close()
    texte = "\n".join(lignes_sortie)
    return re.sub(r"\n{3,}", "\n\n", texte).strip()


def marquer_les_articles_abroges(texte: str) -> str:
    """Découpe le texte en articles et entoure les articles abrogés de
    marqueurs, pour que le découpage les repère sans réanalyser le texte."""
    lignes = texte.split("\n")
    blocs = []
    bloc_courant = []

    def cloturer_bloc():
        if not bloc_courant:
            return
        contenu_bloc = "\n".join(bloc_courant)
        if ARTICLE_ABROGE.search(contenu_bloc):
            blocs.append(f"{MARQUEUR_DEBUT_ABROGE}\n{contenu_bloc}\n{MARQUEUR_FIN_ABROGE}")
        else:
            blocs.append(contenu_bloc)

    for ligne in lignes:
        if DEBUT_ARTICLE.match(ligne.strip()):
            cloturer_bloc()
            bloc_courant = [ligne]
        else:
            bloc_courant.append(ligne)
    cloturer_bloc()

    return "\n".join(blocs)


def nettoyer_pdf(chemin_pdf: Path) -> str:
    texte_sans_bruit = extraire_texte_brut(chemin_pdf)
    return marquer_les_articles_abroges(texte_sans_bruit)


def nettoyer_un_fichier(chemin_pdf: Path, dossier_sortie: Path) -> None:
    sortie = dossier_sortie / (chemin_pdf.stem + ".txt")
    texte_propre = nettoyer_pdf(chemin_pdf)
    sortie.write_text(texte_propre, encoding="utf-8")

    nb_articles_abroges = texte_propre.count(MARQUEUR_DEBUT_ABROGE)
    print(f"Écrit : {sortie} ({len(texte_propre)} caractères, {nb_articles_abroges} article(s) abrogé(s) marqué(s))")


def main():
    parser = argparse.ArgumentParser(
        description="Nettoie un PDF Légifrance/Journal officiel, ou tous les PDF d'un dossier."
    )
    parser.add_argument("pdf", type=Path, help="Chemin d'un PDF, ou d'un dossier contenant des PDF (ex: data/)")
    parser.add_argument(
        "--sortie",
        type=Path,
        default=Path("data/clean"),
        help="Dossier de sortie pour les .txt (par défaut : data/clean/)",
    )
    args = parser.parse_args()

    args.sortie.mkdir(parents=True, exist_ok=True)

    if args.pdf.is_dir():
        pdfs = sorted(args.pdf.glob("*.pdf"))
        if not pdfs:
            print(f"Aucun .pdf trouvé dans {args.pdf}")
            return
        for chemin_pdf in pdfs:
            nettoyer_un_fichier(chemin_pdf, args.sortie)
    else:
        nettoyer_un_fichier(args.pdf, args.sortie)


if __name__ == "__main__":
    main()
