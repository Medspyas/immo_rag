"""Fenêtres d'un article long, vectorisées en plus de l'article entier.

Le modèle d'embeddings (e5) ne lit que les 512 premiers tokens d'un texte,
soit environ 1 500 à 2 000 caractères : la fin d'un article long est invisible
pour la recherche par le sens. L'article 6 de la loi Hoguet (4 500 caractères)
pose à la fin de son I la règle « aucune somme n'est due avant la conclusion
de l'opération » : la question q29 ne le trouvait qu'au rang 95.

Un article long reçoit donc, en plus du vecteur de son texte entier, un
vecteur par fenêtre de son texte ; à la recherche, il prend le score de son
meilleur vecteur. Les fenêtres ne servent qu'à retrouver l'article : la
recherche renvoie toujours l'article entier, seul transmis au reclasseur, au
LLM et à la vérification des citations.

Mesuré le 2026-10-16 sur le golden dataset (corpus complet, 40 questions) :
fenêtres de 1 000 caractères, MRR 0,765 → 0,790, hit@1 67,5 % → 70 %,
hit@5 92,5 % → 95 % (q29 introuvable → 1re), aucune question en recul, scores
hors corpus inchangés. Fenêtres de 1 500 caractères : MRR 0,778 (q32 recule) ;
de 2 000 : MRR 0,782 (q08 perdue). Fenêtres de trois phrases, mesurées le
2026-10-05 : MRR en baisse, trop courtes pour porter le sens d'une règle.
Un vecteur par élément d'énumération (chapeau + « c) … »), mesuré le
2026-10-17 pour q23 (art. 25 de la loi de 1965, liste de 15 décisions) :
q23 retrouvée au rang 6, mais q31 perdue, MRR 0,790 → 0,769, même limité aux
articles longs. Écarté. Les mêmes fenêtres dans l'index BM25 (score de
l'article = meilleure fenêtre), mesurées le 2026-10-19 : art. 25 rang BM25
42 → 29, mais q23 toujours hors des résultats, q28 et q31 perdues, MRR
0,790 → 0,740. Écarté.

Contenu du fichier :
- fenetres_article : les fenêtres d'un article, vide pour un article court.
"""
from __future__ import annotations

# Taille d'une fenêtre, en caractères : tient dans les 512 tokens d'e5.
TAILLE_FENETRE = 1000
# Écart entre deux débuts de fenêtre : un tiers de chevauchement, pour qu'une
# phrase coupée au bord d'une fenêtre se retrouve entière dans la suivante.
PAS_FENETRE = 670
# En dessous, l'article tient entier dans ce qu'e5 lit : pas de fenêtre.
LONGUEUR_ARTICLE_LONG = 1800


def fenetres_article(texte: str) -> list[str]:
    """Fenêtres d'un article long, à partir de PAS_FENETRE caractères : le
    début de l'article est déjà couvert par le vecteur du texte entier.

    Chaque fenêtre commence au début d'un mot ; la dernière atteint la fin de
    l'article. Un article court n'en a aucune.
    """
    if len(texte) <= LONGUEUR_ARTICLE_LONG:
        return []
    fenetres: list[str] = []
    position = PAS_FENETRE
    while position < len(texte):
        debut = texte.rfind(" ", 0, position) + 1
        fenetres.append(texte[debut : debut + TAILLE_FENETRE])
        if debut + TAILLE_FENETRE >= len(texte):
            break
        position += PAS_FENETRE
    return fenetres
