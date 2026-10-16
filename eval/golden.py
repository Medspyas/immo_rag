"""Chargement et validation du jeu de questions de référence (« golden dataset »).

Le fichier golden_dataset.jsonl contient une question par ligne, au format JSON.
Chaque cas associe une formulation métier à la ou aux sources attendues (loi +
article). C'est le corrigé qui sert à mesurer la recherche : on pose la question
au système et on vérifie que l'article attendu remonte dans les premiers
résultats.

Quelques cas sont volontairement hors du corpus ingéré (hors_corpus) : ils
vérifient que l'assistant refuse de répondre plutôt que d'inventer.

D'autres sont des questions de suivi (historique non vide) : « Et pour un
meublé ? » n'a de sens qu'après l'échange précédent. Ils mesurent la
réécriture en question autonome (suivi.py), qui demande un appel au LLM : les
mesures de recherche seule (gate, ablation) les laissent de côté.

Contenu du fichier :
- CasGolden : une question de référence.
- charger_golden : lit le fichier .jsonl et renvoie la liste des cas validés.
"""
from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from app.rag.suivi import ECHANGES_MAX, Echange

FICHIER_GOLDEN = Path(__file__).with_name("golden_dataset.jsonl")


class CasGolden(BaseModel):
    """Une question de référence et sa correction.

    doc_id / articles : la source attendue. Pour un cas hors_corpus, doc_id vaut
    None et articles est vide (rien n'est censé être trouvé).

    reponse_type : « reponse » quand une réponse fondée est attendue, « refus »
    quand l'assistant doit décliner (cas hors_corpus).

    historique : les échanges qui précèdent une question de suivi (vide pour
    une question autonome).
    """

    id: str
    question: str
    categorie: str
    doc_id: str | None
    articles: list[str]
    hors_corpus: bool = False
    reponse_type: str
    note: str = ""
    historique: list[Echange] = Field(default_factory=list, max_length=ECHANGES_MAX)

    @model_validator(mode="after")
    def _coherence(self) -> CasGolden:
        if self.hors_corpus:
            if self.doc_id is not None or self.articles:
                raise ValueError(
                    f"{self.id} : un cas hors_corpus ne porte ni doc_id ni article."
                )
            if self.reponse_type != "refus":
                raise ValueError(f"{self.id} : un cas hors_corpus attend un refus.")
        else:
            if not self.doc_id or not self.articles:
                raise ValueError(
                    f"{self.id} : un cas du corpus doit préciser doc_id et articles."
                )
            if self.reponse_type != "reponse":
                raise ValueError(f"{self.id} : un cas du corpus attend une réponse.")
        return self


def charger_golden(chemin: Path | None = None) -> list[CasGolden]:
    """Lit le fichier .jsonl (une question par ligne) et renvoie les cas validés.

    Lève une erreur si une ligne est mal formée, si un cas est incohérent, ou si
    deux cas partagent le même identifiant.
    """
    chemin = chemin or FICHIER_GOLDEN
    cas: list[CasGolden] = []
    for numero, ligne in enumerate(
        chemin.read_text(encoding="utf-8").splitlines(), start=1
    ):
        ligne = ligne.strip()
        if not ligne:
            continue
        try:
            cas.append(CasGolden.model_validate(json.loads(ligne)))
        except ValueError as e:
            raise ValueError(f"{chemin.name}, ligne {numero} : {e}") from e

    identifiants = [c.id for c in cas]
    doublons = sorted({i for i in identifiants if identifiants.count(i) > 1})
    if doublons:
        raise ValueError(f"Identifiants en double dans le golden : {doublons}")
    return cas


def questions_autonomes(cas: list[CasGolden]) -> list[CasGolden]:
    """Les cas sans historique : ceux qu'une recherche seule peut traiter."""
    return [c for c in cas if not c.historique]
