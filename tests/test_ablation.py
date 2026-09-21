"""Tests de l'étude d'ablation.

Aucun modèle réel : les embeddings sont une doublure déterministe, le corpus
est réduit à deux courtes lois écrites dans un dossier temporaire. On vérifie le
câblage (découpage → matrice → recherche) et le calcul des scores, pas la
qualité du modèle.
"""
from __future__ import annotations

import numpy as np

from eval.ablation import (
    KS,
    _diagnostic_hors_corpus,
    evaluer_systeme,
    recherche_dense,
    recherche_hybride,
)
from eval.corpus_memoire import construire_corpus
from eval.golden import CasGolden

LOI_A = """Loi A

Article 1
Le locataire donne congé avec un préavis de trois mois.

Article 2
Le dépôt de garantie est restitué dans un délai d'un mois.
"""

LOI_B = """Loi B

Article 1
L'agent immobilier doit détenir une carte professionnelle.
"""


class _EmbeddingsFake:
    """Vecteur déterministe : une dimension par mot-clé suivi, valeur = nombre
    d'occurrences. Deux textes qui partagent des mots-clés obtiennent des
    vecteurs proches, ce qui suffit à exercer la recherche par le sens."""

    MOTS = ("préavis", "congé", "dépôt", "garantie", "carte", "professionnelle")

    def _vecteur(self, texte: str) -> list[float]:
        bas = texte.lower()
        v = np.array([float(bas.count(m)) for m in self.MOTS], dtype=np.float32)
        norme = np.linalg.norm(v)
        return (v / norme).tolist() if norme else v.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vecteur(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vecteur(text)


def _corpus(tmp_path):
    (tmp_path / "loi_a.txt").write_text(LOI_A, encoding="utf-8")
    (tmp_path / "loi_b.txt").write_text(LOI_B, encoding="utf-8")
    return construire_corpus(
        _EmbeddingsFake(),
        fichiers=["loi_a.txt", "loi_b.txt"],
        dossier=tmp_path,
        cache_dir=tmp_path / "cache",
    )


def test_construire_corpus_decoupe_tous_les_articles(tmp_path):
    corpus = _corpus(tmp_path)
    assert len(corpus.chunks) == 3
    assert {(c.doc_id, c.metadata["article"]) for c in corpus.chunks} == {
        ("loi_a", "1"),
        ("loi_a", "2"),
        ("loi_b", "1"),
    }


def test_recherche_dense_remonte_l_article_sur_le_bon_sujet(tmp_path):
    corpus = _corpus(tmp_path)
    sources = recherche_dense(corpus)("préavis de départ du locataire", 3)
    assert sources[0] == ("loi_a", "1")


def test_recherche_hybride_renvoie_des_couples_doc_article(tmp_path):
    corpus = _corpus(tmp_path)
    sources = recherche_hybride(corpus)("carte professionnelle", 3)
    assert ("loi_b", "1") in sources


class _RerankerFake:
    """Garde l'ordre de la fusion et attribue un score fixe : suffit à exercer
    le câblage du diagnostic, pas la qualité du reclassement."""

    def rerank(self, query: str, documents: list[str]) -> list[tuple[int, float]]:
        return [(i, 0.9) for i in range(len(documents))]


def test_diagnostic_ajoute_le_score_de_reclassement_quand_un_reranker_est_fourni(tmp_path):
    corpus = _corpus(tmp_path)
    cas = [
        CasGolden(id="c1", question="préavis du locataire", categorie="x",
                  doc_id="loi_a", articles=["1"], reponse_type="reponse"),
        CasGolden(id="c2", question="question sans rapport", categorie="x",
                  doc_id=None, articles=[], reponse_type="refus",
                  hors_corpus=True),
    ]

    sans = _diagnostic_hors_corpus(corpus, cas)
    assert "score de reclassement" not in sans

    avec = _diagnostic_hors_corpus(corpus, cas, reranker=_RerankerFake())
    assert "Meilleur score de reclassement (0–1) :" in avec
    assert "0.900" in avec


def test_evaluer_systeme_calcule_hit_et_mrr():
    cas = [
        CasGolden(id="c1", question="q1", categorie="x", doc_id="d",
                  articles=["1"], reponse_type="reponse"),
        CasGolden(id="c2", question="q2", categorie="x", doc_id="d",
                  articles=["2"], reponse_type="reponse"),
    ]

    # c1 : cible au rang 1 ; c2 : cible absente.
    reponses = {"q1": [("d", "1"), ("d", "9")], "q2": [("d", "8"), ("d", "7")]}

    res = evaluer_systeme("test", lambda question, k: reponses[question], cas)

    assert res.n_questions == 2
    assert res.hit_at[1] == 0.5
    assert all(res.hit_at[k] == 0.5 for k in KS)
    assert res.mrr == (1.0 + 0.0) / 2
