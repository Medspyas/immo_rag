"""Tests de la chaîne d'ingestion.

Aucune base ni aucun modèle réel : les embeddings et la base vectorielle sont
remplacés par des doublures qui se contentent d'enregistrer ce qu'on leur passe.
On vérifie l'enchaînement des étapes, pas la qualité des vecteurs.
"""
from __future__ import annotations

import json
import uuid

from app.config import Settings
from app.rag.chunking import RawChunk
from app.rag.fenetres import fenetres_article
from app.rag.generator import (
    CONSIGNE_COMPLETUDE,
    CONSIGNE_ROLES_ARTICLE,
    CONSIGNE_ROLES_QUESTION,
)
from app.rag.nli import ScoreNLI
from app.rag.pipeline import (
    construire_index_bm25,
    construire_retriever,
    documents_bm25,
    fichiers_du_corpus,
    identifiant_chunk,
    ingerer_document,
    lire_source,
    preparer_morceaux,
    repondre,
)
from app.rag.retriever import HybridRetriever
from app.rag.suivi import CONSIGNE_SUIVI, Echange
from app.rag.vector_store import Chunk

# Deux articles, dont un abrogé (marqueurs posés par le nettoyage en amont).
TEXTE = """Loi d'essai

Article 1
Le locataire donne congé avec un préavis de trois mois.

=== ARTICLE ABROGÉ (n'est plus en vigueur — ne pas citer comme droit actuel) ===
Article 2
Ancienne règle sur le dépôt de garantie.
=== FIN ARTICLE ABROGÉ ===

Article 3
Le dépôt de garantie est restitué dans un délai d'un mois.
"""


class _EmbeddingsFake:
    """Renvoie un vecteur trivial par texte et retient les textes reçus, pour
    vérifier ce qui est réellement vectorisé."""

    def __init__(self) -> None:
        self.textes_recus: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.textes_recus = list(texts)
        return [[float(i), 0.0, 0.0] for i, _ in enumerate(texts)]


class _VectorStoreFake:
    """Enregistre l'ordre des appels et les morceaux insérés."""

    def __init__(self) -> None:
        self.appels: list[str] = []
        self.morceaux: list[Chunk] = []

    def delete_by_doc(self, doc_id: str) -> None:
        self.appels.append(f"delete:{doc_id}")

    def upsert(self, chunks: list[Chunk]) -> None:
        self.appels.append(f"upsert:{len(chunks)}")
        self.morceaux = list(chunks)


class _ContextGeneratorFake:
    def resumer_document(self, debut_document: str) -> str:
        return "[contexte]"


def test_preparer_morceaux_un_par_article_hors_preambule():
    morceaux = preparer_morceaux(TEXTE, "loi-essai", embeddings=_EmbeddingsFake())
    assert len(morceaux) == 2
    assert [m.metadata["article"] for m in morceaux] == ["1", "3"]


def test_preparer_morceaux_ecarte_l_article_abroge():
    morceaux = preparer_morceaux(TEXTE, "loi-essai", embeddings=_EmbeddingsFake())
    assert "2" not in [m.metadata["article"] for m in morceaux]


def test_preparer_morceaux_identifiants_stables_entre_deux_appels():
    a = preparer_morceaux(TEXTE, "loi-essai", embeddings=_EmbeddingsFake())
    b = preparer_morceaux(TEXTE, "loi-essai", embeddings=_EmbeddingsFake())
    assert [m.id for m in a] == [m.id for m in b]
    assert a[0].id == identifiant_chunk("loi-essai", "1")


def test_preparer_morceaux_sans_contexte_vectorise_le_texte_brut():
    embeddings = _EmbeddingsFake()
    preparer_morceaux(TEXTE, "loi-essai", embeddings=embeddings)
    assert embeddings.textes_recus[0].startswith("Article 1")


def test_preparer_morceaux_avec_contexte_prefixe_le_texte_vectorise():
    embeddings = _EmbeddingsFake()
    morceaux = preparer_morceaux(
        TEXTE,
        "loi-essai",
        embeddings=embeddings,
        contextualizer=_ContextGeneratorFake(),
    )
    assert embeddings.textes_recus[0].startswith("[contexte]")
    assert morceaux[0].content.startswith("[contexte]")


def test_preparer_morceaux_refuse_un_texte_sans_article():
    try:
        preparer_morceaux("Un texte sans le moindre article.", "vide",
                          embeddings=_EmbeddingsFake())
    except ValueError as e:
        assert "Aucun article" in str(e)
    else:
        raise AssertionError("une erreur était attendue")


def test_ingerer_document_supprime_avant_d_inserer():
    store = _VectorStoreFake()
    ingerer_document(TEXTE, "loi-essai", embeddings=_EmbeddingsFake(),
                     vector_store=store)
    assert store.appels == ["delete:loi-essai", "upsert:2"]


def test_ingerer_document_rapport_chiffre():
    store = _VectorStoreFake()
    rapport = ingerer_document(TEXTE, "loi-essai", embeddings=_EmbeddingsFake(),
                               vector_store=store)
    assert rapport.doc_id == "loi-essai"
    assert rapport.articles_indexes == 2
    assert rapport.articles_abroges == 0
    assert rapport.contextualise is False


def test_ingerer_document_signale_la_contextualisation():
    store = _VectorStoreFake()
    rapport = ingerer_document(TEXTE, "loi-essai", embeddings=_EmbeddingsFake(),
                               vector_store=store,
                               contextualizer=_ContextGeneratorFake())
    assert rapport.contextualise is True


def test_fichiers_du_corpus_ecarte_les_textes_non_consolides(tmp_path):
    for nom in ("loi_b.txt", "loi_ALUR.txt", "loi_a.txt", "notes.md"):
        (tmp_path / nom).write_text("x", encoding="utf-8")
    assert fichiers_du_corpus(tmp_path) == ["loi_a.txt", "loi_b.txt"]


def test_fichiers_du_corpus_prefere_le_texte_de_l_api(tmp_path):
    for nom in ("loi_a.txt", "loi_a.json", "loi_b.txt"):
        (tmp_path / nom).write_text("x", encoding="utf-8")
    assert fichiers_du_corpus(tmp_path) == ["loi_a.json", "loi_b.txt"]


def test_lire_source_json_et_txt_donnent_les_memes_articles(tmp_path):
    (tmp_path / "loi.txt").write_text(TEXTE, encoding="utf-8")
    texte = {
        "title": "Loi d'essai",
        "sections": [],
        "articles": [{"num": "2", "content": "<p>Le préavis est de trois mois.</p>",
                      "intOrdre": 1, "etat": "VIGUEUR"}],
    }
    (tmp_path / "loi_api.json").write_text(json.dumps(texte), encoding="utf-8")
    _, articles_txt = lire_source(tmp_path / "loi.txt")
    texte_api, articles_api = lire_source(tmp_path / "loi_api.json")
    assert articles_txt[0].doc_id == "loi"
    assert [(a.doc_id, a.article) for a in articles_api] == [("loi_api", "2")]
    assert "trois mois" in texte_api


def test_preparer_morceaux_utilise_les_articles_fournis():
    articles = [RawChunk(doc_id="loi", article="7", content="Article 7\nTexte.")]
    morceaux = preparer_morceaux("", "loi", embeddings=_EmbeddingsFake(), articles=articles)
    assert [m.metadata["article"] for m in morceaux] == ["7"]


def test_documents_bm25_reprend_id_contenu_et_metadonnees():
    morceaux = preparer_morceaux(TEXTE, "loi-essai", embeddings=_EmbeddingsFake())
    docs = documents_bm25(morceaux)
    assert [d.id for d in docs] == [m.id for m in morceaux]
    assert docs[0].metadata["article"] == "1"
    assert "préavis" in docs[0].content


def test_documents_bm25_indexe_les_questions_sans_toucher_au_contenu():
    morceaux = preparer_morceaux(TEXTE, "loi-essai", embeddings=_EmbeddingsFake())
    docs = documents_bm25(morceaux, {("loi-essai", "1"): ["Combien de temps dure le préavis ?"]})
    assert docs[0].complement == "Combien de temps dure le préavis ?"
    assert "Combien de temps" not in docs[0].content
    assert all(d.complement == "" for d in docs[1:])


def test_documents_bm25_indexe_le_chemin_de_sections():
    articles = [RawChunk(doc_id="cgi", article="32", content="Article 32\nAbattement de 30 %.",
                         section="Section II : Revenus imposables > I : Revenus fonciers")]
    morceaux = preparer_morceaux("", "cgi", embeddings=_EmbeddingsFake(), articles=articles)
    docs = documents_bm25(morceaux, {("cgi", "32"): ["Micro-foncier ?"]})
    assert docs[0].complement == "Section II : Revenus imposables > I : Revenus fonciers\nMicro-foncier ?"
    assert "Revenus fonciers" not in docs[0].content


# --- construire_index_bm25 ---------------------------------------------------


class _MorceauStockeFake:
    def __init__(self, id, doc_id, content, metadata):
        self.id = id
        self.doc_id = doc_id
        self.content = content
        self.metadata = metadata


class _VectorStoreLectureFake:
    def __init__(self, morceaux):
        self._morceaux = morceaux

    def tous(self):
        return self._morceaux


def test_construire_index_bm25_retrouve_un_morceau_range():
    # Corpus dédié (pas TEXTE) : avec seulement 2 documents dont 1 porte le
    # terme cherché, le calcul BM25 classique (idf = log((N-n+0.5)/(n+0.5)))
    # tombe pile sur idf = 0 pour un terme présent dans la moitié du corpus,
    # un artefact de petit corpus qui n'a rien à voir avec ce qui est testé ici.
    stockes = [
        _MorceauStockeFake(uuid.uuid4(), "loi-essai", "Le préavis est de trois mois.", {"article": "1"}),
        _MorceauStockeFake(uuid.uuid4(), "loi-essai", "Le dépôt de garantie est restitué.", {"article": "2"}),
        _MorceauStockeFake(uuid.uuid4(), "loi-essai", "Les charges de copropriété sont réparties.", {"article": "3"}),
    ]
    index = construire_index_bm25(_VectorStoreLectureFake(stockes))
    resultats = index.search("préavis")
    assert resultats
    assert resultats[0].doc_id == "loi-essai"


def test_construire_index_bm25_base_vide_ne_trouve_rien():
    index = construire_index_bm25(_VectorStoreLectureFake([]))
    assert index.search("préavis") == []


# --- construire_retriever ----------------------------------------------------


class _RechercheDenseFake:
    def search(self, query_embedding, k):
        return []


class _RechercheSparseFake:
    def search(self, query, k):
        return []


class _RerankerFake:
    def rerank(self, query, documents):
        return []


def test_construire_retriever_transmet_le_seuil_avec_reclasseur():
    retriever = construire_retriever(
        _settings(seuil_garde_fou_retrieval=0.42),
        _EmbeddingsFake(),
        _RechercheDenseFake(),
        _RechercheSparseFake(),
        reranker=_RerankerFake(),
    )
    assert isinstance(retriever, HybridRetriever)
    assert retriever._seuil_rerank == 0.42


def test_construire_retriever_transmet_le_nombre_de_candidats():
    retriever = construire_retriever(
        _settings(k_par_source=30, k_avant_rerank=25),
        _EmbeddingsFake(),
        _RechercheDenseFake(),
        _RechercheSparseFake(),
        reranker=_RerankerFake(),
    )
    assert retriever._k_par_source == 30
    assert retriever._k_avant_rerank == 25


def test_construire_retriever_ignore_le_seuil_sans_reclasseur():
    retriever = construire_retriever(
        _settings(seuil_garde_fou_retrieval=0.42),
        _EmbeddingsFake(),
        _RechercheDenseFake(),
        _RechercheSparseFake(),
    )
    assert retriever._seuil_rerank is None


# --- repondre (orchestration d'une requête) ---------------------------------


def _settings(**overrides) -> Settings:
    base = {"database_url": "postgresql://x", "groq_api_key": "fake-key"}
    base.update(overrides)
    return Settings(**base)


class _ResultatFake:
    def __init__(self, doc_id, content, article="15"):
        self.id = article
        self.doc_id = doc_id
        self.content = content
        self.metadata = {"article": article, "abroge": False}


class _RetrieverFake:
    def __init__(self, resultats):
        self._resultats = resultats
        self.question_recue = None
        self.k_recu = None

    def search(self, question, k=10):
        self.question_recue = question
        self.k_recu = k
        return self._resultats


class _LLMFake:
    def __init__(self, sortie: str):
        self._sortie = sortie

    def generer(self, consigne: str, question: str) -> str:
        return self._sortie


class _NLIToujoursFonde:
    def evaluer(self, premisses: list[str], hypothese: str) -> list[ScoreNLI]:
        return [ScoreNLI(implication=0.99, contradiction=0.0) for _ in premisses]


_ARTICLE_15 = _ResultatFake(
    "loi-89-462", "Le délai de préavis est de trois mois.", article="15"
)

_REPONSE_OK = json.dumps(
    {
        "refus": False,
        "reponse": "Le délai de préavis est de trois mois.",
        "citations": [
            {"doc_id": "loi-89-462", "article": "15", "passage": "Le délai de préavis est de trois mois."}
        ],
    }
)


def test_repondre_enchaine_recherche_generation_et_garde_fous():
    retriever = _RetrieverFake([_ARTICLE_15])
    reponse = repondre(
        "Quel préavis ?", retriever, _LLMFake(_REPONSE_OK), _NLIToujoursFonde(), _settings()
    )
    assert reponse.refus is False
    assert retriever.question_recue == "Quel préavis ?"


def test_repondre_sans_resultat_refuse_sans_appeler_le_modele():
    retriever = _RetrieverFake([])
    appele = []
    class _LLMEspion:
        def generer(self, consigne, question):
            appele.append(True)
            return _REPONSE_OK
    reponse = repondre("Quel préavis ?", retriever, _LLMEspion(), _NLIToujoursFonde(), _settings())
    assert reponse.refus is True
    assert appele == []


def test_repondre_citation_inventee_bascule_en_refus():
    retriever = _RetrieverFake([_ARTICLE_15])
    sortie = json.dumps(
        {
            "refus": False,
            "reponse": "Le préavis est de six mois.",
            "citations": [
                {"doc_id": "loi-89-462", "article": "15", "passage": "le délai de préavis est de six mois"}
            ],
        }
    )
    reponse = repondre("Quel préavis ?", retriever, _LLMFake(sortie), _NLIToujoursFonde(), _settings())
    assert reponse.refus is True
    assert "introuvable" in reponse.motif


def test_repondre_affirmation_non_fondee_bascule_en_refus():
    class _NLIJamaisFonde:
        def evaluer(self, premisses, hypothese):
            return [ScoreNLI(implication=0.0, contradiction=0.0) for _ in premisses]

    retriever = _RetrieverFake([_ARTICLE_15])
    reponse = repondre(
        "Quel préavis ?", retriever, _LLMFake(_REPONSE_OK), _NLIJamaisFonde(), _settings()
    )
    assert reponse.refus is True
    assert "non fondée" in reponse.motif


class _LLMSuivi:
    """Réécrit la question de suivi, puis répond comme _LLMFake."""

    def __init__(self, reecrite: str):
        self._reecrite = reecrite

    def generer(self, consigne: str, question: str) -> str:
        if consigne == CONSIGNE_SUIVI:
            return json.dumps({"question": self._reecrite})
        return _REPONSE_OK


def test_repondre_reecrit_la_question_de_suivi():
    retriever = _RetrieverFake([_ARTICLE_15])
    historique = [Echange(question="Quel préavis pour le locataire ?", reponse="Trois mois.")]
    reponse = repondre(
        "Et pour un meublé ?", retriever, _LLMSuivi("Quel préavis pour un meublé ?"),
        _NLIToujoursFonde(), _settings(), historique=historique,
    )
    assert retriever.question_recue == "Quel préavis pour un meublé ?"
    assert reponse.question_comprise == "Quel préavis pour un meublé ?"


def test_repondre_sans_historique_ne_signale_aucune_reecriture():
    reponse = repondre(
        "Quel préavis ?", _RetrieverFake([_ARTICLE_15]), _LLMFake(_REPONSE_OK),
        _NLIToujoursFonde(), _settings(),
    )
    assert reponse.question_comprise is None


def test_repondre_transmet_k_au_retriever():
    retriever = _RetrieverFake([_ARTICLE_15])
    repondre("Quel préavis ?", retriever, _LLMFake(_REPONSE_OK), _NLIToujoursFonde(), _settings(), k=3)
    assert retriever.k_recu == 3


def test_repondre_prend_k_dans_la_config_par_defaut():
    retriever = _RetrieverFake([_ARTICLE_15])
    repondre(
        "Quel préavis ?", retriever, _LLMFake(_REPONSE_OK), _NLIToujoursFonde(),
        _settings(k_articles_generation=4),
    )
    assert retriever.k_recu == 4


def test_repondre_est_tracee_quand_langfuse_est_configure(monkeypatch):
    """repondre porte @trace (observability/tracing.py) : un appel doit ouvrir
    un span Langfuse quand un client est disponible, sans changer son résultat."""
    journal: list[str] = []

    class _SpanFake:
        def __enter__(self):
            journal.append("entree")
            return self

        def __exit__(self, *exc):
            journal.append("sortie")
            return False

        def update(self, output=None):
            pass

    class _ClientFake:
        def start_as_current_span(self, name):
            journal.append(f"nom:{name}")
            return _SpanFake()

    monkeypatch.setattr(
        "app.observability.tracing._client_langfuse", lambda: _ClientFake()
    )
    retriever = _RetrieverFake([_ARTICLE_15])
    reponse = repondre(
        "Quel préavis ?", retriever, _LLMFake(_REPONSE_OK), _NLIToujoursFonde(), _settings()
    )
    assert reponse.refus is False
    assert journal == ["nom:repondre", "entree", "sortie"]


def test_preparer_morceaux_vectorise_les_fenetres_des_articles_longs():
    long = " ".join(f"mot{i}" for i in range(600))
    texte = f"Loi d'essai\n\nArticle 1\nLe loyer est payé chaque mois.\n\nArticle 2\n{long}\n"
    morceaux = preparer_morceaux(texte, "loi-essai", embeddings=_EmbeddingsFake())
    assert morceaux[0].fenetres == []
    assert len(morceaux[1].fenetres) == len(fenetres_article(morceaux[1].content)) > 0


def test_construire_index_bm25_indexe_les_questions_des_articles():
    stockes = [
        _MorceauStockeFake(uuid.uuid4(), "loi-essai", "Le préavis est de trois mois.", {"article": "1"}),
        _MorceauStockeFake(uuid.uuid4(), "loi-essai", "Le dépôt de garantie est restitué.", {"article": "2"}),
        _MorceauStockeFake(uuid.uuid4(), "loi-essai", "Les charges sont réparties.", {"article": "3"}),
    ]
    questions = {("loi-essai", "2"): ["Quand récupérer sa caution ?"]}
    index = construire_index_bm25(_VectorStoreLectureFake(stockes), questions)
    assert index.search("caution")[0].id == stockes[1].id


class _LLMSelonConsigne:
    """Répond selon la consigne reçue : rédactions successives d'un côté,
    verdict de complétude de l'autre. Retient les messages de rédaction."""

    def __init__(self, redactions: list[str], completude: str):
        self._redactions = list(redactions)
        self._completude = completude
        self.messages_redaction: list[str] = []
        self.appels_completude = 0

    def generer(self, consigne: str, question: str) -> str:
        if consigne == CONSIGNE_COMPLETUDE:
            self.appels_completude += 1
            return self._completude
        self.messages_redaction.append(question)
        return self._redactions.pop(0)


_INCOMPLETE = json.dumps(
    {
        "complete": False,
        "regle_generale": "Le délai de préavis est de trois mois.",
        "manque": "Le préavis est réduit à un mois en zone tendue.",
    }
)


def _toujours() -> Settings:
    return _settings(controle_completude="toujours")


def test_repondre_controle_cible_ignore_une_reponse_ordinaire():
    # Mode par défaut : une réponse sans dérogation ni régime particulier ne
    # paie pas l'appel au juge (quota gratuit, 2026-10-22).
    llm = _LLMSelonConsigne([_REPONSE_OK], _INCOMPLETE)
    repondre("Quel préavis ?", _RetrieverFake([_ARTICLE_15]), llm, _NLIToujoursFonde(), _settings())
    assert llm.appels_completude == 0


def test_repondre_reponse_incomplete_est_redigee_une_seconde_fois():
    seconde = json.dumps(
        {
            "refus": False,
            "reponse": "Le délai de préavis est de trois mois, réduit à un mois en zone tendue.",
            "citations": [
                {"doc_id": "loi-89-462", "article": "15", "passage": "Le délai de préavis est de trois mois."}
            ],
        }
    )
    llm = _LLMSelonConsigne([_REPONSE_OK, seconde], _INCOMPLETE)
    reponse = repondre("Quel préavis ?", _RetrieverFake([_ARTICLE_15]), llm, _NLIToujoursFonde(), _toujours())
    assert "zone tendue" in reponse.reponse
    # L'élément omis est signalé au LLM ; une seule seconde rédaction.
    assert len(llm.messages_redaction) == 2
    assert "réduit à un mois en zone tendue" in llm.messages_redaction[1]
    assert llm.appels_completude == 1


def test_repondre_reponse_complete_garde_la_premiere_redaction():
    llm = _LLMSelonConsigne([_REPONSE_OK], json.dumps({"complete": True, "manque": ""}))
    reponse = repondre("Quel préavis ?", _RetrieverFake([_ARTICLE_15]), llm, _NLIToujoursFonde(), _settings())
    assert reponse.reponse == "Le délai de préavis est de trois mois."
    assert len(llm.messages_redaction) == 1


def test_repondre_sans_controle_de_completude():
    llm = _LLMSelonConsigne([_REPONSE_OK], _INCOMPLETE)
    repondre(
        "Quel préavis ?", _RetrieverFake([_ARTICLE_15]), llm, _NLIToujoursFonde(),
        _settings(controle_completude="non"),
    )
    assert llm.appels_completude == 0


def test_repondre_seconde_redaction_refusee_garde_la_premiere():
    # Fausse alerte du contrôle : la seconde rédaction échoue aux garde-fous
    # (citation inventée), la première, déjà vérifiée, est gardée (q16).
    inventee = json.dumps(
        {
            "refus": False,
            "reponse": "Le préavis est de six mois.",
            "citations": [{"doc_id": "loi-89-462", "article": "15", "passage": "passage absent de l'article"}],
        }
    )
    llm = _LLMSelonConsigne([_REPONSE_OK, inventee, inventee], _INCOMPLETE)
    reponse = repondre("Quel préavis ?", _RetrieverFake([_ARTICLE_15]), llm, _NLIToujoursFonde(), _toujours())
    assert reponse.refus is False
    assert reponse.reponse == "Le délai de préavis est de trois mois."


def test_repondre_seconde_refusee_rappelle_le_regime_particulier():
    # t15 : la première rédaction, gardée, ne doit pas passer pour la règle
    # commune quand l'article cité relève d'un régime particulier.
    article = _ResultatFake("loi-89-462", "Le délai de préavis est de trois mois.", article="15")
    article.metadata["section"] = "Titre > Dispositions particulières aux logements meublés"
    inventee = json.dumps(
        {
            "refus": False,
            "reponse": "Le préavis est de six mois.",
            "citations": [{"doc_id": "loi-89-462", "article": "15", "passage": "passage absent de l'article"}],
        }
    )
    # Article de régime particulier : l'affirmation passe aussi par le juge.
    juge_ok = json.dumps({"fondees": [True]})
    llm = _LLMSelonConsigne([_REPONSE_OK, juge_ok, inventee, inventee], _INCOMPLETE)
    reponse = repondre("Quel préavis ?", _RetrieverFake([article]), llm, _NLIToujoursFonde(), _settings())
    assert reponse.reponse.startswith("Le délai de préavis est de trois mois.")
    assert "Précision : L'article 15 relève de : « Dispositions particulières aux logements meublés »." in reponse.reponse



# --- Situation de la question et cas de l'article cité (v108) -----------------

_ART_553 = _ResultatFake("code_civil", "Toutes constructions sur un terrain sont présumées faites par le propriétaire.", article="553")
_ART_554 = _ResultatFake(
    "code_civil",
    "Le propriétaire du sol qui a fait des constructions avec des matériaux qui ne lui appartenaient pas doit en payer la valeur.",
    article="554",
)
_ART_555 = _ResultatFake(
    "code_civil",
    "Lorsque les constructions ont été faites par un tiers avec des matériaux appartenant à ce dernier, "
    "le propriétaire du fonds a le droit d'obliger le tiers à les enlever.",
    article="555",
)


def _redaction(article: str, passage: str, texte: str) -> str:
    return json.dumps(
        {"refus": False, "reponse": texte, "citations": [{"doc_id": "code_civil", "article": article, "passage": passage}]}
    )


class _LLMRoles:
    """Rédactions successives ; rôles de la question et de l'article figés."""

    def __init__(self, redactions: list[str], roles_article: list[str]):
        self._redactions = list(redactions)
        self._roles_article = roles_article
        self.messages_redaction: list[str] = []

    def generer(self, consigne: str, question: str) -> str:
        if consigne == CONSIGNE_ROLES_QUESTION:
            return json.dumps({"roles": [
                {"role": "propriétaire du terrain", "personne": "le voisin"},
                {"role": "auteur de la construction", "personne": "le voisin"},
                {"role": "propriétaire des matériaux", "personne": "l'utilisateur"},
            ]})
        if consigne == CONSIGNE_ROLES_ARTICLE:
            return json.dumps({
                "phrase": "Lorsque les constructions ont été faites par un tiers avec des matériaux appartenant à ce dernier",
                "roles": [{"role": "x", "personne": p} for p in self._roles_article],
            })
        self.messages_redaction.append(question)
        return self._redactions.pop(0)


_QUESTION_V108 = "Mon voisin a construit sur son terrain avec mes matériaux : puis-je les reprendre ?"
_FAUSSE = _redaction("555", "le propriétaire du fonds a le droit d'obliger le tiers à les enlever", "Le propriétaire du fonds a le droit d'obliger le tiers à les enlever.")
_JUSTE = _redaction("554", "doit en payer la valeur", "Le propriétaire du sol doit en payer la valeur.")


def test_repondre_cas_de_l_article_different_redige_de_nouveau():
    llm = _LLMRoles([_FAUSSE, _JUSTE], ["le propriétaire du fonds", "un tiers", "un tiers"])
    reponse = repondre(
        _QUESTION_V108, _RetrieverFake([_ART_553, _ART_555, _ART_554]), llm, _NLIToujoursFonde(),
        _settings(controle_completude="non"),
    )
    assert reponse.citations[0].article == "554"
    assert len(llm.messages_redaction) == 2
    assert "Contrôle des rôles" in llm.messages_redaction[1]
    assert "auteur de la construction" in llm.messages_redaction[1]


def test_repondre_meme_article_apres_ecart_signale_refuse():
    llm = _LLMRoles([_FAUSSE, _FAUSSE], ["le propriétaire du fonds", "un tiers", "un tiers"])
    reponse = repondre(
        _QUESTION_V108, _RetrieverFake([_ART_553, _ART_555, _ART_554]), llm, _NLIToujoursFonde(),
        _settings(controle_completude="non"),
    )
    assert reponse.refus
    assert "555" in reponse.motif
    assert len(llm.messages_redaction) == 2


def test_repondre_redaction_apres_ecart_confiee_au_redacteur_des_roles():
    # v108 (2026-11-11) : les gpt-oss recitaient l'art. 555 même avec l'écart
    # signalé ; la rédaction refaite passe par le rédacteur dédié (qwen).
    llm = _LLMRoles([_FAUSSE], ["le propriétaire du fonds", "un tiers", "un tiers"])
    redacteur = _LLMRoles([_JUSTE], [])
    reponse = repondre(
        _QUESTION_V108, _RetrieverFake([_ART_553, _ART_555, _ART_554]), llm, _NLIToujoursFonde(),
        _settings(controle_completude="non"), redacteur_roles=redacteur,
    )
    assert reponse.citations[0].article == "554"
    assert len(llm.messages_redaction) == 1
    assert len(redacteur.messages_redaction) == 1
    assert "Contrôle des rôles" in redacteur.messages_redaction[0]


class _LLMRolesCompletude(_LLMRoles):
    """Comme _LLMRoles, avec une omission signalée par le contrôle de complétude."""

    def generer(self, consigne: str, question: str) -> str:
        if consigne == CONSIGNE_COMPLETUDE:
            return json.dumps({
                "complete": False,
                "regle_generale": "Toutes constructions sur un terrain sont présumées faites par le propriétaire.",
                "manque": "La présomption de l'article 553.",
            })
        return super().generer(consigne, question)


def test_repondre_completude_garde_l_ecart_des_roles_et_le_controle():
    # v108 en réel (2026-11-10) : la rédaction de complétude revenait à
    # l'art. 555 écarté par le contrôle des rôles.
    llm = _LLMRolesCompletude([_FAUSSE, _JUSTE, _FAUSSE], ["le propriétaire du fonds", "un tiers", "un tiers"])
    reponse = repondre(
        _QUESTION_V108, _RetrieverFake([_ART_553, _ART_555, _ART_554]), llm, _NLIToujoursFonde(),
        _toujours(),
    )
    assert len(llm.messages_redaction) == 3
    assert "Contrôle des rôles" in llm.messages_redaction[2]
    assert reponse.citations[0].article == "554"


def test_repondre_cas_de_l_article_conforme_garde_la_reponse():
    llm = _LLMRoles([_FAUSSE], ["le propriétaire du sol", "le propriétaire du sol", "autrui"])
    reponse = repondre(
        _QUESTION_V108, _RetrieverFake([_ART_553, _ART_555, _ART_554]), llm, _NLIToujoursFonde(),
        _settings(controle_completude="non"),
    )
    assert reponse.citations[0].article == "555"
    assert len(llm.messages_redaction) == 1


def test_repondre_sans_situation_personnelle_pas_de_controle_des_roles():
    llm = _LLMRoles([_FAUSSE], ["le propriétaire du fonds", "un tiers", "un tiers"])
    repondre(
        "Un voisin construit avec les matériaux d'autrui : que prévoit la loi ?",
        _RetrieverFake([_ART_553, _ART_555, _ART_554]), llm, _NLIToujoursFonde(),
        _settings(controle_completude="non"),
    )
    assert len(llm.messages_redaction) == 1
