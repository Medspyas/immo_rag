"""Configuration du projet.

Tous les réglages sont lus depuis le fichier .env (un modèle figure dans
.env.example). La classe Settings énumère les réglages attendus et interrompt
le démarrage si l'un d'eux manque ou si une clé est mal orthographiée.

Contrainte de projet : seuls des composants gratuits sont retenus. Chaque
réglage de fournisseur n'accepte donc qu'une valeur locale ou une offre
gratuite ; aucun service payant n'est proposé.
"""
from __future__ import annotations

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="forbid")

    database_url: str

    # Embeddings : modèle local et gratuit. Un autre modèle local se choisit
    # via embeddings_model (et embeddings_dimension si sa taille diffère).
    embeddings_provider: Literal["local"] = "local"
    embeddings_model: str = "intfloat/multilingual-e5-base"
    embeddings_dimension: int = 768
    # Préfixes attendus par la famille e5 devant la question et devant
    # l'article. Vides pour un modèle sans cette convention (BAAI/bge-m3).
    embeddings_prefixe_requete: str = "query: "
    embeddings_prefixe_passage: str = "passage: "

    # Reclassement des résultats de recherche : modèle local et gratuit. Un
    # autre modèle local se choisit via rerank_model. "onnx" (OnnxReranker,
    # reranker.py) exécute le même modèle sur la carte graphique ; "local"
    # reste le défaut, qui fonctionne sur toute machine.
    rerank_provider: Literal["local", "onnx"] = "local"
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    # Dossier du modèle exporté en ONNX (scripts/exporter_reclasseur_onnx.py),
    # utilisé seulement si rerank_provider="onnx".
    rerank_onnx_dir: str = ".cache/reclasseur_onnx"
    # Longueur maximale, en tokens, de chaque paire (question, morceau) soumise
    # au reclasseur. bge-reranker-v2-m3 accepte jusqu'à 8192 tokens et traite
    # sinon les articles longs en entier, ce qui rend une passe CPU très lente.
    # 512 borne ce coût et aligne la fenêtre du reclasseur sur celle des embeddings e5.
    rerank_max_length: int = 512

    # Garde-fou étage 1 (cf. CLAUDE.md) : sous ce score de reclassement, un
    # morceau est écarté des résultats plutôt que soumis au LLM générateur.
    # Mesuré sur le golden dataset (ablation --rerank, 2026-09-10) : score du
    # meilleur passage médiane 0,939 pour les questions du corpus, max 0,234
    # pour les questions hors corpus. 0.3 sépare nettement les deux groupes.
    seuil_garde_fou_retrieval: float = 0.3

    # Résultats demandés à chaque recherche (sens, mots-clés) avant fusion, et
    # candidats de la fusion soumis au reclasseur. Mesuré sur le golden dataset
    # (ablation --rerank, 2026-10-03) : 30/30 au lieu de 20/20 fait remonter
    # l'article 18 de la loi de 1965 (missions du syndic, 23e en recherche par
    # le sens) — hit@5 83,9 % → 87,1 %, MRR 0,655 → 0,686, questions hors
    # corpus inchangées (max 0,13). 40/40 n'apportait rien de plus alors.
    # Le 2026-10-23, 40/40 retenu : l'article 977 (barème de l'IFI, « tarif »,
    # 20e par le sens, absent des mots-clés) entrait juste au-dessous des 30
    # candidats pour « Quels sont les taux de l'impôt sur la fortune
    # immobilière (IFI) ? » (q48, question de suivi réécrite) ; avec 40, 1er.
    # Gate : hit@5 et hit@10 inchangés, MRR 0,802 → 0,792 (q32 passe du 1er au
    # 2e rang, toujours parmi les articles envoyés au LLM), hors corpus inchangé.
    k_par_source: int = 40
    k_avant_rerank: int = 40

    # Nombre d'articles envoyés au LLM pour rédiger la réponse. Chaque article
    # consomme du quota Groq gratuit (8000 tokens/minute). Mesuré sur le golden
    # dataset (2026-09-29) : 5 au lieu de 10 réduit les tokens d'entrée de 38 %
    # (médiane 3757 -> 2325) pour une seule question dont l'article attendu
    # sort du lot (23/28 -> 22/28) ; 3 en écarte trop (19/28).
    k_articles_generation: int = 5

    # Garde-fou étage 2 (cf. CLAUDE.md) : vérifie que chaque affirmation de la
    # réponse générée est bien impliquée par le texte des articles cités,
    # via un modèle NLI local et gratuit. Un modèle anglais-only échouerait
    # silencieusement sur des affirmations en français.
    nli_provider: Literal["local"] = "local"
    nli_model: str = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
    nli_max_length: int = 512
    # Au-dessus de seuil_nli_haut, l'affirmation est jugée fondée sans recours
    # au LLM. En dessous, le score du NLI seul ne permet pas de trancher :
    # repli sur un juge LLM (cf. CLAUDE.md, "cas limites").
    seuil_nli_haut: float = 0.7
    # Même au-dessus de seuil_nli_haut, une affirmation qu'un extrait de
    # l'article contredit au-delà de ce seuil passe au juge LLM : signaux
    # opposés, le NLI seul ne tranche pas (cf. generator.verifier_fidelite).
    seuil_nli_contradiction: float = 0.7

    # LLM du contextual retrieval, de la génération de réponse, et du juge de
    # repli du garde-fou étage 2. Groq offre un accès gratuit ; le volume
    # d'usage doit rester dans ses limites.
    llm_provider: Literal["groq"] = "groq"
    llm_model: str = "openai/gpt-oss-20b"
    # Modèles de relais quand llm_model a épuisé son quota par minute : Groq
    # compte le quota séparément pour chaque modèle, deux modèles doublent
    # donc le débit (cf. generator.LLMEnCascade). qwen/qwen3.8-27b est écarté :
    # sa limite de 1000 tokens de sortie par minute est sous le max_tokens de
    # 1024 que réserve chaque appel. Liste vide : un seul modèle, sans cascade.
    llm_modeles_relais: list[str] = ["openai/gpt-oss-120b"]
    # Juge de repli du garde-fou étage 2 : essayé en premier pour le juge, les
    # autres modèles servant de relais. Mesuré le 2026-10-02 : le 20b rejetait
    # « l'état des lieux est obligatoire » (art. 3-2 : « est établi »), le
    # 120b l'accepte, et les deux refusent les réponses fausses testées.
    llm_modele_juge: str = "openai/gpt-oss-120b"
    # Après les garde-fous, le juge relit la réponse en partant des articles
    # cités : une omission qui la rend trompeuse (règle générale tue derrière
    # un plafond majoré, q58) déclenche une seconde rédaction
    # (generator.verifier_completude). « cible » : seulement quand la réponse
    # s'appuie sur une dérogation ou un régime particulier
    # (generator.completude_a_verifier) ; « toujours » épuisait le quota
    # gratuit du juge (golden arrêté à q17 le 2026-10-22).
    controle_completude: Literal["cible", "toujours", "non"] = "cible"

    groq_api_key: str | None = None

    # Origines autorisées à appeler l'API depuis un navigateur (front, phase 3).
    # Format JSON dans .env, ex. ["http://localhost:5173"] ; vide par défaut.
    cors_origins: list[str] = []
