"""Exporte le reclasseur BAAI/bge-reranker-v2-m3 au format ONNX.

Pourquoi ce script existe
-------------------------
Le reclasseur (reranker.py) tourne par défaut sur le processeur via
sentence-transformers. C'est lent : ~20 s pour reclasser une vingtaine de
passages (mesuré par eval/latence.py), contre ~0,6 s une fois exporté en ONNX
et exécuté sur la carte graphique.

ONNX Runtime sait exécuter le même modèle sur une carte graphique AMD ou Intel
sous Windows (fournisseur « DirectML »). Il faut pour cela le modèle converti
au format ONNX. Cette conversion se fait une seule fois, ici.

Pourquoi un environnement isolé
-------------------------------
L'outil de conversion (optimum) impose une version ancienne de la bibliothèque
transformers, qui traîne des failles corrigées dans la version utilisée par le
projet. Le script crée donc un environnement Python jetable, y installe l'outil,
fait la conversion, récupère les fichiers produits, puis supprime l'environnement.
Rien de tout cela n'entre dans les dépendances du projet.

Résultat
--------
Les fichiers du modèle ONNX (model.onnx + poids + tokenizer) sont copiés dans
.cache/reclasseur_onnx/. C'est le dossier que OnnxReranker (reranker.py) charge
quand RERANK_PROVIDER=onnx.

Usage
-----
    python scripts/exporter_reclasseur_onnx.py

Compter ~10 min (téléchargement de l'outil) et ~2 Go d'espace disque temporaire.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

MODELE = "BAAI/bge-reranker-v2-m3"
OPSET = 18
DESTINATION = Path(__file__).resolve().parent.parent / ".cache" / "reclasseur_onnx"

# Outil de conversion + son moteur. Ils ne vivent que dans l'environnement
# jetable, jamais dans les dépendances du projet.
DEPENDANCES_EXPORT = ["optimum-onnx", "onnx", "onnxruntime"]


def _run(cmd: list[str]) -> None:
    print("  $", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    if DESTINATION.exists() and any(DESTINATION.iterdir()):
        print(f"{DESTINATION} existe déjà. Le supprimer d'abord pour réexporter.")
        return

    with tempfile.TemporaryDirectory(prefix="export_onnx_") as tmp:
        tmp = Path(tmp)
        venv = tmp / "venv"
        sortie = tmp / "modele"

        print("1/4 Création de l'environnement jetable…")
        _run([sys.executable, "-m", "venv", str(venv)])
        bin_dir = venv / ("Scripts" if sys.platform == "win32" else "bin")
        py = bin_dir / ("python.exe" if sys.platform == "win32" else "python")
        optimum_cli = bin_dir / ("optimum-cli.exe" if sys.platform == "win32" else "optimum-cli")

        print("2/4 Installation de l'outil de conversion…")
        _run([str(py), "-m", "pip", "install", "-q", "--upgrade", "pip"])
        _run([str(py), "-m", "pip", "install", "-q", *DEPENDANCES_EXPORT])

        print("3/4 Conversion du modèle (peut prendre quelques minutes)…")
        _run([
            str(optimum_cli),
            "export", "onnx",
            "-m", MODELE,
            "--task", "text-classification",
            "--opset", str(OPSET),
            str(sortie),
        ])

        print(f"4/4 Copie vers {DESTINATION}…")
        DESTINATION.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(sortie, DESTINATION)

    print("Terminé. Régler RERANK_PROVIDER=onnx dans .env pour l'utiliser.")


if __name__ == "__main__":
    main()
