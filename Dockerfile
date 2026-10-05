FROM python:3.12-slim

# Pas de fichiers .pyc ni de sortie retenue en mémoire tampon : les journaux
# apparaissent tout de suite dans `docker compose logs`.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# PyTorch en version processeur : la version par défaut embarque CUDA
# (plusieurs Go) inutile dans ce conteneur.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app app
COPY init.sql .

# Utilisateur sans privilèges : une faille dans l'API ne donne pas la main
# root sur le conteneur.
RUN useradd --create-home immo && mkdir -p /app/.cache/huggingface && chown -R immo /app/.cache
USER immo

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
