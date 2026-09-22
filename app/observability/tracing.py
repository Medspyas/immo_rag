"""Traçage optionnel des appels (Langfuse).

Le décorateur @trace enregistre un appel de fonction (durée, entrée, sortie)
dans Langfuse lorsqu'il est configuré (cf. CLAUDE.md, observabilité). Sans
configuration, il ne fait rien d'autre qu'appeler la fonction normalement :
le code tracé n'a jamais à savoir si le traçage est actif, et les tests des
autres modules n'ont pas besoin de connaître Langfuse pour continuer à
fonctionner.

Volontairement lu depuis l'environnement (LANGFUSE_*) plutôt que depuis
app.config.Settings : Settings exige des réglages obligatoires (database_url)
propres au cœur du produit, alors que le traçage est une préoccupation
transversale qui doit rester utilisable même quand ce cœur n'est pas
configurable dans le contexte courant (ex. un script d'évaluation autonome).

Contenu du fichier :
- trace : décorateur, no-op si Langfuse n'est pas configuré.
"""
from __future__ import annotations

import functools
import os
from collections.abc import Callable
from typing import TypeVar

F = TypeVar("F", bound=Callable)


def _client_langfuse():
    """Renvoie le client Langfuse si les deux clés sont présentes dans
    l'environnement, sinon None.

    Import paresseux : le paquet langfuse (facultatif, non listé dans
    requirements.txt tant que l'observabilité n'est pas branchée) n'est requis
    que si le traçage est réellement activé.
    """
    cle_publique = os.environ.get("LANGFUSE_PUBLIC_KEY")
    cle_secrete = os.environ.get("LANGFUSE_SECRET_KEY")
    if not cle_publique or not cle_secrete:
        return None

    from langfuse import Langfuse

    return Langfuse(
        public_key=cle_publique,
        secret_key=cle_secrete,
        host=os.environ.get("LANGFUSE_HOST"),
    )


def trace(nom: str | None = None) -> Callable[[F], F]:
    """Décorateur : trace un appel de fonction dans Langfuse s'il est
    configuré, ne fait rien d'autre qu'appeler la fonction sinon.

    nom identifie l'appel dans Langfuse ; le nom de la fonction décorée sert
    de valeur par défaut.
    """

    def decorateur(fonction: F) -> F:
        @functools.wraps(fonction)
        def emballage(*args, **kwargs):
            client = _client_langfuse()
            if client is None:
                return fonction(*args, **kwargs)

            with client.start_as_current_span(name=nom or fonction.__name__) as span:
                resultat = fonction(*args, **kwargs)
                span.update(output=resultat)
                return resultat

        return emballage

    return decorateur
