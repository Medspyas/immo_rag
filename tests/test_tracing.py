from app.observability.tracing import trace


def test_sans_configuration_la_fonction_s_execute_normalement(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    @trace()
    def additionner(a, b):
        return a + b

    assert additionner(2, 3) == 5


def test_une_seule_cle_presente_reste_no_op(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    @trace()
    def additionner(a, b):
        return a + b

    assert additionner(2, 3) == 5


def test_preserve_le_nom_et_la_docstring_de_la_fonction(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    @trace()
    def ma_fonction():
        """Une docstring quelconque."""

    assert ma_fonction.__name__ == "ma_fonction"
    assert ma_fonction.__doc__ == "Une docstring quelconque."


class _SpanFake:
    def __init__(self, journal, nom):
        self._journal = journal
        self._nom = nom

    def __enter__(self):
        self._journal.append(f"entree:{self._nom}")
        return self

    def __exit__(self, *exc):
        self._journal.append(f"sortie:{self._nom}")
        return False

    def update(self, output=None):
        self._journal.append(f"sortie_valeur:{output}")


class _ClientFake:
    def __init__(self, journal):
        self._journal = journal

    def start_as_current_span(self, name):
        return _SpanFake(self._journal, name)


def test_configure_ouvre_un_span_et_transmet_le_resultat(monkeypatch):
    journal: list[str] = []
    monkeypatch.setattr(
        "app.observability.tracing._client_langfuse", lambda: _ClientFake(journal)
    )

    @trace("mon_appel")
    def additionner(a, b):
        return a + b

    assert additionner(2, 3) == 5
    assert journal == ["entree:mon_appel", "sortie_valeur:5", "sortie:mon_appel"]


def test_nom_par_defaut_est_celui_de_la_fonction(monkeypatch):
    journal: list[str] = []
    monkeypatch.setattr(
        "app.observability.tracing._client_langfuse", lambda: _ClientFake(journal)
    )

    @trace()
    def ma_fonction():
        return None

    ma_fonction()
    assert journal[0] == "entree:ma_fonction"
