import httpx
import pytest

from app.rag.legifrance import (
    URL_JETON,
    ClientLegifrance,
    articles_legifrance,
    html_en_texte,
)


def _article(num, contenu, ordre, etat="VIGUEUR"):
    return {"num": num, "content": contenu, "intOrdre": ordre, "etat": etat}


# Forme réduite d'une réponse de /consult/legiPart : sections imbriquées,
# articles dans le désordre (l'ordre du texte est donné par intOrdre).
TEXTE = {
    "title": "Loi n° 70-9 du 2 janvier 1970",
    "appellations": ["loi Hoguet 1 [1970]", "loi Hoguet"],
    "articles": [],
    "sections": [
        {
            "title": "Titre II : Dispositions diverses.",
            "intOrdre": 20,
            "etat": "VIGUEUR",
            "articles": [_article("3", "<p>Troisième.</p>", 1)],
            "sections": [],
        },
        {
            "title": "Titre Ier : Des activités.",
            "intOrdre": 10,
            "etat": "VIGUEUR",
            "articles": [
                _article("2", "<p>Deuxième.</p>", 2),
                _article("1", "<p>Premier alinéa.</p><p>Second alinéa.</p>", 1),
                _article("1-1", "<p>Ancien.</p>", 3, etat="ABROGE"),
            ],
            "sections": [],
        },
        {
            "title": "Titre III : Ancien titre",
            "intOrdre": 30,
            "etat": "ABROGE",
            "articles": [_article("3", "<p>Copie laissée dans une section abrogée.</p>", 1)],
            "sections": [],
        },
    ],
}


def test_html_en_texte_une_ligne_par_paragraphe():
    html = "<p>I. - Le  bail <a href='x'>prévu</a> à l'article 3.</p><p></p><p>II. - Suite<br/>et fin.</p>"
    assert html_en_texte(html) == "I. - Le bail prévu à l'article 3.\nII. - Suite\net fin."


def test_html_en_texte_garde_une_ligne_de_tableau_sur_une_ligne():
    html = (
        "<p>1. Le tarif est fixé à :</p><table><tbody>"
        "<tr><th><p>Fraction</p></th><th><p>Tarif</p><p>(en pourcentage)</p></th></tr>"
        "<tr><td><p>N'excédant pas 800 000 €</p></td><td><p>0</p></td></tr>"
        "</tbody></table><p>2. Suite.</p>"
    )
    assert html_en_texte(html) == (
        "1. Le tarif est fixé à :\n"
        "Fraction | Tarif (en pourcentage)\n"
        "N'excédant pas 800 000 € | 0\n"
        "2. Suite."
    )


def test_articles_dans_l_ordre_du_texte_sans_abroges():
    articles = articles_legifrance(TEXTE, "Loi_hoguet")
    assert [a.article for a in articles] == ["1", "2", "3"]
    assert articles[0].content == "Article 1\nPremier alinéa.\nSecond alinéa."
    assert all(a.doc_id == "Loi_hoguet" for a in articles)


def test_article_d_une_section_abrogee_ecarte():
    articles = articles_legifrance(TEXTE, "Loi_hoguet")
    assert "Copie" not in articles[2].content


def test_section_et_titre_avec_nom_d_usage():
    article = articles_legifrance(TEXTE, "Loi_hoguet")[0]
    assert article.section == "Titre Ier : Des activités"
    assert article.titre == "Loi n° 70-9 du 2 janvier 1970 (loi Hoguet)"


def test_numero_en_double_garde_une_seule_fois():
    texte = {
        "title": "Code",
        "articles": [_article("16 ", "<p>A.</p>", 1), _article("16", "<p>B.</p>", 2)],
        "sections": [],
    }
    articles = articles_legifrance(texte, "code")
    assert [(a.article, a.content) for a in articles] == [("16", "Article 16\nA.")]


def test_client_demande_un_jeton_puis_le_reutilise():
    recues = []

    def _serveur(requete):
        recues.append(requete)
        if str(requete.url) == URL_JETON:
            return httpx.Response(200, json={"access_token": "jeton-1", "expires_in": 3600})
        return httpx.Response(200, json={"title": "Loi"})

    client = ClientLegifrance("id", "secret", http=httpx.Client(transport=httpx.MockTransport(_serveur)))
    assert client.texte_consolide("LEGITEXT1", "2026-10-28") == {"title": "Loi"}
    client.texte_consolide("LEGITEXT2", "2026-10-28")
    assert sum(str(r.url) == URL_JETON for r in recues) == 1
    assert recues[1].headers["Authorization"] == "Bearer jeton-1"
    assert b'"textId":"LEGITEXT1"' in recues[1].content.replace(b" ", b"")


def test_client_sans_identifiants_refuse():
    with pytest.raises(ValueError):
        ClientLegifrance("", "")
