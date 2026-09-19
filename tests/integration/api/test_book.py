"""Textbook endpoints and the SQL-injection guarantee (İP-8, api.md §3.7)."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
import httpx
import pytest

from tests.integration.api.conftest import account, auth_header, load_book

pytestmark = pytest.mark.integration


@pytest.fixture
async def loaded(client: httpx.AsyncClient, app: FastAPI, book_datasets: Path) -> dict[str, str]:
    """A signed-in student and the two sample chapters loaded into the database."""
    _user, token = await account(client, app, "kitap@ornek.edu.tr")
    load_book(app, book_datasets)
    return auth_header(token)


async def test_api_book_fetch_turev_tanimi_returns_definition_unit(
    client: httpx.AsyncClient, loaded: dict[str, str]
) -> None:
    """ "türev tanımını getir" resolves to the Definition unit (the İP-8 criterion phrase)."""
    response = await client.post(
        "/book/resolve", json={"text": "türev tanımını getir"}, headers=loaded
    )
    assert response.status_code == 200, response.text
    hits = response.json()
    assert hits, "the İP-8 command must resolve"
    unit = hits[0]["unit"]
    assert unit["id"] == "tanim-2.4"
    assert unit["type"] == "Definition"
    assert unit["title_tr"]
    assert unit["rendered"]["body_html"]
    assert unit["jsonld"]


async def test_api_book_fetch_through_compute_text(
    client: httpx.AsyncClient, loaded: dict[str, str]
) -> None:
    """The same phrase through ``POST /compute/text`` returns ``kind:"book"``, never the CAS."""
    response = await client.post(
        "/compute/text", json={"text": "türev tanımını getir"}, headers=loaded
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["kind"] == "book"
    assert body["result"] is None
    assert body["command"]["op"] == "book_fetch"
    assert [unit["id"] for unit in body["book_units"]][:1] == ["tanim-2.4"]


async def test_api_book_search_and_unit_and_related(
    client: httpx.AsyncClient, loaded: dict[str, str]
) -> None:
    """Search, unit lookup and the one-hop neighbourhood all answer."""
    found = await client.get("/book/search", params={"q": "türev tanımı"}, headers=loaded)
    assert found.status_code == 200
    assert found.json()[0]["unit"]["id"] == "tanim-2.4"

    unit = await client.get("/book/units/tanim-2.4", headers=loaded)
    assert unit.status_code == 200
    assert unit.json()["chapter"] == "bolum-2"

    related = await client.get("/book/units/tanim-2.4/related", headers=loaded)
    assert related.status_code == 200
    assert isinstance(related.json(), list)

    missing = await client.get("/book/units/yok-9.9", headers=loaded)
    assert missing.status_code == 404
    assert missing.json()["code"] == "NOT_FOUND"


async def test_api_book_chapters_lists_the_loaded_tree(
    client: httpx.AsyncClient, loaded: dict[str, str]
) -> None:
    """The chapter tree reflects what was loaded."""
    response = await client.get("/book/chapters", headers=loaded)
    assert response.status_code == 200
    chapters = {item["id"] for item in response.json()}
    assert {"bolum-1", "bolum-2"} <= chapters


async def test_api_book_export_requires_staff(
    client: httpx.AsyncClient, app: FastAPI, loaded: dict[str, str]
) -> None:
    """JSON-LD and Turtle exports are instructor/admin only and re-parse as graphs."""
    denied = await client.get("/book/export.jsonld", headers=loaded)
    assert denied.status_code == 403

    _user, token = await account(client, app, "hoca@ornek.edu.tr", "instructor")
    jsonld = await client.get("/book/export.jsonld", headers=auth_header(token))
    assert jsonld.status_code == 200
    assert "@graph" in jsonld.json()

    turtle = await client.get("/book/export.ttl", headers=auth_header(token))
    assert turtle.status_code == 200
    assert turtle.headers["content-type"].startswith("text/turtle")
    assert "@prefix" in turtle.text


@pytest.mark.parametrize(
    "needle",
    ["' OR 1=1 --", 'x"; DROP TABLE book_units; --', "%' UNION SELECT * FROM users --"],
)
async def test_api_sql_injection_string_in_search_is_parameterized(
    client: httpx.AsyncClient, loaded: dict[str, str], needle: str
) -> None:
    """An injection string is data: it returns nothing and leaves every table in place."""
    response = await client.get("/book/search", params={"q": needle}, headers=loaded)
    assert response.status_code == 200, response.text
    assert response.json() == []

    still_there = await client.get("/book/units/tanim-2.4", headers=loaded)
    assert still_there.status_code == 200

    resolved = await client.post("/book/resolve", json={"text": needle}, headers=loaded)
    assert resolved.status_code == 200
    assert resolved.json() == []


async def test_api_book_search_rejects_an_unknown_type_filter(
    client: httpx.AsyncClient, loaded: dict[str, str]
) -> None:
    """The ``type`` filter is an allow-list, not a free string."""
    response = await client.get(
        "/book/search", params={"q": "türev", "type": "Uydurma"}, headers=loaded
    )
    assert response.status_code == 422
    assert response.json()["code"] == "VALIDATION_ERROR"
