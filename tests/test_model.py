import pytest


def test_predict(client):
    response = client.post(
        url="/predict",
        json={
            "text": "I am feeling happy today!",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert "label" in body
    assert "confidence" in body


def test_predict_empty_text(client):
    response = client.post(
        url="/predict",
        json={
            "text": "",
        },
    )
    assert response.status_code == 422


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_missing_text(client):
    response = client.post(url="/predict", json={})
    assert response.status_code == 422
