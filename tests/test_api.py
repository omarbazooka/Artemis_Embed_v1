from fastapi.testclient import TestClient
import app as app_module

client = TestClient(app_module.app)


def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["service"] == "Artemis Embed v1"


def test_model_metadata():
    response = client.get("/api/model")
    assert response.status_code == 200
    body = response.json()
    assert body["dimensions"] == [768, 512, 256, 128]
    assert body["pooling"] == "masked_mean"
