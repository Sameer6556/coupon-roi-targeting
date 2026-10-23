"""API tests against the committed model artifacts (skipped if they haven't been built)."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "artifacts" / "models"
pytestmark = pytest.mark.skipif(
    not (MODELS / "uplift_model.joblib").exists(), reason="run `couponroi all` first"
)


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    import api.main as mod

    return TestClient(mod.app), mod


def test_health(client):
    c, _ = client
    r = c.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_score_example(client):
    c, mod = client
    example = mod.ScoreRequest.model_config["json_schema_extra"]["example"]
    body = c.post("/score", json=example).json()
    assert body["is_regular"] is True
    assert 0 <= body["lapse_probability"] <= 1
    assert len(body["lapse_drivers"]) == 3
    assert body["recommendation"] in {"mail", "skip"}
    # higher cost can only make mailing look worse
    pricier = {**example, "business": {"cost_per_booklet": 50.0}}
    assert c.post("/score", json=pricier).json()["recommendation"] == "skip"


def test_non_regular_gets_no_lapse_score(client):
    c, mod = client
    example = mod.ScoreRequest.model_config["json_schema_extra"]["example"]
    f = {**example["features"], "active_weeks_16w": 3, "active_weeks_8w": 2}
    body = c.post("/score", json={"features": f}).json()
    assert body["is_regular"] is False and body["lapse_probability"] is None


def test_validation_rejects_impossible_values(client):
    c, mod = client
    example = mod.ScoreRequest.model_config["json_schema_extra"]["example"]
    f = {**example["features"], "active_weeks_8w": 12}
    assert c.post("/score", json={"features": f}).status_code == 422


def test_unknown_household_is_404(client):
    c, _ = client
    assert c.get("/households/99999999").status_code == 404
