import pytest
import json

from app.config import settings
from app.disease_info import INFO, LABELS

from .conftest import make_image

CLASSES = set(json.loads(settings.class_names_path.read_text()))


def test_every_class_has_info():
    assert CLASSES <= set(INFO) and CLASSES <= set(LABELS)



def test_health(client):
    r = client.get("/api/health")
    assert r.json()["status"] == "ok" and r.json()["model_loaded"] is True


def test_register_login_me(client):
    body = {"name": "A", "email": "Dup@Example.com", "password": "password123"}
    assert client.post("/api/auth/register", json=body).status_code == 201
    assert client.post("/api/auth/register", json=body).status_code == 409
    bad = client.post("/api/auth/login", data={"username": "dup@example.com", "password": "wrongpass1"})
    assert bad.status_code == 401
    ok = client.post("/api/auth/login", data={"username": "dup@example.com", "password": "password123"})
    assert ok.status_code == 200
    me = client.get("/api/users/me", headers={"Authorization": f"Bearer {ok.json()['access_token']}"})
    assert me.json()["email"] == "dup@example.com"


def test_register_validation(client):
    assert client.post("/api/auth/register", json={"name": "A", "email": "nope", "password": "password123"}).status_code == 422
    assert client.post("/api/auth/register", json={"name": "A", "email": "a@b.co", "password": "short"}).status_code == 422


def test_requires_auth(client):
    assert client.post("/api/predict", files={"file": ("a.jpg", make_image(), "image/jpeg")}).status_code == 401
    assert client.get("/api/predictions", headers={"Authorization": "Bearer garbage"}).status_code == 401


@pytest.mark.parametrize("fmt,ctype", [("JPEG", "image/jpeg"), ("PNG", "image/png"), ("WEBP", "image/webp")])
def test_predict_formats(client, auth, fmt, ctype):
    h, _ = auth
    r = client.post("/api/predict", headers=h, files={"file": ("leaf", make_image(fmt), ctype)})
    assert r.status_code == 201, r.text
    d = r.json()
    assert set(d["probabilities"]) == CLASSES
    assert abs(sum(d["probabilities"].values()) - 1) < 1e-3
    assert d["predicted_class"] in d["probabilities"]
    assert d["confidence"] == pytest.approx(max(d["probabilities"].values()))
    assert d["info"]["advice"]
    img = client.get(f"/api/predictions/{d['id']}/image", headers=h)
    heat = client.get(f"/api/predictions/{d['id']}/heatmap", headers=h)
    assert img.status_code == heat.status_code == 200
    assert heat.headers["content-type"] == "image/png" and heat.content[:4] == b"\x89PNG"


def test_predict_is_deterministic(client, auth):
    h, _ = auth
    img = make_image()
    a = client.post("/api/predict", headers=h, files={"file": ("a.jpg", img, "image/jpeg")}).json()
    b = client.post("/api/predict", headers=h, files={"file": ("a.jpg", img, "image/jpeg")}).json()
    assert a["probabilities"] == pytest.approx(b["probabilities"])


def test_invalid_inputs(client, auth):
    h, _ = auth
    assert client.post("/api/predict", headers=h, files={"file": ("a.txt", b"hello", "text/plain")}).status_code == 415
    assert client.post("/api/predict", headers=h, files={"file": ("a.jpg", b"not an image", "image/jpeg")}).status_code == 422
    assert client.post("/api/predict", headers=h, files={"file": ("a.jpg", b"", "image/jpeg")}).status_code == 400
    assert client.post("/api/predict", headers=h).status_code == 422


def test_history_isolation_stats_delete(client, auth):
    h, _ = auth
    ids = [client.post("/api/predict", headers=h, files={"file": ("a.jpg", make_image(), "image/jpeg")}).json()["id"] for _ in range(2)]
    hist = client.get("/api/predictions", headers=h).json()
    assert [p["id"] for p in hist] == ids[::-1]  # newest first
    st = client.get("/api/stats", headers=h).json()
    assert st["total"] == 2 and sum(st["by_class"].values()) == 2 and len(st["last_7_days"]) == 7
    assert st["last_7_days"][-1]["count"] == 2

    other = client.post("/api/auth/register", json={"name": "B", "email": "other@example.com", "password": "password123"}).json()
    oh = {"Authorization": f"Bearer {other['access_token']}"}
    for path in ("", "/image", "/heatmap"):
        assert client.get(f"/api/predictions/{ids[0]}{path}", headers=oh).status_code == 404
    assert client.delete(f"/api/predictions/{ids[0]}", headers=oh).status_code == 404
    assert client.get("/api/predictions", headers=oh).json() == []

    assert client.delete(f"/api/predictions/{ids[0]}", headers=h).status_code == 204
    assert client.get(f"/api/predictions/{ids[0]}", headers=h).status_code == 404


def _post(client, h):
    return client.post("/api/predict", headers=h, files={"file": ("a.jpg", make_image(), "image/jpeg")})


def test_corn_flag_matches_class(client, auth):
    h, _ = auth
    d = _post(client, h).json()
    assert d["is_corn_leaf"] == (d["predicted_class"] != "Not_Leaf")


def test_low_confidence_is_rejected_as_not_corn(client, auth, monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "reject_below_confidence", 1.01)  # nothing can pass
    h, _ = auth
    r = _post(client, h)
    assert r.status_code == 201
    d = r.json()
    assert d["is_corn_leaf"] is False and d["predicted_class"] == "Not_Leaf"
    assert d["label"] == "Not a corn leaf" and d["info"]["symptoms"] == [] and d["low_confidence"] is False
    st = client.get("/api/stats", headers=h).json()
    assert st["not_leaf"] == 1 and st["diseased"] == 0 and st["healthy"] == 0
    assert st["average_confidence"] == 0 and st["by_class"]["Not_Leaf"] == 1
    assert client.get("/api/predictions", headers=h).json()[0]["is_corn_leaf"] is False


def test_reject_reason_logic(client, monkeypatch):
    """Decision logic, independent of what any particular model predicts."""
    import numpy as np
    from app.config import settings
    from app.main import app
    from app.ml import OodGate, Result

    clf = app.state.classifier
    mk = lambda cls="Healthy", conf=0.95, dist=None: Result(cls, conf, {}, b"", b"", dist)
    monkeypatch.setattr(settings, "reject_below_confidence", 0.55)
    monkeypatch.setattr(clf, "ood", None)
    assert clf.reject_reason(mk()) is None
    assert clf.reject_reason(mk(conf=0.40)) == "low_confidence"
    assert clf.reject_reason(mk(cls="Not_Leaf")) == "model_not_leaf"

    gate = object.__new__(OodGate)
    gate.means, gate.precision, gate.threshold = np.zeros((1, 1280)), np.eye(1280), 100.0
    monkeypatch.setattr(clf, "ood", gate)
    assert clf.reject_reason(mk(dist=50.0)) is None
    assert clf.reject_reason(mk(dist=150.0)) == "unfamiliar_image"
    monkeypatch.setattr(settings, "ood_scale", 2.0)  # more lenient: threshold becomes 200
    assert clf.reject_reason(mk(dist=150.0)) is None
    assert gate.distance(np.full(1280, 2.0)) == 4.0 * 1280  # squared distance to the zero mean


def test_ood_gate_is_applied_end_to_end(client, auth, monkeypatch):
    import numpy as np
    from app.main import app
    from app.ml import OodGate
    gate = object.__new__(OodGate)
    gate.means, gate.precision, gate.threshold = np.zeros((1, 1280)), np.eye(1280), 0.0  # everything is "too far"
    monkeypatch.setattr(app.state.classifier, "ood", gate)
    h, _ = auth
    d = _post(client, h).json()
    assert d["is_corn_leaf"] is False and d["predicted_class"] == "Not_Leaf"
