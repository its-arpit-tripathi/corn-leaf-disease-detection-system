import io
import os
import tempfile

_tmp = tempfile.mkdtemp()
os.environ["DATABASE_URL"] = f"sqlite:///{_tmp}/test.db"
os.environ["UPLOAD_DIR"] = f"{_tmp}/uploads"
os.environ["SECRET_KEY"] = "test-secret-key-with-sufficient-length-123456"

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image


@pytest.fixture(scope="session")
def client():
    from app.main import app
    with TestClient(app) as c:  # runs lifespan: loads the real model once
        yield c


def make_image(fmt="JPEG", size=(300, 200)) -> bytes:
    rng = np.random.default_rng(0)
    arr = np.full((*size[::-1], 3), (60, 140, 50), dtype=np.uint8)
    arr[40:70, 60:200] = (150, 110, 60)
    arr = np.clip(arr + rng.integers(-15, 15, arr.shape), 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, fmt)
    return buf.getvalue()


@pytest.fixture()
def auth(client):
    import uuid
    email = f"{uuid.uuid4().hex[:8]}@example.com"
    r = client.post("/api/auth/register", json={"name": "Test", "email": email, "password": "password123"})
    assert r.status_code == 201
    return {"Authorization": f"Bearer {r.json()['access_token']}"}, email


@pytest.fixture(autouse=True)
def no_confidence_floor(monkeypatch):
    """Synthetic test images are not real leaves; keep the confidence floor off unless a test turns it on."""
    from app.config import settings
    monkeypatch.setattr(settings, "reject_below_confidence", 0.0)
