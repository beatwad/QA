import httpx
import pytest

pytestmark = pytest.mark.integration


def test_qdrant_is_up(config):
    resp = httpx.get(config.qdrant.url + "/collections", timeout=5)
    assert resp.status_code == 200
