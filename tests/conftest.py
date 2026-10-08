import httpx
import pytest

from rag.config import load_config


@pytest.fixture(scope="session")
def config():
    return load_config()


def _reachable(url: str) -> bool:
    try:
        httpx.get(url, timeout=2).raise_for_status()
        return True
    except httpx.HTTPError:
        return False


def pytest_collection_modifyitems(config, items):
    llm_items = [i for i in items if "llm" in i.keywords]
    if llm_items and not _reachable(load_config().llm.base_url + "/models"):
        skip = pytest.mark.skip(reason="vLLM not reachable")
        for item in llm_items:
            item.add_marker(skip)
