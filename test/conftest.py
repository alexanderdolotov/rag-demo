import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # the project root

import retrieval  # noqa: E402

try:
    import neo4j_stories as ns
except ImportError:          # neo4j / nltk not installed: the Neo4j tests skip themselves
    ns = None

if ns is not None:
    ns.PREFIX = "Test"       # every label and index the tests create lives next to, never in, the loaded corpus


@pytest.fixture(scope="session", autouse=True)
def _remove_test_namespace_from_neo4j():
    yield
    if ns is not None and ns.reachable():
        driver = ns.connect()
        try:
            ns.drop_namespace(driver)
        finally:
            driver.close()


@pytest.fixture(scope="session")
def embedder():
    """The sentence-transformers model, loaded once per run; None when the package is not installed (the vector tests skip themselves)."""
    return retrieval.load_embedder()
