import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # the project root

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
