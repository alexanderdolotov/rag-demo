"""
Load the story files (stories/) into OpenSearch and Neo4j, once, so both databases hold exactly the same corpus.

    python data_load.py             # load what is missing; does nothing when both already hold this corpus;
                                        # refuses to touch a database that holds a different one
    python data_load.py --reload    # first delete the story data in both databases, then load again
    python data_load.py --status    # what each database holds compared with the files; changes nothing
    python data_load.py --no-rewrites   # load only the original version of each story (see below)

--no-rewrites loads the originals and leaves the rewrites out of both databases (a story folder then needs no rewrite files). Both
databases record that, next to the fingerprint of what was loaded, so ask_stories.py works on whichever you loaded; test/compare_stories.py
needs the rewrites and says so. Switching between the two needs --reload, like any other change to the corpus.

--reload deletes only this project's data: the OpenSearch index `stories` and the Neo4j nodes labelled Story, StoryVersion,
StoryTerm and StoryCorpus (the indexes and constraints starting with story_ / term_ are rebuilt). Anything else in either
database is left alone. Both databases record a fingerprint of the files they were loaded from, so ask_stories.py and
test/compare_stories.py can refuse to run against a database that no longer matches the files.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
import corpus  # noqa: E402
import neo4j_stories as ns  # noqa: E402
import opensearch_stories as os_s  # noqa: E402
import retrieval  # noqa: E402

INDEX = "stories"


# ---------------------------------------------------------------------------
# What is loaded
# ---------------------------------------------------------------------------
def has_rewrites(docs: List[Dict]) -> bool:
    return any(d["kind"] == "rewrite" for d in docs)


def in_scope(docs: List[Dict], rewrites: bool) -> List[Dict]:
    """All the versions, or only the originals."""
    return docs if rewrites else [d for d in docs if d["kind"] == "original"]


def _scope(rewrites: Optional[bool]) -> str:
    return "with its rewrites" if rewrites else "without its rewrites (the originals only)"


def opensearch_state(index: str, disk_hash: str, versions: Optional[int] = None) -> Dict[str, object]:
    """{"status": 'empty' | 'same' | 'incomplete' | 'different', "docs": n, "hash": fingerprint it was loaded from, "rewrites": whether the
    rewrites were loaded (data loaded before this was recorded counts as yes)}.
    'incomplete': the fingerprint matches but, given the number of `versions` in the files, documents are missing."""
    state = os_s.index_state(index)
    if state is None:
        return {"status": "empty", "docs": 0, "hash": None, "rewrites": None}
    loaded = state["meta"].get("corpus_hash")
    if loaded != disk_hash:
        status = "different"
    else:
        status = "incomplete" if versions is not None and state["docs"] != versions else "same"
    return {"status": status, "docs": state["docs"], "hash": loaded, "rewrites": state["meta"].get("rewrites", True)}


def neo4j_state(driver, disk_hash: str) -> Dict[str, object]:
    meta = ns.read_meta(driver)
    if meta is None:
        leftovers = ns._run(driver, "MATCH (v:StoryVersion) RETURN count(v) AS n")[0]["n"]     # a graph from before the fingerprint existed
        return {"status": "different" if leftovers else "empty", "docs": leftovers, "hash": None, "rewrites": None}
    return {"status": "same" if meta.get("corpus_hash") == disk_hash else "different", "docs": meta["versions"],
            "hash": meta.get("corpus_hash"), "rewrites": meta.get("rewrites", True),
            "scenario": f"targets={meta['targets']}, learn_from={meta['learn_from']}"}


def problems(docs: List[Dict], index: str, driver) -> List[str]:
    """Why a database cannot be searched as the loaded corpus; empty when both hold exactly these files (with the rewrites if `docs` has
    them, the originals only if it has not)."""
    disk, wants = corpus.corpus_hash(docs), has_rewrites(docs)
    found = []
    for name, state in (("OpenSearch", opensearch_state(index, disk, len(docs))), ("Neo4j", neo4j_state(driver, disk))):
        if state["status"] == "empty":
            found.append(f"{name} holds no story data")
        elif state["status"] == "incomplete":
            found.append(f"{name} holds {state['docs']} documents but the files have {len(docs)} versions (an interrupted load, or documents removed since)")
        elif state["hash"] is None:
            found.append(f"{name} holds story data without a fingerprint (an interrupted load, or not made by data_load.py)")
        elif state["rewrites"] != wants:
            found.append(f"{name} holds the corpus {_scope(state['rewrites'])}, not {_scope(wants)}")
        elif state["status"] == "different":
            found.append(f"{name} was loaded from different story files (loaded {state['hash']}, files {disk})")
    meta = ns.read_meta(driver)
    if meta and (meta["targets"], meta["learn_from"]) != ("all", "all"):
        found.append(f"the Neo4j graph is in the scenario targets={meta['targets']}, learn_from={meta['learn_from']}, not the loaded corpus")
    return found


def loaded_scope(index: str, driver) -> Optional[bool]:
    """True if both databases hold the rewrites, False if both hold the originals only; None when nothing is loaded or they disagree."""
    state, graph = os_s.index_state(index), ns.read_meta(driver)
    scopes = {meta.get("rewrites", True) for meta in ((state or {}).get("meta"), graph) if meta}
    return scopes.pop() if len(scopes) == 1 else None


def check(docs: List[Dict], index: str, driver) -> Tuple[List[Dict], List[str]]:
    """For a script that searches whatever was loaded: (the story files in the scope the databases were loaded with, problems()). If nothing is
    loaded, or the two databases disagree about the scope, the full corpus is what is checked against, so the problems say what is wrong."""
    wanted = in_scope(docs, loaded_scope(index, driver) is not False)
    return wanted, problems(wanted, index, driver)


# ---------------------------------------------------------------------------
# Changing it
# ---------------------------------------------------------------------------
def truncate(index: str, driver) -> Dict[str, int]:
    """Delete this project's data in both databases; returns what was removed."""
    state = os_s.index_state(index)
    removed = {"opensearch_docs": state["docs"] if state else 0,
               "neo4j_nodes": ns._run(driver, "MATCH (n) WHERE n:Story OR n:StoryVersion OR n:StoryTerm OR n:StoryCorpus RETURN count(n) AS n")[0]["n"]}
    os_s.drop_index(index)
    ns.wipe(driver)
    return removed


def load(docs: List[Dict], index: str, driver, embedder, targets=("opensearch", "neo4j")) -> None:
    meta = {"corpus_hash": corpus.corpus_hash(docs), "rewrites": has_rewrites(docs),
            **{k: v for k, v in corpus.stats(docs).items() if k in ("stories", "versions")}}
    if "opensearch" in targets:
        os_s.build_index(index, embedder, docs, meta)
    if "neo4j" in targets:
        ns.build_graph(driver, ns.prepare(embedder, docs), "all", "all", embedder, meta)


def run(docs: List[Dict], index: str, driver, embedder, reload: bool = False) -> Dict[str, object]:
    """The whole decision: what to delete and what to load. Raises RuntimeError rather than overwrite a different corpus."""
    disk = corpus.corpus_hash(docs)
    result: Dict[str, object] = {"removed": None, "loaded": []}
    if reload:
        result["removed"] = truncate(index, driver)
    wants = has_rewrites(docs)
    states = {"opensearch": opensearch_state(index, disk, len(docs)), "neo4j": neo4j_state(driver, disk)}
    different = [name for name, st in states.items() if st["status"] in ("different", "incomplete")]
    if different:
        other_scope = [n for n in different if states[n]["hash"] is not None and states[n]["rewrites"] != wants]
        hint = f" ({' and '.join(other_scope)} hold it {_scope(states[other_scope[0]]['rewrites'])})" if other_scope else ""
        raise RuntimeError(f"{' and '.join(different)} already hold a different corpus or an incomplete one{hint}; run with --reload to replace it")
    needed = [name for name, st in states.items() if st["status"] == "empty"]
    if needed:
        load(docs, index, driver, embedder, tuple(needed))
    result["loaded"] = needed
    return result


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
def describe(docs: List[Dict], embedder) -> List[str]:
    s = corpus.stats(docs)
    lines = [f"files: {s['stories']} stories, {s['versions']} versions{'' if has_rewrites(docs) else ', originals only'} "
             f"({s['min_words']}-{s['max_words']} words each, {s['total_words']:,} in all), fingerprint {s['hash']}"]
    cut = corpus.over_window(docs, embedder)
    if cut:
        lines.append(f"WARNING: {len(cut)} version(s) are longer than the embedding model reads ({embedder.max_seq_length} tokens) and would be cut: {', '.join(cut[:5])}")
    return lines


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reload", action="store_true", help="delete the story data in both databases first, then load")
    parser.add_argument("--status", action="store_true", help="show what is loaded and change nothing")
    parser.add_argument("--no-rewrites", action="store_true",
                        help="load only the original version of each story (and check against only the originals); ask_stories.py works on that, "
                             "test/compare_stories.py needs the rewrites")
    parser.add_argument("--stories-dir", default=str(corpus.STORIES_DIR))
    parser.add_argument("--index", type=os_s.index_name, default=INDEX, help="OpenSearch index name")
    args = parser.parse_args(argv)

    try:
        docs = in_scope(corpus.load_corpus(Path(args.stories_dir), require_rewrites=not args.no_rewrites), not args.no_rewrites)
    except corpus.CorpusError as e:
        print(f"stories folder problem: {e}")
        return 2
    if not os_s.reachable():
        print(f"OpenSearch not reachable at {os_s.URL} (docker start opensearch-poc)")
        return 2
    if not ns.reachable():
        print(f"Neo4j not reachable at {ns.URI} (docker start neo4j-poc)")
        return 2
    embedder = retrieval.load_embedder()
    if embedder is None:
        print("sentence-transformers is not installed: pip install -r requirements.txt")
        return 2

    print("\n".join(describe(docs, embedder)))
    driver = ns.connect()
    try:
        disk = corpus.corpus_hash(docs)
        if args.status:
            for name, state in (("OpenSearch", opensearch_state(args.index, disk, len(docs))), ("Neo4j", neo4j_state(driver, disk))):
                extra = (", originals only" if state["rewrites"] is False else "") + (f", {state['scenario']}" if state.get("scenario") else "")
                print(f"{name:10s} {state['status']:9s} {state['docs']} versions loaded{extra}")
            return 0
        try:
            result = run(docs, args.index, driver, embedder, reload=args.reload)
        except RuntimeError as e:
            print(f"not loaded: {e}")
            return 1
        if result["removed"]:
            r = result["removed"]
            print(f"removed: {r['opensearch_docs']} OpenSearch documents (index {args.index}), {r['neo4j_nodes']} Neo4j nodes")
        print(f"loaded into: {', '.join(result['loaded'])}" if result["loaded"] else "nothing to do: both databases already hold these files")
        found = problems(docs, args.index, driver)
        print(("check: both databases hold the same corpus" + ("" if has_rewrites(docs) else " (originals only)")) if not found
              else "check FAILED:\n  " + "\n  ".join(found))
        return 0 if not found else 1
    finally:
        driver.close()


if __name__ == "__main__":
    sys.exit(main())
