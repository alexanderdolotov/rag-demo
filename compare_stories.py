"""
Search engine (OpenSearch) against graph database (Neo4j) on the same story corpus and the same queries.

    python compare_stories.py                 # the story files in stories, in the loaded OpenSearch index and in Neo4j
    python compare_stories.py --explain       # also prints which terms the graph added to each query

Needs OpenSearch (OPENSEARCH_URL) and Neo4j (NEO4J_URI) loaded by data_load.py, see README.md. The OpenSearch arms read the
loaded index. The three Neo4j scenarios are built under their own label prefix ("Cmp"), searched, then deleted again, so the
loaded graph is never touched.

Three scenarios, because what a method may learn from is half the story:
  originals only         only the original texts exist, in the index and in the graph
  rewrites as thesaurus  the answer is always an original, but the graph learns word links from the rewrites (graph arms only)
  all versions           originals and rewrites are all retrievable
The table shows the rank of the first correct story (. = not found), then hit@3 and mean reciprocal rank (mrr).
"""
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
import corpus  # noqa: E402
import data_load  # noqa: E402
import neo4j_stories as ns  # noqa: E402
import opensearch_stories as os_s  # noqa: E402
import retrieval  # noqa: E402

SCENARIOS: List[Tuple[str, str, str]] = [
    ("originals only", "original", "original"),
    ("rewrites as thesaurus", "original", "all"),
    ("all versions", "all", "all"),
]
Ranks = Dict[str, Dict[str, Optional[int]]]   # arm -> query id -> rank of the first correct story


def all_queries(query_sets: Sequence[corpus.QuerySet]) -> List[Tuple[str, str, Tuple[str, ...]]]:
    """Every query of every set, in file order."""
    return [q for s in query_sets for q in s.queries]


def run_scenario(targets: str, learn_from: str, prep, driver, index: str, embedder, queries) -> Tuple[Ranks, "ns.StoryGraph"]:
    """Rank of the first correct story for every arm and every query (id, text, correct story ids) in one scenario."""
    graph = ns.build_graph(driver, prep, targets, learn_from, embedder)
    kind = "original" if targets == "original" else None
    ranks: Ranks = {}
    for qid, text, expect in queries:
        arms: Dict[str, List[Dict]] = {}
        if targets == learn_from:   # an index cannot "learn from rewrites it may not return"
            arms.update(os_s.run_arms(index, text, embedder, kind))
        arms.update(graph.arms(text))
        for arm, hits in arms.items():
            ranks.setdefault(arm, {})[qid] = retrieval.first_relevant_rank(hits, expect)
    return ranks, graph


def summarize(ranks: Dict[str, Optional[int]], qids: Sequence[str]) -> Tuple[int, float]:
    """(hit@3, mean reciprocal rank) over the queries `qids`."""
    return retrieval.summarize([ranks.get(q) for q in qids])


def print_scenario(name: str, ranks: Ranks, graph: "ns.StoryGraph", query_sets: Sequence[corpus.QuerySet]) -> None:
    """One row per arm: the rank of the first correct story for each query, then hit@3 and mrr per query set."""
    sets = [(s.name, [q for q, _, _ in s.queries]) for s in query_sets]
    print(f"== {name}   ({graph.counts['versions']} retrievable texts, {graph.counts['terms']} terms, {graph.counts['co_edges']} co-occurrence edges)")
    print(f"    {'':8s}" + " |".join(" ".join(f"{q:>3s}" for q in ids) for _, ids in sets) + " | " + " ".join(f"{label:>11s}" for label, _ in sets))
    order = ["plain", "stem", "syn", "dense", "hybrid", "story", "n_ft", "n_vec", "n_co", "n_term", "n_hyb", "n_story"]
    for arm in [a for a in order if a in ranks]:
        r = ranks[arm]
        cells = " |".join(" ".join(f"{(str(r[q]) if r[q] else '.'):>3s}" for q in ids) for _, ids in sets)
        totals = "  ".join(f"{summarize(r, ids)[0]:>2d}/{len(ids)} {summarize(r, ids)[1]:5.2f}" for _, ids in sets)
        print(f"    {arm:8s}{cells} | {totals}")
    print()


def explain(graph: "ns.StoryGraph", queries, width: int = 7) -> None:
    for qid, text, expect in queries:
        co = ", ".join(f"{r['word']}" for r in graph.co_expansion(text)[:width]) or "-"
        term = ", ".join(f"{r['word']}<{r['via'][0]}" for r in graph.term_expansion(text) if r["via"][0] != r["word"])[:90] or "-"
        print(f"    {qid:3s} {text[:52]!r:56s} co: {co}")
        print(f"    {'':3s} {'':56s} term: {term}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stories-dir", default=str(corpus.STORIES_DIR))
    parser.add_argument("--index", type=os_s.index_name, default=data_load.INDEX)
    parser.add_argument("--explain", action="store_true", help="print the terms the graph added to each query")
    args = parser.parse_args()

    try:
        docs = corpus.load_corpus(Path(args.stories_dir))
        query_sets = corpus.load_queries(Path(args.stories_dir), docs)
    except corpus.CorpusError as e:
        sys.exit(f"stories folder problem: {e}")
    if not os_s.reachable():
        sys.exit(f"OpenSearch not reachable at {os_s.URL}")
    if not ns.reachable():
        sys.exit(f"Neo4j not reachable at {ns.URI}")
    embedder = retrieval.load_embedder()
    if embedder is None:
        sys.exit("sentence-transformers is not installed: pip install -r requirements.txt")
    driver = ns.connect()
    try:
        found = [p for p in data_load.problems(docs, args.index, driver) if not p.startswith("the Neo4j graph is in the scenario")]
        if found:
            sys.exit("The databases do not hold the story files:\n  " + "\n  ".join(found) + "\nLoad them with: python data_load.py --reload")
        queries = all_queries(query_sets)
        prep = ns.prepare(embedder, docs)
        ns.PREFIX = "Cmp"                       # scenario graphs live next to the loaded one, never in it
        print(f"{len(docs)} texts, {len({d['topic'] for d in docs})} stories (files {corpus.corpus_hash(docs)}); "
              + ", ".join(f"{s.name} = {len(s.queries)} queries" for s in query_sets) + ".  n_* = Neo4j arms, the rest OpenSearch.\n")
        try:
            for name, targets, learn_from in SCENARIOS:
                ranks, graph = run_scenario(targets, learn_from, prep, driver, args.index, embedder, queries)
                print_scenario(name, ranks, graph, query_sets)
                if args.explain and (targets, learn_from) == ("original", "all"):
                    print("    terms added to each query (co = from rewrites, term = nearest words by embedding):")
                    explain(graph, queries)
        finally:
            ns.drop_namespace(driver)
    finally:
        ns.PREFIX = ""
        driver.close()


if __name__ == "__main__":
    main()
