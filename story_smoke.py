"""
Load the dummy stories into OpenSearch and compare ways of finding them.

    python story_smoke.py                 # (re)builds the small in-code `stories_toy` index (not the loaded corpus), then searches it
    python story_smoke.py --no-vectors    # skip the embedding model (keyword arms only)
    python story_smoke.py --keep          # search the existing index without rebuilding it
    OPENSEARCH_URL=http://host:9200 python story_smoke.py

Only the `stories_toy` index (or --index) is created/deleted; the loaded corpus is never touched. Vectors need `pip install sentence-transformers`
(the first run downloads a ~90 MB model).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import opensearch_stories as os_s  # noqa: E402


def show(label, hits, expect, width=4):
    cells = []
    for h in hits[:width]:
        mark = "✓" if h["topic"] in expect else "·"
        via = f"<{h['via']}>" if h.get("via") else ""
        cells.append(f"{mark} {h['topic']}/{h['version']}{via}")
    print(f"    {label:7s} " + "   ".join(cells))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", default="stories_toy")
    parser.add_argument("--no-vectors", action="store_true")
    parser.add_argument("--keep", action="store_true", help="do not rebuild the index")
    args = parser.parse_args()

    if not os_s.reachable():
        sys.exit(f"OpenSearch not reachable at {os_s.URL}")

    embedder = None if args.no_vectors else os_s.load_embedder()
    if embedder is None and not args.no_vectors:
        print("sentence-transformers not installed: running keyword arms only (pip install sentence-transformers)\n")
    if not args.keep:
        n = os_s.build_index(args.index, embedder)
        print(f"loaded {n} story documents into '{args.index}'" + (" with embeddings" if embedder else "") + "\n")

    print("how the index reads a query (story_syn analyzer):")
    for text in ("import duties", "frog jumping"):
        print(f"    {text!r:18} -> {os_s.analyze(args.index, 'story_syn', text)}")
    print()

    for qid, text, expect in os_s.QUERIES:
        print(f"{qid}  {text!r}   correct: {', '.join(expect)}")
        print("  originals only (what the index understands on its own):")
        for arm, hits in os_s.run_arms(args.index, text, embedder, kind="original").items():
            show(arm, hits, expect)
        print("  all versions (originals + rewrites):")
        for arm, hits in os_s.run_arms(args.index, text, embedder).items():
            show(arm, hits, expect)
        print()

    print("scoreboard   hit@3 = correct story in the top 3 of {} queries, mrr = mean reciprocal rank".format(len(os_s.QUERIES)))
    orig = os_s.scoreboard(args.index, embedder, kind="original")
    full = os_s.scoreboard(args.index, embedder)
    print(f"    {'':8s}{'originals only':>22s}{'all versions':>22s}")
    for arm in full:
        a, b = orig.get(arm), full[arm]
        left = f"{int(a['hit3'])}/{int(a['n'])}  mrr {a['mrr']:.2f}" if a else "-"
        right = f"{int(b['hit3'])}/{int(b['n'])}  mrr {b['mrr']:.2f}"
        print(f"    {arm:8s}{left:>22s}{right:>22s}")


if __name__ == "__main__":
    main()
