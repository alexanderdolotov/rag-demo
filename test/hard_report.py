"""
Print, or with --write put into README.md, the tables of the "Hard questions" section: modes and the commands that run them, scores, the
confusion matrix of each mode and the outcome of every question. They are computed from the recorded LLM replies (no API call) and the
real databases, so the README cannot drift from the tests; test_hard_questions.py fails if it does.

    python test/hard_report.py            # print the tables
    python test/hard_report.py --write    # replace the block between the hard-questions markers in README.md
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hard_eval as he  # noqa: E402
from llm_cassette import Cassette  # noqa: E402

import corpus  # noqa: E402
import retrieval  # noqa: E402

README = Path(__file__).resolve().parent.parent / "README.md"


def block(text: str) -> str:
    """What is between the markers of README text (without them), or raises ValueError."""
    start, end = text.index(he.README_START) + len(he.README_START), text.index(he.README_END)
    return text[start:end].strip("\n")


def replaced(text: str, new: str) -> str:
    start, end = text.index(he.README_START) + len(he.README_START), text.index(he.README_END)
    return text[:start] + "\n" + new + "\n" + text[end:]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", action="store_true", help="update README.md in place")
    args = parser.parse_args()
    docs = corpus.load_corpus()
    questions = he.load_questions(docs)
    with he.built_stores(retrieval.load_embedder()) as stores:
        results = he.evaluate(stores, questions, Cassette(he.CASSETTE_FILE))      # replay only: a missing reply stops here, nothing is fetched
    report = he.markdown_report(results, questions, he.stories_of(docs))
    if args.write:
        README.write_text(replaced(README.read_text(encoding="utf-8"), report), encoding="utf-8")
        print(f"README.md updated ({len(report.splitlines())} lines between the markers)")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
