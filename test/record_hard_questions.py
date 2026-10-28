"""
Record the real LLM's replies for the hard questions (test/cassettes/hard_questions.json), which test_hard_questions.py then replays for free.
This is the only thing here that calls OpenRouter (model anthropic/claude-haiku-5.5, key from .env). Replies already recorded are never fetched
again, and the run stops before it would pass --max-calls (every call ever made for the file counts, even for replies since discarded).

    python test/record_hard_questions.py --analysis-only    # only the analysis of each question (one call each), and print how it was read
    python test/record_hard_questions.py                    # everything the modes need; replies already recorded are free
    python test/record_hard_questions.py --only h01,h03       # just those questions
    python test/record_hard_questions.py --max-calls 95
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hard_eval as he  # noqa: E402
from llm_cassette import BudgetExceeded, Cassette  # noqa: E402

import ask_stories as ask  # noqa: E402
import corpus  # noqa: E402
import intent as it  # noqa: E402
import os  # noqa: E402
import retrieval  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-calls", type=int, default=95, help="never make more than this many API calls in total, counting those already made (default 95)")
    parser.add_argument("--analysis-only", action="store_true")
    parser.add_argument("--only", help="comma-separated question ids to record (default: all)")
    parser.add_argument("--model", default=ask.DEFAULT_MODELS["openrouter"])
    args = parser.parse_args()
    ask.load_env()
    if not os.environ.get("OPENROUTER_API_KEY"):
        sys.exit("OPENROUTER_API_KEY is not set (put it in .env)")
    cassette = Cassette(he.CASSETTE_FILE, live=True, max_calls=args.max_calls, model=args.model)
    questions = he.load_questions(corpus.load_corpus())
    if args.only:
        questions = [q for q in questions if q["id"] in args.only.split(",")]
    start = cassette.spent
    try:
        if args.analysis_only:
            for q in questions:
                intent = it.parse_intent(cassette(it.ANALYZE_SYSTEM, q["question"]), q["question"])
                print(f"{q['id']}  {q['question']}")
                print("      " + ("; ".join(f"{c.name}: core {list(c.core)} related {list(c.related)}" for c in intent.concepts) +
                                  f" | must {list(intent.must)} | narrowing {list(intent.context)} | unverifiable {list(intent.unverifiable)}" if intent.understood else "(analysis not understood)"))
        else:
            embedder = retrieval.load_embedder()
            with he.built_stores(embedder) as stores:
                table = he.scores(he.evaluate(stores, questions, cassette), questions)
            print(he.format_table(table))
    except BudgetExceeded as e:
        print(f"\nstopped before passing the limit: {e}")
        return 1
    finally:
        print(f"\n{cassette.fetched} API calls made this run; {cassette.spent} spent in total on this file (started at {start}, limit {args.max_calls})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
