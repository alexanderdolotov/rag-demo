"""
Shared by test_hard_questions.py (replays recorded LLM replies) and record_hard_questions.py (records them): the hard questions, the search modes
to compare, the stores to search, and how a list of results is scored against the stories that should have been found.
"""
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import ask_stories as ask  # noqa: E402
import corpus  # noqa: E402
import neo4j_stories as ns  # noqa: E402
import opensearch_stories as os_s  # noqa: E402

HERE = Path(__file__).resolve().parent
QUESTIONS_FILE = HERE / "hard_questions.json"
CASSETTE_FILE = HERE / "cassettes" / "hard_questions.json"
INDEX = "stories_hard_test"
DBS = ask.DBS

# Each mode is the app run with these arguments: the options are built by ask_stories' own parser, so what the tests measure is what
# `python ask_stories.py <arguments>` does. Everything else is the default: --os-arm hybrid --neo-arm n_hyb --top 5.
MODES = {
    "no LLM, one search": ["--no-llm"],
    "rewrite only": ["--no-rerank"],
    "rewrite + rerank (default)": [],
}
README_START, README_END = "<!-- hard-questions:start -->", "<!-- hard-questions:end -->"


def command(argv: Sequence[str]) -> str:
    return " ".join(["python ask_stories.py"] + list(argv))


def load_questions(docs: Sequence[Dict]) -> List[Dict]:
    """The hard questions; refuses a story that does not exist, a repeated id, or a story that is both relevant and acceptable."""
    questions = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))["questions"]
    known, seen = {d["topic"] for d in docs}, set()
    for q in questions:
        stories = set(q["relevant"]) | set(q["acceptable"])
        if q["id"] in seen or not q["relevant"] or not stories <= known or set(q["relevant"]) & set(q["acceptable"]):
            raise ValueError(f"{q['id']}: bad question entry")
        seen.add(q["id"])
    return questions


@contextmanager
def built_stores(embedder):
    """The corpus in a scratch OpenSearch index and a `Test`-prefixed Neo4j graph (never the loaded ones), removed again afterwards."""
    docs, previous = corpus.load_corpus(), ns.PREFIX
    ns.PREFIX = "Test"
    driver = ns.connect()
    os_s.build_index(INDEX, embedder, docs)
    try:
        graph = ns.build_graph(driver, ns.prepare(embedder, docs), "all", "all", embedder)
        yield ask.Stores(INDEX, graph, embedder, corpus.originals(docs), {d["id"]: d["text"] for d in docs})
    finally:
        ns.drop_namespace(driver)
        driver.close()
        os_s.drop_index(INDEX)
        ns.PREFIX = previous


def evaluate(stores, questions: Sequence[Dict], llm, modes: Dict = MODES) -> Dict[str, Dict[str, "ask.Result"]]:
    """{mode: {question id: Result}}; each mode runs as `ask_stories.py <its arguments>` would, and the LLM is used unless it says --no-llm."""
    out: Dict[str, Dict[str, ask.Result]] = {}
    for name, argv in modes.items():
        args = ask.make_parser().parse_args(argv)
        used = None if args.no_llm else llm
        opts = ask.options_from(args, used)
        out[name] = {q["id"]: ask.ask(q["question"], stores, opts, used) for q in questions}
    return out


def judge_rows(shown: Sequence[str], relevant: Sequence[str], acceptable: Sequence[str] = ()) -> Dict[str, float]:
    """Precision, recall, F1 and reciprocal rank of the stories shown, against the ones that should have been. Acceptable stories are left out
    of the count altogether. Showing nothing scores 0."""
    counted = [t for t in shown if t not in acceptable]
    hits = len(set(counted) & set(relevant))
    precision = hits / len(counted) if counted else 0.0
    recall = hits / len(relevant)
    first = next((i for i, t in enumerate(counted, start=1) if t in relevant), None)
    return {"precision": precision, "recall": recall, "f1": 2 * precision * recall / (precision + recall) if hits else 0.0,
            "mrr": 1 / first if first else 0.0}


def scores(results: Dict[str, Dict[str, "ask.Result"]], questions: Sequence[Dict]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """{mode: {database: {metric: mean over the questions}}}"""
    table: Dict[str, Dict[str, Dict[str, float]]] = {}
    for mode, by_question in results.items():
        table[mode] = {}
        for db in DBS:
            rows = [judge_rows([r["topic"] for r in by_question[q["id"]].rows[db]], q["relevant"], q["acceptable"]) for q in questions]
            table[mode][db] = {m: sum(r[m] for r in rows) / len(rows) for m in ("precision", "recall", "f1", "mrr")}
    return table


def mean(table: Dict[str, Dict[str, Dict[str, float]]], mode: str, metric: str) -> float:
    """A metric for a mode, averaged over both databases."""
    return sum(table[mode][db][metric] for db in DBS) / len(DBS)


def format_table(table: Dict[str, Dict[str, Dict[str, float]]]) -> str:
    lines = [f"{'':30s}{'precision':>11s}{'recall':>9s}{'F1':>7s}{'MRR':>7s}   (mean over the questions and both databases)"]
    for mode in table:
        lines.append(f"{mode:30s}" + "".join(f"{mean(table, mode, m):>{w}.2f}" for m, w in (("precision", 11), ("recall", 9), ("f1", 7), ("mrr", 7))))
    return "\n".join(lines)


# --------------------------------------------------------------------------- confusion matrix and the README block
def confusion(results: Dict[str, Dict[str, "ask.Result"]], questions: Sequence[Dict], stories: Sequence[str]) -> Dict[str, Dict[str, Dict[str, int]]]:
    """{mode: {database: {tp, fn, fp, tn}}} pooled over the questions. A (question, story) pair is a positive when the story is relevant
    and shown (tp), relevant and not shown (fn); not relevant and shown (fp), not relevant and not shown (tn). Acceptable stories are left out."""
    out: Dict[str, Dict[str, Dict[str, int]]] = {}
    for mode, by_question in results.items():
        out[mode] = {db: {"tp": 0, "fn": 0, "fp": 0, "tn": 0} for db in DBS}
        for q in questions:
            relevant, acceptable = set(q["relevant"]), set(q["acceptable"])
            for db in DBS:
                shown = {r["topic"] for r in by_question[q["id"]].rows[db]} - acceptable
                cell = out[mode][db]
                cell["tp"] += len(relevant & shown)
                cell["fn"] += len(relevant - shown)
                cell["fp"] += len(shown - relevant)
                cell["tn"] += len(set(stories) - acceptable - relevant - shown)
    return out


def _pooled(cells: Dict[str, Dict[str, int]]) -> Dict[str, int]:
    return {k: sum(cells[db][k] for db in DBS) for k in ("tp", "fn", "fp", "tn")}


def markdown_report(results: Dict[str, Dict[str, "ask.Result"]], questions: Sequence[Dict], stories: Sequence[str]) -> str:
    """The tables of the README's "Hard questions" section, from the recorded results: modes and the commands that run them, the scores, the
    confusion matrix of each mode and the outcome of each question."""
    table, conf, modes = scores(results, questions), confusion(results, questions, stories), list(results)
    runs, uses = len(questions) * len(DBS), {}
    for mode in modes:
        args = ask.make_parser().parse_args(MODES[mode])
        used = "none" if args.no_llm else " + ".join(n for n, on in (("analysis", not args.no_rewrite), ("scoring", not args.no_rerank)) if on) or "none"
        uses[mode] = used
    out = ["**The modes, and the command that runs each** (the tests build their options from exactly these arguments, with the default "
           "`--os-arm hybrid --neo-arm n_hyb --top 5`):", "", "| Mode | Run the app with | LLM steps |", "|---|---|---|"]
    out += [f"| {mode} | `{command(MODES[mode])}` | {uses[mode]} |" for mode in modes]
    out += ["", f"**Scores** (mean over the {len(questions)} questions and both databases; precision = how much of what is shown is right, "
            "recall = how much of what is right is shown, MRR = how high the first right story is):", "",
            "| Mode | precision | recall | F1 | MRR |", "|---|---|---|---|---|"]
    out += [f"| {mode} | " + " | ".join(f"{mean(table, mode, m):.2f}" for m in ("precision", "recall", "f1", "mrr")) + " |" for mode in modes]
    out += ["", f"**Confusion matrix of each mode**, pooled over {len(questions)} questions x 2 databases = {runs} runs "
            f"(every run asks, for each of the {len(stories)} stories: should it be shown, and was it?):", ""]
    for mode in modes:
        c = _pooled(conf[mode])
        out += [f"`{command(MODES[mode])}`: {mode}", "", "| | shown | not shown |", "|---|---|---|",
                f"| **should be shown** | {c['tp']} (true positive) | {c['fn']} (false negative) |",
                f"| **should not be shown** | {c['fp']} (false positive) | {c['tn']} (true negative) |", "",
                f"Pooled precision {c['tp'] / (c['tp'] + c['fp']):.2f}, recall {c['tp'] / (c['tp'] + c['fn']):.2f}: "
                f"{c['fp']} wrong stories shown for {c['tp']} right ones.", ""]
    out += ["**Per database** (same counts, split):", "", "| Mode | Database | TP | FN | FP | TN |", "|---|---|---|---|---|---|"]
    out += [f"| {mode} | {db} | {conf[mode][db]['tp']} | {conf[mode][db]['fn']} | {conf[mode][db]['fp']} | {conf[mode][db]['tn']} |" for mode in modes for db in DBS]
    out += ["", "**Question by question**: wrong stories shown (FP) and right stories missing (FN), added up over both databases; "
            "`exact` = every right story and nothing else:", "", "| Question | Kind | " + " | ".join(modes) + " |", "|---|---|" + "---|" * len(modes)]
    for q in questions:
        cells = []
        for mode in modes:
            fp = fn = 0
            for db in DBS:
                shown = {r["topic"] for r in results[mode][q["id"]].rows[db]} - set(q["acceptable"])
                fp += len(shown - set(q["relevant"]))
                fn += len(set(q["relevant"]) - shown)
            cells.append("exact" if not fp and not fn else f"FP {fp}, FN {fn}")
        out.append(f"| {q['id']} | {q['kind']} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def stories_of(docs: Sequence[Dict]) -> List[str]:
    return sorted({d["topic"] for d in docs})
