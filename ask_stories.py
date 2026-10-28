"""
Type a question and see what OpenSearch and Neo4j each return for it (top 5, ranked), plus one LLM answer written from
each database's own top 5, so you can feel how the two behave side by side.

    echo 'OPENROUTER_API_KEY=...' > .env       # git-ignored; or export it. Never printed
    python ask_stories.py                      # the LLM analyses the question (concepts by priority, conditions a passage must meet), the databases
                                               # are searched once per concept, the LLM scores every candidate 0-3 and each database's answer is
                                               # written from what scored well
    python ask_stories.py --no-rewrite         # ... without the analysis: the question is searched as typed
    python ask_stories.py --no-rerank          # ... without the scoring: the databases' own order
    python ask_stories.py --no-llm             # retrieval only, no key needed (one search with the question as typed)
    python ask_stories.py --judge              # + a last call says which answer is better supported (a hint, not a measurement)
    python ask_stories.py --provider ollama    # local model instead of OpenRouter (default model qwen3:8b)
    python ask_stories.py --debug              # + the exact requests sent to each database (OpenSearch JSON, Neo4j Cypher + parameters)

The corpus is the story files in stories/, loaded once into both databases by data_load.py (this script refuses to run if
either database no longer matches the files). Each database ranks versions (originals and rewrites); they are collapsed to one
row per story and the text given to the LLM is always the story's original. Beside each result you see the sentence where the words
of your question occur (cut around the first of them when it is long), or the start of the story when none does, as with a match made
by the vectors alone. Scores are each engine's own and cannot be compared across the two databases.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import corpus  # noqa: E402
import data_load  # noqa: E402
import neo4j_stories as ns  # noqa: E402
import opensearch_stories as os_s  # noqa: E402
import pipeline  # noqa: E402
import retrieval  # noqa: E402
from intent import ANALYZE_SYSTEM, Intent, flat_intent, parse_intent  # noqa: E402
from retrieval import Query, Searchable  # noqa: E402

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODELS = {"openrouter": "anthropic/claude-haiku-5.5", "ollama": "qwen3:8b"}
REQUEST_TIMEOUT = 120
ENV_FILE = Path(__file__).resolve().parent / ".env"   # git-ignored; holds OPENROUTER_API_KEY
POOL = 10      # stories each database hands on from one search (the most it can show)
DBS = ("OpenSearch", "Neo4j")
SEARCH_SIZE = 30   # versions requested from each database before they are collapsed to stories

OS_ARMS = {"plain": "raw words", "stem": "stemmed keywords", "syn": "keywords + synonym list", "dense": "vectors",
           "hybrid": "keywords + synonyms and vectors, merged"}
NEO_ARMS = {"n_ft": "stemmed keywords", "n_vec": "vectors", "n_co": "word links learned from rewrites",
            "n_term": "nearest words by embedding", "n_hyb": "word links, nearest words and vectors, merged"}

SAMPLES = ["Is the company being hurt by import duties?", "toad resting at the edge of a stream", "did the firm get money back from customs authorities",
           "what were the night watchman's responsibilities?", "did the company suffer a cyberattack?", "something took off and went into space",
           "who bakes bread, and what do import duties do to margins?"]

ANSWER_SYSTEM = ("You answer questions using only the numbered passages you are given. Cite the passages you use like [1] or [2][3]. "
                 "If the passages do not answer the question, say so plainly instead of guessing. Keep it to 2-4 sentences.")
JUDGE_SYSTEM = ("You compare two answers to the same question. Each answer was written only from its own numbered passages. Say which answer is "
                "better supported by its passages and more useful: A, B or tie, then give one or two sentences of reason.")

LLM = Callable[[str, str], str]   # (system prompt, user prompt) -> reply; failures come back as "[LLM error: ...]" text


# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------
def chat(provider: str, model: str, system: str, user: str, url: str = "http://localhost:11434",
         key: Optional[str] = None, post=requests.post) -> str:
    """One chat completion at temperature 0. Never raises: a failure comes back as '[LLM error: ...]' so the retrieval output still prints."""
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    try:
        if provider == "openrouter":
            key = os.environ.get("OPENROUTER_API_KEY", "") if key is None else key
            if not key:
                return "[LLM error: OPENROUTER_API_KEY is not set]"
            resp = post(OPENROUTER_URL, timeout=REQUEST_TIMEOUT,
                        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                                 "X-Title": "rag_demo"},
                        json={"model": model, "temperature": 0, "messages": messages})
        else:
            resp = post(f"{url}/api/chat", timeout=REQUEST_TIMEOUT,
                        json={"model": model, "stream": False, "options": {"temperature": 0, "num_ctx": 8192}, "messages": messages})
        if resp.status_code >= 400:
            return f"[LLM error: HTTP {resp.status_code} {resp.text[:200]}]"
        body = resp.json()
        text = body["choices"][0]["message"]["content"] if provider == "openrouter" else body["message"]["content"]
        return re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip() or "[LLM error: empty reply]"
    except Exception as e:  # noqa: BLE001 - the caller only needs the message
        return f"[LLM error: {type(e).__name__}: {e}]"


def load_env(path: Path = ENV_FILE) -> None:
    """Put the KEY=VALUE lines of a .env file into the environment; variables that are already set win. A missing file is fine."""
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return
    for line in lines:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            name, value = line.split("=", 1)
            os.environ.setdefault(name.strip().removeprefix("export ").strip(), value.strip().strip("\"'"))


def failed(reply: str) -> bool:
    return reply.startswith("[LLM error")


# ---------------------------------------------------------------------------
# Pure helpers (no services)
# ---------------------------------------------------------------------------
def collapse(hits: Sequence[Dict], originals: Dict[str, str], pool: int = POOL) -> List[Dict]:
    """Versions -> one row per story (its best version's score as it is, see retrieval.story_level), best first; `text` is the story's original."""
    rows = retrieval.story_level(hits)[:pool]
    return [dict(r, text=originals[r["topic"]], native=i) for i, r in enumerate(rows, start=1)]


SNIPPET_WIDTH = 92   # characters of story text shown beside each result
# Words of a question that say nothing about the topic. The engines' own stop list is short ("what" and "were" are not on it), but a sentence
# that merely contains "what" is not the evidence for a match, so these are ignored when looking for the sentence to show.
QUESTION_FILLER = frozenset("""what whats who whom whose which when where why how do does did done doing have has had having been being were am
                               would could should about from i me my we us our you your he him his she her them its also than so just very""".split())
_SENTENCE_END = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"\u201c(])')


def sentence_spans(text: str) -> List[Tuple[int, int]]:
    """(start, end) of each sentence of `text`."""
    spans, start = [], 0
    for m in _SENTENCE_END.finditer(text):
        spans.append((start, m.start()))
        start = m.end()
    return spans + [(start, len(text))]


def excerpt(sentence: str, hit: Tuple[int, int], width: int = SNIPPET_WIDTH) -> str:
    """The whole `sentence` if it fits in `width` characters, else the stretch of it around `hit` (start, end of the matched word, or of the
    stretch holding several): centred on it where the sentence allows, cut between words but never through the hit, and marked with ...
    where text was left out."""
    if len(sentence) <= width:
        return sentence
    room = width - 6                                              # space for a "..." on each side
    start = max(0, min((hit[0] + hit[1]) // 2 - room // 2, len(sentence) - room))
    end = min(len(sentence), start + room)
    while 0 < start < hit[0] and sentence[start - 1] != " ":      # do not start or stop in the middle of a word
        start += 1
    while hit[1] < end < len(sentence) and sentence[end] != " ":
        end -= 1
    return ("..." if start > 0 else "") + sentence[start:end].strip() + ("..." if end < len(sentence) else "")


def evidence(text: str, stems: Set[str], width: int = SNIPPET_WIDTH) -> Optional[str]:
    """The sentence of `text` that holds the most different words of the query (the first such sentence on a tie), cut to `width` around
    them; None when no query word occurs in `text`. A word counts when it has the same stem, so "duties" finds "duty". When the sentence is
    cut, the window is centred on the stretch from the first to the last of the different query words if that fits, else on the first."""
    text = " ".join(text.split())
    best: Optional[Tuple[int, str, Tuple[int, int]]] = None
    for start, end in sentence_spans(text):
        sentence = text[start:end]
        first: Dict[str, Tuple[int, int]] = {}
        for stem, _, s, e in ns.token_spans(sentence):
            if stem in stems:
                first.setdefault(stem, (s, e))
        if first and (best is None or len(first) > best[0]):
            hits = sorted(first.values())
            stretch = (hits[0][0], max(e for _, e in hits))
            best = (len(first), sentence, stretch if stretch[1] - stretch[0] <= width - 6 else hits[0])
    return excerpt(best[1], best[2], width) if best else None


def add_snippets(rows: List[Dict], query: str, stores: "Stores") -> None:
    """Give each row a `snippet`: the sentence where the query's words occur (leaving out QUESTION_FILLER), from the version that matched
    (`via`), else from the original; `snippet_from` names that version. A row where no such word occurs (a match made by the vectors alone,
    or by filler words) gets none."""
    stems = {stem for stem, word in ns.tokens(query) if word not in QUESTION_FILLER}
    for r in rows:
        for version in dict.fromkeys([r["via"], "orig"]):
            text = stores.texts.get(f"{r['topic']}:{version}") or (stores.originals[r["topic"]] if version == "orig" else "")
            found = evidence(text, stems) if text and stems else None
            if found:
                r["snippet"], r["snippet_from"] = found, version
                break


def shown(r: Dict) -> str:
    """The text beside a result: the sentence where the query's words occur (tagged with its version when that is not the one that matched),
    or else the start of the story."""
    if r.get("snippet"):
        return (f"[{r['snippet_from']}] " if r["snippet_from"] != r["via"] else "") + r["snippet"]
    return textwrap.shorten(r["text"], SNIPPET_WIDTH, placeholder="...")


def numbered(rows: Sequence[Dict]) -> str:
    return "\n".join(f"[{i}] {r['text']}" for i, r in enumerate(rows, start=1))


def answer_prompt(question: str, rows: Sequence[Dict], constraints: Sequence[str] = ()) -> str:
    asked = (f"\n\nThe question also asks for: {'; '.join(constraints)}. If the passages do not show whether that holds, say so."
             if constraints else "")
    return f"Passages:\n{numbered(rows)}{asked}\n\nQuestion: {question}"


def overlap(first: Sequence[Dict], second: Sequence[Dict]) -> Dict[str, object]:
    a, b = [r["topic"] for r in first], [r["topic"] for r in second]
    return {"same_first": bool(a) and a[:1] == b[:1], "both": [t for t in a if t in b],
            "only_first": [t for t in a if t not in b], "only_second": [t for t in b if t not in a]}


def judge_prompt(question: str, blocks: Sequence[Tuple[Sequence[Dict], str]]) -> str:
    parts = [f"Question: {question}"]
    for label, (rows, answer) in zip("AB", blocks):
        parts.append(f"Passages for answer {label}:\n{numbered(rows)}\n\nAnswer {label}: {answer}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# What each database is asked (only the searches the chosen method needs)
# ---------------------------------------------------------------------------
def search_opensearch(stores: "Stores", query: Searchable, arm: str, size: int = SEARCH_SIZE) -> List[Dict]:
    """Only the requests this one method needs (the other arms are not run)."""
    return os_s.run_arms(stores.index, query, stores.embedder, size=size, only=[arm])[arm]


def search_neo4j(stores: "Stores", query: Searchable, arm: str, size: int = SEARCH_SIZE) -> List[Dict]:
    return stores.graph.arms(query, size, only=[arm])[arm]


@contextmanager
def capture(sink: List[Tuple[str, str, object]]):
    """Record every request the two libraries send while the block runs, as (kind, text, payload): kind 'http' (path, JSON body) or 'cypher'."""
    os_s.TRACE = lambda method, path, body: sink.append(("http", f"{method} {path}", body))
    ns.TRACE = lambda cypher, params: sink.append(("cypher", cypher, params))
    try:
        yield
    finally:
        os_s.TRACE = ns.TRACE = None


def shorten(value, max_items: int = 30):
    """A copy that prints sensibly: a long list of floats (an embedding) becomes a short note, other long lists are cut."""
    if isinstance(value, dict):
        return {k: shorten(v, max_items) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        if len(value) > 8 and all(isinstance(x, float) for x in value):
            return f"<{len(value)} floats: {', '.join(f'{x:.4f}' for x in value[:4])}, ...>"
        if len(value) > max_items:
            return [shorten(v, max_items) for v in value[:max_items]] + [f"... {len(value) - max_items} more"]
        return [shorten(v, max_items) for v in value]
    return value


def format_trace(entries: Sequence[Tuple[str, str, object]], indent: str = "    ") -> List[str]:
    lines, i = [], 0
    for kind, text, payload in entries:
        if kind == "step":                                        # a header for the requests of one step, not a request itself
            lines.append(f"{indent}-- {text} --")
            continue
        i += 1
        if kind == "http":
            lines.append(f"{indent}{i}. {text}")
            body = json.dumps(shorten(payload)) if payload is not None else ""
        else:
            lines.append(f"{indent}{i}. Cypher")
            for line in textwrap.dedent(text).strip().splitlines():
                if line.strip():
                    wrapped = textwrap.wrap(line, 104, subsequent_indent="    ", break_long_words=False, break_on_hyphens=False)
                    lines += [f"{indent}   | {part}" for part in wrapped]
            body = "parameters: " + json.dumps(shorten(payload))
        lines += textwrap.wrap(body, 112, initial_indent=indent + "   ", subsequent_indent=indent + "   ", break_long_words=False, break_on_hyphens=False)
    return lines


def analysis_note(stores: "Stores", query: str, arm: str) -> List[str]:
    """How the index turns the query into search terms (stop words dropped, stemmed, synonyms added), for the keyword methods."""
    analyzer = {"stem": "story_stem", "syn": "story_syn", "hybrid": "story_syn"}.get(arm)
    return [f'the index analyses "{query}" with {analyzer} as: {os_s.analyze(stores.index, analyzer, query)}'] if analyzer else []


# ---------------------------------------------------------------------------
# One question
# ---------------------------------------------------------------------------
@dataclass
class Options:
    os_arm: str = "hybrid"
    neo_arm: str = "n_hyb"
    top: int = 5
    rewrite: bool = True       # the LLM analyses the question (concepts by priority, conditions) and the databases are searched with that
    rerank: bool = True        # the LLM scores the candidates 0-3 per concept and the weak ones are dropped
    judge: bool = False
    debug: bool = False        # (rewrite and rerank do nothing without an LLM)


@dataclass
class Stores:
    index: str
    graph: "ns.StoryGraph"
    embedder: object
    originals: Dict[str, str]                   # story id -> the original's text, what is sent to the LLM
    texts: Dict[str, str] = field(default_factory=dict)   # "story:version" -> that version's text, where the matching sentence is looked for


@dataclass
class Result:
    question: str
    note: str = ""
    intent: Optional[Intent] = None             # how the LLM read the question, when it was asked to and the reply could be read
    rows: Dict[str, List[Dict]] = field(default_factory=dict)       # database -> rows shown
    ms: Dict[str, float] = field(default_factory=dict)              # database -> time spent searching
    trace: Dict[str, List[Tuple[str, str, object]]] = field(default_factory=dict)   # database -> requests sent (debug only)
    notes: Dict[str, List[str]] = field(default_factory=dict)       # database -> extra debug lines
    judged: Dict[str, bool] = field(default_factory=dict)           # database -> the LLM's scores were used (False: they could not be read)
    answers: Dict[str, str] = field(default_factory=dict)
    answer_s: Dict[str, float] = field(default_factory=dict)
    judge: str = ""
    judge_order: Tuple[str, ...] = ()           # which database the judge saw as A, then B


def _timed(fn: Callable, *args) -> Tuple[object, float]:
    t0 = time.perf_counter()
    out = fn(*args)
    return out, time.perf_counter() - t0


def query_label(query: Query) -> str:
    return " | ".join(f"{tier}: {', '.join(words)}" for tier, words in query.tiers())


def _searcher(stores: Stores, opts: Options, res: Result) -> pipeline.Search:
    """search(database, query) -> one row per story, with the sentence to show; adds the time and, with --debug, the requests to `res`."""
    def search(db: str, query: Searchable) -> List[Dict]:
        fn, arm = (search_opensearch, opts.os_arm) if db == "OpenSearch" else (search_neo4j, opts.neo_arm)
        sink: List[Tuple[str, str, object]] = []
        with capture(sink) if opts.debug else nullcontext():
            hits, seconds = _timed(fn, stores, query, arm)
        rows = collapse(hits, stores.originals)
        add_snippets(rows, query.words(("core", "related")) if isinstance(query, Query) else query, stores)
        res.ms[db] = res.ms.get(db, 0.0) + seconds * 1000
        if opts.debug:
            res.trace.setdefault(db, []).extend(([("step", query_label(query), None)] if isinstance(query, Query) else []) + sink)
            if db == "OpenSearch":                                                        # an extra request, made after the timing
                res.notes.setdefault(db, []).extend(analysis_note(stores, query.words() if isinstance(query, Query) else query, opts.os_arm))
        return rows
    return search


def ask(question: str, stores: Stores, opts: Options, llm: Optional[LLM] = None, rng=random) -> Result:
    """Search both databases and, when an LLM is given, answer from each one's rows. With an LLM the question is first analysed (opts.rewrite:
    concepts by priority and conditions, see intent.py) and the candidates are scored afterwards (opts.rerank); see pipeline.py."""
    res = Result(question)
    search = _searcher(stores, opts, res)
    intent = flat_intent(question)
    if llm and opts.rewrite:
        reply = llm(ANALYZE_SYSTEM, question)
        if failed(reply):
            res.note = f"rewrite failed, the question was searched as typed ({reply})"
        else:
            got = parse_intent(reply, question)
            if got.understood:
                res.intent = intent = got
            else:
                res.note = "the analysis could not be read, the question was searched as typed"

    found = pipeline.search_all(question, intent, DBS, search)
    scores = None
    if llm and opts.rerank:
        scores = pipeline.score_pool(llm, question, intent, found)
        res.judged = {db: scores is not None for db in DBS}
        if scores is None:
            res.note = "; ".join(filter(None, [res.note, "the LLM's scores could not be read, so the databases' own order is shown"]))
    res.rows = {db: pipeline.rank(intent.concepts, found[db], scores, opts.top) for db in DBS}

    if llm:
        unverifiable = res.intent.unverifiable if res.intent else ()
        with ThreadPoolExecutor(2) as ex:   # a database that returned nothing gives the LLM nothing to answer from, so it is not asked
            futures = {name: ex.submit(_timed, llm, ANSWER_SYSTEM, answer_prompt(question, rows, unverifiable))
                       for name, rows in res.rows.items() if rows}
        for name, future in futures.items():
            (res.answers[name], res.answer_s[name]) = future.result()
        if opts.judge and len(res.answers) == len(res.rows) and not any(failed(a) for a in res.answers.values()):
            res.judge_order = ("OpenSearch", "Neo4j") if rng.random() < 0.5 else ("Neo4j", "OpenSearch")   # the order must not always be the same
            res.judge = llm(JUDGE_SYSTEM, judge_prompt(question, [(res.rows[n], res.answers[n]) for n in res.judge_order]))
    return res


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def describe_intent(res: Result) -> List[str]:
    intent = res.intent
    if intent is None:
        return []
    lines = []
    for i, c in enumerate(intent.concepts, start=1):
        words = f"core: {', '.join(c.core)}" + (f" | related: {', '.join(c.related)}" if c.related else "")
        head = "understood" if len(intent.concepts) == 1 else ("main concept" if i == 1 else f"concept {i}")
        lines.append(f"{head}: {c.name}   [{words}]")
    if intent.must:
        lines.append(f"a passage must meet: {'; '.join(intent.must)}")
    if intent.context:
        lines.append(f"narrowing words (weak): {', '.join(intent.context)}")
    if intent.unverifiable:
        lines.append(f"no story can show: {'; '.join(intent.unverifiable)}")
    return lines


def render(res: Result, opts: Options) -> str:
    out = [""]
    out += describe_intent(res)
    if res.note:
        out.append(f"note: {res.note}")
    for name, arm, notes in (("OpenSearch", opts.os_arm, OS_ARMS), ("Neo4j", opts.neo_arm, NEO_ARMS)):
        judged = " - judged by the LLM" if res.judged.get(name) else (" - LLM scores unusable, native order" if name in res.judged else "")
        out.append(f"{name}  [{arm}: {notes[arm]}]  {res.ms[name]:.0f} ms{judged}")
        if opts.debug:
            out.append(f"  debug: {sum(1 for kind, _, _ in res.trace.get(name, []) if kind != 'step')} request(s) sent to {name} (long lists abbreviated)")
            out += format_trace(res.trace.get(name, []))
            for note in res.notes.get(name, []):
                out += textwrap.wrap(note, 112, initial_indent="    ", subsequent_indent="      ", break_long_words=False)
        for rank, r in enumerate(res.rows[name], start=1):
            rel = f"[{r['rel']}/3] " if "rel" in r else ""
            tag = f"  [{r['concept']}]" if r.get("concept") else ""
            was = f"  (was #{r['native']})" if res.judged.get(name) and r["native"] != rank else ""
            out.append(f"  {rank}  {r['topic']:14s} {r['score']:.4f}  via {r['via']:5s} {rel}{shown(r)}{tag}{was}")
        if not res.rows[name]:
            out.append("  (nothing relevant found)" if res.judged.get(name) else "  (nothing matched)")
    both = overlap(res.rows["OpenSearch"], res.rows["Neo4j"])
    out.append(f"agreement: same #1: {'yes' if both['same_first'] else 'no'} | in both: {', '.join(both['both']) or '-'} | "
               f"only OpenSearch: {', '.join(both['only_first']) or '-'} | only Neo4j: {', '.join(both['only_second']) or '-'}")
    out.append("(scores are each engine's own and cannot be compared across the databases)")
    for name in res.answers:
        out += ["", f"answer from {name}'s top {len(res.rows[name])}  ({res.answer_s[name]:.1f} s):", textwrap.indent(textwrap.fill(res.answers[name], 100), "  ")]
    if res.judge:
        out += ["", f"judge (saw {res.judge_order[0]} as A, {res.judge_order[1]} as B; a hint, not a measurement):", textwrap.indent(textwrap.fill(res.judge, 100), "  ")]
    return "\n".join(out)


def make_parser() -> argparse.ArgumentParser:
    """The command line of this script (kept apart from main() so the tests can build the same Options from the same arguments)."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--provider", choices=sorted(DEFAULT_MODELS), default="openrouter")
    parser.add_argument("--model", help="default: anthropic/claude-haiku-5.5 (openrouter) or qwen3:8b (ollama)")
    parser.add_argument("--ollama-url", default=os.environ.get("OLLAMA_URL", "http://localhost:11434"))
    parser.add_argument("--no-llm", action="store_true", help="retrieval only")
    parser.add_argument("--no-rewrite", action="store_true", help="do not let the LLM analyse the question: search it as typed")
    parser.add_argument("--no-rerank", action="store_true", help="do not let the LLM score the candidates: keep each database's own order")
    for flag in ("--one-shot", "--rewrite", "--rerank"):   # the search is always one-shot, and both LLM steps are on by default
        parser.add_argument(flag, action="store_true", help="accepted and does nothing (the default)")
    parser.add_argument("--judge", action="store_true", help="LLM says which answer is better supported")
    parser.add_argument("--os-arm", choices=list(OS_ARMS), default="hybrid")
    parser.add_argument("--neo-arm", choices=list(NEO_ARMS), default="n_hyb")
    parser.add_argument("--top", type=int, default=5, help=f"rows shown per database (1-{POOL})")
    parser.add_argument("--debug", action="store_true", help="print the exact requests sent to each database")
    return parser


def options_from(args: argparse.Namespace, llm: Optional[LLM]) -> Options:
    """What the arguments mean for ask(): both LLM steps are on unless switched off; without an LLM they do nothing and --judge is dropped."""
    return Options(args.os_arm, args.neo_arm, args.top, not args.no_rewrite, not args.no_rerank, args.judge and llm is not None, args.debug)


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    load_env()
    if not 1 <= args.top <= POOL:
        parser.error(f"--top must be between 1 and {POOL}")

    llm: Optional[LLM] = None
    model = args.model or DEFAULT_MODELS[args.provider]
    if not args.no_llm:
        if args.provider == "openrouter" and not os.environ.get("OPENROUTER_API_KEY"):
            print(f"OPENROUTER_API_KEY is not set (put it in {ENV_FILE} or export it): showing retrieval only; or use --provider ollama")
        elif args.provider == "ollama" and not _ollama_up(args.ollama_url):
            print(f"Ollama is not answering at {args.ollama_url}: showing retrieval only")
        else:
            llm = lambda system, user: chat(args.provider, model, system, user, url=args.ollama_url)  # noqa: E731
    if llm is None and args.judge:
        print("--judge needs an LLM and is ignored in this run")

    try:
        docs = corpus.load_corpus()
    except corpus.CorpusError as e:
        sys.exit(f"stories folder problem: {e}")
    if not os_s.reachable():
        sys.exit(f"OpenSearch not reachable at {os_s.URL} (docker start opensearch-poc)")
    if not ns.reachable():
        sys.exit(f"Neo4j not reachable at {ns.URI} (docker start neo4j-poc)")
    embedder = retrieval.load_embedder()
    if embedder is None:
        sys.exit("sentence-transformers is not installed: pip install -r requirements.txt")

    index = data_load.INDEX
    driver = ns.connect()
    try:
        found = data_load.problems(docs, index, driver)
        if found:
            sys.exit("The databases do not hold the story files:\n  " + "\n  ".join(found) + "\nLoad them with: python data_load.py --reload")
        stores = Stores(index, ns.StoryGraph.attach(driver, embedder), embedder, corpus.originals(docs), {d["id"]: d["text"] for d in docs})
        opts = options_from(args, llm)
        ask("warm up", stores, Options(args.os_arm, args.neo_arm), None)   # the first search pays for model and connection start-up
        steps = [name for name, on in (("rewrite", opts.rewrite), ("rerank", opts.rerank), ("judge", opts.judge)) if on and llm]
        print(f"\n{len(docs)} texts / {len(stores.originals)} stories (identical files) loaded in both databases. "
              f"LLM: {f'{args.provider} {model}' if llm else 'off'}. LLM steps: {', '.join(steps) or 'none'}{'; debug on' if opts.debug else ''}.")
        print("Try:\n  " + "\n  ".join(SAMPLES) + "\n(/quit or Ctrl-D to leave)")
        while True:
            try:
                question = input("\nquestion> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if question in ("/quit", "/exit", "/q"):
                break
            if question:
                print(render(ask(question, stores, opts, llm), opts))
    finally:
        driver.close()


def _ollama_up(url: str) -> bool:
    try:
        return requests.get(f"{url}/api/tags", timeout=3).ok
    except requests.RequestException:
        return False


if __name__ == "__main__":
    main()
