"""
Type a question and see what OpenSearch and Neo4j each return for it (top 5, ranked), plus one LLM answer written from
each database's own top 5, so you can feel how the two behave side by side.

    echo 'OPENROUTER_API_KEY=...' > .env       # git-ignored; or export it. Never printed
    python ask_stories.py                      # plain run: the question goes unchanged to both databases
    python ask_stories.py --no-llm             # retrieval only, no key needed
    python ask_stories.py --rewrite            # + the LLM turns the question into a keyword query first (same text to both)
    python ask_stories.py --rerank             # + the LLM re-orders each database's 10 candidates
    python ask_stories.py --judge              # + a third call says which answer is better supported (a hint, not a measurement)
    python ask_stories.py --provider ollama    # local model instead of OpenRouter (default model qwen3:8b)
    python ask_stories.py --debug              # + the exact requests sent to each database (OpenSearch JSON, Neo4j Cypher + parameters)

The corpus is the story files in stories/, loaded once into both databases by data_load.py (this script refuses to run if
either database no longer matches the files). Each database ranks versions (originals and rewrites); they are collapsed to one
row per story and the text shown, and given to the LLM, is always the story's original. Scores are each engine's own and cannot
be compared across the two databases.
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
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import corpus  # noqa: E402
import data_load  # noqa: E402
import neo4j_stories as ns  # noqa: E402
import opensearch_stories as os_s  # noqa: E402

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODELS = {"openrouter": "anthropic/claude-haiku-5.5", "ollama": "qwen3:8b"}
REQUEST_TIMEOUT = 120
ENV_FILE = Path(__file__).resolve().parent / ".env"   # git-ignored; holds OPENROUTER_API_KEY
POOL = 10      # stories each database hands to the optional re-ranker
SEARCH_SIZE = 30   # versions requested from each database before they are collapsed to stories

OS_ARMS = {"plain": "raw words", "stem": "stemmed keywords", "syn": "keywords + synonym list", "dense": "vectors",
           "hybrid": "keywords + synonyms and vectors, merged"}
NEO_ARMS = {"n_ft": "stemmed keywords", "n_vec": "vectors", "n_co": "word links learned from rewrites",
            "n_term": "nearest words by embedding", "n_hyb": "word links, nearest words and vectors, merged"}

SAMPLES = ["Is the company being hurt by import duties?", "toad resting at the edge of a stream", "did the firm get money back from customs authorities",
           "what were the night watchman's responsibilities?", "did the company suffer a cyberattack?", "something took off and went into space"]

ANSWER_SYSTEM = ("You answer questions using only the numbered passages you are given. Cite the passages you use like [1] or [2][3]. "
                 "If the passages do not answer the question, say so plainly instead of guessing. Keep it to 2-4 sentences.")
REWRITE_SYSTEM = ("Turn the user's question into a short search query of 6-12 keywords for a text search engine: keep the key terms and add close "
                  "synonyms and other ways the idea could be worded. Reply with the keywords only, separated by spaces, no explanation.")
RERANK_SYSTEM = ("You rank passages by how directly they answer the question. Reply with a JSON list of the passage numbers, best first, "
                 "for example [3,1,2]. Include every number once. No other text.")
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
    """Versions -> one row per story (best version, rewrites weighted below originals), best first; `text` is the story's original."""
    rows = os_s.story_level(hits)[:pool]
    return [dict(r, text=originals[r["topic"]], native=i) for i, r in enumerate(rows, start=1)]


def numbered(rows: Sequence[Dict]) -> str:
    return "\n".join(f"[{i}] {r['text']}" for i, r in enumerate(rows, start=1))


def answer_prompt(question: str, rows: Sequence[Dict]) -> str:
    return f"Passages:\n{numbered(rows)}\n\nQuestion: {question}"


def clean_rewrite(reply: str) -> str:
    return re.sub(r"\s+", " ", reply.replace('"', " ")).strip()


def parse_ranking(reply: str, n: int) -> Optional[List[int]]:
    """The order of passages 1..n from the model's reply (any numbers it left out go last, in their old order); None if there is no usable list."""
    match = re.search(r"\[[^\[\]]*\]", reply or "")
    if not match:
        return None
    try:
        numbers = json.loads(match.group(0))
    except ValueError:
        return None
    order: List[int] = []
    for x in numbers:
        if type(x) is int and 1 <= x <= n and x not in order:
            order.append(x)
    return order + [i for i in range(1, n + 1) if i not in order] if order else None


def rerank(question: str, rows: List[Dict], llm: LLM) -> Tuple[List[Dict], bool]:
    """Rows re-ordered by the LLM (each keeps its native rank in `native`); the native order and False when the reply is unusable."""
    order = parse_ranking(llm(RERANK_SYSTEM, f"Question: {question}\n\nPassages:\n{numbered(rows)}"), len(rows))
    return ([rows[i - 1] for i in order], True) if order else (rows, False)


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
def search_opensearch(stores: "Stores", query: str, arm: str, size: int = SEARCH_SIZE) -> List[Dict]:
    keywords = lambda field: os_s.bm25(stores.index, field, query, size)  # noqa: E731
    vectors = lambda: os_s.knn(stores.index, os_s.embed(stores.embedder, [query])[0], size)  # noqa: E731
    if arm in ("plain", "stem", "syn"):
        return keywords({"plain": "text", "stem": "text.stem", "syn": "text.syn"}[arm])
    if arm == "dense":
        return vectors()
    return os_s.rrf([keywords("text.syn"), vectors()])                     # hybrid


def search_neo4j(stores: "Stores", query: str, arm: str, size: int = SEARCH_SIZE) -> List[Dict]:
    graph = stores.graph
    if arm in ("n_ft", "n_vec", "n_co", "n_term"):
        return {"n_ft": graph.ft, "n_vec": graph.vec, "n_co": graph.co, "n_term": graph.term}[arm](query, size)
    return os_s.rrf([graph.co(query, size), graph.term(query, size), graph.vec(query, size)])   # n_hyb


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
    lines = []
    for i, (kind, text, payload) in enumerate(entries, start=1):
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
    rewrite: bool = False
    rerank: bool = False
    judge: bool = False
    debug: bool = False


@dataclass
class Stores:
    index: str
    graph: "ns.StoryGraph"
    embedder: object
    originals: Dict[str, str]                   # story id -> the original's text, what is shown and sent to the LLM


@dataclass
class Result:
    question: str
    query: str                                  # the text the databases were searched with
    note: str = ""
    rows: Dict[str, List[Dict]] = field(default_factory=dict)       # database -> top rows
    ms: Dict[str, float] = field(default_factory=dict)              # database -> search time
    trace: Dict[str, List[Tuple[str, str, object]]] = field(default_factory=dict)   # database -> requests sent (debug only)
    notes: Dict[str, List[str]] = field(default_factory=dict)       # database -> extra debug lines
    reranked: Dict[str, bool] = field(default_factory=dict)         # database -> the LLM order was used
    answers: Dict[str, str] = field(default_factory=dict)
    answer_s: Dict[str, float] = field(default_factory=dict)
    judge: str = ""
    judge_order: Tuple[str, ...] = ()           # which database the judge saw as A, then B


def _timed(fn: Callable, *args) -> Tuple[object, float]:
    t0 = time.perf_counter()
    out = fn(*args)
    return out, time.perf_counter() - t0


def ask(question: str, stores: Stores, opts: Options, llm: Optional[LLM] = None, rng=random) -> Result:
    """Search both databases and, when an LLM is given, answer from each one's top rows. The optional steps need the LLM."""
    query, note = question, ""
    if opts.rewrite and llm:
        reply = llm(REWRITE_SYSTEM, question)
        if failed(reply) or not clean_rewrite(reply):
            note = f"rewrite failed, the question was sent as typed ({reply})"
        else:
            query = clean_rewrite(reply)
    res = Result(question, query, note)

    searches = {"OpenSearch": lambda: search_opensearch(stores, query, opts.os_arm),
                "Neo4j": lambda: search_neo4j(stores, query, opts.neo_arm)}
    pool: Dict[str, List[Dict]] = {}
    for name, search in searches.items():
        sink: List[Tuple[str, str, object]] = []
        with capture(sink) if opts.debug else nullcontext():
            hits, seconds = _timed(search)
        pool[name], res.ms[name] = collapse(hits, stores.originals), seconds * 1000
        if opts.debug:
            res.trace[name] = sink
    if opts.debug:
        res.notes["OpenSearch"] = analysis_note(stores, query, opts.os_arm)       # an extra request, made after the timing

    if opts.rerank and llm:
        with ThreadPoolExecutor(2) as ex:
            futures = {name: ex.submit(rerank, question, rows, llm) for name, rows in pool.items()}
        for name, future in futures.items():
            pool[name], res.reranked[name] = future.result()
    res.rows = {name: rows[:opts.top] for name, rows in pool.items()}

    if llm:
        with ThreadPoolExecutor(2) as ex:
            futures = {name: ex.submit(_timed, llm, ANSWER_SYSTEM, answer_prompt(question, rows)) for name, rows in res.rows.items()}
        for name, future in futures.items():
            (res.answers[name], res.answer_s[name]) = future.result()
        if opts.judge and not any(failed(a) for a in res.answers.values()):
            res.judge_order = ("OpenSearch", "Neo4j") if rng.random() < 0.5 else ("Neo4j", "OpenSearch")   # the order must not always be the same
            res.judge = llm(JUDGE_SYSTEM, judge_prompt(question, [(res.rows[n], res.answers[n]) for n in res.judge_order]))
    return res


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def render(res: Result, opts: Options) -> str:
    out = [""]
    if res.query != res.question:
        out.append(f'sent to both databases: "{res.query}"   (rewritten by the LLM)')
    if res.note:
        out.append(f"note: {res.note}")
    for name, arm, notes in (("OpenSearch", opts.os_arm, OS_ARMS), ("Neo4j", opts.neo_arm, NEO_ARMS)):
        reordered = " - re-ordered by the LLM" if res.reranked.get(name) else (" - LLM re-order failed, native order" if name in res.reranked else "")
        out.append(f"{name}  [{arm}: {notes[arm]}]  {res.ms[name]:.0f} ms{reordered}")
        if opts.debug:
            out.append(f"  debug: {len(res.trace.get(name, []))} request(s) sent to {name} (long lists abbreviated)")
            out += format_trace(res.trace.get(name, []))
            for note in res.notes.get(name, []):
                out += textwrap.wrap(note, 112, initial_indent="    ", subsequent_indent="      ", break_long_words=False)
        for rank, r in enumerate(res.rows[name], start=1):
            was = f"  (was #{r['native']})" if res.reranked.get(name) and r["native"] != rank else ""
            out.append(f"  {rank}  {r['topic']:14s} {r['score']:.4f}  via {r['via']:5s} {textwrap.shorten(r['text'], 92, placeholder='...')}{was}")
        if not res.rows[name]:
            out.append("  (nothing matched)")
    both = overlap(res.rows["OpenSearch"], res.rows["Neo4j"])
    out.append(f"agreement: same #1: {'yes' if both['same_first'] else 'no'} | in both: {', '.join(both['both']) or '-'} | "
               f"only OpenSearch: {', '.join(both['only_first']) or '-'} | only Neo4j: {', '.join(both['only_second']) or '-'}")
    out.append("(scores are each engine's own and cannot be compared across the databases)")
    for name in res.answers:
        out += ["", f"answer from {name}'s top {len(res.rows[name])}  ({res.answer_s[name]:.1f} s):", textwrap.indent(textwrap.fill(res.answers[name], 100), "  ")]
    if res.judge:
        out += ["", f"judge (saw {res.judge_order[0]} as A, {res.judge_order[1]} as B; a hint, not a measurement):", textwrap.indent(textwrap.fill(res.judge, 100), "  ")]
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--provider", choices=sorted(DEFAULT_MODELS), default="openrouter")
    parser.add_argument("--model", help="default: anthropic/claude-haiku-5.5 (openrouter) or qwen3:8b (ollama)")
    parser.add_argument("--ollama-url", default=os.environ.get("OLLAMA_URL", "http://localhost:11434"))
    parser.add_argument("--no-llm", action="store_true", help="retrieval only")
    parser.add_argument("--rewrite", action="store_true", help="LLM turns the question into a keyword query before searching")
    parser.add_argument("--rerank", action="store_true", help="LLM re-orders each database's candidates")
    parser.add_argument("--judge", action="store_true", help="LLM says which answer is better supported")
    parser.add_argument("--os-arm", choices=list(OS_ARMS), default="hybrid")
    parser.add_argument("--neo-arm", choices=list(NEO_ARMS), default="n_hyb")
    parser.add_argument("--top", type=int, default=5, help=f"rows shown per database (1-{POOL})")
    parser.add_argument("--debug", action="store_true", help="print the exact requests sent to each database")
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
    if llm is None and (args.rewrite or args.rerank or args.judge):
        print("--rewrite / --rerank / --judge need an LLM and are ignored in this run")

    try:
        docs = corpus.load_corpus()
    except corpus.CorpusError as e:
        sys.exit(f"stories folder problem: {e}")
    if not os_s.reachable():
        sys.exit(f"OpenSearch not reachable at {os_s.URL} (docker start opensearch-poc)")
    if not ns.reachable():
        sys.exit(f"Neo4j not reachable at {ns.URI} (docker start neo4j-poc)")
    embedder = os_s.load_embedder()
    if embedder is None:
        sys.exit("sentence-transformers is not installed: pip install -r requirements.txt")

    index = data_load.INDEX
    driver = ns.connect()
    try:
        found = data_load.problems(docs, index, driver)
        if found:
            sys.exit("The databases do not hold the story files:\n  " + "\n  ".join(found) + "\nLoad them with: python data_load.py --reload")
        stores = Stores(index, ns.StoryGraph.attach(driver, embedder), embedder, corpus.originals(docs))
        opts = Options(args.os_arm, args.neo_arm, args.top, args.rewrite and llm is not None, args.rerank and llm is not None,
                       args.judge and llm is not None, args.debug)
        ask("warm up", stores, Options(args.os_arm, args.neo_arm), None)   # the first search pays for model and connection start-up
        steps = [name for name, on in (("rewrite", opts.rewrite), ("rerank", opts.rerank), ("judge", opts.judge)) if on]
        print(f"\n{len(docs)} texts / {len(stores.originals)} stories (identical files) loaded in both databases. "
              f"LLM: {f'{args.provider} {model}' if llm else 'off'}. Extra steps: {', '.join(steps) or 'none (plain run)'}{'; debug on' if opts.debug else ''}.")
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
