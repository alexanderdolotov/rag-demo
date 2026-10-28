"""
The story corpus on disk, the single source both databases are loaded from (see data_load.py).

    stories/<story_id>/original.txt        the original; this file name is what marks it as the original
    stories/<story_id>/rewrite-1.txt       a rewrite of it in different words ... rewrite-5.txt
    stories/queries.json                   the test queries and which stories answer them (see load_queries)

Plain UTF-8 text, paragraphs separated by blank lines. A story id is the folder name (lowercase letters, digits, underscore).
Documents come back as {id, topic, version, kind, text, path} with version 'orig' or 'rw1'..'rwN' and kind 'original' or
'rewrite', the same shape the rest of this project already uses.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

STORIES_DIR = Path(__file__).resolve().parent / "stories"
ORIGINAL_FILE = "original.txt"
QUERIES_FILE = "queries.json"
_REWRITE = re.compile(r"^rewrite-(\d+)\.txt$")
_STORY_ID = re.compile(r"^[a-z0-9_]+$")


class CorpusError(ValueError):
    """The stories folder does not follow the layout above."""


class QuerySet(NamedTuple):
    name: str
    description: str
    queries: List[Tuple[str, str, Tuple[str, ...]]]   # (query id, text, story ids that count as a correct answer)


def load_corpus(root: Path = STORIES_DIR) -> List[Dict]:
    """Every version of every story, ordered by story id, the original first. Raises CorpusError on anything unexpected."""
    root = Path(root)
    if not root.is_dir():
        raise CorpusError(f"{root} is not a folder")
    docs: List[Dict] = []
    for story in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith((".", "_"))):
        if not _STORY_ID.match(story.name):
            raise CorpusError(f"{story.name}: a story folder name may only use lowercase letters, digits and underscores")
        files = {p.name: p for p in story.iterdir() if p.is_file() and not p.name.startswith(".")}
        if ORIGINAL_FILE not in files:
            raise CorpusError(f"{story.name}: there is no {ORIGINAL_FILE} (the original must be marked by that file name)")
        versions = [("orig", files.pop(ORIGINAL_FILE))]
        numbered: Dict[int, Path] = {}
        for name, path in sorted(files.items()):
            match = _REWRITE.match(name)
            if not match:
                raise CorpusError(f"{story.name}/{name}: expected {ORIGINAL_FILE} or rewrite-<number>.txt")
            number = int(match.group(1))
            if number in numbered:   # rewrite-1.txt and rewrite-01.txt would both become version rw1
                raise CorpusError(f"{story.name}: {numbered[number].name} and {name} are both rewrite {number}")
            numbered[number] = path
        if not numbered:
            raise CorpusError(f"{story.name}: no rewrites")
        versions += [(f"rw{n}", path) for n, path in sorted(numbered.items())]
        for version, path in versions:
            text = path.read_text(encoding="utf-8").strip()
            if not text:
                raise CorpusError(f"{story.name}/{path.name} is empty")
            docs.append({"id": f"{story.name}:{version}", "topic": story.name, "version": version,
                         "kind": "original" if version == "orig" else "rewrite", "text": text, "path": str(path)})
    if not docs:
        raise CorpusError(f"no story folders in {root}")
    return docs


def load_queries(root: Path = STORIES_DIR, docs: Optional[List[Dict]] = None) -> List[QuerySet]:
    """The query sets in queries.json, in file order. With `docs`, also refuses a query that expects a story the corpus does not have
    (such a typo would otherwise score as "not found" forever). Raises CorpusError on anything unexpected."""
    path = Path(root) / QUERIES_FILE
    if not path.is_file():
        raise CorpusError(f"{path} does not exist (the test queries live next to the stories)")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as e:
        raise CorpusError(f"{path.name} is not valid JSON: {e}") from e
    sets_raw = raw.get("sets") if isinstance(raw, dict) else None
    if not isinstance(sets_raw, list) or not sets_raw:
        raise CorpusError(f'{path.name}: expected {{"sets": [ ... ]}} with at least one set')
    known = {d["topic"] for d in docs} if docs is not None else None
    seen: set = set()
    sets: List[QuerySet] = []
    for s in sets_raw:
        name = s.get("name") if isinstance(s, dict) else None
        if not isinstance(name, str) or not name:
            raise CorpusError(f"{path.name}: every set needs a \"name\"")
        queries = s.get("queries")
        if not isinstance(s.get("description"), str) or not isinstance(queries, list) or not queries:
            raise CorpusError(f"{path.name}: {name}: needs a \"description\" and a non-empty \"queries\" list")
        parsed = []
        for q in queries:
            qid, text, expect = (q.get(k) if isinstance(q, dict) else None for k in ("id", "text", "expect"))
            if not (isinstance(qid, str) and qid and isinstance(text, str) and text.strip()
                    and isinstance(expect, list) and expect and all(isinstance(e, str) for e in expect)):
                raise CorpusError(f"{path.name}: {name}: every query needs a string \"id\", a string \"text\" and a non-empty \"expect\" list of story ids: {q!r}")
            if qid in seen:
                raise CorpusError(f"{path.name}: query id {qid!r} is used twice")
            seen.add(qid)
            if known is not None and not set(expect) <= known:
                raise CorpusError(f"{path.name}: {qid} expects {sorted(set(expect) - known)}, which are not stories in the corpus")
            parsed.append((qid, text, tuple(expect)))
        sets.append(QuerySet(name, s["description"], parsed))
    return sets


def corpus_hash(docs: List[Dict]) -> str:
    """A short fingerprint of the ids and texts: equal hashes mean two databases hold the same corpus."""
    digest = hashlib.sha256()
    for d in sorted(docs, key=lambda d: d["id"]):
        digest.update(f"{d['id']}\0{d['text']}\0".encode("utf-8"))
    return digest.hexdigest()[:16]


def originals(docs: List[Dict]) -> Dict[str, str]:
    """{story id: the original's text}."""
    return {d["topic"]: d["text"] for d in docs if d["kind"] == "original"}


def stats(docs: List[Dict]) -> Dict[str, object]:
    words = [len(d["text"].split()) for d in docs]
    return {"stories": len({d["topic"] for d in docs}), "versions": len(docs), "min_words": min(words), "max_words": max(words),
            "total_words": sum(words), "hash": corpus_hash(docs)}


def over_window(docs: List[Dict], embedder) -> Optional[List[str]]:
    """Ids of versions longer than the embedding model reads (they would be silently cut); None without a model."""
    if embedder is None:
        return None
    limit = embedder.max_seq_length
    return [d["id"] for d in docs if len(embedder.tokenizer(d["text"], add_special_tokens=True, verbose=False)["input_ids"]) > limit]
