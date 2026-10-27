"""
The story corpus on disk, the single source both databases are loaded from (see data_load.py).

    stories/<story_id>/original.txt        the original; this file name is what marks it as the original
    stories/<story_id>/rewrite-1.txt       a rewrite of it in different words ... rewrite-5.txt

Plain UTF-8 text, paragraphs separated by blank lines. A story id is the folder name (lowercase letters, digits, underscore).
Documents come back as {id, topic, version, kind, text, path} with version 'orig' or 'rw1'..'rwN' and kind 'original' or
'rewrite', the same shape the rest of this project already uses.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Dict, List, Optional

STORIES_DIR = Path(__file__).resolve().parent / "stories"
ORIGINAL_FILE = "original.txt"
_REWRITE = re.compile(r"^rewrite-(\d+)\.txt$")
_STORY_ID = re.compile(r"^[a-z0-9_]+$")


class CorpusError(ValueError):
    """The stories folder does not follow the layout above."""


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
        numbered = []
        for name, path in files.items():
            match = _REWRITE.match(name)
            if not match:
                raise CorpusError(f"{story.name}/{name}: expected {ORIGINAL_FILE} or rewrite-<number>.txt")
            numbered.append((int(match.group(1)), path))
        if not numbered:
            raise CorpusError(f"{story.name}: no rewrites")
        versions += [(f"rw{n}", path) for n, path in sorted(numbered)]
        for version, path in versions:
            text = path.read_text(encoding="utf-8").strip()
            if not text:
                raise CorpusError(f"{story.name}/{path.name} is empty")
            docs.append({"id": f"{story.name}:{version}", "topic": story.name, "version": version,
                         "kind": "original" if version == "orig" else "rewrite", "text": text, "path": str(path)})
    if not docs:
        raise CorpusError(f"no story folders in {root}")
    return docs


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
