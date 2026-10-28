"""
The LLM side of ask_stories.py: one call to one provider, the keys found in .env, and the fall-back from one key to the next.

    chat()            one completion from one provider: OpenRouter, Anthropic's own API (the official SDK) or a local Ollama.
                      Never raises: a failure comes back as "[LLM error: ...]" text, so the retrieval output still prints.
    keys_in_order()   the keys .env names, in the order the file lists them
    candidates()      the keys that are filled in, in that order, each with its provider and model
    Failover          an LLM (system, user) -> reply over several keys: the first one that works; a key that is rejected, rate
                      limited, erroring or unreachable is given up for the session and the next one answers instead
    connect()         the Failover for a command line, after checking the keys in order with one tiny request each

The rule for the keys: whichever is listed first in .env, is filled in and works. Empty values are skipped, a key the provider rejects
or cannot reach is skipped, and what is left is tried in the order of the file.
"""
from __future__ import annotations

import os
import re
import threading
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import requests

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODELS = {"openrouter": "anthropic/claude-haiku-5.5", "anthropic": "claude-haiku-5-5", "ollama": "qwen3:8b"}
KEY_PROVIDERS = {"OPENROUTER_API_KEY": "openrouter", "ANTHROPIC_API_KEY": "anthropic"}     # the names .env may hold a key under
REQUEST_TIMEOUT = 120
MAX_TOKENS = 16000          # Anthropic: thinking and reply together; the model stops early, this is only the ceiling
ENV_FILE = Path(__file__).resolve().parent / ".env"   # git-ignored; holds the keys

PING_SYSTEM = "Reply with the single word: ok"
# What an error reply looks like when the key or the endpoint is the problem (as opposed to this one request): worth giving the key up for
# the session. HTTP 401/403 rejected, 402 no credit, 404 no such model, 429 rate limited, 5xx provider down; or no connection at all.
_DEAD = re.compile(r"HTTP (401|402|403|404|429|5\d\d)\b|ConnectionError|Timeout|ConnectError|SSLError|not installed|is not set")


def failed(reply: str) -> bool:
    return reply.startswith("[LLM error")


def looks_dead(reply: str) -> bool:
    """True when an error reply says the key was rejected, out of credit, rate limited, or the provider could not be reached or is down."""
    return failed(reply) and bool(_DEAD.search(reply))


# ---------------------------------------------------------------------------
# One call to one provider
# ---------------------------------------------------------------------------
def _strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip() or "[LLM error: empty reply]"


def _anthropic_detail(e) -> str:
    """The readable part of an Anthropic error ("authentication_error: invalid x-api-key"), else whatever the SDK says."""
    error = e.body.get("error") if isinstance(getattr(e, "body", None), dict) else None
    if isinstance(error, dict) and error.get("message"):
        return f"{error.get('type', 'error')}: {error['message']}"
    return str(e.message)


def _anthropic_reply(model: str, system: str, user: str, key: str, sdk=None) -> str:
    """One message through the official anthropic package. No temperature is sent: current Claude models reject any but the default.
    The SDK's own retries are off: moving on to the next key is the retry."""
    if sdk is None:
        try:
            import anthropic as sdk
        except ImportError:
            return "[LLM error: the anthropic package is not installed (pip install anthropic)]"
    client = sdk.Anthropic(api_key=key, timeout=REQUEST_TIMEOUT, max_retries=0)
    try:
        message = client.messages.create(model=model, max_tokens=MAX_TOKENS, system=system, messages=[{"role": "user", "content": user}])
    except sdk.APIStatusError as e:
        return f"[LLM error: HTTP {e.status_code} {_anthropic_detail(e)[:200]}]"
    except sdk.APIConnectionError as e:                                       # includes timeouts
        return f"[LLM error: {type(e).__name__}: could not reach the Anthropic API]"
    return _strip_thinking("".join(block.text for block in message.content if block.type == "text"))


def chat(provider: str, model: str, system: str, user: str, url: str = OLLAMA_URL, key: Optional[str] = None, post=requests.post, sdk=None) -> str:
    """One chat completion. Never raises: a failure comes back as '[LLM error: ...]' so the retrieval output still prints.
    OpenRouter and Ollama run at temperature 0; `post` and `sdk` are for tests."""
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    try:
        if provider == "anthropic":
            key = os.environ.get("ANTHROPIC_API_KEY", "") if key is None else key
            if not key:
                return "[LLM error: ANTHROPIC_API_KEY is not set]"
            return _anthropic_reply(model, system, user, key, sdk)
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
        return _strip_thinking(text)
    except Exception as e:  # noqa: BLE001 - the caller only needs the message
        return f"[LLM error: {type(e).__name__}: {e}]"


# ---------------------------------------------------------------------------
# The keys in .env
# ---------------------------------------------------------------------------
def read_env(path: Path = ENV_FILE) -> List[Tuple[str, str]]:
    """(name, value) of each KEY=VALUE line of a .env file, in file order; comments and lines without = are skipped. A missing file is empty."""
    try:
        lines = Path(path).read_text().splitlines()
    except OSError:
        return []
    pairs = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            name, value = line.split("=", 1)
            pairs.append((name.strip().removeprefix("export ").strip(), value.strip().strip("\"'")))
    return pairs


def load_env(path: Path = ENV_FILE) -> None:
    """Put the KEY=VALUE lines of a .env file into the environment; variables that are already set win. A missing file is fine."""
    for name, value in read_env(path):
        os.environ.setdefault(name, value)


def keys_in_order(path: Path = ENV_FILE) -> List[str]:
    """The names a .env file lists, in file order, each once (the first place it appears decides)."""
    return list(dict.fromkeys(name for name, _ in read_env(path)))


@dataclass(frozen=True)
class Candidate:
    provider: str
    env_name: str          # where the key comes from ("" for a local Ollama, which has none)
    key: str
    model: str
    url: str = ""          # Ollama only

    @property
    def label(self) -> str:
        return f"{self.provider} {self.model}" + (f" ({self.env_name})" if self.env_name else "")


def candidates(order: Sequence[str], environ: Mapping[str, str] = os.environ) -> List[Candidate]:
    """The keys that are filled in, in the order `order` (the names of .env) lists them; a key that is only in the environment, not in
    the file, comes after those. An empty or blank value is no key. The value is the environment's (so an exported variable wins over
    the file's), the position is the file's."""
    names = [n for n in order if n in KEY_PROVIDERS] + [n for n in KEY_PROVIDERS if n not in order]
    out = []
    for name in dict.fromkeys(names):
        key = (environ.get(name) or "").strip()
        if key:
            out.append(Candidate(KEY_PROVIDERS[name], name, key, DEFAULT_MODELS[KEY_PROVIDERS[name]]))
    return out


# ---------------------------------------------------------------------------
# Falling back from one key to the next
# ---------------------------------------------------------------------------
class Failover:
    """An LLM, (system prompt, user prompt) -> reply, over several candidates. Calls go to the current one; when it fails the next
    answers, and if that works it becomes the current one. A candidate whose failure says the key or the provider is the problem
    (looks_dead) is skipped for the rest of the session, so a dead endpoint costs one failed call, not one per question; when every
    candidate has been given up on they are all tried again. Any other failure (an empty reply, a bad request) just moves on to the
    next candidate for that call."""

    def __init__(self, candidates: Sequence[Candidate], chat_fn: Callable = chat, say: Callable[[str], None] = print):
        self.candidates, self._chat, self.say = list(candidates), chat_fn, say
        self.current = 0
        self.down: Dict[int, str] = {}
        self._lock = threading.Lock()

    def _ask(self, c: Candidate, system: str, user: str) -> str:
        return self._chat(c.provider, c.model, system, user, url=c.url or OLLAMA_URL, key=c.key or None)

    def _order(self) -> List[int]:
        with self._lock:
            if len(self.down) >= len(self.candidates):
                self.down.clear()
            first = [self.current] + [i for i in range(len(self.candidates)) if i != self.current]
            return [i for i in first if i not in self.down]

    def __call__(self, system: str, user: str) -> str:
        last = "[LLM error: no LLM is configured]"
        for i in self._order():
            c = self.candidates[i]
            reply = self._ask(c, system, user)
            if not failed(reply):
                with self._lock:
                    self.current = i
                return reply
            last = reply
            if looks_dead(reply):
                with self._lock:
                    self.down[i] = reply
                self.say(f"note: {c.env_name or c.provider} gave up ({reply[12:90].strip()}); trying the next key")
        return last

    def describe(self) -> str:
        names = [c.label for c in self.candidates]
        first = names[self.current]
        rest = [n for i, n in enumerate(names) if i != self.current and i not in self.down]       # a key that failed the check is not a fall-back
        return first + (f", then {', then '.join(rest)} if it fails" if rest else "")


def _ollama_up(url: str) -> bool:
    try:
        return requests.get(f"{url}/api/tags", timeout=3).ok
    except requests.RequestException:
        return False


def connect(provider: str, model: Optional[str], ollama_url: str, order: Sequence[str], environ: Mapping[str, str] = os.environ,
            chat_fn: Callable = chat, ollama_up: Callable[[str], bool] = _ollama_up, say: Callable[[str], None] = print) -> Optional[Failover]:
    """The LLM for this session, or None (and why, through `say`). `provider` "auto": the keys of .env in the file's order, the first
    that answers a tiny request used and the rest kept as fall-backs; anything else: exactly that provider. `model` replaces the model of
    the first candidate only (model names differ between providers)."""
    if provider == "ollama":
        if not ollama_up(ollama_url):
            say(f"Ollama is not answering at {ollama_url}: showing retrieval only")
            return None
        return Failover([Candidate("ollama", "", "", model or DEFAULT_MODELS["ollama"], ollama_url)], chat_fn, say)
    found = candidates(order, environ)
    if provider != "auto":
        found = [c for c in found if c.provider == provider]
    if not found:
        wanted = "OPENROUTER_API_KEY or ANTHROPIC_API_KEY" if provider == "auto" else next(k for k, v in KEY_PROVIDERS.items() if v == provider)
        say(f"no key found: put {wanted} in {ENV_FILE.name} (the first one listed that works is used) or export it; showing retrieval only. "
            "Or use --provider ollama.")
        return None
    if model:
        found[0] = replace(found[0], model=model)
    llm = Failover(found, chat_fn, say)
    for i, c in enumerate(found):                           # a tiny request tells a working key from a rejected or unreachable one
        reply = chat_fn(c.provider, c.model, PING_SYSTEM, "ping", url=c.url or OLLAMA_URL, key=c.key)
        if not failed(reply):
            llm.current = i
            return llm
        llm.down[i] = reply
        say(f"{c.env_name} is not usable ({reply[12:100].strip().rstrip(']')}); " + ("trying the next key" if i + 1 < len(found) else "no key left"))
    say("no key works: showing retrieval only")
    return None
