"""
The LLM client and the fall-back from one key to the next (llm.py). No network and no database: the providers are replaced by fakes, and
the Anthropic SDK (the real one) runs over a mock transport, so the request it would send is checked without sending it.

The rule being tested: the key listed first in .env that is filled in and works is used; a key that is empty, rejected or unreachable
gives way to the next one.
"""
import json
from types import SimpleNamespace

import pytest

import llm

SECRETS = ("sk-secret-AAA", "sk-secret-BBB")


# --------------------------------------------------------------------------- the order of the keys
def test_the_keys_are_listed_in_the_order_the_file_has_them(tmp_path):
    env = tmp_path / ".env"
    env.write_text("# a comment\n\nANTHROPIC_API_KEY=\nexport OPENROUTER_API_KEY = 'x'\nOTHER=1\nnot a pair\nANTHROPIC_API_KEY=again\n")
    assert llm.keys_in_order(env) == ["ANTHROPIC_API_KEY", "OPENROUTER_API_KEY", "OTHER"]        # each once, first appearance decides
    assert llm.keys_in_order(tmp_path / "missing.env") == []


def test_load_env_still_never_overrides_what_is_already_set(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("ANTHROPIC_API_KEY=from-file\nOPENROUTER_API_KEY=from-file\n")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-shell")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    llm.load_env(env)
    assert llm.os.environ["ANTHROPIC_API_KEY"] == "from-shell" and llm.os.environ["OPENROUTER_API_KEY"] == "from-file"


def test_the_candidates_follow_the_file_order_and_skip_empty_keys():
    both = {"ANTHROPIC_API_KEY": "a", "OPENROUTER_API_KEY": "o"}
    assert [c.provider for c in llm.candidates(["ANTHROPIC_API_KEY", "OPENROUTER_API_KEY"], both)] == ["anthropic", "openrouter"]
    assert [c.provider for c in llm.candidates(["OPENROUTER_API_KEY", "ANTHROPIC_API_KEY"], both)] == ["openrouter", "anthropic"]
    assert [c.env_name for c in llm.candidates(["ANTHROPIC_API_KEY", "OPENROUTER_API_KEY"], {"ANTHROPIC_API_KEY": "", "OPENROUTER_API_KEY": "o"})] == ["OPENROUTER_API_KEY"]
    assert llm.candidates(["ANTHROPIC_API_KEY"], {"ANTHROPIC_API_KEY": "   "}) == [] and llm.candidates([], {}) == []


def test_a_key_only_in_the_environment_comes_after_the_files_keys_and_unknown_names_are_ignored():
    environ = {"ANTHROPIC_API_KEY": "a", "OPENROUTER_API_KEY": "o", "SOMETHING_ELSE": "x"}
    assert [c.env_name for c in llm.candidates(["OPENROUTER_API_KEY", "SOMETHING_ELSE"], environ)] == ["OPENROUTER_API_KEY", "ANTHROPIC_API_KEY"]


def test_each_candidate_gets_its_providers_default_model():
    got = {c.provider: c.model for c in llm.candidates(["ANTHROPIC_API_KEY", "OPENROUTER_API_KEY"], {"ANTHROPIC_API_KEY": "a", "OPENROUTER_API_KEY": "o"})}
    assert got == {"anthropic": "claude-haiku-5-5", "openrouter": "anthropic/claude-haiku-5.5"}


# --------------------------------------------------------------------------- when the error means the key is the problem
@pytest.mark.parametrize("reply, dead", [
    ("[LLM error: HTTP 401 invalid x-api-key]", True),
    ("[LLM error: HTTP 403 forbidden]", True),
    ("[LLM error: HTTP 402 insufficient credits]", True),
    ("[LLM error: HTTP 404 model not found]", True),
    ("[LLM error: HTTP 429 rate limited]", True),
    ("[LLM error: HTTP 503 overloaded]", True),
    ("[LLM error: ConnectionError: refused]", True),
    ("[LLM error: ReadTimeout: slow]", True),
    ("[LLM error: APIConnectionError: could not reach the Anthropic API]", True),
    ("[LLM error: the anthropic package is not installed (pip install anthropic)]", True),
    ("[LLM error: HTTP 400 bad request]", False),               # this one request was wrong, not the key
    ("[LLM error: empty reply]", False),
    ("a normal answer mentioning HTTP 401", False),             # only error replies count
])
def test_which_errors_say_the_key_or_the_provider_is_the_problem(reply, dead):
    assert llm.looks_dead(reply) is dead


# --------------------------------------------------------------------------- Anthropic's own API, through the real SDK on a mock transport
def message_json(*blocks, stop="end_turn"):
    return {"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-haiku-5-5", "content": list(blocks),
            "stop_reason": stop, "stop_sequence": None, "usage": {"input_tokens": 3, "output_tokens": 2}}


TEXT = lambda t: {"type": "text", "text": t}  # noqa: E731
THINKING = {"type": "thinking", "thinking": "", "signature": "sig"}


def real_sdk(handler):
    """The installed anthropic package with its HTTP layer replaced by `handler`."""
    anthropic = pytest.importorskip("anthropic")
    try:
        import httpx2 as httpx          # anthropic 1.x
    except ImportError:
        import httpx                    # anthropic 0.x
    return SimpleNamespace(
        Anthropic=lambda **kw: anthropic.Anthropic(http_client=httpx.Client(transport=httpx.MockTransport(handler)), **kw),
        APIStatusError=anthropic.APIStatusError, APIConnectionError=anthropic.APIConnectionError, httpx=httpx)


def test_the_anthropic_request_has_the_documented_shape_sends_no_temperature_and_reads_only_the_text(monkeypatch):
    seen = {}
    holder = {}

    def handler(request):
        seen["url"], seen["headers"], seen["body"] = str(request.url), dict(request.headers), json.loads(request.content)
        return holder["sdk"].httpx.Response(200, json=message_json(THINKING, TEXT("<think>hmm</think>It is [1].")))
    holder["sdk"] = real_sdk(handler)
    reply = llm.chat("anthropic", "claude-haiku-5-5", "SYS", "USER", key=SECRETS[0], sdk=holder["sdk"])
    assert reply == "It is [1]."                                                     # the thinking block is skipped, think tags are stripped
    assert seen["url"].endswith("/v1/messages") and seen["headers"]["x-api-key"] == SECRETS[0] and "anthropic-version" in seen["headers"]
    body = seen["body"]
    assert body["model"] == "claude-haiku-5-5" and body["system"] == "SYS" and body["messages"] == [{"role": "user", "content": "USER"}]
    assert body["max_tokens"] == llm.MAX_TOKENS and "temperature" not in body        # Haiku 5.5 rejects a non-default temperature


def test_an_anthropic_rejection_comes_back_as_error_text_without_the_key(monkeypatch):
    holder = {}

    def handler(request):
        return holder["sdk"].httpx.Response(401, json={"type": "error", "error": {"type": "authentication_error", "message": "invalid x-api-key"}})
    holder["sdk"] = real_sdk(handler)
    reply = llm.chat("anthropic", "claude-haiku-5-5", "S", "U", key=SECRETS[0], sdk=holder["sdk"])
    assert reply == "[LLM error: HTTP 401 authentication_error: invalid x-api-key]"        # the readable part, not the SDK's raw dump
    assert SECRETS[0] not in reply and llm.looks_dead(reply)


def test_an_anthropic_error_without_a_readable_body_still_gives_the_status_and_the_sdks_message():
    holder = {}
    holder["sdk"] = real_sdk(lambda request: holder["sdk"].httpx.Response(529, text="overloaded, try later"))
    reply = llm.chat("anthropic", "claude-haiku-5-5", "S", "U", key=SECRETS[0], sdk=holder["sdk"])
    assert reply.startswith("[LLM error: HTTP 529") and "overloaded" in reply and llm.looks_dead(reply)


def test_an_unreachable_anthropic_api_comes_back_as_a_dead_connection_not_an_exception():
    holder = {}

    def handler(request):
        raise holder["sdk"].httpx.ConnectError("no route")
    holder["sdk"] = real_sdk(handler)
    reply = llm.chat("anthropic", "claude-haiku-5-5", "S", "U", key=SECRETS[0], sdk=holder["sdk"])
    assert "ConnectionError" in reply and llm.looks_dead(reply) and SECRETS[0] not in reply


def test_a_reply_with_no_text_is_an_error_so_the_next_key_gets_a_turn():
    holder = {}
    holder["sdk"] = real_sdk(lambda request: holder["sdk"].httpx.Response(200, json=message_json(THINKING, stop="refusal")))
    assert llm.chat("anthropic", "claude-haiku-5-5", "S", "U", key=SECRETS[0], sdk=holder["sdk"]) == "[LLM error: empty reply]"


def test_a_missing_key_or_a_missing_package_is_reported_not_raised(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert llm.chat("anthropic", "m", "S", "U") == "[LLM error: ANTHROPIC_API_KEY is not set]"
    monkeypatch.setitem(__import__("sys").modules, "anthropic", None)             # makes `import anthropic` fail
    reply = llm.chat("anthropic", "m", "S", "U", key=SECRETS[0])
    assert "not installed" in reply and llm.looks_dead(reply)


# --------------------------------------------------------------------------- falling back
class FakeChat:
    """A chat function: which reply each key gets, and a record of who was asked."""

    def __init__(self, replies):
        self.replies, self.calls = replies, []

    def __call__(self, provider, model, system, user, url="", key=None):
        self.calls.append((provider, key))
        reply = self.replies[key]
        return reply(system, user) if callable(reply) else reply


A = llm.Candidate("anthropic", "ANTHROPIC_API_KEY", SECRETS[0], "claude-haiku-5-5")
B = llm.Candidate("openrouter", "OPENROUTER_API_KEY", SECRETS[1], "anthropic/claude-haiku-5.5")
REJECTED = "[LLM error: HTTP 401 invalid x-api-key]"


def failover(replies, *cands, notes=None):
    chat = FakeChat(replies)
    return llm.Failover(cands or (A, B), chat, say=(notes if notes is not None else []).append), chat


def test_the_first_key_answers_and_the_others_are_never_asked():
    f, chat = failover({SECRETS[0]: "from A", SECRETS[1]: "from B"})
    assert f("s", "u") == "from A" and f("s", "u") == "from A"
    assert chat.calls == [("anthropic", SECRETS[0])] * 2


def test_a_rejected_key_hands_over_to_the_next_and_is_not_asked_again_this_session():
    notes = []
    f, chat = failover({SECRETS[0]: REJECTED, SECRETS[1]: "from B"}, notes=notes)
    assert f("s", "u") == "from B"
    assert f("s", "u") == "from B"
    assert chat.calls == [("anthropic", SECRETS[0]), ("openrouter", SECRETS[1]), ("openrouter", SECRETS[1])]       # A asked once, then given up on
    assert len(notes) == 1 and "ANTHROPIC_API_KEY" in notes[0] and not any(s in notes[0] for s in SECRETS)


def test_a_key_that_cannot_connect_is_given_up_the_same_way():
    f, chat = failover({SECRETS[0]: "[LLM error: ConnectionError: refused]", SECRETS[1]: "from B"})
    assert f("s", "u") == "from B" and f("s", "u") == "from B" and chat.calls.count(("anthropic", SECRETS[0])) == 1


def test_an_error_that_is_this_requests_fault_tries_the_next_key_without_giving_the_first_up():
    f, chat = failover({SECRETS[0]: lambda s, u: "[LLM error: HTTP 400 bad request]" if u == "odd" else "from A", SECRETS[1]: "from B"})
    assert f("s", "odd") == "from B"
    assert f("s", "fine") == "from B"                                    # B answered last, so B is where the next call starts ...
    assert f.down == {}                                                  # ... but A was never written off


def test_when_every_key_fails_the_last_error_comes_back_and_all_are_tried_again_next_time():
    f, chat = failover({SECRETS[0]: REJECTED, SECRETS[1]: "[LLM error: HTTP 503 down]"})
    assert f("s", "u") == "[LLM error: HTTP 503 down]"
    assert len(chat.calls) == 2 and len(f.down) == 2
    chat.replies[SECRETS[0]] = "from A again"
    assert f("s", "u") == "from A again"                                 # nothing was left, so both were tried afresh


def test_describe_names_the_key_in_use_and_only_the_fall_backs_still_in_play():
    f, _ = failover({SECRETS[0]: "x", SECRETS[1]: "y"})
    assert f.describe() == "anthropic claude-haiku-5-5 (ANTHROPIC_API_KEY), then openrouter anthropic/claude-haiku-5.5 (OPENROUTER_API_KEY) if it fails"
    f.current, f.down = 1, {0: REJECTED}
    assert f.describe() == "openrouter anthropic/claude-haiku-5.5 (OPENROUTER_API_KEY)"


# --------------------------------------------------------------------------- choosing the key at the start
def connect(replies, order, environ, provider="auto", model=None, **kw):
    notes, chat = [], FakeChat(replies)
    return llm.connect(provider, model, "http://ollama", order, environ, chat_fn=chat, say=notes.append, **kw), chat, notes


BOTH = {"ANTHROPIC_API_KEY": SECRETS[0], "OPENROUTER_API_KEY": SECRETS[1]}


def test_the_first_key_listed_that_answers_is_used_and_the_rest_are_not_even_asked():
    f, chat, notes = connect({SECRETS[0]: "ok", SECRETS[1]: "ok"}, ["OPENROUTER_API_KEY", "ANTHROPIC_API_KEY"], BOTH)
    assert f.current == 0 and f.candidates[0].provider == "openrouter" and chat.calls == [("openrouter", SECRETS[1])] and notes == []


def test_an_empty_key_is_skipped_without_a_request():
    f, chat, notes = connect({SECRETS[1]: "ok"}, ["ANTHROPIC_API_KEY", "OPENROUTER_API_KEY"], {"ANTHROPIC_API_KEY": "", "OPENROUTER_API_KEY": SECRETS[1]})
    assert len(f.candidates) == 1 and f.candidates[0].provider == "openrouter" and chat.calls == [("openrouter", SECRETS[1])]


def test_a_rejected_first_key_is_skipped_with_the_reason_and_the_next_one_is_used():
    f, chat, notes = connect({SECRETS[0]: REJECTED, SECRETS[1]: "ok"}, ["ANTHROPIC_API_KEY", "OPENROUTER_API_KEY"], BOTH)
    assert f.current == 1 and f.candidates[f.current].provider == "openrouter"
    assert [c[0] for c in chat.calls] == ["anthropic", "openrouter"]
    assert len(notes) == 1 and "ANTHROPIC_API_KEY is not usable" in notes[0] and "401" in notes[0] and "trying the next key" in notes[0]
    assert not any(s in n for n in notes for s in SECRETS)
    assert f.describe() == "openrouter anthropic/claude-haiku-5.5 (OPENROUTER_API_KEY)"      # the rejected key is not offered as a fall-back


def test_an_unreachable_first_key_falls_back_just_as_a_rejected_one_does():
    f, chat, notes = connect({SECRETS[0]: "[LLM error: APIConnectionError: could not reach]", SECRETS[1]: "ok"}, ["ANTHROPIC_API_KEY", "OPENROUTER_API_KEY"], BOTH)
    assert f.candidates[f.current].provider == "openrouter"


def test_no_key_or_no_working_key_means_no_llm_and_a_reason():
    f, chat, notes = connect({}, ["ANTHROPIC_API_KEY"], {"ANTHROPIC_API_KEY": ""})
    assert f is None and chat.calls == [] and "no key found" in notes[0] and "ANTHROPIC_API_KEY" in notes[0]
    f, chat, notes = connect({SECRETS[0]: REJECTED, SECRETS[1]: REJECTED}, ["ANTHROPIC_API_KEY", "OPENROUTER_API_KEY"], BOTH)
    assert f is None and len(chat.calls) == 2 and notes[-1] == "no key works: showing retrieval only" and "no key left" in notes[-2]


def test_naming_a_provider_uses_only_that_providers_key():
    f, chat, notes = connect({SECRETS[1]: "ok"}, ["ANTHROPIC_API_KEY", "OPENROUTER_API_KEY"], BOTH, provider="openrouter")
    assert [c.provider for c in f.candidates] == ["openrouter"] and chat.calls == [("openrouter", SECRETS[1])]
    f, chat, notes = connect({}, ["ANTHROPIC_API_KEY"], {"OPENROUTER_API_KEY": SECRETS[1]}, provider="anthropic")
    assert f is None and "ANTHROPIC_API_KEY" in notes[0] and chat.calls == []


def test_a_model_given_on_the_command_line_replaces_the_first_keys_model_only():
    f, chat, notes = connect({SECRETS[0]: "ok", SECRETS[1]: "ok"}, ["ANTHROPIC_API_KEY", "OPENROUTER_API_KEY"], BOTH, model="claude-sonnet-5-5")
    assert [c.model for c in f.candidates] == ["claude-sonnet-5-5", "anthropic/claude-haiku-5.5"]


def test_a_local_ollama_needs_no_key_and_is_checked_without_generating_anything():
    f, chat, notes = connect({}, [], {}, provider="ollama", ollama_up=lambda url: True)
    assert f.candidates[0].provider == "ollama" and f.candidates[0].url == "http://ollama" and chat.calls == []
    f, chat, notes = connect({}, [], {}, provider="ollama", ollama_up=lambda url: False)
    assert f is None and "Ollama is not answering" in notes[0]
