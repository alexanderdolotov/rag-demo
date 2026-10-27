# rag_demo - OpenSearch against Neo4j for finding concepts in text

A standalone proof of concept: it compares OpenSearch (search engine) and Neo4j (graph database) on one shared corpus of invented stories
(plain files in `stories/`), so you can type a question and see how each database ranks the answers, with an LLM answer written from each
database's top 5. It has nothing to do with the stocks / Open Terminal projects; a RAG over real financial filings would be a separate project.
Everything runs on this Mac against two throwaway Docker containers.

```bash
cd ~/Projects/rag_demo
python3 -m venv .venv && source .venv/bin/activate     # this project's own venv; .venv/ is git-ignored
pip install -r requirements.txt
docker start opensearch-poc neo4j-poc                  # first-time container setup is in the Neo4j section below

python data_load.py                                # load stories/ into both databases (once)
python ask_stories.py                              # type a question, see both databases' top 5 and an LLM answer from each
python compare_stories.py                          # 28 queries x 3 scenarios x every method, as a table
pytest test -v                                     # tests skip themselves when a service is down
```

## One corpus, loaded once into both databases

The corpus is plain files: `stories/<story_id>/original.txt` plus `rewrite-1.txt` ... `rewrite-5.txt` (20 stories, 120 files; format and
the five rewrite styles in [stories/README.md](stories/README.md)). `data_load.py` is the only thing that writes them into the databases:

```bash
python data_load.py             # load what is missing; nothing to do if both already hold these files; refuses to overwrite a different corpus
python data_load.py --reload    # first delete this project's data in both databases, then load again
python data_load.py --status    # what each database holds compared with the files; changes nothing
```

Both databases store a fingerprint of the files they were loaded from, and `ask_stories.py` / `compare_stories.py` refuse to run when it no
longer matches (edit a story, then `--reload`). `--reload` deletes only the OpenSearch index `stories` and the Neo4j nodes labelled
`Story`, `StoryVersion`, `StoryTerm`, `StoryCorpus`; anything else in either database is left alone. A load takes about 15 seconds
(embeddings for 120 texts and about 2,200 words). Versions longer than the embedding window (512 tokens) are reported, not silently cut.

Tests, `compare_stories.py` and the toy smoke run each work in their own namespace (other index names; Neo4j labels with a `Test` / `Cmp`
prefix), so none of them touches the loaded corpus. `neo4j_stories.PREFIX` does that; an unprefixed delete-everything bug would still hit
all of Neo4j, because the community edition has one database only.

## The small in-code test corpus (OpenSearch tests and smoke run)

The unit tests and `story_smoke.py` use a tiny, fast, in-code corpus (`STORIES` in `opensearch_stories.py`: 24 one-sentence texts) of frog / stock / tariff stories, each with LLM-style rewrites in different words and traps
(polysemy, negation, unrelated stories), loaded into a local OpenSearch to compare ways of finding the same concept
when the words differ. See the docstring of `opensearch_stories.py` for the methods compared.

```bash
python story_smoke.py                     # (re)loads the small `stories_toy` index (not the loaded corpus) and prints every query and a scoreboard
python story_smoke.py --no-vectors        # keyword arms only
pytest test/test_opensearch_stories.py -v  # skips itself if OpenSearch is down; uses and removes a `stories_test` index
```

| File | Purpose |
|---|---|
| `opensearch_stories.py` | stories, queries, synonym list, index mapping/analyzers, loaders, search arms, rank fusion, scoring |
| `story_smoke.py` | command-line run: load, search, scoreboard |
| `test/test_opensearch_stories.py` | the findings as assertions (analyzer behaviour, vocabulary gap, synonyms, rewrites, polysemy, vectors) |

Caveats: 24 one-sentence documents and 8 queries is a toy; the synonym list was written knowing the queries, so the keyword-with-synonyms
numbers are optimistic. Treat results as a way to see how each method behaves, not as accuracy figures.


## Neo4j (graph database) and its methods

The corpus goes into Neo4j as a graph (versions, terms, and word links learned from the rewrites) and is searched with the same kinds of
methods as OpenSearch. Query sets, defined in `opensearch_stories.py`: set 1 = the 8 first queries, set 2 = 11 harder ones (`HELDOUT`: words
missing from the corpus, wrong-polarity and wrong-sense distractors), set 3 = 9 about the stories that exist only on disk (`NEW`).
`neo4j_stories.py` has the graph model and the arms (`n_ft`, `n_vec`, `n_co`, `n_term`, `n_hyb`, `n_story`).

```bash
# one-off: a throwaway local Neo4j (no auth, bound to localhost only - do not put real data in it)
docker run -d --name neo4j-poc -p 127.0.0.1:7474:7474 -p 127.0.0.1:7687:7687 -e NEO4J_AUTH=none \
  -e NEO4J_server_memory_heap_max__size=1G -e NEO4J_server_memory_pagecache_size=512m neo4j:latest
docker start opensearch-poc neo4j-poc          # on later days

python compare_stories.py                 # rank table per scenario on the loaded corpus (see "Comparison table")
python compare_stories.py --explain       # also shows which terms the graph added to each query
pytest test/test_neo4j_stories.py -v      # skips itself if either service is down; small in-code corpus, `Test`-prefixed graph
```

Env `NEO4J_URI` (default `bolt://localhost:7687`) and optionally `NEO4J_USER` / `NEO4J_PASSWORD`. Only nodes labelled
`Story`, `StoryVersion`, `StoryTerm`, `StoryCorpus` (and indexes `story_*`, `term_*`) are written or deleted, under the prefixes described above;
the Neo4j *community* edition has a single database, so do not point this at one that holds other data under those labels.
The Neo4j browser is at http://localhost:7474 (try `MATCH (a:StoryTerm {stem:'toad'})-[c:CO]-(b) RETURN a,c,b LIMIT 25`).

| File | Purpose |
|---|---|
| `neo4j_stories.py` | tokenizer, graph build (`targets` / `learn_from` scenarios), the Neo4j search arms |
| `test/test_neo4j_stories.py` | the findings as assertions (index parity, what co-occurrence can and cannot bridge, graph vs vectors) |

Caveat: these queries are single-hop concept lookups, which is not where a graph database is expected to shine
(entity relationships, multi-hop questions, provenance); this benchmark does not test that.

## Ask both databases (live test)

Type a question; each database returns its own top 5 (ranked, best first) and an LLM writes one answer from each database's top 5.
The plain run sends your question unchanged; the optional LLM steps are switched on one at a time with arguments, so you can see what each one adds.

```bash
echo 'OPENROUTER_API_KEY=...' > .env       # once; .env is git-ignored. An exported OPENROUTER_API_KEY wins over the file
python ask_stories.py                     # plain run: native top 5 per database + one answer per database
python ask_stories.py --no-llm            # retrieval only, no key needed
python ask_stories.py --rewrite           # + LLM turns the question into a keyword query (same text sent to both databases)
python ask_stories.py --rerank            # + LLM re-orders each database's 10 candidates ("was #7" shows the native rank)
python ask_stories.py --judge             # + a third call says which answer is better supported (random A/B order; a hint only)
python ask_stories.py --rewrite --rerank --judge
python ask_stories.py --provider ollama   # local model instead (default qwen3:8b; needs `ollama serve` and `ollama pull qwen3:8b`)
python ask_stories.py --os-arm dense --neo-arm n_vec     # compare other methods (see --help for the arm names)
python ask_stories.py --debug              # print the exact requests sent to each database (OpenSearch JSON bodies, Neo4j Cypher + parameters)
```

`--debug` shows only what the chosen method sends (embeddings are abbreviated to their first numbers; the real vector is sent), plus, for the
keyword methods, how the index analyses your words (stop words dropped, stemmed, synonyms added). Default OpenRouter model:
`anthropic/claude-haiku-5.5` (change with `--model`; the older `claude-3.5-haiku` is no longer served).

Versions are collapsed to one row per story with the best version's score (rewrites count 0.6 of an original), and the text shown and sent to
the LLM is the story's original, so near-ties can flip because of that weighting: the `via` column shows which version matched. Scores are each
engine's own, not comparable across databases. Times include embedding the query; Neo4j's merged method makes several round trips.

## Comparison table

`compare_stories.py` runs three scenarios (originals only; rewrites used only as a thesaurus for the graph; originals and rewrites all
retrievable) over three query sets (set 1 = 8 first queries, set 2 = 11 harder ones, set 3 = 9 about the stories that exist only on disk) and
prints the rank of the first correct story per query plus hit@3 and mean reciprocal rank per set. It reads the loaded OpenSearch index; the
Neo4j scenario graphs are built under their own `Cmp` prefix and deleted afterwards. `--explain` adds the words the graph appended to each query.

## Files

| File | Purpose |
|---|---|
| `stories/` | the corpus (20 stories x 6 files) and its README |
| `corpus.py` | reads and validates `stories/`, fingerprints it |
| `data_load.py` | loads both databases from the files (`--reload`, `--status`) |
| `ask_stories.py` | the interactive two-database script (LLM client, optional steps, `--debug`) |
| `compare_stories.py` | the comparison table |
| `opensearch_stories.py`, `neo4j_stories.py` | index / graph building and every search method, per database |
| `story_smoke.py` | quick demo on the small in-code corpus |
| `test/` | `test_corpus.py`, `test_data_load.py`, `test_ask_stories.py`, `test_opensearch_stories.py`, `test_neo4j_stories.py` |
