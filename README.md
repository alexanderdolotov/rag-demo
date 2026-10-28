# rag_demo - OpenSearch against Neo4j for finding concepts in text

A standalone proof of concept: it compares OpenSearch (a search engine) and Neo4j (a graph database) at finding the same concept in
differently worded text. Both hold one shared corpus of invented stories (plain files in `stories/`). You type a question and see what each
database finds. An LLM is optional: it first reads the question, then scores what the databases found, and an answer is written from each
database's results. It has nothing to do with the stocks / Open Terminal projects; a RAG over real financial filings would be a separate
project. Everything runs on one Mac against two throwaway Docker containers. MIT licensed (see `LICENSE`).

```bash
cd ~/Projects/rag_demo
python3 -m venv .venv && source .venv/bin/activate     # this project's own venv; .venv/ is git-ignored
pip install -r requirements.txt
docker start opensearch-poc neo4j-poc                  # first time: create them, see "The two databases"

python data_load.py                                # load stories/ into both databases (once)
python ask_stories.py                              # ask a question: the LLM reads it, both databases are searched, what scores well is shown
python test/compare_stories.py                     # 28 queries x 3 scenarios x every method, as a table
pytest -v                                          # about two minutes; the tests that need a database skip themselves when it is down
```

## The two databases

Two throwaway containers, bound to localhost, with no login (do not put real data in them):

```bash
docker run -d --name opensearch-poc -p 127.0.0.1:9200:9200 -p 127.0.0.1:9600:9600 \
  -e discovery.type=single-node -e DISABLE_SECURITY_PLUGIN=true -e OPENSEARCH_JAVA_OPTS="-Xms2g -Xmx2g" \
  opensearchproject/opensearch:latest
docker run -d --name neo4j-poc -p 127.0.0.1:7474:7474 -p 127.0.0.1:7687:7687 -e NEO4J_AUTH=none \
  -e NEO4J_server_memory_heap_max__size=1G -e NEO4J_server_memory_pagecache_size=512m neo4j:latest
docker start opensearch-poc neo4j-poc          # on later days (the containers keep their data)
```

Developed against Python 3.9, OpenSearch 3.9.0, Neo4j 2026.10.1 (the code uses the `SEARCH` clause that Neo4j's vector and full-text indexes
have in 2026.x, so an older Neo4j will not do) and sentence-transformers 5.1 with `all-MiniLM-L6-v2` (384-number vectors). The `latest` tags move;
pin them if you want the same versions later. Settings, all optional:

| Variable | Default | What for |
|---|---|---|
| `OPENSEARCH_URL` | `http://localhost:9200` | where OpenSearch is |
| `NEO4J_URI` | `bolt://localhost:7687` | where Neo4j is |
| `NEO4J_USER`, `NEO4J_PASSWORD` | unset: no login | Neo4j login (used when the password is set) |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | the model that turns texts and questions into vectors |
| `OPENROUTER_API_KEY` | none; put it in `.env` (git-ignored, see `.env.example`) | the LLM steps with the default provider |
| `OLLAMA_URL` | `http://localhost:11434` | the LLM steps with `--provider ollama` |

The Neo4j browser is at http://localhost:7474 (try `MATCH (a:StoryTerm {stem:'toad'})-[c:CO]-(b) RETURN a,c,b LIMIT 25`). The Neo4j *community*
edition has a single database, so do not point this at one that holds other data under the labels below.

## One corpus, loaded once into both databases

The corpus is plain files: `stories/<story_id>/original.txt` plus `rewrite-1.txt` ... `rewrite-5.txt` (20 stories, 120 files; format and
the five rewrite styles in [stories/README.md](stories/README.md)). `data_load.py` is the only thing that writes them into the databases:

```bash
python data_load.py             # load what is missing; nothing to do if both already hold these files; refuses to overwrite a different corpus
python data_load.py --reload    # first delete this project's data in both databases, then load again
python data_load.py --status    # what each database holds compared with the files; changes nothing
```

Both databases store a fingerprint of the files they were loaded from, and `ask_stories.py` / `test/compare_stories.py` refuse to run when it no
longer matches (edit a story, then `--reload`). The fingerprint covers the story text only: after changing the analysis settings or the
tokenizer, `--reload` is needed too. `--reload` deletes only the OpenSearch index `stories` and the Neo4j nodes labelled
`Story`, `StoryVersion`, `StoryTerm`, `StoryCorpus` (and the indexes `story_*`, `term_*`, which are rebuilt); anything else in either database
is left alone (`--index` must be one plain index name: a wildcard, a comma list or `_all` is refused). An index is stamped with the
fingerprint only after its last document is in, and a count that does not match the files is reported, so an interrupted load is never mistaken
for a finished one. A load takes about 20 seconds (embeddings for 120 texts and about 2,200 words). Versions longer than the embedding window
(512 tokens) are reported, not silently cut.

The tests and `test/compare_stories.py` each work in their own namespace (other index names such as `stories_test` and `stories_hard_test`; Neo4j
labels with a `Test` / `Cmp` prefix), so none of them touches the loaded corpus. `neo4j_stories.PREFIX` does that; an unprefixed
delete-everything bug would still hit all of Neo4j, because the community edition has one database only.

## Ask both databases (live test)

Type a question; each database returns its own results, ranked, and an LLM writes one answer from each database's results. With an LLM
the search is **one-shot with concept priority**: the LLM analyses the question first and scores the candidates afterwards, and the databases
are each searched once per concept, in one round. Without an LLM (`--no-llm`, or no key) the question is searched once as typed.

```bash
echo 'OPENROUTER_API_KEY=...' > .env       # once; .env is git-ignored. An exported OPENROUTER_API_KEY wins over the file
python ask_stories.py                     # analysis + one search per concept + scoring + one answer per database from what scored well
python ask_stories.py --no-rewrite        # ... question searched as typed (the scoring still judges it)
python ask_stories.py --no-rerank         # ... databases' own order, nothing scored or cut
python ask_stories.py --no-llm            # retrieval only, no key needed (one search, question as typed)
python ask_stories.py --judge             # + a last call says which answer is better supported (random A/B order; a hint only)
python ask_stories.py --provider ollama   # local model instead (default qwen3:8b; needs `ollama serve` and `ollama pull qwen3:8b`)
python ask_stories.py --os-arm dense --neo-arm n_vec     # compare other methods (see --help for the arm names)
python ask_stories.py --debug              # print the exact requests sent to each database (OpenSearch JSON bodies, Neo4j Cypher + parameters)
```

`--one-shot`, `--rewrite` and `--rerank` are accepted and do nothing: the search is always one-shot and both LLM steps are on by default. Which
arguments reproduce which test mode is in the table under "Hard questions".

### How a question is searched

1. **Analysis (one LLM call).** The question becomes *concepts*, **most important first**, each with **core** words (what an answer must be about)
   and **related** words; plus **conditions** a passage must meet and that can be judged from its meaning ("benefited, not hurt", "because of
   shipping, not policy"), **narrowing** words (place, time, manner: weak evidence) and things **no story can show** ("near the user"). Spelling is
   corrected here, so "breal" reaches "bread". If the reply cannot be read, the question itself is the one concept.
2. **One search per concept per database**, all in one round. Words count by importance (`retrieval.TIER_WEIGHT`: core 1, related 0.33, narrowing
   0.07): OpenSearch gets one boosted match clause per tier, Neo4j puts the weights into its scoring, and the vector arms embed the core words
   rather than the whole question.
3. **Scoring (one LLM call per concept).** Every story either database found for any concept is scored 0-3 against that concept *and* the
   conditions, keyed by story id, listed by name rather than in the databases' order. The opposite outcome ("hurt" for "benefited") breaks a
   condition and scores 1. The score is the same whichever database found the story, so it is asked once. With several concepts each one is
   told about the others and that a condition applies only to the part of the question it concerns.
4. **What is shown.** Per concept, the stories that scored 2 or 3, best first; the concepts take turns, the main concept first, so a second
   subject is not crowded out; each row says which concept it answers. If nothing is strong, the weak ones (score 1); otherwise "nothing
   relevant found" instead of a padded list. Anything no story can show is passed to the answer step, which says when the passages do not settle it.

The threshold (`pipeline.STRONG`), the weights and the prompts (`intent.py`) are constants you can change.

What is on screen: versions are collapsed to one row per story with the best version's score as it is (a tie goes to the original); the text sent
to the LLM is always the story's original, and the `via` column says which version matched. The line beside each result is the sentence where the
words of your question occur (matched by stem, so "duties" finds "duty"), taken from the version that matched, else from the original, and cut
around those words with `...` when it is longer than a line; a snippet from a version other than `via` is tagged, for example `[orig]`. When none
of your words occurs (a match made by the vectors alone, such as `ceo`, a word no story contains) you see the start of the story instead. Scores
are each engine's own, not comparable across databases. `--debug` shows only what the chosen method sends (embeddings are abbreviated to their
first numbers; the real vector is sent), plus how the index analyses your words (stop words dropped, stemmed, synonyms added). The default
OpenRouter model is `anthropic/claude-haiku-5.5` (change with `--model`).

## OpenSearch methods (tests)

The OpenSearch tests run on the same `stories/` corpus and the first query set from `stories/queries.json`, loaded into their own scratch
index, to compare ways of finding the same concept when the words differ (the stories have LLM-style rewrites in different words and traps:
polysemy, negation, unrelated stories). See the docstring of `opensearch_stories.py` for the methods compared.

```bash
pytest test/test_opensearch_stories.py -v  # skips itself if OpenSearch is down; uses and removes a `stories_test` index
```

Caveats: 20 invented stories and 28 queries is a toy; the synonym list was written knowing the first queries, so the keyword-with-synonyms
numbers are optimistic. Treat results as a way to see how each method behaves, not as accuracy figures.

## Neo4j (graph database) and its methods

The corpus goes into Neo4j as a graph (versions, terms, and word links learned from the rewrites) and is searched with the same kinds of
methods as OpenSearch. Query sets, in `stories/queries.json`: set 1 = the 8 first queries, set 2 = 11 harder ones (less overlap in wording,
wrong-polarity and wrong-sense distractors), set 3 = 9 about the stories added after the first eleven.
`neo4j_stories.py` has the graph model and the arms (`n_ft`, `n_vec`, `n_co`, `n_term`, `n_hyb`, `n_story`). Text analysis: OpenSearch's
`stem` / `syn` arms and Neo4j's `n_ft` both use Lucene's English analysis (possessives dropped, Porter stemmer); the graph arms (`n_co`,
`n_term`) use their own tokenizer with NLTK's Snowball stemmer, so a few stems differ (`wednesday` / `wednesdai`).

```bash
python test/compare_stories.py            # rank table per scenario on the loaded corpus (see "Comparison table")
python test/compare_stories.py --explain  # also shows which terms the graph added to each query
pytest test/test_neo4j_stories.py -v      # skips itself if either service is down; runs on `stories/` in a `Test`-prefixed graph
```

Caveat: these queries are single-hop concept lookups, which is not where a graph database is expected to shine
(entity relationships, multi-hop questions, provenance); this benchmark does not test that.

## Comparison table

`test/compare_stories.py` runs three scenarios (originals only; rewrites used only as a thesaurus for the graph; originals and rewrites all
retrievable) over the three query sets of `stories/queries.json` (28 queries) and prints the rank of the first correct story per query plus
hit@3 and mean reciprocal rank per set. It reads the loaded OpenSearch index; the Neo4j scenario graphs are built under their own `Cmp` prefix
and deleted afterwards. `--explain` adds the words the graph appended to each query. It is a report, not a pass/fail test.

## Which database is better?

For finding the right passages in this project, **neither wins on results, and OpenSearch is the better tool for the job**: it gets the same
results with about half the code and a fraction of the stored data. Neo4j would be the better tool for a different job, questions about
relationships, which this benchmark never asks. The evidence is below. It is a snapshot taken from `test/compare_stories.py` and a timing run on
the loaded corpus (28 queries from `stories/queries.json`); unlike the hard-question tables, no test regenerates it.

**Accuracy**: mean reciprocal rank (how high the first right story is; 1.00 is perfect), "originals only" / "all versions" as defined under
"Comparison table":

| Method | OpenSearch | Neo4j |
|---|---|---|
| plain words | `plain` 0.61 / 0.90 | - |
| stemmed keywords | `stem` 0.63 / 0.88 | `n_ft` 0.62 / 0.88 |
| keywords + a hand-written synonym list | `syn` 0.71 / 0.89 | - |
| word links learned from the rewrites | - | `n_co` 0.59 / 0.88 |
| nearest words by embedding | - | `n_term` 0.69 / 0.89 |
| **vectors** | `dense` 0.90 / 0.95 | `n_vec` 0.90 / 0.95 |
| everything combined | `hybrid` 0.80 / 0.95 | `n_hyb` 0.75 / 0.95 |

On the hard questions (tables above) the two databases are identical in the default mode and within 0.02 F1 of each other for a plain search.

**Speed, size and effort** (one Mac, loaded corpus):

| | OpenSearch | Neo4j |
|---|---|---|
| one keyword search / one vector search (median) | 2 ms / 11 ms | 2 ms / 11 ms |
| one combined search (median) | 20 ms | 43 ms (8-15 Cypher round trips) |
| what is stored | 120 documents, 437 KB | 2,345 nodes and 326,681 relationships, 315,148 of them word links |
| code written for it | `opensearch_stories.py`, about 190 lines | `neo4j_stories.py`, about 400 lines |

Why it comes out like this:

- **The accuracy ties because the parts that do the work are the same.** The vectors come from the same model and are compared the same way, so both
  databases return the same ranking (a test pins it): vectors alone put the right story in the top 3 for all 28 queries. Both keyword searches use
  Lucene's English analysis, so they match within 0.01. The combining step is our own code in both.
- **The graph-specific methods did not earn their keep.** `n_co` is the weakest method in the table. The word links join words that occur in the same
  story, and in 20 stories nearly every word shares a story with hundreds of others, so a two-word question is widened by about 270 words. Learning
  the links from the rewrites helps (0.59 to 0.67 when the rewrites are used as a thesaurus) but stays well under plain vectors. The links also grow
  with the square of a story's vocabulary, which would not survive real filings.
- **OpenSearch needed far less from us.** Its text scoring, analyzers, synonyms and nearest-neighbour search are built in. Neo4j brings vector and
  full-text indexes, but its tokenizer, text scoring, word links, word-vector index and combining step are all code in this repository. That is also
  why its combined search needs several round trips; at 20 stories the difference is invisible.

**Where Neo4j would win.** Questions about relationships: who succeeded whom, which company acquired which, what two firms have in common. The
stories contain exactly that structure (Reyes handing over to Nair, Verita and Ardent settling, Fairmont's four acquisitions) but no question here asks
for it, so Neo4j never got its best case. A sensible design may be OpenSearch to find the passages and Neo4j to hold the people, companies and deals
extracted from them. A fair test would be a set of hand-judged relationship questions run on both; it does not exist yet.

Caveats: 20 stories and 28 queries. One query moves the score by 0.03 to 0.1, so gaps like hybrid 0.80 against 0.75 are noise. This tests one use of
a graph (word links inside the text), not graph databases in general. The timings are one run on one laptop, indicative only.

## Tests

```bash
pytest -v                                  # everything: about two minutes
pytest test/test_corpus.py test/test_retrieval.py test/test_intent.py test/test_pipeline.py test/test_readme.py     # no database needed: seconds
pytest test/test_hard_questions.py -v      # the hard-question benchmark, below
```

| File | What it checks | Needs |
|---|---|---|
| `test_corpus.py` | the story-file loader's rules, the real story files, `queries.json` | nothing |
| `test_retrieval.py` | rank fusion, versions to stories, picking arms, the tiered `Query`, hit@3 / MRR | nothing |
| `test_intent.py` | the LLM's analysis and scores: what is accepted, what degrades to "search as typed", the prompts | nothing |
| `test_pipeline.py` | the one-shot concept-priority logic, with a scripted search and a scripted LLM | nothing |
| `test_readme.py` | the files and `ask_stories.py` commands this README names are real | nothing |
| `test_data_load.py` | loading both databases from a 3-story corpus: refusals, reload, interrupted loads | both databases |
| `test_opensearch_stories.py` | analyzers, index names, the findings on OpenSearch | OpenSearch |
| `test_neo4j_stories.py` | tokens, the graph, the comparison matrix as assertions | both |
| `test_ask_stories.py` | helpers, the LLM client with a fake HTTP layer, the whole question-to-answer flow with a scripted LLM | both |
| `test_hard_questions.py` | the hard-question benchmark, replaying recorded LLM replies | both |

No test calls an LLM. The only code that does is `test/record_hard_questions.py`, below.

## Hard questions (does the LLM rewriter + reranker earn its keep?)

`test/hard_questions.json` has 13 questions a plain search handles badly: typos, two topics in one sentence, wrong-polarity and wrong-cause traps,
ambiguous words ("bank", "duties"), vocabulary gaps. Each has the stories a good answer must be built from, judged by hand (plus, for one question,
a story that is acceptable either way and is not counted). `test/test_hard_questions.py` runs each mode on both databases and scores what is
shown. The tables below are **generated** from the recorded results by `python test/hard_report.py --write`, and a test fails if they drift.

<!-- hard-questions:start -->
**The modes, and the command that runs each** (the tests build their options from exactly these arguments, with the default `--os-arm hybrid --neo-arm n_hyb --top 5`):

| Mode | Run the app with | LLM steps |
|---|---|---|
| no LLM, one search | `python ask_stories.py --no-llm` | none |
| rewrite only | `python ask_stories.py --no-rerank` | analysis |
| rewrite + rerank (default) | `python ask_stories.py` | analysis + scoring |

**Scores** (mean over the 13 questions and both databases; precision = how much of what is shown is right, recall = how much of what is right is shown, MRR = how high the first right story is):

| Mode | precision | recall | F1 | MRR |
|---|---|---|---|---|
| no LLM, one search | 0.32 | 0.96 | 0.47 | 0.92 |
| rewrite only | 0.35 | 1.00 | 0.50 | 0.94 |
| rewrite + rerank (default) | 1.00 | 1.00 | 1.00 | 1.00 |

**Confusion matrix of each mode**, pooled over 13 questions x 2 databases = 26 runs (every run asks, for each of the 20 stories: should it be shown, and was it?):

`python ask_stories.py --no-llm`: no LLM, one search

| | shown | not shown |
|---|---|---|
| **should be shown** | 41 (true positive) | 3 (false negative) |
| **should not be shown** | 87 (false positive) | 387 (true negative) |

Pooled precision 0.32, recall 0.93: 87 wrong stories shown for 41 right ones.

`python ask_stories.py --no-rerank`: rewrite only

| | shown | not shown |
|---|---|---|
| **should be shown** | 44 (true positive) | 0 (false negative) |
| **should not be shown** | 84 (false positive) | 390 (true negative) |

Pooled precision 0.34, recall 1.00: 84 wrong stories shown for 44 right ones.

`python ask_stories.py`: rewrite + rerank (default)

| | shown | not shown |
|---|---|---|
| **should be shown** | 44 (true positive) | 0 (false negative) |
| **should not be shown** | 0 (false positive) | 474 (true negative) |

Pooled precision 1.00, recall 1.00: 0 wrong stories shown for 44 right ones.

**Per database** (same counts, split):

| Mode | Database | TP | FN | FP | TN |
|---|---|---|---|---|---|
| no LLM, one search | OpenSearch | 20 | 2 | 44 | 193 |
| no LLM, one search | Neo4j | 21 | 1 | 43 | 194 |
| rewrite only | OpenSearch | 22 | 0 | 42 | 195 |
| rewrite only | Neo4j | 22 | 0 | 42 | 195 |
| rewrite + rerank (default) | OpenSearch | 22 | 0 | 0 | 237 |
| rewrite + rerank (default) | Neo4j | 22 | 0 | 0 | 237 |

**Question by question**: wrong stories shown (FP) and right stories missing (FN), added up over both databases; `exact` = every right story and nothing else:

| Question | Kind | no LLM, one search | rewrite only | rewrite + rerank (default) |
|---|---|---|---|---|
| h01 | typo and a condition no passage can meet | FP 6, FN 0 | FP 6, FN 0 | exact |
| h02 | two topics, vocabulary gap on both | FP 6, FN 0 | FP 6, FN 0 | exact |
| h03 | wrong-polarity trap | FP 8, FN 0 | FP 8, FN 0 | exact |
| h04 | right-subject, wrong-cause traps | FP 8, FN 0 | FP 8, FN 0 | exact |
| h05 | abstract category over several stories | FP 4, FN 2 | FP 2, FN 0 | exact |
| h06 | ambiguous word | FP 8, FN 0 | FP 8, FN 0 | exact |
| h07 | vocabulary gap and an ambiguous word | FP 8, FN 0 | FP 8, FN 0 | exact |
| h09 | two topics in the same field | FP 6, FN 0 | FP 6, FN 0 | exact |
| h11 | two topics, vocabulary gap on both | FP 6, FN 0 | FP 6, FN 0 | exact |
| h12 | two topics, one of them ambiguous | FP 6, FN 0 | FP 6, FN 0 | exact |
| h13 | held out: two topics, a same-field trap | FP 6, FN 0 | FP 6, FN 0 | exact |
| h14 | held out: two topics, vocabulary gap on both | FP 7, FN 1 | FP 6, FN 0 | exact |
| h15 | held out: several typos | FP 8, FN 0 | FP 8, FN 0 | exact |
<!-- hard-questions:end -->

How to read the confusion matrix: each run asks, for every story, "should it be shown for this question?" and notes whether it was. The two
numbers that matter are the false positives (noise a person has to read past) and the false negatives (an answer the person never sees).

What the tables show: a plain search already finds most of what it should (few false negatives) but cannot stop: its false positives are the padding of a
fixed-size list, plus traps such as the wrong-polarity story ranked first for "benefited from tariffs" and a finance story shown for a river-bank
question. The rewriter's concept queries find the stories a plain search sometimes misses (h05, h14); the scoring is what removes the false positives.

Limits: 13 questions, 20 stories, one judge model at temperature 0; a corpus this small makes the cut easy for an LLM, so a perfect score here is
not accuracy on real filings. Questions h13-h15 were held out when the design was written, but h13 and h14 then exposed a real flaw (conditions
were applied to the wrong subject; fixed in `intent.score_prompt`), so only h15 is untouched. Two further questions were dropped to fit the API budget.
A staged multi-step variant (core words first, widen only when too few strong results) scored worse than a single scoring pass on the first version
of this benchmark (F1 0.59 against 0.85) and was removed: a search with the LLM's core words alone often missed the right stories (h07, h11, h12),
and in 3 of 26 scoring calls the model returned the wrong number of scores (scores are now keyed by story id, not listed by position).

The tests replay LLM replies recorded once from the real model (`test/cassettes/hard_questions.json`), so they are deterministic, need no key and
cost nothing. If a prompt, the corpus or the candidates change, a recorded reply is no longer found and the test says so. Recording is the only thing
that calls OpenRouter (model `anthropic/claude-haiku-5.5`); it stops before it would pass `--max-calls` and counts every call ever made for the file,
including replies since discarded:

```bash
python test/record_hard_questions.py --analysis-only     # one call per question: how the model reads it
python test/record_hard_questions.py --only h01,h03      # just those questions
python test/record_hard_questions.py --max-calls 100     # everything the modes need; replies already recorded are free
python test/hard_report.py --write                       # refresh the tables above from the recorded replies
```

## Files

| File | Purpose |
|---|---|
| `stories/` | the corpus (20 stories x 6 files), the test queries (`queries.json`) and its README |
| `corpus.py` | reads and validates `stories/` and its queries, fingerprints the corpus |
| `data_load.py` | loads both databases from the files (`--reload`, `--status`) |
| `ask_stories.py` | the interactive two-database script (LLM client, the question-to-answer flow, display, `--debug`) |
| `opensearch_stories.py`, `neo4j_stories.py` | index / graph building and every search method, per database |
| `retrieval.py` | what the two share: embedding model, rank fusion, versions to stories, picking arms, hit@3 / MRR, the tiered `Query` |
| `intent.py` | the LLM's analysis of a question and its 0-3 scores: prompts and parsers, no services |
| `pipeline.py` | the one-shot search with concept priority: one search per concept, score, cut, take turns (search and LLM are passed in) |
| `test/compare_stories.py` | the comparison table (a report, not pass/fail; `test_neo4j_stories.py` reuses its scenario runner) |
| `test/hard_questions.json`, `test/cassettes/hard_questions.json` | the hard questions with their right answers, and the recorded LLM replies |
| `test/hard_eval.py`, `test/llm_cassette.py` | the benchmark's modes, scoring and confusion matrix; the replaying (and, when asked, recording) LLM stand-in |
| `test/record_hard_questions.py`, `test/hard_report.py` | record the replies (calls OpenRouter); generate the README tables (does not) |
| `test/conftest.py`, `pytest.ini` | the shared embedding-model fixture and the `Test` Neo4j namespace; pytest's settings |
| `requirements.txt`, `.env.example`, `LICENSE` | dependencies; the settings template; MIT |
