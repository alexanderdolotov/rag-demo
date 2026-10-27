# Story corpus

The single source both databases are loaded from (`python data_load.py`). 20 stories, each in its own folder:

```
stories/<story_id>/original.txt        the original - this file name is what marks it as the original
stories/<story_id>/rewrite-1.txt       five rewrites of it, in different words ... rewrite-5.txt
```

Plain UTF-8 text, paragraphs separated by blank lines. The folder name is the story id (lowercase letters, digits, underscore).
`corpus.py` refuses any other file name, an empty file, a missing original or no rewrites. Editing a file changes the corpus
fingerprint, so `ask_stories.py` / `compare_stories.py` will tell you to run `data_load.py --reload`.

## The five rewrites (same facts, different words)

| File | Style |
|---|---|
| `rewrite-1.txt` | plain-language retelling with everyday synonyms |
| `rewrite-2.txt` | formal / technical register |
| `rewrite-3.txt` | first person or spoken voice |
| `rewrite-4.txt` | short summary (about 50-70 words) |
| `rewrite-5.txt` | outcome first, different order |

Originals are 150-260 words, so a whole version fits the embedding model's window (set to 512 tokens) and needs no chunking; real filings will.

## The stories

| Group | Stories |
|---|---|
| Animals and nature | `frog_log`, `frog_still`, `river_bank`, `mountain_hike` |
| Everyday | `bread`, `lost_dog`, `rocket`, `village_customs`, `job_duties` |
| Company news (8-K / earnings style, fictional companies) | `tariff_margin` (refund of duties lifts margin), `tariff_hurt` (new duties squeeze margin), `stock_jump`, `bank_rates`, `buyback`, `cyber_incident`, `ceo_departure`, `supply_chain`, `guidance_cut`, `litigation`, `regional_bank` |

Deliberate traps for the retrieval methods: `job_duties` (a watchman's *duties*) and `village_customs` (*customs* as traditions) share
words with the import-duty stories; `supply_chain` is a cost squeeze that is not tariffs; `river_bank` and `regional_bank` share "bank"
with `bank_rates`; `frog_still` is about a frog that does *not* jump. The originals deliberately avoid the words their rewrites use
(for example the `frog_log` original never says toad, hopped or trunk), which is what makes the vocabulary gap measurable;
`test/test_corpus.py` guards that.

Everything here is invented for testing; any resemblance to a real company or person is accidental.
