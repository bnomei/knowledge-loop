# ideation-loop

[![Git Tag](https://img.shields.io/github/v/tag/bnomei/ideation-loop?sort=semver)](https://github.com/bnomei/ideation-loop/tags)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/badge/packaging-uv-2E3440?logo=uv)](https://docs.astral.sh/uv/)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Discord](https://flat.badgen.net/badge/discord/bnomei?color=7289da&icon=discord&label)](https://discordapp.com/users/bnomei)
[![Buymecoffee](https://flat.badgen.net/badge/icon/donate?icon=buymeacoffee&color=FF813F&label)](https://www.buymeacoffee.com/bnomei)

`ideation-loop` is a local ideation-first exploration loop runner built on
the OpenAI API. It keeps a persistent "world state" in JSON, asks the model
for follow-up questions, drafts structured answers, scores them for usefulness,
grounding, reuse, and overclaiming, deduplicates similar content, and renders
a lineage graph of the surviving artifacts.

This repository is for running focused ideation and exploration loops over
time. A single state file represents one exploration.

This is a script-first local `uv` app, not a published PyPI package. Git tags
mark source snapshots for the repository and tracked examples.

## How It Works

```mermaid
flowchart TD
    start[Run main.py] --> load[Load or create state JSON]
    load --> context[Summarize strongest artifacts and registry concepts]
    context --> questions[Generate exploit and explore questions]
    questions --> draft[Draft one structured artifact per question]
    draft --> review[Score usefulness, reuse, grounding, overclaim, and dedup]
    review --> keep{Artifact survives?}
    keep --> update[Append artifact and update registry and reuse]
    keep --> drop[Drop near-clone or weak draft]
    update --> save[Write updated state JSON]
    drop --> save
    save --> graph[Optionally render lineage graph PNG]
    graph --> next[Stop or run next iteration]
```

The loop is optimized for disciplined ideation, not truth-verification. It is
best used to generate better frames, mechanisms, scenario ideas, and
next-check questions rather than to build a definitive research archive.

By default:

- the script runs `10` iterations
- each iteration generates `3` new questions
- state is written to `world_state.json`
- the graph is written to `lineage_graph.png`

## What A Run Produces

The canonical output is the updated state JSON file. Stdout is only progress
logging, and the graph PNG is only a visualization layer.

After a normal run you get:

- an updated state file with the latest `iteration`, `artifacts`, `registry`, and optional `seed`
- optional graph output at `lineage_graph.png` or `path/to/state_lineage_graph.png`
- console lines showing accepted or rejected artifacts for that run

The runner does not emit a separate report object. If you want downstream
automation or agent analysis, use the state JSON as the source of truth.

## What A "World State" Is

A world state is the JSON file that stores one running exploration.

It contains:

- `iteration`: the current iteration count
- `artifacts`: the surviving question and answer records plus their scores
- `registry`: reusable definitions discovered during the run
- optional `seed`: topic guidance for that exploration

Newer runs may also persist extra epistemic fields on artifacts, such as
evidence type, evidence strength, assumptions, a competing hypothesis, and a
main failure case. Older state files without these fields still load normally.
Older state files can also be rejudged with the current scoring logic by using
`--rejudge-existing`.

Important behavior:

- a world state is a working set, not a permanent archive of everything ever generated
- the loop prunes artifacts over time, so low-value or displaced artifacts may disappear
- if a state has a stored `seed`, future runs of that same state continue using it
- `explore` questions open new frames, while `exploit` questions deepen or stress-test the strongest current line

## Inspect Results With jq

Because the output is structured JSON, `jq` is the ideal way to slice it for
shell workflows and AI agents. Most agents should read a focused JSON view,
not the entire state file.

Basic run summary:

```bash
jq '{
  iteration,
  artifact_count: (.artifacts | length),
  registry_count: (.registry | length),
  seed_topic: (.seed.topic // null)
}' world_state.json
```

Top artifacts by fate:

```bash
jq '[
  .artifacts
  | sort_by(.fate)
  | reverse
  | .[:5]
  | .[]
  | {
      id,
      mode: .question.mode,
      question: .question.text,
      fate,
      novelty,
      grounding_score,
      overclaim_penalty
    }
]' world_state.json
```

Compact JSON slice for an AI agent:

```bash
jq -c '{
  iteration,
  seed,
  top_artifacts: (
    .artifacts
    | sort_by(.fate)
    | reverse
    | .[:8]
    | map({
        id,
        question: .question.text,
        answer,
        fate,
        evidence_type,
        evidence_strength,
        claims: [.claims[].text],
        open_questions
      })
  ),
  registry: (
    .registry
    | map({
        id,
        name,
        meaning,
        status
      })
  )
}' world_state.json
```

Use `jq -c` when you want one compact JSON object that an agent can consume
directly without extra cleanup.

## Requirements

- `uv`
- Python `3.9+`
- an OpenAI API key

## Release Model

- use `uv sync --frozen` for reproducible local installs
- treat `pyproject.toml` as the source of truth for the app version
- treat Git tags such as `v0.1.0` as repository release snapshots, not PyPI releases

If you do not already have `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Quickstart

1. Install dependencies.
2. Create an `.env` file with your API key.
3. Run the default exploration.

```bash
uv sync --frozen
cp .env.example .env
uv run --env-file .env python -u main.py
```

`uv sync --frozen` is the supported install path for this repository. The
checked-in `uv.lock` is part of the release contract for local runs and CI.

That command will:

- load or create `world_state.json`
- run `10` iterations
- update `lineage_graph.png`

## Common Workflows

### Start The Default Exploration

Use the repository-default state file:

```bash
uv run --env-file .env python -u main.py
```

### Continue An Existing Exploration

Use the same state file again.

```bash
uv run --env-file .env python -u main.py \
  --state-file runs/cross-lingual-cot-trust.json \
  --iters 3
```

That continues the existing exploration, keeps its stored seed if it has one,
and writes back to the same file.

### Pull In Current Information With OpenAI Web Search

Web search is optional and off by default. When enabled, only question
generation and artifact drafting may use OpenAI's built-in `web_search` tool.
The scoring and judging passes stay local to the current state.

Use it when the topic benefits from fresh facts, current events, or recent
developments:

```bash
uv run --env-file .env python -u main.py \
  --state-file runs/current-topic.json \
  --web-search \
  --iters 3
```

If you want to constrain search to specific sources:

```bash
uv run --env-file .env python -u main.py \
  --state-file runs/current-topic.json \
  --web-search \
  --web-search-domain www.anthropic.com \
  --web-search-domain openai.com \
  --iters 3
```

### Rejudge An Older State With The Current Scoring Logic

Use this when you want an older state to benefit from the current judges and
fate scoring before you continue it.

Rescore only:

```bash
uv run --env-file .env python -u main.py \
  --state-file runs/cross-lingual-cot-trust.json \
  --rejudge-existing \
  --iters 0
```

Rescore first, then continue the loop:

```bash
uv run --env-file .env python -u main.py \
  --state-file runs/cross-lingual-cot-trust.json \
  --rejudge-existing \
  --iters 3
```

This re-scores existing artifacts with the current judges and updates their
stored metrics. It does not rewrite the underlying question or answer text.
Because each existing artifact is judged again, this can add noticeable model
cost on larger states.

### Start A New Seeded Exploration

Use a separate state file so you do not disturb the default run.

```bash
uv run --env-file .env python -u main.py \
  --state-file runs/medical-calibration.json \
  --seed-topic "Trust calibration for medical advice without visible CoT" \
  --seed-goal "Prefer answer-level summaries and verification prompts over raw reasoning traces" \
  --seed-include "non-expert users" \
  --seed-include "verification scaffolds" \
  --seed-avoid "broad AGI safety"
```

### Seed From A JSON Document

Use a richer seed file when you want more explicit guidance.

```bash
uv run --env-file .env python -u main.py \
  --state-file runs/cross-lingual-cot-trust.json \
  --seed-file seed.example.json
```

See [`seed.example.json`](seed.example.json) for the expected shape.

### Initialize A Seeded State Without Running Iterations

This is useful when you want to create the state first and run it later.

```bash
uv run --env-file .env python -u main.py \
  --state-file runs/cross-lingual-cot-trust.json \
  --seed-file seed.example.json \
  --iters 0
```

### Intentionally Retarget A Populated State

By default, the runner refuses to apply a new seed to a populated unseeded
state. This is a safety guard so you do not accidentally retarget the current
`world_state.json`.

If you really want to reseed an existing populated state:

```bash
uv run --env-file .env python -u main.py \
  --state-file runs/cross-lingual-cot-trust.json \
  --seed-file seed.example.json \
  --replace-seed
```

## Files Written By The Script

For the default run:

- `world_state.json`: the persisted state for the default exploration
- `lineage_graph.png`: the lineage graph for the default exploration

For alternate state files:

- `--state-file path/to/foo.json` writes state to `path/to/foo.json`
- if `--graph-file` is omitted, the graph defaults to `path/to/foo_lineage_graph.png`
- prefer `runs/*.json` for live local explorations; `runs/` is git-ignored
- treat `states/*.json` as checked-in example snapshots, not your primary scratch area

## CLI Reference

Show the full help:

```bash
uv run --env-file .env python -u main.py --help
```

Available options:

- `--iters ITERS`
  Number of iterations to run. Default: `10`.
- `--state-file STATE_FILE`
  Load from and save to a different state file instead of `world_state.json`.
- `--graph-file GRAPH_FILE`
  Write the graph to a specific PNG path instead of the derived default.
- `--git`
  Commit state-file updates after each save.
- `--no-graph`
  Skip graph rendering.
- `--quiet`
  Suppress routine progress logs and keep only errors plus final save paths.
- `--redact-output`
  Replace question text in progress logs and graph labels with stable redacted
  ids.
- `--seed-file SEED_FILE`
  Load a JSON seed document and persist it into the selected state.
- `--seed-topic SEED_TOPIC`
  Provide a seed topic inline from the CLI.
- `--seed-goal SEED_GOAL`
  Add a seed goal inline from the CLI.
- `--seed-include TEXT`
  Repeatable hint for concepts or subtopics to prefer.
- `--seed-avoid TEXT`
  Repeatable hint for concepts or subtopics to avoid.
- `--seed-question TEXT`
  Repeatable starter question to bias the exploration.
- `--replace-seed`
  Allow replacing or adding a seed on an already populated state file.
- `--rejudge-existing`
  Re-score existing artifacts in the selected state with the current judges
  before running iterations.
- `--web-search`
  Allow question and artifact generation to use OpenAI's built-in web search
  tool. Off by default.
- `--web-search-domain TEXT`
  Repeatable allowed-domain filter for built-in web search. Requires
  `--web-search`.

## Seed File Format

A seed file is JSON with this shape:

```json
{
  "topic": "Trust calibration for medical advice without visible chain-of-thought",
  "goal": "Explore brief answer-level summaries and verification prompts as safer alternatives to visible reasoning traces.",
  "include": ["non-expert users", "verification scaffolds", "uncertainty cues"],
  "avoid": ["broad AGI safety", "general philosophy of mind"],
  "seed_questions": [
    "Do short contrastive summaries calibrate trust better than full chain-of-thought?",
    "Which verification prompts actually increase external checking in safety-relevant tasks?"
  ],
  "seed_definitions": [
    {
      "name": "verification scaffold",
      "meaning": "A lightweight interface element that helps a user check an answer with an external calculation, source, or checklist."
    }
  ]
}
```

Only `topic` is strictly required. The rest are optional guidance fields.

## Configuration

Runtime model configuration comes from environment variables:

- `OPENAI_API_KEY`
  Required. Used for all OpenAI API calls.
- `OPENAI_MODEL`
  Optional. Defaults to `gpt-5.1`.
- `OPENAI_EMBED_MODEL`
  Optional. Defaults to `text-embedding-3-small`.

The script does not load `.env` by itself. Use `uv run --env-file .env ...` if
you want `.env` values injected into the process. Startup now validates this
configuration early and fails before the first model call if the required
variables are missing or empty.

## Costs And Runtime Characteristics

Each question currently triggers multiple model calls:

- one structured draft generation
- two positive scoring passes
- one adversarial scoring pass
- embedding calls for deduplication when needed

That means longer runs can consume a meaningful number of tokens and API
requests.

Also note:

- the loop keeps an in-memory embedding cache for the current process
- the script rewrites the full state JSON on each save
- graph rendering imports `matplotlib` and `networkx`, so first runs may spend a
  moment building local caches
- use `--quiet` in shared shells or CI when you do not want routine prompt text
  echoed to stdout
- use `--redact-output` when you want logs and graph labels to stay traceable
  without exposing question text

## Notes

- Use `python -u` if you want live progress lines while the loop is running.
- A seed is stored inside the selected state file, so you do not need to pass
  the seed again when continuing that exploration.
- The default behavior is intentionally conservative: the existing
  [`world_state.json`](world_state.json)
  is not reseeded unless you explicitly opt in with `--replace-seed`.
