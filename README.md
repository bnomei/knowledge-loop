# ideation-loop

[![Git Tag](https://img.shields.io/github/v/tag/bnomei/ideation-loop?sort=semver)](https://github.com/bnomei/ideation-loop/tags)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/badge/packaging-uv-2E3440?logo=uv)](https://docs.astral.sh/uv/)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Discord](https://flat.badgen.net/badge/discord/bnomei?color=7289da&icon=discord&label)](https://discordapp.com/users/bnomei)
[![Buymecoffee](https://flat.badgen.net/badge/icon/donate?icon=buymeacoffee&color=FF813F&label)](https://www.buymeacoffee.com/bnomei)

`ideation-loop` is a local, stateful ideation runner built on the OpenAI API.
It keeps one exploration in a JSON file, proposes follow-up questions, drafts
structured answers, scores them, deduplicates weak repeats, and optionally
renders a lineage graph.

Use it when you want:

- an exploration that improves over multiple runs instead of one-shot prompting
- reusable structured output in JSON instead of a pile of chat transcripts
- something easy to inspect with `jq` or feed into another AI agent

It is not built for:

- fact-verified research archives
- polished narrative reports
- retrieval over large document corpora

This is a script-first local `uv` app, not a published PyPI package. Git tags
mark repository snapshots, not PyPI releases.

## Quickstart

Needs Python `3.9+`, `uv`, and an OpenAI API key.

Fastest path:
[download the current `main` branch zip](https://github.com/bnomei/ideation-loop/archive/refs/heads/main.zip),
then extract `main.py`.

```bash
curl -L -o ideation-loop-main.zip \
  https://github.com/bnomei/ideation-loop/archive/refs/heads/main.zip
unzip -j ideation-loop-main.zip '*/main.py'
export OPENAI_API_KEY="your_openai_api_key_here"
uv run \
  --with "openai>=2.30,<3" \
  --with "pydantic>=2.12,<3" \
  --with "matplotlib>=3.9,<4" \
  --with "networkx>=3.2,<4" \
  python -u main.py
```

Repo workflow:

```bash
uv sync --frozen
cp .env.example .env
uv run --env-file .env python -u main.py
```

If you do not already have `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

`uv sync --frozen` is the supported install path for this repository. The
checked-in `uv.lock` is part of the supported local and CI environment.

## What You Get

After a normal run you get:

- `world_state.json`: the canonical output for the exploration
- `lineage_graph.png`: an optional visualization of artifact relationships
- stdout progress lines showing accepted or rejected artifacts

For non-default runs:

- use `--state-file runs/foo.json` to keep live work in the git-ignored `runs/` directory
- if `--graph-file` is omitted, the graph defaults to `runs/foo_lineage_graph.png`
- treat `states/*.json` as checked-in examples, not your main scratch space

The state file is the source of truth. Stdout is just progress logging, and
the PNG is only a visualization layer.

Typical shape of the saved JSON:

```json
{
  "iteration": 10,
  "artifacts": [
    {
      "question": {
        "text": "What mechanism would make this idea robust outside lab conditions?",
        "mode": "exploit"
      },
      "answer": "A stronger version would rely on ...",
      "fate": 0.81,
      "claims": [
        {
          "text": "Robustness depends on explicit verification steps."
        }
      ]
    }
  ],
  "registry": [
    {
      "name": "verification scaffold",
      "meaning": "A lightweight structure that helps a user check an answer."
    }
  ]
}
```

## Common Tasks

### Continue A Run

```bash
uv run --env-file .env python -u main.py \
  --state-file runs/cross-lingual-cot-trust.json \
  --iters 3
```

Use the same state file again to continue the same exploration. If that state
already has a stored seed, it keeps using it.

### Start A Seeded Run

```bash
uv run --env-file .env python -u main.py \
  --state-file runs/medical-calibration.json \
  --seed-topic "Trust calibration for medical advice without visible CoT" \
  --seed-goal "Prefer answer-level summaries and verification prompts over raw reasoning traces" \
  --seed-include "non-expert users" \
  --seed-include "verification scaffolds" \
  --seed-avoid "broad AGI safety"
```

### Seed From JSON

```bash
uv run --env-file .env python -u main.py \
  --state-file runs/cross-lingual-cot-trust.json \
  --seed-file seed.example.json
```

### Use OpenAI Web Search

Web search is optional and off by default. When enabled, only question
generation and artifact drafting may use OpenAI's built-in `web_search` tool.
The judging and scoring passes still operate against the current state.

```bash
uv run --env-file .env python -u main.py \
  --state-file runs/current-topic.json \
  --web-search \
  --iters 3
```

Restrict web search to specific domains:

```bash
uv run --env-file .env python -u main.py \
  --state-file runs/current-topic.json \
  --web-search \
  --web-search-domain www.anthropic.com \
  --web-search-domain openai.com \
  --iters 3
```

### Rejudge An Existing State

Rescore an older state with the current scoring logic without generating new
artifacts:

```bash
uv run --env-file .env python -u main.py \
  --state-file runs/cross-lingual-cot-trust.json \
  --rejudge-existing \
  --iters 0
```

Rescore first, then continue:

```bash
uv run --env-file .env python -u main.py \
  --state-file runs/cross-lingual-cot-trust.json \
  --rejudge-existing \
  --iters 3
```

### Initialize A Seeded State Without Running

```bash
uv run --env-file .env python -u main.py \
  --state-file runs/cross-lingual-cot-trust.json \
  --seed-file seed.example.json \
  --iters 0
```

### Replace The Seed On A Populated State

By default, the runner refuses to retarget a populated unseeded state. Use
`--replace-seed` only when you want to intentionally repurpose an existing
state file.

```bash
uv run --env-file .env python -u main.py \
  --state-file runs/cross-lingual-cot-trust.json \
  --seed-file seed.example.json \
  --replace-seed
```

## Read Results With jq

Because the canonical output is JSON, `jq` is the easiest way to inspect runs
from the shell. It is also the best handoff format for AI agents: instead of
feeding an agent the full state file, slice out the parts it actually needs.

Basic summary:

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

Use `jq -c` when you want one compact JSON object that another tool or agent
can consume directly.

## How It Works

High-level flow:

```text
state.json
  -> summarize strongest artifacts and registry concepts
  -> generate a small batch of exploit and explore questions
  -> draft one structured artifact per question
  -> score usefulness, reuse, grounding, overclaim, and dedup
  -> keep survivors and drop near-clones
  -> write updated state.json
  -> optionally render lineage_graph.png
```

The loop is optimized for disciplined ideation, not truth-verification. It is
best used to generate better frames, mechanisms, scenario ideas, and
next-check questions rather than to build a definitive research archive.

By default:

- the script runs `10` iterations
- each iteration generates `3` new questions
- the default state file is `world_state.json`
- the default graph file is `lineage_graph.png`

## What A World State Is

A world state is the JSON file that stores one running exploration.

Core fields:

- `iteration`: current iteration count
- `artifacts`: surviving question and answer records plus scores and metadata
- `registry`: reusable definitions discovered during the run
- `seed`: optional topic guidance stored with the state

Important behavior:

- a world state is a working set, not a permanent archive of everything ever generated
- the loop prunes artifacts over time, so weaker artifacts may disappear
- if a state has a stored seed, future runs of that same state continue using it
- `explore` questions open new frames, while `exploit` questions deepen or stress-test the strongest current line

Newer runs may also persist extra epistemic fields such as evidence type,
evidence strength, assumptions, a competing hypothesis, and a main failure
case. Older state files without these fields still load normally.

## CLI Essentials

Show the full help:

```bash
uv run --env-file .env python -u main.py --help
```

The most important flags are:

- `--iters ITERS`
  Number of iterations to run. Default: `10`.
- `--state-file PATH`
  Load from and save to a different state file instead of `world_state.json`.
- `--seed-file PATH`
  Load a JSON seed document and persist it into the selected state.
- `--seed-topic TEXT`
  Provide a seed topic inline from the CLI.
- `--rejudge-existing`
  Re-score existing artifacts in the selected state before running iterations.
- `--web-search`
  Allow question and artifact generation to use OpenAI's built-in web search tool.
- `--web-search-domain TEXT`
  Restrict built-in web search to specific domains. Requires `--web-search`.
- `--no-graph`
  Skip graph rendering.
- `--quiet`
  Suppress routine progress logs and keep only errors plus final save paths.
- `--redact-output`
  Replace question text in progress logs and graph labels with stable redacted ids.
- `--git`
  Commit state-file updates after each save.

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
you want `.env` values injected into the process. Startup validates this
configuration early and fails before the first model call if required values
are missing or empty.

## Costs And Runtime Notes

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
- graph rendering imports `matplotlib` and `networkx`, so first runs may spend a moment building local caches
- use `--quiet` in shared shells or CI when you do not want routine prompt text echoed to stdout
- use `--redact-output` when you want logs and graph labels to stay traceable without exposing question text

## Project Notes

- use `python -u` if you want live progress lines while the loop is running
- a seed is stored inside the selected state file, so you do not need to pass it again when continuing that exploration
- the default behavior is intentionally conservative: an existing `world_state.json` is not reseeded unless you explicitly opt in with `--replace-seed`
- treat `pyproject.toml` as the source of truth for the app version
- treat Git tags such as `v0.1.1` as repository snapshots, not PyPI releases
