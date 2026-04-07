"""Minimal ideation and exploration loop with explicit stages and persisted lineage.

High-level flow:
1. Summarize the best current artifacts and definitions into a compact context.
2. Ask the model for a small batch of follow-up questions.
3. Draft one structured artifact per question.
4. Deduplicate questions and claims before they enter long-lived state.
5. Score the draft for usefulness and reuse, then apply an adversarial penalty.
6. Materialize the draft as an artifact, update lineage/registry state, and
   persist the world state plus an optional lineage graph.

Most future changes land in three places:
- `LoopConfig` for runtime knobs and thresholds.
- `process_question()` for the per-question pipeline.
- `run_iteration()` for iteration-level orchestration.
"""

import argparse
import json
import math
import os
import re
import tempfile
import uuid
import hashlib
import subprocess
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Literal, Optional, TypeVar

import matplotlib.pyplot as plt
import networkx as nx
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError

# -------------------------
# Config
# -------------------------


@dataclass(frozen=True)
class LoopConfig:
    """Centralized knobs for prompt sizes, thresholds, and output files."""

    model: str = os.getenv("OPENAI_MODEL", "gpt-5.1")
    embed_model: str = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    state_file: Path = Path("world_state.json")
    graph_file: Path = Path("lineage_graph.png")
    max_population: int = 50
    top_context: int = 8
    questions_per_iteration: int = 3
    explore_questions_per_iteration: int = 1
    lineage_context_artifacts: int = 20
    reference_context_limit: int = 200
    registry_context_limit: int = 50
    context_leader_slots: int = 3
    context_explore_slots: int = 2
    context_underused_slots: int = 2
    context_frame_breaker_slots: int = 1
    registry_anchor_slots: int = 16
    registry_recent_slots: int = 12
    registry_underused_slots: int = 12
    question_embed_similarity: float = 0.90
    claim_embed_similarity: float = 0.92
    question_lexical_prefilter: float = 0.45
    claim_lexical_prefilter: float = 0.45
    artifact_dedup_reject_threshold: float = 0.85
    novelty_bonus_weight: float = 0.12
    novelty_dominant_neighbors: int = 3
    novelty_corpus_neighbors: int = 5
    explore_retention_floor: int = 10
    explore_grounding_weight: float = 0.03
    exploit_grounding_weight: float = 0.08
    explore_overclaim_weight: float = 0.18
    exploit_overclaim_weight: float = 0.28
    registry_match_threshold: float = 0.72
    registry_meaning_conflict_threshold: float = 0.45
    registry_name_weight: float = 0.6
    registry_meaning_weight: float = 0.4
    hazard_min_artifacts: int = 20
    hazard_every_iterations: int = 5
    hazard_drop_leaders: int = 3
    web_search_max_tool_calls: int = 3


CONFIG = LoopConfig()

MODEL = CONFIG.model
EMBED_MODEL = CONFIG.embed_model
STATE_FILE = CONFIG.state_file
GRAPH_FILE = CONFIG.graph_file
WEB_SEARCH_ENABLED = False
WEB_SEARCH_ALLOWED_DOMAINS: tuple[str, ...] = ()

MAX_POP = CONFIG.max_population
TOP_CONTEXT = CONFIG.top_context
QUESTION_BATCH_SIZE = CONFIG.questions_per_iteration

# dedup thresholds
QUESTION_EMBED_SIM = CONFIG.question_embed_similarity
CLAIM_EMBED_SIM = CONFIG.claim_embed_similarity
QUESTION_LEXICAL_PREFILTER = CONFIG.question_lexical_prefilter
CLAIM_LEXICAL_PREFILTER = CONFIG.claim_lexical_prefilter

client: Optional[OpenAI] = None


def new_id() -> str:
    return str(uuid.uuid4())


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_client() -> OpenAI:
    global client
    if client is None:
        client = OpenAI()
    return client


def default_graph_file_for(state_file: Path) -> Path:
    default_state = LoopConfig.state_file
    default_graph = LoopConfig.graph_file
    if state_file == default_state:
        return default_graph
    return state_file.with_name(f"{state_file.stem}_lineage_graph.png")


def configure_runtime_paths(
    state_file: Optional[Path] = None,
    graph_file: Optional[Path] = None,
) -> None:
    """Update the runtime output paths without changing the default behavior."""
    global CONFIG, STATE_FILE, GRAPH_FILE

    chosen_state = (state_file or CONFIG.state_file).expanduser()
    chosen_graph = (graph_file or default_graph_file_for(chosen_state)).expanduser()

    CONFIG = replace(CONFIG, state_file=chosen_state, graph_file=chosen_graph)
    STATE_FILE = CONFIG.state_file
    GRAPH_FILE = CONFIG.graph_file


def configure_web_search(
    enabled: bool = False,
    allowed_domains: Optional[List[str]] = None,
) -> None:
    """Set runtime web-search behavior for generation without changing defaults."""
    global WEB_SEARCH_ENABLED, WEB_SEARCH_ALLOWED_DOMAINS
    WEB_SEARCH_ENABLED = enabled
    WEB_SEARCH_ALLOWED_DOMAINS = tuple(dict.fromkeys(allowed_domains or []))


def validate_runtime_config(needs_api: bool) -> None:
    """Fail fast on missing or empty runtime configuration."""
    if not MODEL.strip():
        raise ValueError("OPENAI_MODEL must not be empty.")
    if not EMBED_MODEL.strip():
        raise ValueError("OPENAI_EMBED_MODEL must not be empty.")
    if not needs_api:
        return

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is required. "
            "The script does not auto-load .env; use "
            "'uv run --env-file .env ...' or export OPENAI_API_KEY."
        )


# -------------------------
# Base model
# -------------------------


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


StructuredSchema = TypeVar("StructuredSchema", bound=StrictModel)
EvidenceType = Literal[
    "conjecture", "mechanism", "synthesis", "mixed", "evidence-backed"
]
EvidenceStrength = Literal["unknown", "low", "mixed", "moderate", "high"]
QuestionKind = Literal[
    "gap", "refinement", "challenge", "contradiction", "generalization"
]
QuestionMode = Literal["exploit", "explore"]


# -------------------------
# DSL
# -------------------------


class Definition(StrictModel):
    name: str
    meaning: str


class RegistryDefinition(StrictModel):
    id: str = Field(default_factory=new_id)
    name: str
    meaning: str
    aliases: List[str] = Field(default_factory=list)
    fingerprint: str = ""
    status: Literal["active", "conflicted"] = "active"


class Claim(StrictModel):
    id: str = Field(default_factory=new_id)
    text: str
    confidence: float = Field(ge=0.0, le=1.0)


class Question(StrictModel):
    id: str = Field(default_factory=new_id)
    text: str
    kind: QuestionKind
    mode: QuestionMode = "exploit"


class TopicSeed(StrictModel):
    topic: str
    goal: str = ""
    include: List[str] = Field(default_factory=list)
    avoid: List[str] = Field(default_factory=list)
    seed_questions: List[str] = Field(default_factory=list)
    seed_definitions: List[Definition] = Field(default_factory=list)


class Artifact(StrictModel):
    id: str = Field(default_factory=new_id)
    created_at: str = Field(default_factory=now_utc_iso)

    question: Question
    answer: str
    definitions: List[Definition] = Field(default_factory=list)
    claims: List[Claim] = Field(default_factory=list)

    parents: List[str] = Field(default_factory=list)
    references: List[str] = Field(default_factory=list)
    referenced_definition_ids: List[str] = Field(default_factory=list)
    evidence_type: EvidenceType = "synthesis"
    evidence_strength: EvidenceStrength = "unknown"
    assumptions: List[str] = Field(default_factory=list)
    competing_hypothesis: str = ""
    main_failure_case: str = ""
    verification_targets: List[str] = Field(default_factory=list)
    open_questions: List[str] = Field(default_factory=list)

    score1: float = 0.0
    score2: float = 0.0
    adversarial: float = 0.0
    dedup: float = 0.0
    novelty: float = 0.0
    grounding_score: float = 0.0
    overclaim_penalty: float = 0.0
    reuse: int = 0
    fate: float = 0.0


class State(StrictModel):
    iteration: int = 0
    artifacts: List[Artifact] = Field(default_factory=list)
    registry: List[RegistryDefinition] = Field(default_factory=list)
    seed: Optional[TopicSeed] = None


# -------------------------
# Structured output schemas
# -------------------------


class QBatch(StrictModel):
    questions: List["PlannedQuestion"]


class PlannedQuestion(StrictModel):
    text: str
    kind: QuestionKind
    mode: QuestionMode


class Draft(StrictModel):
    question: Question
    answer: str
    definitions: List[Definition] = Field(default_factory=list)
    claims: List[Claim] = Field(default_factory=list)
    parents: List[str] = Field(default_factory=list)
    references: List[str] = Field(default_factory=list)
    evidence_type: EvidenceType = "synthesis"
    evidence_strength: EvidenceStrength = "unknown"
    assumptions: List[str] = Field(default_factory=list)
    competing_hypothesis: str = ""
    main_failure_case: str = ""
    verification_targets: List[str] = Field(default_factory=list)
    open_questions: List[str] = Field(default_factory=list)


class Score(StrictModel):
    score: float = Field(ge=0.0, le=1.0)


class Adv(StrictModel):
    penalty: float = Field(ge=0.0, le=1.0)


class GroundingReview(StrictModel):
    grounding_score: float = Field(ge=0.0, le=1.0)
    overclaim_penalty: float = Field(ge=0.0, le=1.0)
    unsupported_claims: List[str] = Field(default_factory=list)


@dataclass(frozen=True)
class DraftEvaluation:
    """Scores attached to a draft before it becomes a persisted artifact."""

    score1: float
    score2: float
    adversarial: float
    dedup: float
    novelty: float
    grounding_score: float
    overclaim_penalty: float
    reuse: int
    question_mode: QuestionMode = "exploit"

    @property
    def fate(self) -> float:
        return compute_fate(
            self.score1,
            self.score2,
            self.adversarial,
            self.dedup,
            self.novelty,
            self.grounding_score,
            self.overclaim_penalty,
            self.question_mode,
            self.reuse,
        )


# -------------------------
# Text helpers
# -------------------------

STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "to",
    "in",
    "on",
    "for",
    "with",
    "is",
    "are",
    "be",
    "by",
    "that",
    "this",
    "it",
    "as",
    "at",
    "from",
    "if",
    "then",
    "than",
    "into",
    "out",
    "up",
    "down",
    "how",
    "what",
    "why",
    "when",
}


def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    parts = [p for p in text.split() if p and p not in STOPWORDS]
    return " ".join(parts)


def lexical_similarity(a: str, b: str) -> float:
    sa = set(normalize(a).split())
    sb = set(normalize(b).split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# -------------------------
# Embeddings
# -------------------------

_embedding_cache: dict[str, List[float]] = {}


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    dot = sum(a * b for a, b in zip(v1, v2))
    n1 = math.sqrt(sum(a * a for a in v1))
    n2 = math.sqrt(sum(b * b for b in v2))
    if n1 == 0 or n2 == 0:
        return 0.0
    return dot / (n1 * n2)


def embed_texts(texts: List[str]) -> List[List[float]]:
    missing = [t for t in texts if t not in _embedding_cache]
    if missing:
        resp = get_client().embeddings.create(
            model=EMBED_MODEL,
            input=missing,
        )
        for txt, item in zip(missing, resp.data):
            _embedding_cache[txt] = item.embedding
    return [_embedding_cache[t] for t in texts]


def max_embedding_similarity(
    text: str, others: List[str], lexical_prefilter: float
) -> float:
    if not others:
        return 0.0

    candidates = [o for o in others if lexical_similarity(text, o) >= lexical_prefilter]
    if not candidates:
        return 0.0

    vectors = embed_texts([text] + candidates)
    base = vectors[0]
    return max(cosine_similarity(base, v) for v in vectors[1:])


def top_k_embedding_similarity(
    text: str,
    others: List[str],
    k: int = 3,
) -> float:
    """Average the top-k semantic similarities without lexical prefiltering."""
    if not others:
        return 0.0

    vectors = embed_texts([text] + others)
    base = vectors[0]
    similarities = sorted(
        (cosine_similarity(base, vector) for vector in vectors[1:]),
        reverse=True,
    )
    top = similarities[: max(1, min(k, len(similarities)))]
    return sum(top) / len(top)


# -------------------------
# Persistence
# -------------------------


def load_state() -> State:
    if not STATE_FILE.exists():
        return State()
    return State.model_validate_json(STATE_FILE.read_text(encoding="utf-8"))


def run_git_command(args: List[str]) -> None:
    """Run a git command and surface stderr when it fails."""
    try:
        subprocess.run(
            ["git", *args],
            check=True,
            text=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        stdout = (exc.stdout or "").strip()
        details = stderr or stdout or "unknown git error"
        raise RuntimeError(f"git {' '.join(args)} failed: {details}") from exc


def save_state(state: State, git: bool = False) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    serialized = state.model_dump_json(indent=2, exclude_none=True)
    temp_path: Optional[Path] = None

    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=STATE_FILE.parent,
            prefix=f".{STATE_FILE.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(serialized)
            temp_path = Path(handle.name)
        temp_path.replace(STATE_FILE)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()

    if git:
        run_git_command(["add", str(STATE_FILE)])
        run_git_command(
            [
                "commit",
                "-m",
                f"update {datetime.now(timezone.utc).isoformat()}",
            ]
        )


# -------------------------
# OpenAI structured call
# -------------------------


def build_web_search_tool() -> dict[str, object]:
    """Responses API tool config for optional built-in web search."""
    tool: dict[str, object] = {"type": "web_search"}
    if WEB_SEARCH_ALLOWED_DOMAINS:
        tool["filters"] = {"allowed_domains": list(WEB_SEARCH_ALLOWED_DOMAINS)}
    return tool


def llm(
    prompt: str,
    schema: type[StructuredSchema],
    tokens: int = 1000,
    allow_web_search: bool = False,
) -> StructuredSchema:
    """Call the Responses API with structured parsing and one concise retry."""
    current_prompt = prompt
    current_tokens = tokens

    for attempt in range(2):
        try:
            request: dict[str, object] = {
                "model": MODEL,
                "input": current_prompt,
                "text_format": schema,
                "max_output_tokens": current_tokens,
            }
            if allow_web_search and WEB_SEARCH_ENABLED:
                request["tools"] = [build_web_search_tool()]
                request["tool_choice"] = "auto"
                request["max_tool_calls"] = CONFIG.web_search_max_tool_calls
                request["include"] = ["web_search_call.action.sources"]

            resp = get_client().responses.parse(
                **request
            )
            if resp.output_parsed is None:
                raise ValueError(f"No parsed output returned for {schema.__name__}")
            return resp.output_parsed
        except ValidationError:
            if attempt == 1:
                raise

            current_tokens = max(current_tokens * 2, current_tokens + 800)
            current_prompt = (
                prompt
                + "\n\nRetry instruction: return the same schema more concisely. "
                + "Keep strings short and avoid unnecessary detail."
            )

    raise RuntimeError(f"Unreachable llm retry state for {schema.__name__}")


# -------------------------
# Context
# -------------------------


def top_artifacts(state: State, n: int = TOP_CONTEXT) -> List[Artifact]:
    """Return the current highest-fate artifacts that shape future generations."""
    return sorted(state.artifacts, key=lambda a: a.fate, reverse=True)[:n]


def stable_iteration_rank(namespace: str, state: State, value: str) -> str:
    """Deterministic pseudo-random ordering keyed to the current iteration."""
    return fingerprint(f"{namespace}:{state.iteration}:{value}")


def take_unique_artifacts(
    selected: List[tuple[Artifact, str]],
    selected_ids: set[str],
    candidates: List[Artifact],
    count: int,
    role: str,
) -> None:
    """Append up to count unseen artifacts tagged with the given context role."""
    if count <= 0:
        return

    taken = 0
    for artifact in candidates:
        if artifact.id in selected_ids:
            continue
        selected.append((artifact, role))
        selected_ids.add(artifact.id)
        taken += 1
        if taken >= count:
            return


def select_context_artifacts(
    state: State,
    n: int = TOP_CONTEXT,
) -> List[tuple[Artifact, str]]:
    """Mix leaders, underused branches, frame-breakers, and sampled survivors."""
    if n <= 0 or not state.artifacts:
        return []

    selected: List[tuple[Artifact, str]] = []
    selected_ids: set[str] = set()
    artifacts = list(state.artifacts)

    leaders = sorted(
        artifacts, key=lambda artifact: (artifact.fate, artifact.reuse), reverse=True
    )
    take_unique_artifacts(
        selected,
        selected_ids,
        leaders,
        min(CONFIG.context_leader_slots, n),
        "leader",
    )

    remaining_slots = max(n - len(selected), 0)
    if remaining_slots:
        explores = sorted(
            [artifact for artifact in artifacts if artifact.question.mode == "explore"],
            key=lambda artifact: (artifact.fate, artifact.novelty, artifact.reuse),
            reverse=True,
        )
        take_unique_artifacts(
            selected,
            selected_ids,
            explores,
            min(CONFIG.context_explore_slots, remaining_slots),
            "explore",
        )

    remaining_slots = max(n - len(selected), 0)
    if remaining_slots:
        underused = sorted(
            artifacts,
            key=lambda artifact: (
                artifact.reuse,
                len(artifact.parents) + len(artifact.references),
                -artifact.fate,
                artifact.created_at,
            ),
        )
        take_unique_artifacts(
            selected,
            selected_ids,
            underused,
            min(CONFIG.context_underused_slots, remaining_slots),
            "underused",
        )

    remaining_slots = max(n - len(selected), 0)
    if remaining_slots:
        frame_breakers = [
            artifact
            for artifact in artifacts
            if artifact.question.kind
            in {"challenge", "contradiction", "generalization"}
        ]
        frame_breakers = sorted(
            frame_breakers,
            key=lambda artifact: stable_iteration_rank(
                "frame-breaker", state, artifact.id
            ),
        )
        take_unique_artifacts(
            selected,
            selected_ids,
            frame_breakers,
            min(CONFIG.context_frame_breaker_slots, remaining_slots),
            "frame_breaker",
        )

    remaining_slots = max(n - len(selected), 0)
    if remaining_slots:
        sampled = sorted(
            artifacts,
            key=lambda artifact: stable_iteration_rank(
                "artifact-sample", state, artifact.id
            ),
        )
        take_unique_artifacts(
            selected, selected_ids, sampled, remaining_slots, "sampled"
        )

    return selected[:n]


def definition_usage_counts(state: State) -> dict[str, int]:
    """Count how often each registry definition is referenced by current artifacts."""
    counts = {registry.id: 0 for registry in state.registry}
    for artifact in state.artifacts:
        for definition_id in artifact.referenced_definition_ids:
            counts[definition_id] = counts.get(definition_id, 0) + 1
    return counts


def take_unique_registry_entries(
    selected: List[tuple[RegistryDefinition, str, int]],
    selected_ids: set[str],
    candidates: List[RegistryDefinition],
    count: int,
    role: str,
    usage_counts: dict[str, int],
) -> None:
    """Append up to count unseen registry entries tagged with their sampling role."""
    if count <= 0:
        return

    taken = 0
    for registry in candidates:
        if registry.id in selected_ids:
            continue
        selected.append((registry, role, usage_counts.get(registry.id, 0)))
        selected_ids.add(registry.id)
        taken += 1
        if taken >= count:
            return


def select_registry_entries(
    state: State,
    n: Optional[int] = None,
) -> List[tuple[RegistryDefinition, str, int]]:
    """Mix anchor, underused, recent, and sampled registry concepts."""
    if not state.registry:
        return []

    limit = n or CONFIG.registry_context_limit
    usage_counts = definition_usage_counts(state)
    index_by_id = {registry.id: idx for idx, registry in enumerate(state.registry)}

    selected: List[tuple[RegistryDefinition, str, int]] = []
    selected_ids: set[str] = set()

    anchors = sorted(
        state.registry,
        key=lambda registry: (
            usage_counts.get(registry.id, 0),
            registry.status == "active",
            -index_by_id[registry.id],
        ),
        reverse=True,
    )
    take_unique_registry_entries(
        selected,
        selected_ids,
        anchors,
        min(CONFIG.registry_anchor_slots, limit),
        "anchor",
        usage_counts,
    )

    remaining_slots = max(limit - len(selected), 0)
    if remaining_slots:
        underused = sorted(
            state.registry,
            key=lambda registry: (
                usage_counts.get(registry.id, 0),
                -index_by_id[registry.id],
                registry.status != "active",
            ),
        )
        take_unique_registry_entries(
            selected,
            selected_ids,
            underused,
            min(CONFIG.registry_underused_slots, remaining_slots),
            "underused",
            usage_counts,
        )

    remaining_slots = max(limit - len(selected), 0)
    if remaining_slots:
        recent = list(reversed(state.registry))
        take_unique_registry_entries(
            selected,
            selected_ids,
            recent,
            min(CONFIG.registry_recent_slots, remaining_slots),
            "recent",
            usage_counts,
        )

    remaining_slots = max(limit - len(selected), 0)
    if remaining_slots:
        sampled = sorted(
            state.registry,
            key=lambda registry: stable_iteration_rank(
                "registry-sample", state, registry.id
            ),
        )
        take_unique_registry_entries(
            selected,
            selected_ids,
            sampled,
            remaining_slots,
            "sampled",
            usage_counts,
        )

    return selected[:limit]


def topic_seed_payload(seed: TopicSeed) -> dict[str, object]:
    """Trim the seed down to the fields that should steer question generation."""
    return {
        "topic": seed.topic,
        "goal": seed.goal,
        "include": seed.include[:10],
        "avoid": seed.avoid[:10],
        "seed_questions": seed.seed_questions[:5],
        "seed_definitions": [
            definition.model_dump() for definition in seed.seed_definitions[:10]
        ],
    }


def topic_seed_prompt(state: State) -> str:
    """Optional seed instructions that keep a new exploration on-topic."""
    if state.seed is None:
        return ""

    return f"""
Topic seed:
{json.dumps(topic_seed_payload(state.seed), ensure_ascii=False)}

Seed rules:
- stay near the seeded topic unless a closely related contradiction, refinement, or generalization is clearly useful
- prefer questions and answers that advance the seeded goal
- respect the include and avoid hints when present
"""


def context_blob(state: State) -> str:
    """Compact summary of the strongest artifacts and the current registry."""
    context_artifacts = select_context_artifacts(state)
    registry_entries = select_registry_entries(state)
    payload = {
        "artifacts": [
            {
                "id": a.id,
                "context_role": role,
                "question": a.question.text,
                "question_mode": a.question.mode,
                "answer": a.answer[:1500],
                "claims": [c.text for c in a.claims[:5]],
                "definitions": [d.name for d in a.definitions[:5]],
                "evidence_type": a.evidence_type,
                "evidence_strength": a.evidence_strength,
                "competing_hypothesis": a.competing_hypothesis[:240],
                "main_failure_case": a.main_failure_case[:240],
                "verification_targets": a.verification_targets[:3],
                "open_questions": a.open_questions[:3],
                "novelty": a.novelty,
                "grounding_score": a.grounding_score,
                "overclaim_penalty": a.overclaim_penalty,
                "fate": a.fate,
                "reuse": a.reuse,
            }
            for a, role in context_artifacts
        ],
        "registry": [
            {
                "id": r.id,
                "context_role": role,
                "name": r.name,
                "meaning": r.meaning[:300],
                "aliases": r.aliases[:5],
                "status": r.status,
                "usage_count": usage_count,
            }
            for r, role, usage_count in registry_entries
        ],
    }
    if state.seed is not None:
        payload["seed"] = topic_seed_payload(state.seed)
    return json.dumps(payload, ensure_ascii=False)


def question_batch_plan() -> str:
    """Describe the exploit/explore quota for each question batch."""
    explore_count = min(CONFIG.explore_questions_per_iteration, QUESTION_BATCH_SIZE)
    exploit_count = max(QUESTION_BATCH_SIZE - explore_count, 0)
    return (
        "Batch composition:\n"
        f"- exactly {exploit_count} exploit questions: deepen, sharpen, or stress-test the strongest current lines of inquiry\n"
        f"- exactly {explore_count} explore questions: challenge the dominant framing, import a neighboring lens, or test a neglected assumption\n"
        "- explore questions should usually be contradiction, challenge, or generalization questions\n"
        "- explore questions should not be mere parameter tweaks or renamings of the current dominant architecture\n"
        "- use mode=exploit or mode=explore on each question\n"
    )


def build_question_prompt(state: State) -> str:
    """Keep the question-generation prompt in one place for easier prompt edits."""
    return f"""
Generate {QUESTION_BATCH_SIZE} new useful ideation questions.

Rules:
- avoid duplicates
- avoid vague philosophy
- prefer gaps, contradictions, refinements, challenges, or generalizations
- questions should be answerable in plain language
- reuse existing concepts when they genuinely clarify or connect lines of inquiry
- prefer a new framing when it exposes a materially different mechanism, assumption, or failure mode
- treat leader artifacts and anchor registry entries as the current dominant frame, not as ground truth

{question_batch_plan()}

{topic_seed_prompt(state)}

Context:
{context_blob(state)}
"""


def build_artifact_prompt(state: State, question: Question) -> str:
    """Build the structured drafting prompt from the reusable lineage surface."""
    source_artifacts = [
        artifact
        for artifact, _ in select_context_artifacts(
            state, CONFIG.lineage_context_artifacts
        )
    ]
    known_ids = [artifact.id for artifact in source_artifacts]
    known_claim_ids = [
        claim.id for artifact in source_artifacts for claim in artifact.claims
    ][: CONFIG.reference_context_limit]
    mode_rules = (
        """
Mode rules for explore questions:
- prefer hypotheses, frameworks, counterfactuals, or design directions that open up the space
- novelty is welcome when it is explicit about uncertainty and limits
- do not fake confidence just to make the idea sound stronger
- verification_targets should name the smallest concrete checks a human could run next
- open_questions should capture the most useful unresolved issues
"""
        if question.mode == "explore"
        else """
Mode rules for exploit questions:
- prefer the strongest grounded synthesis available from the current context
- sharpen or stress-test the main line of inquiry without pretending certainty
- novelty is welcome only when it materially improves the explanation or exposes a weak point
- verification_targets should focus on checking assumptions, edge cases, or likely weak links
- open_questions should capture the most decision-relevant unknowns that remain
"""
    )

    return f"""
Answer the question with structured output.

Rules:
- prefer reusing existing definitions
- only add new definitions when necessary
- claims should be explicit and reusable
- parents must be selected from known artifact ids
- references must be selected from known claim ids
- if no lineage applies, use empty arrays
- classify the answer honestly as conjecture, mechanism, synthesis, mixed, or evidence-backed
- set evidence strength realistically; prefer lower confidence labels when support is thin
- list the main assumptions that would have to hold
- include one serious competing hypothesis or alternative explanation
- include one main failure case or boundary condition
- include 1-3 verification targets that would most reduce uncertainty for a human reviewer
- include 0-3 open questions that would matter for the next round of ideation
- do not present speculation as established evidence

{mode_rules}

{topic_seed_prompt(state)}

Known artifact ids:
{json.dumps(known_ids)}

Known claim ids:
{json.dumps(known_claim_ids)}

Question:
{question.model_dump_json()}

Context:
{context_blob(state)}
"""


def build_score_prompt(draft: Draft, rubric: str) -> str:
    """Shared prompt wrapper for the positive scoring passes."""
    return f"""
{rubric}

Artifact:
{draft.model_dump_json()}
"""


def build_adversary_prompt(draft: Draft) -> str:
    """Prompt for the adversarial scoring pass."""
    return f"""
Find flaws, hidden assumptions, brittleness, or ambiguity.
Return only a penalty score 0..1.

Artifact:
{draft.model_dump_json()}
"""


def build_grounding_prompt(draft: Draft) -> str:
    """Prompt for the ideation-friendly grounding and overclaim pass."""
    mode_guidance = (
        """
This is an explore artifact in an ideation loop.
- allow plausible speculation and new framings
- reward epistemic honesty
- penalize only claims that sound more established than the artifact's own evidence labels justify
- low grounding is acceptable if uncertainty is explicit and verification targets are good
"""
        if draft.question.mode == "explore"
        else """
This is an exploit artifact in an ideation loop.
- prefer grounded synthesis over rhetorical confidence
- penalize unsupported certainty, invented precision, and claims that outrun the stated evidence
- reward artifacts that stay useful while clearly marking uncertainty
"""
    )

    return f"""
Judge this artifact for ideation-safe grounding.

Return:
- grounding_score: how appropriately grounded and reality-sensitive the artifact is for its own claims
- overclaim_penalty: how much it overstates confidence, precision, or factual support
- unsupported_claims: up to 3 short examples of claims that sound too strong for the stated evidence

Important:
- this is not a truth-verification task
- do not punish an artifact just for being speculative or exploratory
- do punish bluffing, invented certainty, fake quantification, or presenting conjecture as settled fact

{mode_guidance}

Artifact:
{draft.model_dump_json()}
"""


# -------------------------
# Generation
# -------------------------


def normalize_question_modes(questions: List[Question]) -> List[Question]:
    """Enforce the exploit/explore quota even when the model is sloppy about labels."""
    if not questions:
        return questions

    explore_target = min(CONFIG.explore_questions_per_iteration, len(questions))
    explore_indexes = [
        idx for idx, question in enumerate(questions) if question.mode == "explore"
    ]

    normalized = list(questions)
    if len(explore_indexes) < explore_target:
        preferred_indexes = [
            idx
            for idx, question in enumerate(normalized)
            if question.kind in {"challenge", "contradiction", "generalization"}
            and idx not in explore_indexes
        ]
        fallback_indexes = [
            idx
            for idx in range(len(normalized))
            if idx not in explore_indexes and idx not in preferred_indexes
        ]
        promote_indexes = (preferred_indexes + fallback_indexes)[
            : explore_target - len(explore_indexes)
        ]
        for idx in promote_indexes:
            normalized[idx] = normalized[idx].model_copy(update={"mode": "explore"})

    elif len(explore_indexes) > explore_target:
        for idx in explore_indexes[explore_target:]:
            normalized[idx] = normalized[idx].model_copy(update={"mode": "exploit"})

    return normalized


def gen_questions(state: State) -> List[Question]:
    result = llm(
        build_question_prompt(state),
        QBatch,
        tokens=1000,
        allow_web_search=True,
    )
    return normalize_question_modes(
        [
            Question(
                text=planned.text,
                kind=planned.kind,
                mode=planned.mode,
            )
            for planned in result.questions
        ]
    )


def gen_artifact(state: State, q: Question) -> Draft:
    return llm(
        build_artifact_prompt(state, q),
        Draft,
        tokens=1800,
        allow_web_search=True,
    )


# -------------------------
# Evaluation
# -------------------------


def judge1(d: Draft) -> float:
    return llm(
        build_score_prompt(
            d,
            "Score 0..1 for ideation usefulness, clarity, and internal coherence. "
            "Reward artifacts that would help a human think, design, or explore better, even when they are speculative.",
        ),
        Score,
        tokens=120,
    ).score


def judge2(d: Draft) -> float:
    return llm(
        build_score_prompt(
            d,
            "Score 0..1 for generative value, transferability, and future reuse in ideation. "
            "Reward artifacts that open productive next steps, alternative framings, or better follow-up questions.",
        ),
        Score,
        tokens=120,
    ).score


def adversary(d: Draft) -> float:
    return llm(build_adversary_prompt(d), Adv, tokens=120).penalty


def grounding_review(draft: Draft) -> GroundingReview:
    return llm(build_grounding_prompt(draft), GroundingReview, tokens=220)


def novelty_anchor_texts(state: State) -> List[str]:
    """Seed and topic anchors that define the current exploration boundary."""
    anchors: List[str] = []

    if state.seed is not None:
        anchors.append(state.seed.topic)
        if state.seed.goal:
            anchors.append(state.seed.goal)
        anchors.extend(state.seed.seed_questions[:5])
        anchors.extend(
            f"{definition.name}: {definition.meaning}"
            for definition in state.seed.seed_definitions[:5]
        )

    return [anchor for anchor in anchors if anchor]


def novelty_score(draft: Draft, state: State) -> float:
    """Reward on-topic conceptual distance instead of lexical novelty."""
    dominant_questions = [
        artifact.question.text
        for artifact in top_artifacts(state, CONFIG.lineage_context_artifacts)
    ]
    corpus_questions = existing_question_texts(state)
    if not corpus_questions:
        return 0.5

    dominant_similarity = top_k_embedding_similarity(
        draft.question.text,
        dominant_questions,
        k=CONFIG.novelty_dominant_neighbors,
    )
    corpus_similarity = top_k_embedding_similarity(
        draft.question.text,
        corpus_questions,
        k=CONFIG.novelty_corpus_neighbors,
    )
    anchor_pool = dominant_questions + novelty_anchor_texts(state)
    topic_affinity = (
        top_k_embedding_similarity(draft.question.text, anchor_pool, k=1)
        if anchor_pool
        else corpus_similarity
    )

    distance_from_dominant = max(0.0, 1.0 - dominant_similarity)
    continuity = max(topic_affinity, corpus_similarity)
    novelty = distance_from_dominant * (continuity**1.5)
    return max(0.0, min(1.0, novelty))


# -------------------------
# Registry
# -------------------------


def find_registry_match(state: State, d: Definition) -> Optional[RegistryDefinition]:
    best = None
    best_score = 0.0

    for reg in state.registry:
        name_score = max(
            lexical_similarity(d.name, reg.name),
            max(
                (lexical_similarity(d.name, alias) for alias in reg.aliases),
                default=0.0,
            ),
        )
        meaning_score = lexical_similarity(d.meaning, reg.meaning)
        score = (
            CONFIG.registry_name_weight * name_score
            + CONFIG.registry_meaning_weight * meaning_score
        )

        if score > best_score:
            best_score = score
            best = reg

    return best if best_score >= CONFIG.registry_match_threshold else None


def register_defs(state: State, artifact: Artifact) -> None:
    """Attach draft definitions to the shared registry and record the ids used."""
    referenced_ids = []

    for d in artifact.definitions:
        match = find_registry_match(state, d)
        if match:
            if d.name != match.name and d.name not in match.aliases:
                match.aliases.append(d.name)

            if (
                lexical_similarity(d.meaning, match.meaning)
                < CONFIG.registry_meaning_conflict_threshold
            ):
                match.status = "conflicted"

            referenced_ids.append(match.id)
        else:
            reg = RegistryDefinition(
                name=d.name,
                meaning=d.meaning,
                aliases=[],
                fingerprint=fingerprint(d.name + "::" + d.meaning),
            )
            state.registry.append(reg)
            referenced_ids.append(reg.id)

    artifact.referenced_definition_ids = referenced_ids


# -------------------------
# Dedup
# -------------------------


def existing_question_texts(state: State) -> List[str]:
    return [a.question.text for a in state.artifacts]


def existing_claim_texts(state: State) -> List[str]:
    return [c.text for a in state.artifacts for c in a.claims]


def dedup_question_penalty(state: State, q: Question) -> float:
    others = existing_question_texts(state)
    return max_embedding_similarity(q.text, others, QUESTION_LEXICAL_PREFILTER)


def dedup_claims(state: State, claims: List[Claim]) -> tuple[List[Claim], float]:
    existing = existing_claim_texts(state)
    kept: List[Claim] = []
    max_penalty = 0.0

    for claim in claims:
        local_others = [c.text for c in kept]
        sim_existing = max_embedding_similarity(
            claim.text, existing, CLAIM_LEXICAL_PREFILTER
        )
        sim_local = max_embedding_similarity(
            claim.text, local_others, CLAIM_LEXICAL_PREFILTER
        )
        sim = max(sim_existing, sim_local)
        max_penalty = max(max_penalty, sim)

        if sim < CLAIM_EMBED_SIM:
            kept.append(claim)

    return kept, max_penalty


def sanitize_draft(state: State, d: Draft) -> tuple[Draft, float]:
    """Remove redundant claims and return the strongest dedup penalty observed."""
    q_pen = dedup_question_penalty(state, d.question)
    deduped_claims, c_pen = dedup_claims(state, d.claims)
    cleaned = d.model_copy(update={"claims": deduped_claims})
    return cleaned, max(q_pen, c_pen)


def should_reject_draft(dedup: float) -> bool:
    """Drop near-clones before they enter long-lived state."""
    return dedup >= CONFIG.artifact_dedup_reject_threshold


# -------------------------
# Lineage and reuse
# -------------------------


def artifact_index(state: State) -> dict[str, Artifact]:
    return {a.id: a for a in state.artifacts}


def claim_owner_index(state: State) -> dict[str, Artifact]:
    out = {}
    for a in state.artifacts:
        for c in a.claims:
            out[c.id] = a
    return out


def compute_reuse(d: Draft, state: State) -> int:
    count = 0
    idx = artifact_index(state)
    cidx = claim_owner_index(state)

    for pid in d.parents:
        if pid in idx:
            count += 1

    for rid in d.references:
        if rid in cidx:
            count += 1

    return count


def apply_reuse_tracking(state: State, artifact: Artifact) -> None:
    idx = artifact_index(state)
    cidx = claim_owner_index(state)

    for pid in artifact.parents:
        if pid in idx:
            idx[pid].reuse += 1

    for rid in artifact.references:
        owner = cidx.get(rid)
        if owner:
            owner.reuse += 1


# -------------------------
# Fate
# -------------------------


def compute_fate(
    s1: float,
    s2: float,
    adv: float,
    dedup: float,
    novelty: float,
    grounding_score: float,
    overclaim_penalty: float,
    question_mode: QuestionMode,
    reuse: int,
) -> float:
    grounding_weight = (
        CONFIG.exploit_grounding_weight
        if question_mode == "exploit"
        else CONFIG.explore_grounding_weight
    )
    overclaim_weight = (
        CONFIG.exploit_overclaim_weight
        if question_mode == "exploit"
        else CONFIG.explore_overclaim_weight
    )

    return max(
        ((s1 + s2) / 2.0)
        - abs(s1 - s2) * 0.4
        - adv * 0.7
        - dedup * 0.4
        + novelty * CONFIG.novelty_bonus_weight
        + grounding_score * grounding_weight
        - overclaim_penalty * overclaim_weight
        + min(reuse * 0.05, 0.25),
        0.0,
    )


# -------------------------
# Visualization
# -------------------------


def build_graph(state: State) -> nx.DiGraph:
    g = nx.DiGraph()

    for a in state.artifacts:
        label = a.question.text[:48] + ("..." if len(a.question.text) > 48 else "")
        g.add_node(
            a.id,
            label=label,
            fate=a.fate,
            reuse=a.reuse,
        )

    claim_to_owner = {}
    for a in state.artifacts:
        for c in a.claims:
            claim_to_owner[c.id] = a.id

    for a in state.artifacts:
        for pid in a.parents:
            if pid in g:
                g.add_edge(pid, a.id, kind="parent")

        for rid in a.references:
            owner = claim_to_owner.get(rid)
            if owner and owner in g and owner != a.id:
                g.add_edge(owner, a.id, kind="reference")

    return g


def draw_graph(state: State, path: Optional[Path] = None) -> None:
    graph_path = path or GRAPH_FILE
    if not state.artifacts:
        return

    g = build_graph(state)
    if not g.nodes:
        return

    plt.figure(figsize=(16, 10))
    pos = nx.spring_layout(g, seed=42, k=1.2)

    node_sizes = [500 + (g.nodes[n].get("fate", 0.0) * 2500) for n in g.nodes]
    edge_colors = [
        "black" if g.edges[e].get("kind") == "parent" else "gray" for e in g.edges
    ]
    labels = {n: g.nodes[n]["label"] for n in g.nodes}

    nx.draw_networkx_nodes(g, pos, node_size=node_sizes)
    nx.draw_networkx_edges(g, pos, arrows=True, edge_color=edge_colors, alpha=0.6)
    nx.draw_networkx_labels(g, pos, labels=labels, font_size=8)

    plt.title(f"Artifact Lineage Graph — iteration {state.iteration}")
    plt.axis("off")
    plt.tight_layout()
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(graph_path, dpi=200)
    plt.close()


# -------------------------
# Hazard / prune
# -------------------------


def should_run_hazard(iteration: int) -> bool:
    return (
        iteration % CONFIG.hazard_every_iterations == CONFIG.hazard_every_iterations - 1
    )


def hazard(state: State) -> None:
    """Occasionally delete the current leaders to force new exploration."""
    if len(state.artifacts) < CONFIG.hazard_min_artifacts:
        return
    ranked = sorted(state.artifacts, key=lambda a: a.fate, reverse=True)
    state.artifacts = ranked[CONFIG.hazard_drop_leaders :]


def prune(state: State) -> None:
    """Keep the top population while reserving room for explore artifacts."""
    if len(state.artifacts) <= MAX_POP:
        state.artifacts.sort(key=lambda a: a.fate, reverse=True)
        return

    ranked = sorted(state.artifacts, key=lambda artifact: artifact.fate, reverse=True)
    explores = [artifact for artifact in ranked if artifact.question.mode == "explore"]
    explore_keep = min(CONFIG.explore_retention_floor, len(explores), MAX_POP)

    kept: List[Artifact] = explores[:explore_keep]
    kept_ids = {artifact.id for artifact in kept}

    for artifact in ranked:
        if artifact.id in kept_ids:
            continue
        kept.append(artifact)
        if len(kept) >= MAX_POP:
            break

    kept.sort(key=lambda artifact: artifact.fate, reverse=True)
    state.artifacts = kept


# -------------------------
# Seed + CLI
# -------------------------


def load_seed_file(path: Path) -> TopicSeed:
    return TopicSeed.model_validate_json(path.expanduser().read_text(encoding="utf-8"))


def inline_seed_from_args(args: argparse.Namespace) -> Optional[TopicSeed]:
    if args.seed_topic is None:
        return None

    return TopicSeed(
        topic=args.seed_topic,
        goal=args.seed_goal or "",
        include=args.seed_include,
        avoid=args.seed_avoid,
        seed_questions=args.seed_question,
    )


def merge_seed(
    file_seed: Optional[TopicSeed],
    inline_seed: Optional[TopicSeed],
) -> Optional[TopicSeed]:
    if file_seed is None:
        return inline_seed
    if inline_seed is None:
        return file_seed

    merged = file_seed.model_copy(deep=True)
    merged.topic = inline_seed.topic
    if inline_seed.goal:
        merged.goal = inline_seed.goal
    if inline_seed.include:
        merged.include = inline_seed.include
    if inline_seed.avoid:
        merged.avoid = inline_seed.avoid
    if inline_seed.seed_questions:
        merged.seed_questions = inline_seed.seed_questions
    return merged


def resolve_seed(args: argparse.Namespace) -> Optional[TopicSeed]:
    file_seed = load_seed_file(args.seed_file) if args.seed_file else None
    return merge_seed(file_seed, inline_seed_from_args(args))


def apply_seed(
    state: State,
    seed: Optional[TopicSeed],
    replace_seed: bool = False,
) -> tuple[State, bool]:
    """Persist a newly supplied seed without silently retargeting live runs."""
    if seed is None or state.seed == seed:
        return state, False

    if state.seed is not None and not replace_seed:
        raise ValueError(
            "Selected state already has a different seed. "
            "Use --replace-seed or choose a different --state-file."
        )

    if state.seed is None and state.artifacts and not replace_seed:
        raise ValueError(
            "Refusing to seed a populated state with no existing seed. "
            "Use a fresh --state-file or pass --replace-seed to retarget it."
        )

    return state.model_copy(update={"seed": seed}), True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal OpenAI-backed ideation and exploration loop runner."
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=10,
        help="How many iterations to run. Default: 10.",
    )
    parser.add_argument(
        "--state-file",
        type=Path,
        help="Persist and load state from this JSON file instead of world_state.json.",
    )
    parser.add_argument(
        "--graph-file",
        type=Path,
        help="Write the lineage graph to this PNG file. Defaults to a state-specific graph path.",
    )
    parser.add_argument(
        "--git",
        action="store_true",
        help="Commit state-file updates after each save.",
    )
    parser.add_argument(
        "--no-graph",
        action="store_true",
        help="Skip lineage graph rendering.",
    )
    parser.add_argument(
        "--seed-file",
        type=Path,
        help="Load a TopicSeed JSON document and persist it into the selected state file.",
    )
    parser.add_argument(
        "--seed-topic",
        help="Inline topic seed for a new exploration.",
    )
    parser.add_argument(
        "--seed-goal",
        help="Optional goal that clarifies what the seed should optimize for.",
    )
    parser.add_argument(
        "--seed-include",
        action="append",
        default=[],
        help="Repeatable hint for subtopics to prefer.",
    )
    parser.add_argument(
        "--seed-avoid",
        action="append",
        default=[],
        help="Repeatable hint for subtopics to avoid.",
    )
    parser.add_argument(
        "--seed-question",
        action="append",
        default=[],
        help="Repeatable starter question to bias the new exploration.",
    )
    parser.add_argument(
        "--replace-seed",
        action="store_true",
        help="Allow replacing or adding a seed on an already populated state file.",
    )
    parser.add_argument(
        "--rejudge-existing",
        action="store_true",
        help="Re-score existing artifacts in the selected state with the current judges before running iterations.",
    )
    parser.add_argument(
        "--web-search",
        action="store_true",
        help="Allow question and artifact generation to use OpenAI's built-in web search tool. Off by default.",
    )
    parser.add_argument(
        "--web-search-domain",
        action="append",
        default=[],
        help="Repeatable allowed-domain filter for built-in web search. Requires --web-search.",
    )

    args = parser.parse_args()
    used_inline_seed_fields = any(
        [
            args.seed_goal,
            args.seed_include,
            args.seed_avoid,
            args.seed_question,
        ]
    )
    if used_inline_seed_fields and args.seed_topic is None and args.seed_file is None:
        parser.error(
            "--seed-topic is required when using inline seed flags without --seed-file."
        )
    if args.replace_seed and args.seed_file is None and args.seed_topic is None:
        parser.error(
            "--replace-seed requires --seed-file or inline --seed-topic input."
        )
    if args.web_search_domain and not args.web_search:
        parser.error("--web-search-domain requires --web-search.")
    return args


# -------------------------
# Main loop
# -------------------------


def evaluate_draft(draft: Draft, state: State, dedup: float) -> DraftEvaluation:
    """Run the evaluation passes that determine whether a draft survives."""
    grounding = grounding_review(draft)
    return DraftEvaluation(
        score1=judge1(draft),
        score2=judge2(draft),
        adversarial=adversary(draft),
        dedup=dedup,
        novelty=novelty_score(draft, state),
        grounding_score=grounding.grounding_score,
        overclaim_penalty=grounding.overclaim_penalty,
        reuse=compute_reuse(draft, state),
        question_mode=draft.question.mode,
    )


def materialize_artifact(draft: Draft, evaluation: DraftEvaluation) -> Artifact:
    """Convert an in-memory draft plus scores into a persisted artifact record."""
    artifact = Artifact(
        question=draft.question,
        answer=draft.answer,
        definitions=draft.definitions,
        claims=draft.claims,
        parents=draft.parents,
        references=draft.references,
        evidence_type=draft.evidence_type,
        evidence_strength=draft.evidence_strength,
        assumptions=draft.assumptions,
        competing_hypothesis=draft.competing_hypothesis,
        main_failure_case=draft.main_failure_case,
        verification_targets=draft.verification_targets,
        open_questions=draft.open_questions,
        score1=evaluation.score1,
        score2=evaluation.score2,
        adversarial=evaluation.adversarial,
        dedup=evaluation.dedup,
        novelty=evaluation.novelty,
        grounding_score=evaluation.grounding_score,
        overclaim_penalty=evaluation.overclaim_penalty,
        reuse=evaluation.reuse,
    )
    artifact.fate = evaluation.fate
    return artifact


def artifact_to_draft(artifact: Artifact) -> Draft:
    """Project a persisted artifact back into the draft schema for re-evaluation."""
    return Draft(
        question=artifact.question,
        answer=artifact.answer,
        definitions=artifact.definitions,
        claims=artifact.claims,
        parents=artifact.parents,
        references=artifact.references,
        evidence_type=artifact.evidence_type,
        evidence_strength=artifact.evidence_strength,
        assumptions=artifact.assumptions,
        competing_hypothesis=artifact.competing_hypothesis,
        main_failure_case=artifact.main_failure_case,
        verification_targets=artifact.verification_targets,
        open_questions=artifact.open_questions,
    )


def dedup_penalty_for_draft(state: State, draft: Draft) -> float:
    """Score how redundant a draft is against the current artifact population."""
    question_penalty = dedup_question_penalty(state, draft.question)
    _, claim_penalty = dedup_claims(state, draft.claims)
    return max(question_penalty, claim_penalty)


def rejudge_existing_artifacts(state: State) -> bool:
    """Re-score persisted artifacts with the current judges without regenerating content."""
    if not state.artifacts:
        print("\nno existing artifacts to rejudge")
        return False

    print(f"\n--- rejudging {len(state.artifacts)} existing artifacts ---")
    original_artifacts = list(state.artifacts)
    refreshed: List[Artifact] = []

    for artifact in original_artifacts:
        peer_state = state.model_copy(
            update={
                "artifacts": [
                    candidate
                    for candidate in original_artifacts
                    if candidate.id != artifact.id
                ]
            }
        )
        draft = artifact_to_draft(artifact)
        evaluation = evaluate_draft(
            draft,
            peer_state,
            dedup_penalty_for_draft(peer_state, draft),
        )
        updated = artifact.model_copy(
            update={
                "score1": evaluation.score1,
                "score2": evaluation.score2,
                "adversarial": evaluation.adversarial,
                "dedup": evaluation.dedup,
                "novelty": evaluation.novelty,
                "grounding_score": evaluation.grounding_score,
                "overclaim_penalty": evaluation.overclaim_penalty,
                "reuse": evaluation.reuse,
                "fate": evaluation.fate,
            }
        )
        refreshed.append(updated)
        print(f"rejudged: {format_artifact_progress(updated)}")

    state.artifacts = refreshed
    prune(state)
    return True


def process_question(
    state: State, question: Question
) -> tuple[Optional[Artifact], float]:
    """Execute the full per-question pipeline and append the surviving artifact."""
    draft = gen_artifact(state, question)
    draft, dedup = sanitize_draft(state, draft)
    evaluation = evaluate_draft(draft, state, dedup)
    if should_reject_draft(evaluation.dedup):
        return None, evaluation.dedup
    artifact = materialize_artifact(draft, evaluation)

    register_defs(state, artifact)
    apply_reuse_tracking(state, artifact)
    state.artifacts.append(artifact)
    return artifact, evaluation.dedup


def format_artifact_progress(artifact: Artifact) -> str:
    """Stable progress line for per-question logging."""
    return (
        f"{artifact.question.text} -> "
        f"fate={artifact.fate:.3f} dedup={artifact.dedup:.3f} "
        f"novelty={artifact.novelty:.3f} grounding={artifact.grounding_score:.3f} "
        f"overclaim={artifact.overclaim_penalty:.3f}"
    )


def format_rejected_progress(question: Question, dedup: float) -> str:
    """Stable progress line for rejected near-duplicate drafts."""
    return (
        f"{question.text} -> "
        f"rejected dedup={dedup:.3f} threshold={CONFIG.artifact_dedup_reject_threshold:.2f}"
    )


def persist_iteration(state: State, git: bool, render_graph: bool) -> None:
    """Flush the current state to disk and optionally refresh the lineage graph."""
    save_state(state, git=git)

    if render_graph:
        try:
            draw_graph(state)
        except Exception as e:
            print(f"graph render failed: {e}")


def run_iteration(state: State, git: bool = False, render_graph: bool = True) -> None:
    """Run one full ideation iteration over a generated batch of questions."""
    print(f"\n--- iteration {state.iteration} ---")

    questions = gen_questions(state)

    for question in questions:
        try:
            artifact, dedup = process_question(state, question)
            if artifact is None:
                print(format_rejected_progress(question, dedup))
            else:
                print(format_artifact_progress(artifact))
        except Exception as e:
            print(f"artifact failed: {e}")

    if should_run_hazard(state.iteration):
        print("hazard triggered")
        hazard(state)

    prune(state)
    state.iteration += 1
    persist_iteration(state, git=git, render_graph=render_graph)


def run(
    iters: int = 10,
    git: bool = False,
    render_graph: bool = True,
    seed: Optional[TopicSeed] = None,
    replace_seed: bool = False,
    rejudge_existing: bool = False,
) -> None:
    """Run several iterations of the ideation loop against the persisted state."""
    if iters < 0:
        raise ValueError("--iters must be zero or greater.")
    validate_runtime_config(needs_api=iters > 0 or rejudge_existing)

    state, seed_changed = apply_seed(
        load_state(),
        seed,
        replace_seed=replace_seed,
    )
    persisted = False

    if rejudge_existing and rejudge_existing_artifacts(state):
        persist_iteration(state, git=git, render_graph=render_graph)
        persisted = True

    if seed_changed and iters == 0 and not persisted:
        persist_iteration(state, git=git, render_graph=render_graph)
        persisted = True

    for _ in range(iters):
        run_iteration(state, git=git, render_graph=render_graph)
        persisted = True

    if persisted:
        print(f"\nsaved state: {STATE_FILE}")
        if render_graph:
            print(f"saved graph: {GRAPH_FILE}")
    else:
        print("\nno changes persisted")


if __name__ == "__main__":
    cli_args = parse_args()
    configure_runtime_paths(cli_args.state_file, cli_args.graph_file)
    configure_web_search(
        enabled=cli_args.web_search,
        allowed_domains=cli_args.web_search_domain,
    )
    try:
        run(
            iters=cli_args.iters,
            git=cli_args.git,
            render_graph=not cli_args.no_graph,
            seed=resolve_seed(cli_args),
            replace_seed=cli_args.replace_seed,
            rejudge_existing=cli_args.rejudge_existing,
        )
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from exc
