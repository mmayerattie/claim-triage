# Design decisions log

Running record of choices and their trade-offs; the README summarises these.

## Retrieval / embeddings

- Model: `BAAI/bge-small-en-v1.5` via fastembed (quantized ONNX, 384-dim, CLS pooling, L2-normalised, cosine).
  Runs locally on CPU inside the API container: no embedding API, no per-call cost, no extra credential.
  Weights are downloaded at image build time and the hub is disabled at runtime (`HF_HUB_OFFLINE=1`).
- Quantized precision is a deliberate trade-off for the current dataset size. With 30 historical
  claims, the offline check already reaches 94% top-1 / 97% top-3 damage-type agreement, so a larger or
  full-precision model adds image size and latency for little measurable gain. In a production setting
  with thousands of claims and subtler distinctions, we would switch to a higher-precision model
  (e.g. full-precision `bge-large-en-v1.5` or `bge-base`) and re-run the benchmark before adopting it.
- One vector per claim, no chunking (each record is ~100–170 words, well under the 512-token limit).
- Embedded text = damage type + areas + description + adjuster notes. Resolution, cost, policy and
  dates are stored as payload only: they describe outcomes, not incident similarity, and `covered_full`
  appears in 24/30 records.
- Query instruction prefix (BGE's "Represent this sentence…"): fastembed does not add it; we do.
  Measured: identical results with and without on the 35 development queries.
- Incremental indexing. Each point stores a SHA-256 of its source record. At startup only added,
  modified or removed claims are re-embedded and a WARNING names them. `make reindex` forces a full rebuild.
- Weak precedents are flagged, not presented as similar. Nearest-neighbour search always returns
  the top 3, even when nothing is truly similar. Measured top-1 scores on development queries:
  unrelated text 0.45–0.53, novel damage (meteorite, dog digging) 0.64–0.68, genuine precedents
  0.69–0.88. So: `>= 0.70` strong, `0.60–0.70` weak (cited, labelled, human review), `< 0.60` not cited.
  Both thresholds are settings (`SIMILARITY_STRONG`, `SIMILARITY_FLOOR`) and must be re-checked
  against the agent's own (longer) queries.
- A score threshold cannot catch "similar wording, different cause". IMG_029 (car through a fence)
  scores 0.74 against vandalism claims. A second, structured check (no retrieved claim shares the
  assessed damage type) flags these for human review at report assembly.

## LLM

- Start with OpenAI `gpt-5.5` via Portkey; switch only if measured
  results fall short. The client is model-agnostic: changing `LLM_MODEL` is the only required change
  (no provider-specific features; structured final output via a tool call, optional reasoning effort).
- Reasoning effort `none`. gpt-5.5 on `/v1/chat/completions` rejects function tools combined with
  reasoning ("use /v1/responses or set reasoning_effort to 'none'"). We keep Chat Completions (portable
  across Portkey models) with reasoning off; the ReAct loop's explicit purpose-before-action step provides
  structured deliberation. If accuracy is lacking, the fallback is a tool-free reasoning call for the visual
  assessment (hybrid) or the Responses API (OpenAI-specific).
- Measured intake cost: ~2,380 input / ~80 output tokens per single-photo check (~$0.015 at $5/$30 per M), ~2.7 s.

## Claim handling (replaces the single human-review flag)

- Problem measured: the first full run flagged 31/35 claims for human review; 27 of those came only
  from free-text notes the model added. A flag raised on 89% of claims adds no value to adjusters.
- Design: every claim gets a full triage; each situation gets the response that resolves it —
  catastrophic → `senior_adjuster` route; no photo → low confidence + photo request to the claimant;
  evidence conflict / undetermined cause → medium confidence + at most two claimant questions;
  no or weak precedent → medium confidence; everything else → advisory notes (no alarm).
  Route and confidence are computed by code from these rules, never self-reported by the model.
- Severity rubric: minor = a single component repairable/replaceable on its own; severity also reflects
  contents loss and number of rooms; when evidence fits two adjacent levels choose the higher
  (under-rating is the costlier error for an insurer).
- Measured on re-run (35 cases, single run each, same cases used for diagnosis — likely optimistic):
  severity 0.79 → 0.86, cost 0.56 → 0.62; over-ratings 10 → 6; under-ratings 3 → 4 but none by two
  levels; routed 2/35, medium confidence 4/35, claimant questions 3/35.
