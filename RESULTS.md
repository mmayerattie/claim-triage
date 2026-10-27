# Results

## How to run

From `starter-kit/`: `make setup`, then `make test` (80 offline tests, no paid calls).

## Models and embedding method

- LLM: OpenAI GPT-5.5 via Portkey (Chat Completions, reasoning off, strict tool schemas).
- Embeddings: local `BAAI/bge-small-en-v1.5` (fastembed, quantized ONNX, 384 dimensions, cosine similarity in Qdrant), one vector per historical claim (30 points).

## What was tested

- Own suite: 80/80 passing.
- End-to-end acceptance checks (infrastructure, agent loop and trace structure, RAG, accuracy, guardrails): all passing.
- Live checks: image only, image + text, image + text + PDF (kitchen + dining room, 4,000–16,000 USD), text only (`is_valid_image: null`), all five rejection cases (HTTP 400 with the decisive flag false).

## Accuracy on the development set

35 labelled development claims, 35 returned, 0 failed.

| Metric | Run 1 | Run 2 (final) |
|---|---|---|
| Damage type | 1.00 | 1.00 |
| Severity (exact 1, adjacent 0.5) | 0.79 | 0.86 (25 exact) |
| Repair-cost accuracy | 0.56 | 0.62 |
| Over-rated / under-rated | 10 / 3 (one by two levels) | 6 / 4 (all by one level) |
| Claims routed to a person | 31 flagged for review | 2 (catastrophic, senior adjuster) |

- Run 2 added a clearer severity rubric and the handling rules.
- Remaining under-ratings: IMG_005, IMG_007, IMG_021, IMG_027.
- Cost accuracy per claim is 1 − |our midpoint − expected midpoint| ÷ expected midpoint, floored at 0. One severity level moves the range about fourfold, so a severity miss scores close to 0. The metric is also asymmetric: over-estimating drops to 0 quickly, while under-estimating keeps partial credit. Choosing the higher severity when unsure is the safer choice for an insurer, but this metric penalises it more than under-rating. IMG_019 listed 6 areas against 3 labelled, so its cost score is 0.
- Retrieval (offline, 35 development descriptions): correct damage type ranked first 94.3%, in the top 3 97.1%.

## End-to-end P50 latency

14,182 ms (run 2) and 13,608 ms (run 1), client-side over 35 sequential requests including network and gateway. Server-side trace duration p50 is about 11.3 s.

## Cost

About 0.057 USD per claim on average (about 4.4 model calls, 14.7k input tokens of which 45% cached, 440 output tokens), computed from recorded token usage at 5 / 0.50 / 30 USD per million input / cached / output tokens. Not counted: rejected submissions (about 0.015 USD each).

## Assumptions and limitations

- Labels are synthetic references; some severity misses are defensible judgement differences.
- One run per configuration, and the rubric was revised after studying these same 35 cases, so run 2 is likely optimistic for unseen claims. No held-out set, no load test.
- The IMG_019 label counts roof + interior + exterior; our rule is "count each distinct region once".
- Similarity thresholds (0.60 / 0.70) were calibrated on short development queries.
- Business assumptions (tie-break upward, catastrophic routing) are mine and need validation with a claims team.

## What I would do next

- Validate severity on held-out claims and test a reasoning call for the visual assessment.
- Separate the claimant and adjuster interfaces.
- Cheaper model for intake and safety; async queue.
- Policy lookup instead of trusting the request.
