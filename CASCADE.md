# Low-cost Jev / DeepSeek evaluator

This fork adds an opt-in cascade for development evaluations. The original GPT-5.5
runner remains the default. Cascade scores are a separate evaluation series and
are not official GPT-5.5 leaderboard submissions.

## Run

Install the repository dependencies with `uv sync` (Python >=3.10). Set
`OPENROUTER_API_KEY` in your environment or the local `.env` file. The cascade does
not need an OpenAI API key. Use the usual `report/<model>/idx-<task>.md` or `.docx`
layout and the original tasks JSONL:

```sh
uv run python run_evaluation.py --judge cascade \
  --pdf_dir report \
  --tasks_jsonl tasks_and_rubrics.jsonl \
  --out_jsonl cascade-results.jsonl \
  --cache_dir .cascade-cache \
  --budget_usd 2

uv run python aggregate_scores.py \
  --input cascade-results.jsonl \
  --tasks-file tasks_and_rubrics.jsonl
```

`python run_cascade_evaluation.py` is an equivalent entry point. Its defaults are
shown above. The cumulative budget covers **all attempts in the cache**, including
previous runs and conservative reservations for requests with unknown billing.
It is not reset when the process restarts. The amount is a local pre-dispatch
spending guard based on provider price ceilings, not a provider-side account cap.

The cascade runs reports sequentially and processes at most 50 criteria per
request. It reads the full extracted report without silent character truncation.
DOCX extraction is text-only, using the upstream paragraph/table extractor.
Duplicate criterion texts, duplicate model/task files and missing task IDs fail
rather than silently dropping or overwriting criteria.

## Fixed evaluation policy

1. Run `typesafe/jev-1.13` through the native OpenRouter `/api/alpha/decisions`
   endpoint, restricted to TypeSafe. Preserve its three-way choices, probabilities
   and confidence.
2. If **any** Jev decision in the report is `blocked` (`-1`), reevaluate **every**
   criterion in the report with DeepSeek and the source-attribution instructions.
   This check happens after all Jev batches, including the last one.
3. Otherwise retain decisions with finite confidence **>= 0.8**. Repack the
   remaining criteria across batches within that report, then evaluate them with
   `deepseek/deepseek-v4.1-flash-1017`, medium reasoning, 32,768 max output tokens.
4. DeepSeek routing is restricted to `inference-net/fp8` then `nebius/fp8`, with
   OpenRouter provider fallbacks enabled. TypeSafe has no provider fallback.
   Every request sets `data_collection: deny`.

The upstream scoring prompt is reused verbatim. Source review adds publication
identity matching (including mirrors and DOIs), section/table citation scope and
explicit attribution checks. A blocked URL mentioned only in a disclaimer is not
sufficient to mark a criterion blocked.

## Outputs, restart and failures

The JSONL retains the upstream `{model, idx, result: {task, scores, ...}}` shape,
so `aggregate_scores.py` works unchanged. Each score also records its route and
Jev confidence; DeepSeek decisions preserve evidence and reasons. Native Jev
choices have empty reason/evidence fields because that endpoint does not generate
textual evidence. No supporting quotation is fabricated.

The cache stores report/task/configuration hashes, exact request bodies, raw
responses (including actual serving provider), every attempt and its cost, and
per-report results. It contains report text: keep it private. The default cache
is ignored by Git; custom cache paths must also be excluded before committing.
API keys and request authorization headers are never written to the cache.

Rerun the **same command** to resume. Successful requests are validated and reused
without new model calls or double billing. Changed reports, rubrics, output paths
or judge settings require a separate cache/output; this prevents mixing series.
The JSONL is atomically replaced only after all reports in the current input set
succeed, so restarts do not append duplicate rows. After a failure, any previous
export remains an older snapshot: consult `summary.json` and the nonzero exit
status. Completed intermediate reports remain available in the cache.

A known terminal API error (including an error object inside HTTP 200) or an
invalid response gets at most **one** additional attempt, with both responses and
costs retained. Authentication failures stop immediately. Timeout, connection
loss, undecodable responses, interrupted requests, or missing billing stop for
inspection and **are not automatically resent**. Unknown charges remain reserved
at their upper bounds. Never delete a request journal to get around this guard:
reconcile it with provider records first. A second invalid/terminal response also
requires inspection; the retry allowance does not reset on restart.

A cache allows one writer. If a process was killed, inspect the PID in `run.lock`
and confirm it is no longer running before removing that lock. This does not
resolve any `dispatching` attempt: inspect those separately. Provider availability
and price ceilings are checked before the first new request of each run.

## Verification and limits

```sh
uv run python -m unittest discover -s tests -v
```

Tests use synthetic responses and make no network/model calls. They cover routing,
confidence boundaries, cross-batch source review, score validation, billing,
bounded retries, unknown outcomes, cache locking, Markdown/DOCX input and CLI
resume behavior.

The implementation was also replayed locally against 97 previously scored reports
(6,714 criteria): all 292 accepted request payloads and all final scores/routes
matched the original evaluator exactly, with zero new API calls. The private
reports and raw responses are not included in this fork.

Selection was exploratory: an 18-report/six-question pilot achieved 92.91% exact
three-way agreement with a **partially corrected GPT reference**, not a fully
human-labeled held-out set. Jev confidence is a routing signal, not a demonstrated
probability of correctness. Keep close rankings and blocked-source cases under
review; do not mix these scores with historical GPT-5.5 results.
