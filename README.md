# traceaudit

Find instruction breaks, bad tool use, and wasted work in one agent trace.

Input: a task, its instructions, and an execution log. Output: structured findings, each with a family, step indices, a verbatim quote, and why it was wrong given what the agent knew at that step.

## Audit one trace

```bash
python -m pip install -e ".[dev]"
```

Put your key in `.env` next to this README (never commit it):

```
OPENROUTER_API_KEY=sk-or-...
```

Then audit a single trace file:

```bash
traceaudit explain examples/openai_chat.json
```

That prints the report and writes structured output to `traceaudit-out/explain/`. Swap in your own file. If the trace does not carry the task and the rules the agent was under, supply them:

```bash
traceaudit explain mytrace.json --task @task.txt --instructions @policy.md
```

Roughly $1.07 per trace on the development set, p95 $2.15, hard abort at $3. Longer replayed traces sit near $1.50–$2.30; short traces stay under a dollar. `--deterministic` costs nothing and runs local rules only. Full guide: **[docs/usage.md](docs/usage.md)**.

## Other commands

```bash
traceaudit doctor                    # check install, dataset, key
traceaudit formats                   # the 20 trace formats it reads
traceaudit normalize mytrace.json    # show what was actually parsed
traceaudit demo --deterministic      # offline floor, no API
traceaudit demo                      # live audit of the dev set
traceaudit run --dataset data/dev --labels data/labels/dev --out results/run
```

`run` and `explain` both take the path as a bare argument or as `--dataset`. Useful flags: `--traces id1,id2`, `--labelled`, `--source`, `--config`, `--model`, `--mock`, `--show-report`.

## Outputs

The terminal reports each trace as it finishes -- confirmed findings broken down by family, abstentions, cost, latency -- then a run summary: totals by family, cost and latency mean/p95, and where the files went.

Written to `--out` (default `traceaudit-out/`, gitignored):

- `reports/<trace_id>.md` -- read this first: verdict, what the agent was asked to do, then each finding with its step range, the quoted rule, the supporting excerpts, and for inefficiencies the cheaper alternative
- `findings.jsonl`, `findings.json` -- one record per trace, findings included
- `findings.csv` -- one row per finding, including `rule_ref`, `available_info` and the evidence quotes
- `summary.md`, `summary.json` -- run totals, plus per-trace cost and latency
- `eval.md`, `eval.json` -- precision, recall, F1, clean-trace FPs, cost, latency (with `--labels`)

Families: `instruction_violation`, `incorrect_tool_use`, `evidence_contradiction`, `unsupported_success`, `redundant_action`, `ignored_feedback`, `other`.

`insufficient_evidence` is an abstention, not an accusation. It appears in its own section of every report, is counted separately, and is never scored as a false positive.

## How it works

1. **Mechanical lane** — local rules only. Undeclared tools, schema misses, formation errors, identical-success repeats, phantom-failure claims, and same-args retries after a validation error confirm without a model. Running first lets later proposals on the same steps drop as duplicates.
2. **Derive** — one call compiles checks from *this* task, *these* instructions, *these* tools, and step excerpts: family, violation condition, `justified_when`, source excerpt. Split into an obligation lane and a residual lane. Weak presentation and tone items are dropped.
3. **Two agents in parallel** — A audits obligations (instruction rules compiled into typed rules plus a full tool-call outline). B audits the residual: efficiency, adaptation, success claims. Each sees the other lane as secondary, so neither is blind to a family. Both get a full-budget evidence index plus neighborhood excerpts. Both must locate steps, quote evidence, and name a cheaper alternative for inefficiencies.
4. **Judge** — merges both agents into confirmed / insufficient_evidence / rejected, with neighborhood excerpts around cited steps.
5. **Critique** — a second pass only while spend is still under $1.50. Reviews unused checks, abstentions, and disagreements. New proposals still go through quick-check.
6. **Falsify** — a bounded pass that tries to disprove surviving LLM findings. Only an explicit disproof becomes an abstention.
7. **Quick-check** — local gates, not another model. Quote must appear in the cited step, steps must exist, each family must meet its own definition, capture-gap guesses are dropped, and one rule at overlapping steps yields one finding.

Without a key, or with `--deterministic`, the mechanical lane is the whole run.

## Models and cost

Target mean **$1.50** per trace. Hard abort at **$3** per trace so the judge still runs on long traces. Session cap **$40** (`TRACEAUDIT_MAX_SPEND`). `--deterministic` is $0.

| Role | Model | In / out per 1M |
|---|---|---|
| Derive | `anthropic/claude-opus-5` | $5 / $25 |
| Agent A (obligations) | `anthropic/claude-opus-5` | $5 / $25 |
| Agent B (residual) | `openai/gpt-5.6-sol` | $5 / $30 |
| Judge | `anthropic/claude-opus-5` | $5 / $25 |
| Critique (if spend still under $1.50) | `anthropic/claude-opus-5` | $5 / $25 |
| Error classify | `openai/gpt-4.1-mini` | $0.40 / $1.60 |
| Falsify | `anthropic/claude-opus-5` | $5 / $25 |
| Quick-check | local code | $0 |

Any OpenAI-compatible endpoint works. Token math is in [docs/report.md](docs/report.md).

## Evaluation

Development set: 17 traces, 7 gold findings, 12 clean, one annotator.

| Run | P | R | F1 | TP / FP / FN | Cost | Notes |
|---|---:|---:|---:|---|---|---|
| Deterministic floor | 1.000 | 0.714 | 0.833 | 5 / 0 / 2 | $0 | unchanged tripwire |
| Live, all 17 traces | 1.000 | 1.000 | 1.000 | 7 / 0 / 0 | $18.18 accounted, $1.07 mean, p95 $2.15 | development only |

0 false positives on the 12 clean traces. 35 abstentions, tracked separately. Uncached latency on the paid 17-trace pass was about 198s mean / 316s p95; the confirmation pass that applied the last local gates mixed cache hits with a few fresh critique calls. Incremental API this campaign was about **$22**, plus about **$8** from earlier slices, ≈ **$30** toward the quality budget.

Both rows are **development** numbers. Held-out labels are not in this repo, and development F1 must not be quoted as held-out F1. Protocol: [docs/heldout.md](docs/heldout.md). Details: [docs/report.md](docs/report.md).

## Public labelled corpora

Neither corpus is vendored. Converters and taxonomy mappings ship in `scripts/` and `data/labels/mappings/`.

```bash
python scripts/fetch_public_corpus.py --list
python scripts/fetch_public_corpus.py --corpus trail    # gated: accept terms, then huggingface-cli login
python scripts/convert_trail_labels.py --src data/raw/trail
traceaudit run --dataset data/raw/trail-dataset --labels data/labels/trail --labelled --out results/trail
```

Who&When Pro measures recall and localisation only; it injects one error per trace. Excluded error types are dropped from gold, not counted as misses; `eval.md` prints the scored share. See [docs/labeled_trace_corpora.md](docs/labeled_trace_corpora.md).

## Tests

```bash
pytest
```

## Docs

- [docs/usage.md](docs/usage.md) — audit your own trace: formats, task/instructions, reading the output, cost
- [docs/report.md](docs/report.md) — design, token budget, recorded numbers, limitations
- [docs/heldout.md](docs/heldout.md) — held-out protocol
- [docs/patterns.md](docs/patterns.md) — patterns seen in manual review (reference, not executed)
- [docs/labeled_trace_corpora.md](docs/labeled_trace_corpora.md) — external corpora and their fit
