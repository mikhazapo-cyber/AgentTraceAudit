# Report

## Approach

The system does one job: given a task, the agent's instructions, and an execution trace, it reports proven errors and inefficiencies, grouped by family.

Architecture is three model calls.

1. **Two high-power auditors, in parallel.** Auditor A (Claude Opus 5) looks first at correctness, policy, and evidence. Auditor B (GPT-5.6 Sol) looks first at efficiency, adaptation, and success claims. Both see the same pack: task, instructions, tool schemas, a tool-call outline, and as many steps as the input budget fits. Each first derives the checks that *this* task, *these* instructions, and *these* tools make testable, then locates occurrences.
2. **One judge** (Claude Opus 5) sees both derived-check lists, both proposal lists, and neighborhood excerpts around cited steps. On short and medium traces the judge also gets the full pack. It confirms, rejects, or marks insufficient evidence, assigns the parent family and a finer `error_class`, and a 0–100 confidence. Confirms below `TRACEAUDIT_MIN_CONFIDENCE` (default 70) become abstentions. Eval still matches parent family plus overlapping steps.
3. **Local proof only.** A confirm still needs locatable steps and at least one quote that appears in a cited step. Inefficiency families need a cheaper alternative that existed then. Failure here becomes an abstention, not a rejection and not a new finding.

Ingest is one walker: canonical `{steps, kind}`, OpenAI chat `messages` + `tool_calls`, Anthropic content blocks, OpenAI Responses `function_call` items, APIGen `conversations`, AgentInstruct `{from,value}` with `Action:` / `Act: bash`, Toucan stringified `messages` / `available_tools`, OpenClaw/Tavii JSONL event logs, a thin OpenInference/OTEL span tree (`spans` / nested `child_spans`), and generic message lists. Speaker `name` on a message is kept on the canonical step so multi-agent logs can quote who acted. A folder with `index.json` or `trace.jsonl` bundles loads as a dataset. Task and instructions can sit in the file or be passed on the CLI. Long instruction dumps keep must/never/required/have-to lines first.

### Quality–cost trade-off

OpenRouter list prices used for planning (September 2026): Opus 5 $5 / $25 per 1M, GPT-5.6 Sol $5 / $30 per 1M. OpenRouter adds a small platform fee on top.

A planning allocation for a long trace, aimed at the **$1.50** mean:

| Call | Typical in / out | USD |
|---|---:|---:|
| Auditor A (Opus 5, xhigh) | ~70k / 6k | ~0.50 |
| Auditor B (Sol, xhigh) | ~70k / 6k | ~0.53 |
| Judge (Opus 5, high) | ~50k / 4k | ~0.35 |
| **Total** | | **≈ 1.38** |

There is no `max_tokens` on any call. The mean is held by pack size: about 62% of the remaining $1.50 is input (two auditor copies of one pack, then a judge pack from whatever is left), and about 38% is reserved for uncapped output. Long instruction dumps are reduced to must/never/required lines so the steps stay in context. Short traces cannot reach $1.50 without padding; they cost less. The hard abort is **$3**. If the two auditors already hit that abort, the judge is skipped.

Two labs, not two copies of one model, is the recall spend: correlated misses drop. The judge is the precision spend: one auditor can be wrong; the judge must still see proof.

## Examples of automatically derived checks

These are the *kind* of checks the auditors emit for a single trace. They are not a catalogue the product executes.

From a weather instruction ("Answer only from tool results. Do not invent weather numbers."):

- `evidence_contradiction` — Condition: a stated temperature differs from `get_weather`. `justified_when`: the tool was not called, or returned no number. Source: "Answer only from tool results."

From a retail policy ("Before any refund you must authenticate the user by calling find_user_id"):

- `instruction_violation` — Condition: `refund_order` is called and no earlier step is `find_user_id`. `justified_when`: `find_user_id` already returned a user id. Source: "Never call refund_order unless a user_id has been returned."

From a send-email policy ("Do not claim a message was sent unless send_email returned ok"):

- `unsupported_success` — Condition: the agent says the mail was sent after only `draft_email`. `justified_when`: `send_email` returned ok. Source: "Do not claim a message was sent unless send_email returned ok."

From a conversion tool ("Do not repeat a successful conversion"):

- `redundant_action` — Condition: the same tool is called again with identical arguments after a successful result. Alternative: reuse the result from the earlier step. `justified_when`: the first call errored, or the arguments changed.

## Evaluation

Confirmed findings only. Strict match is the same family plus overlapping steps. `insufficient_evidence` is tracked separately and is never a false positive. `error_class` and `confidence` are recorded but not scored.

Repository tests check the contract, not a live labelled score: auditors run in parallel, the judge confirms or drops, a quote that is not in a cited step becomes an abstention, a justified retry is not confirmed, and ingest accepts canonical JSON and OpenAI-style messages.

A labelled folder run (`--labels`) scores development sets only. `data/dev` is the original 17-trace set. `data/checked` is a larger checked sample; `data/precision/index.json` lists it for offline experiments. Those numbers are not held-out. The eval report also includes a **mappable** P/R/F1 on gold findings whose family is not `other` (injected review excluded); that subset is not held-out either.

Held-out labels are not in this repository. After a labelled run, report:

- precision, recall, F1 (strict: family + overlapping steps)
- false positives on traces labelled clean
- abstention count (not in P/R/F1)
- mean and p95 cost per trace (USD)
- mean and p95 latency per trace (seconds)

Target for a held-out labelled subset: F1 around **0.90**, mean cost **$1.50**, p95 cost **$3**.

## Limitations

- Instruction violations need a quotable rule. Implicit norms are abstained.
- Ingest covers common chat, tool-call, event-log, and OpenInference/OTEL span trees. Exotic shapes should be exported as messages or as `{steps: [{kind, ...}]}`.
- Quote matching is whitespace-normalized English. A judge quote that rephrases the step becomes an abstention.
- Two providers through OpenRouter: a run depends on both unless `TRACEAUDIT_MODEL` overrides every stage.
- Short traces spend less than $1.50. That is correct, not under-use.
- Three large-context calls are slower than a cheap classifier. Uncached latency is typically a few minutes per long trace.
- The judge is the precision bottleneck. If it fails, proposals are abstained rather than auto-confirmed.
