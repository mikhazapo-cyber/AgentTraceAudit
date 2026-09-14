# Report

## Approach

Findings must be provable from the trace. Everything below exists to keep the bar high: cite a rule or a tool result, cite the steps, quote the text, and abstain otherwise.

The pipeline spends toward a **$1.50** mean per trace, with a **$3** hard abort.

1. **Mechanical lane.** Local rules confirm without a model: undeclared tool, schema miss, formation error, identical-success repeat, phantom-failure claim, same-args retry after a validation error. It runs first so agent proposals on the same steps drop as duplicates instead of being re-filed under a second family.
2. **Derive.** One strong call reads THIS task, THESE instructions, THESE tools, and step excerpts from the run, and emits checks with a family, violation condition, `justified_when`, and a source excerpt. Checks then split into an obligation lane (instruction rules) and a residual lane (evidence, efficiency, success claims).
3. **Two agents in parallel.** Agent A owns obligations: instruction sentences compiled by `obligations.py` into typed rules (`restricted_tool`, `prerequisite`, `generic`), each bound to the steps that could settle it, plus a full tool-call outline so a required-but-absent call is provable. Agent B owns the residual. Each sees the other lane marked secondary, so neither is blind to a family. Both get an evidence index that uses the char budget (large per-step clips when the trace fits) plus neighborhood excerpts around retrieved steps. Both must locate steps, quote evidence, and name a cheaper alternative for inefficiencies. Presumption of correctness: abstain rather than guess.
4. **Judge.** Sees both agents, the derived checks, and neighborhood excerpts around cited steps, and returns the merged confirmed / insufficient_evidence / rejected set. Confirms only when steps are locatable, a rule applies, and the information was available at the time.
5. **Critique.** A second-pass review that runs only while spend is still under the $1.50 target. It works through unused derived checks, abstentions, and auditor disagreements against the same full evidence pack. New proposals still go through quick-check.
6. **Classify.** A cheap `gpt-4.1-mini` pass tags surviving error results with an `error_kind`. It does not confirm or reject.
7. **Falsify.** A bounded Opus pass tries to disprove surviving LLM findings. Only an explicit `disproved` verdict becomes an abstention. Scripted handlers that return `confirmed` are not treated as disproof.
8. **Quick-check.** Local code, not another model. Details below.

Without a key, or with `--deterministic`, the mechanical lane is the whole run.

### Why the obligation lane

The rules the system used to miss were already on the checklist. One auditor scoring fifteen mixed obligations waived them. A short dedicated brief with its own evidence is an attention fix, not a new detector.

Long policies also carry many sentences a trace cannot settle: tone, posture, presentation. Obligations are ranked by how strongly they tie to observable tools, arguments, and sequencing, and the weak tail is dropped before an auditor sees it.

### Quick-check gates

One gate serves both the mechanical lane and the judged proposals: a judge verdict does not exempt a finding from the proof its family requires. A finding that fails becomes an abstention, except that a misfiled family is a rejection, since abstaining would imply the claim might still be provable.

- Quote must appear verbatim in the cited step; steps must exist.
- `redundant_action` needs a repeat with **identical arguments**. Different arguments are search.
- `ignored_feedback` needs the **same call with the same arguments after an error**. Not retrying, retrying differently, or narrating failure after success belongs elsewhere.
- `instruction_violation` must cite a rule about an **observable action**: it names a tool, gates a tool, or orders one step before another. Phrasing and formatting rules cannot be proven from a trace. A required utterance (state that X, end with a question, propose an approach, ask a clarification) is wording even when the same paragraph later names a tool. Channel protocol about when a silent reply or `NO_REPLY` is allowed is not a domain-tool obligation. A violation whose every cited call the environment rejected is also abstained, because the attempt changed nothing and the failure is already reported. Skill/playbook/readme routing and markdown tables that name a slug that is not an exact declared domain tool are treated as process guidance, not a tool obligation.
- `evidence_contradiction` from the LLM path needs a **value-level** conflict: both sides carry a number (or both a boolean) and they disagree. Category relabels (alumni vs current employer, “profiles” as a count with no opposing number) abstain. Mechanical contradictions are unchanged.
- `unsupported_success` must be a factual completed-action claim (sent, created, paid, refunded), not a characterisation.
- `other` has no testable definition, so a model cannot confirm one.
- Inefficiencies need a named cheaper alternative that existed at that step.
- One rule violated at overlapping steps is one finding, whatever family each auditor filed. The instruction family wins, because the shared evidence is a rule.

### What the system will not do

Confirm on capture loss or a missing schema. Treat a missing `decision` field as a violation. Send raw 100-492KB bundles. Merge different families that share a step but not a rule. Depend on a bug catalogue from the labelled set: the detection rules are structural, and the one remaining prose fallback is generic runtime vocabulary rather than phrasing copied from a scored trace.

Every family is held to one gate. `quick_check` and the mechanical lane call the same `_family_gate`, so a judge-confirmed finding cannot skip the proof its family requires: `incorrect_tool_use` needs a schema violation, an undeclared tool, or a runtime refusal; `evidence_contradiction` needs both the tool-result quote and the contradicting claim; `instruction_violation` needs a rule that is actually in the instructions and that constrains an action.

### Ingest

20 harness formats (19 adapter modules; both `nemotron-*` share one) plus a generic fallback. `openinference` is required for TRAIL: the `otel-genai` adapter skips spans without `gen_ai.*`. See [labeled_trace_corpora.md](labeled_trace_corpora.md).

Adapter detection ranks profiles by coverage ratio, then hit count, then name, so a Chat Completions log is not claimed as LangChain. Loaders accept UTF-8 BOM and already-canonical traces. A file that is not a trace is not defaulted to `openai-chat`.

Tavii traces can recover earlier turns from `openclaw.prompt_messages`. Replay steps are tagged `meta.provenance = prompt_replay`. The mechanical lane skips replay for schema, unknown-tool, and formation checks, because those need the captured tool list, not a reconstructed one. Auditors still see the recovered conversation.

## Models

OpenRouter list prices, September 2026, cross-checked against the OpenAI, Anthropic, Google, and DeepSeek pages.

| Model | In / out per 1M | Role |
|---|---|---|
| `anthropic/claude-opus-5` | $5 / $25 | Default for derive, agent A, judge, critique, and falsify. Best careful citation and policy application at frontier price. 1M context. |
| `openai/gpt-5.6-sol` | $5 / $30 | Agent B. Complementary lab, strong long-horizon reasoning for alternatives and success claims. |
| `anthropic/claude-sonnet-4.6` | $3 / $15 | Fallback if Opus is unavailable. |
| `google/gemini-3.1-pro-preview` | $2 / $12 | Strong long context, too cheap to be the primary auditor here. |
| `openai/gpt-4.1-mini` | $0.40 / $1.60 | Error-kind classify only. |
| `deepseek/deepseek-v4-pro` | ~$0.66 / ~$1.98 | Capable and inexpensive, not the quality default. |

Default endpoint is OpenRouter (`OPENROUTER_API_KEY`, `https://openrouter.ai/api/v1`), which adds roughly 5.5% on top of the list rates in `PRICES_PER_MTOK`.

## Token budget

`max_trace_chars` is 200k. Ingest keeps up to 24k characters per step. Instruction packs go to 64k characters. The evidence index starts at large per-step clips (12k/18k) and only shrinks if the whole index would exceed the budget; older packs started at 960/1400 and left most of a 160k budget unused. Derive sees those excerpts, not a names-only outline. Auditors and the judge also get neighborhood excerpts around retrieved or cited steps. A critique pass runs only while spend is still under $1.50.

Planning estimate for a long replayed trace (the Tavii end of this set):

| Call | In / out (measured, tavii-030 probe) | USD |
|---|---:|---:|
| Derive (Opus 5, high) | 99k / 10k | 0.76 |
| Agent A (Opus 5, xhigh) | 152k / 9k | 0.98 |
| Agent B (Sol, xhigh) | 89k / 6k | 0.28 |
| Judge (Opus 5, high) | 45k / 3k | 0.30 |
| Falsify (Opus 5) | 9k / 2k | 0.08 |
| **Total** | | **≈ 2.40** |

On a short external trace the same stages have almost no extra evidence to send. Critique then fills some leftover budget (~$0.11 on external-021). The $3 abort is the safety rail.

Measured on the 17-trace development set after these packs (`traceaudit-out/phase4-final`): mean **$1.07**, p95 **$2.15**. Seven longer Tavii traces averaged **$1.75** (range $1.47–$2.31). Ten short external traces averaged **$0.59** (range $0.40–$0.84). The overall mean stays under $1.50 because the short traces have no more content to send; we did not pad them.

## Derived-check examples

Compiled from a single trace's task, instructions, and tools. Illustrative, not a catalogue the detector executes.

From AgentInstruct-style tool text:

- `instruction_violation` — Condition: `get_attributes` is called when the question is not a superlative accumulation. `justified_when`: the question asks for an extreme. Source: "Please only use it if the question seeks for a superlative accumulation".
- `incorrect_tool_use` — Condition: the argument to `get_relations` is a type or category rather than an entity or variable. `justified_when`: the argument is an entity already bound in the dialogue.

From a retail policy:

- `instruction_violation` — Condition: a gift card is applied without a balance check covering the price difference. `justified_when`: no gift card is offered, or `get_user_details` already showed sufficient balance. Source: "If the user provides a gift card, it must have enough balance...".

From a weather tool:

- `evidence_contradiction` — Condition: a stated temperature differs from `get_weather`. `justified_when`: the tool was not called, or returned no number.

## Evaluation

Confirmed findings only. Strict match is same family plus overlapping steps. `insufficient_evidence` is tracked separately and never counts as a false positive.

### Development

17 labelled traces: 5 carry 7 gold findings, 12 are clean. Single annotator.

Deterministic floor, measured in-repo with no API:

| Mode | P | R | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| strict | 1.000 | 0.714 | 0.833 | 5 | 0 | 2 |

Zero false positives on the 12 clean traces. The two misses are instruction-precondition golds (`external-009`, `external-017`) that need the live path.

Live confirmation (`traceaudit-out/phase4-final`, default Opus 5 + GPT-5.6 Sol, fuller evidence packs, xhigh auditor reasoning, critique under $1.50, plus the wording / channel-protocol / clarification-playbook gates):

| Mode | P | R | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| strict | 1.000 | 1.000 | 1.000 | 7 | 0 | 0 |

All 7 golds confirmed, including both instruction-precondition golds. Zero false positives on the 12 clean traces. 35 abstentions (the higher-spend path raises more proposals; gates turn the extras into abstentions). Accounted cost **$18.18**, mean **$1.07**, p95 **$2.15**.

Uncached latency on the paid 17-trace pass (`phase4-full`) was **198s** mean / **316s** p95. The confirmation pass that applied the last local gates mixed cache hits with a few fresh critique calls (**44s** mean).

This campaign's incremental API (cache misses only): probe **$3.02** + full 17 **$15.72** + gate rescore **$3.01** = **$21.75**. Prior paid slices were about **$8**. Combined incremental toward the $30 quality budget is about **$30**. The leftover is a few tens of cents, not most of the budget. Cached rescoring still records the original call cost in `summary.json`.

How the live path got there:

| Phase | What | Strict | Incremental new API |
|---|---|---|---|
| prior 1–3 | earlier live slices + process-guidance / value-conflict gates | P 1.000 / R 1.000 / F1 1.000 on cache replay | ~$8 |
| 4 probe | `tavii-030` + `external-021`, new packs | 1 clean FP on `tavii-030` (required closing question) | $3.02 |
| 4 full | all 17, mostly uncached | P 0.778 / R 1.000 / F1 0.875 (FPs on `tavii-011` NO_REPLY protocol and `tavii-022` propose-then-dispatch) | $15.72 |
| 4 final | same proposals, utterance / channel-protocol / clarification-playbook gates | P 1.000 / R 1.000 / F1 1.000 | $3.01 |

The three dropped proposals were structural, not a bug catalogue: a required closing question is wording; when a harness may return `NO_REPLY` is channel protocol; a propose / ask / then-dispatch loop is conversation even if it names `dispatch_search`. Process-guidance and value-level contradiction gates from the previous campaign stayed in.

The lanes bought recall (7/7, up from 5/7) and the gates bought precision. Holding model proposals fixed and turning the gates off drops precision: the phase-4-full proposals scored P 0.778 with 2 false positives before the last three structural tests.

These are development numbers. Do not quote F1 1.000 as held-out.

### Generalising beyond the labelled set

The detector originally recognised several problems by phrases lifted from the
very traces it was scored on. That reads as accuracy on the development set and
as nothing at all on a new harness, so those catalogues were replaced with
structural tests:

| Removed | Why it was overfitting | What replaced it |
|---|---|---|
| Phantom-failure phrase list ("let me correct", "not properly addressing", "my apologies") | Six of the phrases appear in `external-041` and nowhere else in the set; that trace is where the finding scored | A call succeeds, the agent speaks, then re-issues byte-identical arguments. Redoing work you already hold only makes sense if you believe it failed |
| Hollow-argument name list, including the camelCase fields `recommendedApproach` and `uncertaintyAreas` | Copied verbatim from `external-042`'s tool schema | Nothing: an empty required argument is already a typed schema violation, so the list was redundant |
| "you may make a mistake" in the refusal vocabulary | One harness's coaching sentence, quoted from `external-009` | Declared-schema validation, then a test for the runtime naming an argument the call never supplied |
| Weak-success suppressor ("comprehensive", "strategy", …) | A negative catalogue tuned to suppress one development false positive | Nothing: requiring a completed-action verb already excludes characterisations, so the list changed no outcome |
| Required closing question, `NO_REPLY` protocol, propose-then-dispatch loop | The higher-spend path confirmed these on development traces | Utterance / channel-protocol / conversation-playbook tests. The regression cases use different tools and wording than the labelled traces |

The replacements are language-independent, which the regression tests enforce:
`tests/test_generalization.py` detects the same problems using wording the
labelled set does not contain, and rejects the same non-problems.

One prose dependency remains and is worth stating plainly. `external-009`'s
environment reports a bad call as "cannot be executed", and nothing structural
separates that from a missing-resource error: the arguments satisfy the schema,
the runtime echoes the argument back, and the agent's later correction looks the
same as recovering from a genuine 404. Formation-versus-resource is therefore
still decided by a short generic refusal vocabulary. `tavii-022`, which used to
rely on the same vocabulary, is now caught structurally.

### Held-out

Held-out labels are not in this repository. Do not quote development F1 as held-out F1. Run the protocol in [heldout.md](heldout.md), then write precision, recall, F1, clean-trace FPs, abstention count, and mean/p95 cost here.

## Limitations

- Instruction violations need a quotable rule. Implicit norms are abstained.
- Telling a malformed call from a missing resource still depends on a short English refusal vocabulary. A runtime that reports refusals in another language, or only through a status code, falls back to the schema and unsupplied-argument tests.
- Obligation mining, quote verification, and the family gates all assume English word boundaries and Latin-script tokens.
- The obligation lane only reaches rules with an observable subject. A harness that encodes real requirements as style guidance will be under-audited.
- Obligation ranking splits instructions by sentence, so a rule spread across a bulleted list may compile as a fragment.
- Incomplete recordings (`loss` events, unpaired calls) are abstained, not guessed.
- A correct finding filed under the wrong family scores as FP plus FN under strict match.
- 17 development traces cannot stand in for unseen human labels. F1 1.000 is a development confirmation after gates that were written while looking at those traces. The last three gates were written after the higher-spend path created new FPs on this same set.
- Short traces cannot reach a $1.50 mean without padding. The measured $1.07 mean is the mix of $1.75 Tavii traces and $0.59 external traces.
- Two complementary labs improve independence but couple the run to two providers through OpenRouter.
- The shipped `.env` may set `TRACEAUDIT_COST_CAP=1.5`. That aborts the live path before the judge. Raise it to `3` (the config default) for a full audit.
- Falsify only acts on an explicit disproof. A model that hedges leaves the finding in place.
