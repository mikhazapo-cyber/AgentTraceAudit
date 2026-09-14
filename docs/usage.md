# Audit your own trace

A practical guide for: *I have one agent trace file. What did my agent do wrong?*

## 1. Install and set a key

```bash
python -m pip install -e ".[dev]"
cp .env.example .env          # then edit .env
```

Put your key in `.env` in the project directory:

```
OPENROUTER_API_KEY=sk-or-...
```

Any OpenAI-compatible endpoint works. `OPENAI_API_KEY` is also read, and
`TRACEAUDIT_BASE_URL` overrides the endpoint. Check it took:

```bash
traceaudit doctor
```

`mode: live` means the key was found. `mode: deterministic` means it was not, and
only the local mechanical lane will run.

## 2. Audit one trace

```bash
traceaudit explain mytrace.json
```

That is the whole thing. It reads the trace, works out what to check, audits it,
prints the report, and writes the structured output to `traceaudit-out/explain/`.

Try it on the shipped sample first:

```bash
traceaudit explain examples/openai_chat.json
```

`run` does the same for a folder and can score against labels:

```bash
traceaudit run mytraces/ --out results/mine
traceaudit run --dataset data/dev --labels data/labels/dev --out results/dev
```

Both accept the path as a bare argument or as `--dataset`.

## 3. Supply the task and the rules if the trace does not carry them

Findings are only as good as the rules the auditor can see. If your trace has no
system prompt or policy in it, most instruction violations are unprovable and get
abstained rather than reported.

```bash
traceaudit explain mytrace.json --task @task.txt --instructions @policy.md
```

- `--task` -- what the agent was asked to do.
- `--instructions` -- the rules, policy, or system prompt it was under.
- Either flag takes literal text, or `@path` to read a file. `@` is the reliable
  form; use it whenever the value is a path.
- Both override whatever the trace already contained. The run prints which of
  them you supplied, so it is never a silent override.

Instructions are worth supplying even when they feel obvious. A rule that names a
tool, gates a tool, or orders one step before another is provable from a trace. A
rule about tone or formatting is not, and will be abstained on purpose.

## 4. Supported input formats

The format is detected from the file and printed at the start of every run. Pass
`--source <name>` to force one, and `traceaudit formats` to list them.

| Name | What it is |
|---|---|
| `openai-chat` | OpenAI Chat Completions (`messages` + `tool_calls`) |
| `openai-responses` | OpenAI Responses API (`output[].type`) |
| `anthropic-messages` | Anthropic Messages (`content[].type=tool_use`) |
| `google-gemini` | Gemini (`contents[].parts[].functionCall`) |
| `langchain` | LangChain / LangGraph serialized messages |
| `autogen` | AutoGen chat history |
| `metagpt` | MetaGPT roles / actions / environment |
| `bedrock-trajectory` | AWS Bedrock Agent Runtime trajectory |
| `mlflow-trace` | MLflow trace spans |
| `otel-genai` | OpenTelemetry `gen_ai.*` spans |
| `openinference` | OpenInference / Arize Phoenix spans (also TRAIL) |
| `mcp-events` | MCP JSON-RPC request/response capture |
| `litellm-telemetry` | LiteLLM telemetry spans |
| `tavii` | Tavii `semantic_trace_bundle` |
| `agentinstruct` | AgentInstruct Think/Act and ALFWorld text |
| `apigen-mt` | APIGen-MT `{system, tools, conversations}` |
| `nemotron-interactive` | Nemotron interactive, with `reasoning_content` |
| `nemotron-tool-calling` | Nemotron tool-calling, with `reasoning_content` |
| `toucan-mcp` | Toucan MCP (JSON-encoded messages + `available_tools`) |
| `canonical` | traceaudit's own normalized JSON, and the generic fallback |

Aliases: `trail`, `phoenix` and `openinference-otlp` map to `openinference`;
`semantic_trace_bundle` maps to `tavii`.

JSON and JSONL are both fine. A folder needs no index file: every trace-shaped
file in it is picked up.

If detection is wrong, or you want to see exactly what was read:

```bash
traceaudit normalize mytrace.json --limit 4000
```

That prints the canonical steps. If the step count looks too low, the wrong
adapter was chosen -- force the right one with `--source`.

## 5. Read the output

The terminal gives you the verdict. Per trace:

```
    [1/3] my-run-01             ok                   $0.412    98.4s  session $0.412
           2 confirmed (instruction_violation 1, redundant_action 1) | 1 abstained
```

Then a run summary with confirmed totals by family, abstentions, cost and latency
mean/p95, and where the files went.

In `--out`:

| File | What it is for |
|---|---|
| `reports/<trace_id>.md` | Read this first. Verdict, the ask, then each finding. |
| `findings.jsonl` | One JSON record per trace. For programmatic use. |
| `findings.csv` | One row per finding, including `rule_ref` and the evidence quotes. |
| `summary.md`, `summary.json` | Run totals, per-trace cost and latency. |
| `eval.md`, `eval.json` | Precision, recall, F1, clean-trace FPs (only with `--labels`). |

Each confirmed finding tells you:

- **family** -- `instruction_violation`, `incorrect_tool_use`,
  `evidence_contradiction`, `unsupported_success`, `redundant_action`,
  `ignored_feedback`, `other`.
- **steps** -- where it happened, as a step or a step range.
- **the rule** -- quoted from your instructions, or named as a local mechanical
  check. The report says which, so a schema check is never presented as though
  you had written that rule.
- **supporting excerpts** -- quoted from the trace, labelled with the step kind so
  you can see whether the evidence is a tool result or the agent's own words.
- **what the agent could see at that point** -- so a finding cannot rest on
  information that arrived later.
- **a cheaper alternative** -- for inefficiencies, what it should have done
  instead, chosen from what was available at that step.

### confirmed vs abstained

Two outcomes, and they are not the same claim:

- **confirmed** -- provable from this trace. This is a finding.
- **abstained** (`insufficient_evidence`) -- the auditor could not settle it.
  **Not an accusation.** Listed in its own section, counted separately, and never
  scored as a false positive.

A trace with 0 confirmed and 3 abstained means nothing was proven against the
agent. Treat it as clean.

## 6. Cost and time

Core model calls per trace: derive, two auditors in parallel, and a judge. A critique pass runs only while spend is still under $1.50. A cheap classify pass and a bounded falsify pass may also run.

| | Value | Set by |
|---|---|---|
| Target mean | $1.50 per trace | config |
| Measured mean on the 17-trace dev set | $1.07 per trace ($1.75 on the 7 longer Tavii logs, $0.59 on the 10 short external logs) | measured |
| Measured p95 | $2.15 per trace | measured |
| Per-trace hard abort | $3.00 default; a local `.env` of `1.5` will abort the live path | `TRACEAUDIT_COST_CAP` |
| Session cap | $40 default; this quality campaign used 30 | `TRACEAUDIT_MAX_SPEND` |
| Uncached latency | about 198s mean / 316s p95 | measured |

Short traces cost less; a 4-step trace like the sample is well under the mean.
Actual billed spend is printed per trace and recorded in `summary.json`. Every run
starts by printing the caps in force, so raise them in `.env` before you start
rather than discovering a cap mid-run.

To spend nothing:

```bash
traceaudit explain mytrace.json --deterministic   # local rules only, $0
traceaudit explain mytrace.json --mock            # stubbed models, $0, exercises the plumbing
```

`--deterministic` still proves schema violations, undeclared tools, malformed
calls and identical repeats. It cannot read your instructions, so it will miss
most instruction violations.

## 7. When something goes wrong

| Symptom | Cause and fix |
|---|---|
| `no API key found` notice | Key not in `.env`. Set `OPENROUTER_API_KEY`, or pass `--api-key`. |
| `does not look like an agent trace` | The file parses but holds no messages or spans. Check you exported the trace, then try `--source <name>`. |
| `no steps could be read` | Wrong adapter. Run `traceaudit normalize` and force `--source`. |
| `is empty (0 bytes)` | The export produced nothing. |
| Everything abstained | The trace carries no rules. Supply `--instructions @policy.md`. |
| `stopping: session spend hit the cap` | Raise `TRACEAUDIT_MAX_SPEND` in `.env`. |

## 8. Scoring against your own labels

Write one gold JSON per trace into a labels directory, then:

```bash
traceaudit run mytraces/ --labels mylabels/ --out results/mine
traceaudit eval --results results/mine/findings.jsonl --labels mylabels/
```

`eval.md` reports precision, recall, F1, false positives on clean traces,
abstentions separately, and cost and latency both per trace and in aggregate. The
held-out protocol is in [heldout.md](heldout.md); development numbers must not be
quoted as held-out numbers.
