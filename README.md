# traceaudit

Audit one AI agent trace for errors and inefficiencies. You give it the trace and the instructions the agent was under. It reports confirmed problems, each tied to steps and quoted evidence.

## How it works

Two high-power models read the same pack in parallel. A judge keeps only what is proven and assigns each kept item to a family.

1. **Auditor A** (`anthropic/claude-opus-5`) and **Auditor B** (`openai/gpt-5.6-sol`) each derive checks for *this* task, then locate findings with step indices and verbatim quotes.
2. **Judge** (`anthropic/claude-opus-5`) merges the two audits, assigns **family** and **error_class**, and scores **confidence** (integer 0–100). Below `TRACEAUDIT_MIN_CONFIDENCE` (default 70) a confirm becomes an abstention.
3. A local proof check still requires locatable steps and at least one quote that appears in a cited step. Inefficiencies must name a cheaper alternative that already existed. Failure here is an abstention, not a new finding.

There is no per-call output token cap. Pack size holds a **$1.50** mean: requirements first, then as many steps as the leftover input budget fits. About 38% of the target is left for uncapped reasoning and JSON. The hard abort is **$3**. If the two auditors already hit that abort, the judge is skipped.

Families: `instruction_violation`, `incorrect_tool_use`, `evidence_contradiction`, `unsupported_success`, `redundant_action`, `ignored_feedback`, `other`.

`insufficient_evidence` is an abstention. It is listed separately and is never treated as a finding. If a run confirms nothing, the report does not print an evidence section.

## What has been done

This tree is the slim product, not the earlier multi-stage pipeline.

- The live path is two auditors plus a judge. Extra review passes and mechanical lanes were removed.
- Ingest is one walker: OpenAI-style chat and tool calls, event-log JSONL, Toucan, AgentInstruct, plus canonical `{steps, kind}` and a thin OpenInference/OTEL span tree.
- Findings use a closed family set, a finer `error_class`, and a 0–100 confidence. Eval still matches parent family plus overlapping steps.
- A checked public sample is vendored: AgentRx (τ-retail), Who&When, AgentErrorBench, an AEGIS injected slice, and construction examples. Pair traces with gold in `data/labels/checked`. `data/precision/index.json` lists that sample for offline experiments.
- Offline eval reports a **mappable** subset (gold family ≠ `other`, injected review excluded). That subset is not held-out.
- Peer-agree was tightened: an abstention is promoted only when both auditors independently proved the same family on overlapping steps and the judge did not name a concrete look-alike or disproof.
- The judge pack keeps a tool-call outline even when the leftover budget is tight, so a required call that never happened stays visible.

There is no published held-out score in this repository. The only known live labelled slice was **2 traces** (P **1.00** / R **0.67** / F1 **0.80**). Later folder runs stopped on HTTP **402**. Do not quote that slice as a held-out result.

`data/dev` is the original 17-trace development set (12 clean, 5 dirty). `pytest` checks the contract and that gold steps still land on recovered traces. It does not call a model.

## Boot

Python 3.10+ and an [OpenRouter](https://openrouter.ai/) key (or any OpenAI-compatible endpoint).

**macOS / Linux**

```bash
git clone https://github.com/mikhazapo-cyber/AgentTraceAudit.git
cd AgentTraceAudit
python3 -m pip install -e .
export OPENROUTER_API_KEY=sk-or-...
traceaudit
```

**Windows (PowerShell)**

```powershell
git clone https://github.com/mikhazapo-cyber/AgentTraceAudit.git
cd AgentTraceAudit
py -3 -m pip install -e .
$env:OPENROUTER_API_KEY = "sk-or-..."
traceaudit
```

A `.env` file next to this README also works (`OPENROUTER_API_KEY=...`). Never commit the key. See `.env.example`.

With no arguments on a TTY, the CLI asks for:

1. API key (blank uses the environment / `.env`)
2. Trace file or folder
3. Instructions (literal text, `@path`, or Enter to use those already in the trace)
4. Task (same options)

Flags that exist (`traceaudit --help`):

| Flag | What it does |
|---|---|
| `TRACE` | One file, a folder, or a dataset with `index.json` |
| `--task` / `--instructions` | Literal text or `@path` |
| `--api-key` | OpenRouter (or OpenAI-compatible) key |
| `--out` | Output directory (default `traceaudit-out`) |
| `--config` | Optional config JSON |
| `--labels` | Folder of gold JSON. Scores are development-only, not held-out |
| `--yes` | Skip the multi-trace spend confirmation |
| `--fresh` | Ignore finished traces already in `--out` and audit them again |

While a run is in progress the CLI prints auditors, then judge, so a long call does not look hung. A folder of more than one trace asks before spending unless you pass `--yes`. A folder run resumes finished traces already in `--out`. Transient 429/5xx/timeouts are retried; a 402 stops the session.

## Example

`examples/contradict.json` already carries its task and instructions. Override the policy with `examples/policy.md` when you want to:

```bash
traceaudit examples/contradict.json --instructions @examples/policy.md --api-key sk-or-...
```

Windows: `py -3 -m traceaudit examples/contradict.json --instructions @examples/policy.md`.

`examples/clean-retry.json` is a justified retry and should confirm nothing.

## Checked data (optional rebuild)

In the tree now:

| Path | What it is |
|---|---|
| `data/dev` + `data/labels/dev` | Original 17-trace development set |
| `data/checked` + `data/labels/checked` | Vendored checked sample (101 traces) |
| `data/precision/index.json` | Offline index of that checked sample |
| `examples/` | Construction traces used for boot and as the construction slice |

Sources and mapping notes: `data/checked/README.md`. AgentRx, Who&When, AgentErrorBench, an AEGIS injected slice, and the construction examples. TRAIL is gated and is not vendored.

Rebuild the checked folders without spending audit credits (Hugging Face download only):

```bash
python3 -m pip install -e ".[corpora]"
python3 scripts/import_checked_traces.py --convert
```

Windows: `py -3 -m pip install -e ".[corpora]"` then `py -3 scripts/import_checked_traces.py --convert`.

`py -3 -m pytest -q` stays offline and does not spend credits.

This command **does** spend API credits, because it runs the live auditors:

```bash
traceaudit data/checked --labels data/labels/checked --yes --out traceaudit-out/checked
```

Do not run that unless you intend to pay. The files themselves are just traces and labels.

## Output

Confirmed findings only count as findings. Each has:

- **family** and **error_class**
- **confidence** (0–100)
- **steps** in the trace
- **why** it is a problem given what the agent knew then
- **excerpts** from the cited steps (only when something was confirmed)
- for inefficiencies, the **cheaper alternative** that already existed

Abstentions are written separately. They are never false positives.

Written to `--out` (default `traceaudit-out/`):

- `reports/<trace_id>.md`
- `findings.jsonl` / `findings.json` / `findings.csv` (CSV is one row per confirmed finding)
- `summary.md` / `summary.json` — confirmed counts by family, mean and p95 cost and latency
