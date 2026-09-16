# traceaudit

Audit one AI agent trace. Two high-power models independently decide what to check for this task and locate problems. A judge confirms what to keep and assigns each finding to a family.

You pass an API key, a trace, and the instructions the agent was under. You get confirmed errors and inefficiencies, grouped by family, each tied to steps and quoted evidence.

## Boot (macOS, Linux, Windows)

Python 3.10+ and an [OpenRouter](https://openrouter.ai/) key (or any OpenAI-compatible endpoint). Clone, install editable, set a key, run `traceaudit`.

**macOS / Linux**

```bash
git clone https://github.com/YOUR_ORG/AgentTraceAudit.git
cd AgentTraceAudit
python3 -m pip install -e .
export OPENROUTER_API_KEY=sk-or-...
traceaudit
```

**Windows (PowerShell)**

```powershell
git clone https://github.com/YOUR_ORG/AgentTraceAudit.git
cd AgentTraceAudit
py -3 -m pip install -e .
$env:OPENROUTER_API_KEY = "sk-or-..."
traceaudit
```

The CLI then asks for:

1. API key (blank uses the environment / `.env`)
2. Trace file or folder
3. Instructions (literal text, `@path`, or Enter to use those already in the trace)
4. Task (same options)

You can skip the prompts:

```bash
traceaudit examples/contradict.json --instructions @examples/policy.md --api-key sk-or-...
```

While a run is in progress the CLI prints each stage (auditors, then judge) so a long call does not look hung. Multi-trace folders ask before spending; pass `--yes` to skip that.

```bash
traceaudit data/dev --labels data/labels/dev --yes --out traceaudit-out/dev
```

`--labels` scores confirmed findings against gold (same family + overlapping steps). Those numbers are development-only, not held-out. Eval also prints a **mappable** subset (gold family ≠ `other`, injected review excluded); that is also not held-out. `insufficient_evidence` is an abstention and is never a false positive.

A `.env` file next to this README also works:

```
OPENROUTER_API_KEY=sk-or-...
```

Never commit the key.

## Checked public traces (offline sample)

The original 17-trace development set is `data/dev` + `data/labels/dev`. A larger checked sample lives in:

- `data/checked` — vendored traces
- `data/labels/checked` — gold JSON (family + steps)
- `data/precision` — offline index of that sample (`index.json`; 101 traces, not held-out)

Sources (licenses and mapping notes in `data/checked/README.md`):

| Corpus | What is in the tree |
|---|---|
| [microsoft/AgentRx](https://huggingface.co/datasets/microsoft/AgentRx) | τ-retail failed traces with step-level categories |
| [Kevin355/Who_and_When](https://huggingface.co/datasets/Kevin355/Who_and_When) | smaller multi-agent traces (who/when/reason only → family `other`) |
| [davide221/agenterrorbench](https://huggingface.co/datasets/davide221/agenterrorbench) | size-capped traces with locatable decision steps |
| [Fancylalala/AEGIS](https://huggingface.co/datasets/Fancylalala/AEGIS) | injected slice (not naturalistic gold) |
| `examples/` | construction: contradict, auth-skip, clean-retry, clean-search, clean-verify, clean-skill |

**TRAIL** ([PatronusAI/TRAIL](https://huggingface.co/datasets/PatronusAI/TRAIL)) is gated and is not vendored. Runtime ingest can flatten a local OpenInference/OTEL span tree; the converter writes TRAIL only on a private checkout with `--allow-trail` after you fetch into gitignored `data/raw/trail`.

Rebuild the checked folders without OpenRouter (Hugging Face download only):

```bash
python3 -m pip install -e ".[corpora]"   # huggingface_hub, datasets
python3 scripts/import_checked_traces.py --convert
```

On Windows: `py -3 -m pip install -e ".[corpora]"` then `py -3 scripts/import_checked_traces.py --convert`.

`py -3 -m pytest -q` stays offline. This command **does** spend API credits, because it runs the live auditors:

```bash
traceaudit data/checked --labels data/labels/checked --yes --out traceaudit-out/checked
```

Do not run that unless you intend to pay. The files themselves are just traces and labels.

## What you get

Confirmed findings, each with:

- **family** — `instruction_violation`, `incorrect_tool_use`, `evidence_contradiction`, `unsupported_success`, `redundant_action`, `ignored_feedback`, or `other`
- **error_class** — a finer closed label under that family (for example `bad_arguments`, `skipped_required_check`, `contradicts_tool_output`)
- **confidence** — integer 0–100: probability this is a real error of that class given the cited steps. The judge may lower it. Below `TRACEAUDIT_MIN_CONFIDENCE` (default 70) a confirm becomes `insufficient_evidence`, not a false positive. Eval still matches parent family + overlapping steps.
- **steps** in the trace
- **why** it is a problem given what the agent knew then
- **excerpts** from instructions or tool results
- for inefficiencies, the **cheaper alternative** that already existed

`insufficient_evidence` is an abstention. It is listed separately and is never treated as a finding.

Written to `--out` (default `traceaudit-out/`):

- `reports/<trace_id>.md`
- `findings.jsonl` / `findings.json` / `findings.csv`
- `summary.md` / `summary.json` — confirmed counts by family, mean and p95 cost and latency

## How it works

Three model calls. There is no per-agent output token cap. The **$1.50 mean** is held by how much input we send: explicit requirements first, then as many steps as the leftover input budget fits. About 38% of the target is left for uncapped reasoning and JSON. The hard abort is **$3**.

1. **Auditor A** (`anthropic/claude-opus-5`) and **Auditor B** (`openai/gpt-5.6-sol`) run in parallel. Each derives checks for *this* run, then locates findings with step indices and verbatim quotes.
2. **Judge** (`anthropic/claude-opus-5`) merges the two audits, keeps only proven items, assigns family and `error_class`, scores confidence, and rejects justified look-alikes.
3. A local proof check (steps exist, at least one quote appears in a cited step, inefficiencies name an alternative) can only downgrade a confirm to abstain.

A folder run resumes finished traces already in `--out`. Pass `--fresh` to redo them. Transient 429/5xx/timeouts are retried; a 402 stops the session.

Short traces cost less because there is less to send; they are not padded.

See [docs/report.md](docs/report.md) for the approach, examples of derived checks, evaluation notes, and limitations.

## Examples

```bash
traceaudit examples/contradict.json
traceaudit examples/auth-skip.json --instructions @examples/policy.md
```

`examples/contradict.json` already carries its task and instructions. `examples/clean-retry.json` is a justified retry and should confirm nothing.
