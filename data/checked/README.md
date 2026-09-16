# Checked traces

Human-labelled or construction-verified traces for tests and offline precision experiments. Pair with `data/labels/checked`. The original 17-trace development set in `data/dev` is unchanged.

`data/precision/index.json` lists this sample (101 traces: 71 human, 24 injected, 6 construction; 4 clean / 97 dirty). There is no live LLM spend path. Numbers are not held-out.

Rebuild (needs `huggingface_hub` and `datasets` in the venv):

```powershell
py -3 scripts/import_checked_traces.py --convert --fetch-aegis
```

## Sources

| Corpus | License | What we took | Label kind |
|---|---|---|---|
| [microsoft/AgentRx](https://huggingface.co/datasets/microsoft/AgentRx) + [GitHub trajectories](https://github.com/microsoft/AgentRx) | CC BY 4.0 | All 29 τ-retail failed traces with step-level categories | human |
| [Kevin355/Who_and_When](https://huggingface.co/datasets/Kevin355/Who_and_When) | MIT | 24 smaller traces (hand-crafted + algorithm-generated) | human (who/when/reason only) |
| [davide221/agenterrorbench](https://huggingface.co/datasets/davide221/agenterrorbench) | unspecified public research mirror of AgentErrorBench | 18 size-capped traces with locatable decision steps | human |
| [Fancylalala/AEGIS](https://huggingface.co/datasets/Fancylalala/AEGIS) | MIT | 24-row injected slice (not the full 9,533; cap ~30) | injected |
| `examples/` | MIT (this repo) | clean-retry, clean-search, clean-verify, clean-skill, contradict, auth-skip | construction |

**TRAIL** ([PatronusAI/TRAIL](https://huggingface.co/datasets/PatronusAI/TRAIL), MIT) is gated and the card forbids resharing outside a gated HF repo. It is not vendored. The converter can ingest it later from gitignored `data/raw/trail` (`--allow-trail` on a private checkout). AgentRx Magentic-One is skipped (labelled 44 traces are not compact in-tree; GitHub samples are not that gold set and several are large OCR dumps). Who&When Pro and AgenTracer are not public. TraceLab is skipped (sanitized args).

## Family mapping

Source types are mapped only when the correspondence is honest. Everything else that is still a real labelled error becomes `other`. Findings whose step cannot be recovered after ingest are dropped, never guessed.

- instruction / spec / role / **Instruction Adherence** / **Plan Adherence** / constraint-ignorance / AEGIS FM-1.1, FM-1.2, FM-2.3 → `instruction_violation` (`broke_explicit_rule` or `role_or_spec_deviation`). AgentRx **Intent-Plan Misalignment**, **Underspecified User Intent**, and **Intent Not Supported** stay `other`.
- bad tool / invalid invocation / format / parameter error → `incorrect_tool_use`
- tool-output misread / invention vs observation / hallucination / outcome misinterpretation → `evidence_contradiction`
- false done / missing verification / AEGIS FM-3.x / progress_misjudge → `unsupported_success`
- repetition / AEGIS FM-1.3, FM-2.1 → `redundant_action`
- ignore other agents / AEGIS FM-2.5 → `ignored_feedback`
- Who&When (no taxonomy) and remaining labelled errors → `other`. `mistake_step` is a 0-based history index; gold is remapped after ingest.
- inconclusive / empty type / step not on the recovered trace → skip

`insufficient_evidence` is never written as a gold finding.
