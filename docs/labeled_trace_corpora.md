# External labelled corpora

The evaluator matches confirmed findings against gold of the form `{family, steps, description}` (`src/traceaudit/eval/matching.py`). A usable corpus needs a full trajectory, **step indices**, and an error type that maps onto the seven families. Clean traces matter as much as positives, since FP-on-clean is the precision gate.

This is a catalogue, not a claim that anything here has been scored. Converters for TRAIL and Who&When Pro are in `scripts/`. None of these sets replaces held-out human labels in this project's schema.

Shipped set: 17 traces in `data/dev/` and `data/labels/dev/` (5 positive with 7 gold findings, 12 clean). Single annotator, development only.

## Acquisition order

| # | Corpus | Why | Catch |
|---|---|---|---|
| 1 | [AgentRx](https://huggingface.co/datasets/microsoft/AgentRx) `tau_retail` | Same domain as `external-017`. Lists **all** failures per trace. CC-BY-4.0. | Failures only. Needs successful runs as clean controls. |
| 2 | [TRAIL](https://huggingface.co/datasets/PatronusAI/TRAIL) | Closest task: all errors per trace (841 on 148), span-level evidence, four expert annotators. Adapter present. | Gated Hub. Eval only. Long traces, ~4 clean. |
| 3 | [LongRCA-Bench](https://huggingface.co/datasets/CLoud5-real/longrca-bench) VitaBench | Observed, not injected, failures with a human earliest step and rationale. | First error only. |
| 4 | [TELBench](https://huggingface.co/datasets/NJU-LINK/TELBench) | Expert span labels for unsupported and contradicted claims. | Spans, not step indices. Needs an aligner. |
| 5 | [AgentProcessBench](https://github.com/RUCBM/AgentProcessBench) | 1,000 trajectories, human step labels. | Process quality, not our families. Use `neutral` as must-not-flag. |

## Fitness

**Directly usable** means a converter emits our `Label` JSON with no second human pass. **Partial** means step indices exist but the family must be mapped, or only the first error is labelled. **Calibration only** means the defect was injected.

### Directly usable

**TRAIL** (Patronus, 2025) — [Hub](https://huggingface.co/datasets/PatronusAI/TRAIL), [paper](https://arxiv.org/abs/2505.08638). 148 traces, 841 human errors, span id plus category, evidence, and impact. Use the `openinference` adapter, not `otel-genai`. Convert with `scripts/fetch_public_corpus.py --corpus trail` then `scripts/convert_trail_labels.py`; mapping in `data/labels/mappings/trail_to_traceveri.json`. Of 20 leaf types, 5 map directly, 3 are localisation only, 12 are excluded. `eval.md` prints the scored share.

**AgentRx** (Microsoft, 2026) — [Hub](https://huggingface.co/datasets/microsoft/AgentRx), [paper](https://arxiv.org/abs/2602.02475). Failed traces with `failures[]` carrying `step_number` and `failure_category`. Use all failures, not only `root_cause`. No converter yet.

### Partial

| Corpus | Label | Note |
|---|---|---|
| [Who&When](https://huggingface.co/datasets/Kevin355/Who_and_When) | who plus one decisive step | First error only. |
| [TraceElephant](https://huggingface.co/datasets/TraceElephant/TraceElephant) | responsible agent plus decisive step | Same shape. |
| [LongRCA-Bench](https://huggingface.co/datasets/CLoud5-real/longrca-bench) | earliest step plus rationale | Prefer VitaBench and SWE slices. |
| [TELBench](https://huggingface.co/datasets/NJU-LINK/TELBench) | error span ids | Align spans to steps first. |
| [AgentProcessBench](https://github.com/RUCBM/AgentProcessBench) | correct / neutral / error | Precision set: do not flag `neutral`. |

### Calibration only

| Corpus | Labels made by | Use |
|---|---|---|
| [Who&When Pro](https://huggingface.co/datasets/Leoxx/whowhen_pro) | injecting one error into a successful prefix | Recall and localisation only. Converter: `scripts/convert_whowhen_labels.py`. |
| [CORRECT-Error](https://huggingface.co/datasets/yifanyu/CORRECT-Error) | schema-guided injection | Same warning. |

Do not present Who&When Pro as held-out quality: its labels are exact by construction and easier than organic errors. It does cover `ignored_feedback` and `unsupported_success`, which TRAIL cannot score strictly, so converted labels set `gold_completeness: decisive_error_only` and the evaluator warns that precision is not interpretable. Take precision from TRAIL, AgentRx, or the shipped dev set.

## Family map

Map onto the seven families; do not add new ones.

| Their label | Our family |
|---|---|
| Instruction or plan adherence, non-compliance, goal deviation | `instruction_violation` |
| Invalid invocation, tool selection, call formatting, wrong argument | `incorrect_tool_use` |
| Invented information, tool-output hallucination, misread result | `evidence_contradiction` |
| Fabricated success, claimed action with no result | `unsupported_success` |
| Repeated identical calls, resource abuse | `redundant_action` |
| Ignored error, no adaptation after a refused call | `ignored_feedback` |
| Underspecified user, intent not supported | abstain |
| System or API 5xx, timeout, environment | usually not an agent finding |

## Clean traces

Most RCA corpora are failures only, which leaves FP-on-clean undefined. Sources of clean traces:

- The 12 shipped clean labels.
- τ-bench runs with `reward > 0`, after review: outcome-correct is not process-clean.
- TRAIL traces with zero annotated errors (~4).
- AgentProcessBench steps marked `correct` or `neutral`.

## Held-out construction

See [heldout.md](heldout.md). Do not mix Who&When Pro into a held-out precision table. Useful order: reviewed AgentRx retail failures plus reviewed-clean τ-bench successes first, then TRAIL as a second GAIA/SWE set.

## Converter contract

Emit one JSON per trace matching `data/labels/dev/`:

```json
{
  "trace_id": "...",
  "clean": false,
  "findings": [
    {"family": "instruction_violation", "steps": [15], "description": "...", "severity": "minor"}
  ],
  "excluded_findings": [],
  "localization_only": [],
  "source_benchmark": "TRAIL",
  "mapping_version": "1",
  "gold_completeness": "complete"
}
```

Keep the source category in `notes`. Do not drop a second family on the same trace: the matcher is one-to-one **within** a family.
