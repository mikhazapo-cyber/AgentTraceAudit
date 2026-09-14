# Held-out protocol

Run this after development is frozen. Do not tune prompts, thresholds, or rules on held-out labels.

## Layout

```
data/heldout/            # index.json + traces/, same as data/dev, or loose JSON files
data/labels/heldout/     # one JSON per trace_id, same schema as data/labels/dev
```

One label file per trace:

```json
{
  "trace_id": "example-001",
  "clean": false,
  "split": "heldout",
  "findings": [
    {
      "family": "instruction_violation",
      "steps": [4, 5],
      "description": "...",
      "severity": "major"
    }
  ]
}
```

`family` must be one of `instruction_violation`, `incorrect_tool_use`, `evidence_contradiction`, `unsupported_success`, `redundant_action`, `ignored_feedback`, `other`.

## Run

```bash
traceaudit run --dataset data/heldout --labels data/labels/heldout \
               --out results/heldout --config configs/default.json
traceaudit eval --results results/heldout/findings.jsonl --labels data/labels/heldout
```

## Record

Write these into the **Held-out** section of [report.md](report.md), never the Development section:

- precision, recall, F1 (strict and relaxed)
- false positives on clean traces
- `insufficient_evidence` count (abstentions, not FPs)
- mean and p95 cost per trace
- mean and p95 latency
- audit mode (`live`, `deterministic`, or `mock`)

For a public corpus (TRAIL, Who&When Pro, AgentRx), convert with the mappings in `data/labels/mappings/` and print the scored share. Excluded error types must not count as false negatives. Who&When Pro is `gold_completeness: decisive_error_only`, so its precision is not interpretable. See [labeled_trace_corpora.md](labeled_trace_corpora.md).

## Do not

- Quote development F1 as held-out F1.
- Add trace-id-specific rules after seeing held-out misses.
- Report `--mock` as an accuracy measurement. It only checks plumbing.
