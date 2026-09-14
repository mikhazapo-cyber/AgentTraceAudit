# Deterministic development evaluation

Mode `deterministic`, 2026-09-14, no API calls:

```
traceaudit demo --deterministic --out results/dev-det
```

17 traces, 7 gold findings, 12 clean. Confirmed findings only; 0 abstentions.

| Mode | P | R | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| strict | 1.000 | 0.714 | 0.833 | 5 | 0 | 2 |
| relaxed | 1.000 | 0.714 | 0.833 | 5 | 0 | 2 |

Per family (strict):

| Family | P | R | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| incorrect_tool_use | 1.00 | 1.00 | 1.00 | 3 | 0 | 0 |
| evidence_contradiction | 1.00 | 1.00 | 1.00 | 1 | 0 | 0 |
| redundant_action | 1.00 | 1.00 | 1.00 | 1 | 0 | 0 |
| instruction_violation | - | 0.00 | 0.00 | 0 | 0 | 2 |

Zero false positives on the 12 clean traces. Cost $0. Latency 0.02s mean and p95.

Both misses are instruction-precondition golds: `external-009` (`get_attributes` only for superlatives) and `external-017` (gift-card balance check). Mechanical rules cannot reach them; the live path can.

Development result, not held-out.
