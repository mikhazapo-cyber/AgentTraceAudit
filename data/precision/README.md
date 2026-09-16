# Precision-experiment sample

Offline index of the checked traces in `data/checked` + `data/labels/checked`.

```powershell
traceaudit data/checked --labels data/labels/checked --yes --out traceaudit-out/checked
```

That command spends API credits only if you run it. `pytest` does not. This folder itself is just the listing (101 traces). Numbers here are not held-out. AEGIS rows are injected. Eval also reports a mappable subset (family ≠ `other`, not injected); that is not held-out either.
