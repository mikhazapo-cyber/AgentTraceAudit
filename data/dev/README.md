# Development traces (17)

The labelled subset used while building the system: 5 traces carry 7 gold findings, 12 are clean. Pair with `data/labels/dev`.

One annotator, one review. These are development labels, not a held-out set.

```
traceaudit data/dev --labels data/labels/dev --yes --out traceaudit-out/dev
```
