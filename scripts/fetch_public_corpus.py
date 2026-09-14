#!/usr/bin/env python3
"""Fetch public labelled agent-trace corpora (not redistributed in this repo)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

CORPORA = {
    "trail": {
        "hf": "PatronusAI/TRAIL",
        "note": "Gated. Accept the dataset terms, then `huggingface-cli login`. "
        "The card forbids resharing; write only to gitignored data/raw/trail.",
        "dest": "data/raw/trail",
    },
    "whowhen": {
        "hf": "Leoxx/whowhen_pro",
        "note": "CC-BY-4.0. Single decisive error per trace; use for localisation/recall, not precision.",
        "dest": "data/raw/whowhen",
    },
}
ALIASES = {
    "who-when-pro": "whowhen",
    "who_when_pro": "whowhen",
    "trail-benchmark": "trail",
}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--list", action="store_true")
    p.add_argument("--corpus", choices=sorted(set(CORPORA) | set(ALIASES)))
    args = p.parse_args()
    if args.list or not args.corpus:
        for name, info in CORPORA.items():
            print(f"{name}\n  {info['hf']}\n  {info['note']}\n  dest {info['dest']}\n")
        return 0
    info = CORPORA[ALIASES.get(args.corpus, args.corpus)]
    dest = Path(info["dest"])
    dest.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("pip install huggingface_hub  # then huggingface-cli login if the corpus is gated", file=sys.stderr)
        return 2
    snapshot_download(repo_id=info["hf"], repo_type="dataset", local_dir=str(dest))
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
