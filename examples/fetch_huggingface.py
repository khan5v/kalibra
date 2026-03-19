#!/usr/bin/env python3
"""Fetch a HuggingFace dataset and convert to Kalibra JSONL.

Each row becomes a trace. All fields are preserved as-is.
Run `kalibra inspect --suggest` on the output to discover field mappings.

Requirements:  pip install datasets

Examples:
    python kalibra-demo/fetch_huggingface.py Intelligent-Internet/ii-agent_gaia-benchmark_validation
    python kalibra-demo/fetch_huggingface.py \\
        AlexCuadron/SWE-Bench-Verified-O1-reasoning-high-results \\
        --split test

Then:
    kalibra inspect kalibra-demo/dataset.jsonl --suggest
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Fetch HuggingFace dataset → Kalibra JSONL")
    parser.add_argument("dataset", help="Dataset name (e.g. org/name)")
    parser.add_argument("--split", default="train", help="Split to load (default: train)")
    parser.add_argument("--output-dir", default="kalibra-demo", help="Output directory")
    parser.add_argument("--limit", type=int, default=None, help="Max rows to download")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("pip install datasets", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {args.dataset} (split={args.split})...")
    ds = load_dataset(args.dataset, split=args.split, token=os.environ.get("HF_TOKEN"))
    rows = list(ds)
    if args.limit:
        rows = rows[:args.limit]
    print(f"Loaded {len(rows)} rows — columns: {list(rows[0].keys()) if rows else '(empty)'}")

    traces = [row_to_trace(row) for row in rows]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = args.dataset.split("/")[-1][:60]
    path = out_dir / f"{slug}.jsonl"

    with open(path, "w") as f:
        for t in traces:
            f.write(json.dumps(t, default=str) + "\n")

    print(f"Wrote {len(traces)} traces to {path}")
    print(f"\nNext: kalibra inspect {path} --suggest")


def row_to_trace(row: dict) -> dict:
    """Convert a row to a Kalibra trace. Preserve everything, guess nothing."""
    trace: dict = {}
    for k, v in row.items():
        if isinstance(v, (dict, list)):
            trace[k] = json.dumps(v, default=str) if v else None
        else:
            trace[k] = v
    return trace


if __name__ == "__main__":
    main()
