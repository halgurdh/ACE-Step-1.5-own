#!/usr/bin/env python3
"""Standalone CLI wrapper around Meta's audiobox-aesthetics predictor.

Scores one or more WAV files in a single model load (batched) and prints a
JSON object mapping each input path to its {CE, CU, PC, PQ} scores:
  CE = Content Enjoyment, CU = Content Usefulness,
  PC = Production Complexity, PQ = Production Quality.
"""

import argparse
import json

from audiobox_aesthetics.infer import initialize_predictor


def main() -> int:
    parser = argparse.ArgumentParser(description="Score WAV files with audiobox-aesthetics.")
    parser.add_argument("--input", action="append", required=True, help="WAV path(s) to score.")
    args = parser.parse_args()

    predictor = initialize_predictor()
    results = predictor.forward([{"path": path} for path in args.input])
    print(json.dumps(dict(zip(args.input, results))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
