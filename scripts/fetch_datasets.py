"""Download the public real datasets into data/raw and verify checksums.

Usage: python scripts/fetch_datasets.py [--root data/raw] [--force]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentq.datasets.fetch import fetch_datasets  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/raw")
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for path in fetch_datasets(args.root, args.force):
        print(f"ok  {path}")


if __name__ == "__main__":
    main()
