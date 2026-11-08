"""Verify published dataset fingerprints without importing MLX or downloading data."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="also check optional datasets present locally")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    inventory = json.loads((root / "docs/dataset_fingerprints.json").read_text())
    failed = False
    for item in inventory["datasets"]:
        if not item["bundled"] and not args.all:
            continue
        path = root / item["path"]
        if not path.exists():
            print(f"{'MISSING REQUIRED' if item['bundled'] else 'SKIP OPTIONAL'} {item['path']}")
            failed |= item["bundled"]
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        matches = actual == item["sha256"]
        failed |= not matches
        print(f"{'OK' if matches else 'MISMATCH'} {item['path']} ({item['examples']} expected examples)")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
