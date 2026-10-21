#!/usr/bin/env python3
"""Verify downloaded files against Hub sizes and published LFS SHA-256."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "weights/source")
    args = parser.parse_args()
    status = json.loads((args.root / "download-status.json").read_text())
    report = {"repositories": {}, "all_required_sources_present": True}
    for name, entry in status.items():
        if entry["status"] != "complete":
            report["repositories"][name] = {"status": entry["status"], "repo": entry["repo"]}
            report["all_required_sources_present"] = False
            continue
        checks = []
        for file in entry["files"]:
            path = Path(entry["path"]) / file["path"]
            assert path.stat().st_size == file["bytes"], path
            digest = hashlib.sha256()
            with path.open("rb") as source:
                while block := source.read(8 * 1024 * 1024):
                    digest.update(block)
            actual = digest.hexdigest()
            expected = file.get("expected_sha256")
            if expected is not None:
                assert actual == expected, f"Hub SHA-256 mismatch: {path}"
            checks.append({"path": str(path), "bytes": file["bytes"], "sha256": actual,
                           "hub_lfs_sha256_matched": expected is not None})
            print(f"VERIFIED {name}/{file['path']}", flush=True)
        report["repositories"][name] = {"status": "verified", "revision": entry["revision"],
                                          "bytes": entry["bytes"], "files": checks}
    (args.root / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    return int(not report["all_required_sources_present"])


if __name__ == "__main__":
    raise SystemExit(main())
