#!/usr/bin/env python3
"""Build selective oQ5 component checkpoints using the installed oMLX engine."""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys
import shutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "weights/source")
    parser.add_argument("--output", type=Path, default=ROOT / "weights/UniForge3D-oQ5")
    parser.add_argument("--omlx-root", type=Path, default=Path("/Users/op/AI/omlx"))
    parser.add_argument("--component", action="append")
    parser.add_argument("--resume", action="store_true", help="Discard only incomplete generated component directories and retry")
    args = parser.parse_args()
    from uniforge3d_mlx.source import COMPONENTS, component_source
    from uniforge3d_mlx.quantization import convert_component
    import mlx.core as mx
    import torch
    torch.set_num_threads(4)
    mx.set_cache_limit(256 * 1024 * 1024)
    args.output.mkdir(parents=True, exist_ok=True)
    status_path = args.output / "conversion-status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    for component in args.component or COMPONENTS:
        if component not in COMPONENTS:
            raise ValueError(f"Unknown component {component}")
        if (args.output / component / "manifest.json").is_file():
            print(f"EXISTS {component}; run audit for verification", flush=True)
            continue
        partial = args.output / (component + ".partial")
        if args.resume and partial.exists():
            if partial.is_symlink() or partial.resolve().parent != args.output.resolve():
                raise ValueError(f"Unsafe partial output path: {partial}")
            shutil.rmtree(partial)
        try:
            source, paths = component_source(args.source, component)
        except FileNotFoundError as error:
            status[component] = {"status": "waiting_for_source", "error": str(error)}
            print(f"WAITING {component}: {error}", flush=True)
        else:
            print(f"CONVERTING {component}: {len(list(source))} tensors", flush=True)
            manifest = convert_component(source, component, args.output, omlx_root=args.omlx_root, source_paths=paths)
            status[component] = {"status": "converted", "matrix_counts": manifest["matrix_counts"],
                                 "output_payload_bytes": manifest["output_payload_bytes"],
                                 "tensor_count": manifest["tensor_count"]}
            del source
            gc.collect()
            mx.clear_cache()
        temp = status_path.with_suffix(".tmp")
        temp.write_text(json.dumps(status, indent=2) + "\n")
        os.replace(temp, status_path)
    return int(any(v["status"] != "converted" for v in status.values()))


if __name__ == "__main__":
    raise SystemExit(main())
