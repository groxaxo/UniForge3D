"""Stagewise, resumable UniForge3D image-to-textured-GLB inference on Apple Silicon."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

import mlx.core as mx
import numpy as np

from .checkpoint import load_checkpoint
from .export_timings import export_timing_breakdown
from .decoding import decode_sparse, decode_structure
from .dinov3 import DinoV3Encoder
from .flow import JointStructureFlow, RoutedFlow
from .geometry import shape_handoff
from .ops import release_memory
from .preprocessing import InputBatch, preprocess
from .routing import route_views
from .sampling import sample_euler
from .vggt import VGGT, adapt_features


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as file:
        for data in iter(lambda: file.read(8 * 1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def write_json(path, data):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(
        json.dumps(
            data,
            indent=2,
            default=lambda x: x.item() if isinstance(x, np.generic) else str(x),
        )
        + "\n"
    )
    os.replace(temporary, path)


class Pipeline:
    def __init__(
        self,
        bundle,
        output_dir,
        *,
        config_path=None,
        seed=42,
        steps=None,
        resume=False,
        high_res_precision="mixed",
        log=print,
    ):
        self.bundle = Path(bundle).resolve()
        self.output = Path(output_dir).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        config_path = (
            Path(config_path)
            if config_path
            else Path(__file__).resolve().parents[1]
            / "uniforge3d/configs/pipeline.json"
        )
        self.config = json.loads(config_path.read_text())
        if steps is not None:
            if steps < 1:
                raise ValueError("Step count must be positive")
            for stage in (
                "sparse_structure",
                "structured_latent_512",
                "structured_latent_1024",
                "texture",
            ):
                self.config[stage]["sampler"]["params"]["steps"] = steps
        self.seed = int(seed)
        self.resume = resume
        if high_res_precision not in ("float32", "mixed"):
            raise ValueError("High resolution precision must be float32 or mixed")
        self.high_res_precision = high_res_precision
        self.log = log
        self.report = {}
        self._cache_hits = set()

    def save(self, name, **arrays):
        target = self.output / (name + ".npz")
        temporary = self.output / (name + ".tmp.npz")
        np.savez(temporary, **{k: np.array(v) for k, v in arrays.items()})
        os.replace(temporary, target)
        self.report.setdefault("artifacts", {})[target.name] = {
            "sha256": sha256(target),
            "bytes": target.stat().st_size,
        }
        self.write_report()

    def cached(self, name):
        path = self.output / (name + ".npz")
        if not path.exists():
            return None
        expected = self.report.get("artifacts", {}).get(path.name)
        if expected is None or sha256(path) != expected["sha256"]:
            raise ValueError(
                f"Cannot resume: missing provenance or changed artifact {path.name}"
            )
        self.log(f"Using verified checkpoint {name}")
        self._cache_hits.add(name)
        with np.load(path, allow_pickle=False) as data:
            return {key: data[key] for key in data.files}

    def write_report(self):
        self.report["peak_mlx_memory_bytes"] = max(
            self.report.get("peak_mlx_memory_bytes", 0), mx.get_peak_memory()
        )
        write_json(self.output / "run.json", self.report)

    def stage(self, name, fn):
        self.log(f"Starting {name}")
        start = time.perf_counter()
        try:
            result = fn()
        except (Exception, KeyboardInterrupt) as error:
            self.report.update(
                status="interrupted"
                if isinstance(error, KeyboardInterrupt)
                else "failed",
                failed_stage=name,
                error=f"{type(error).__name__}: {error}",
            )
            self.write_report()
            raise
        elapsed = round(time.perf_counter() - start, 3)
        cache_name = "preprocessed" if name == "preprocess" else name
        cached = cache_name in self._cache_hits
        if not cached:
            self.report.setdefault("timings_seconds", {})[name] = elapsed
        self.report["attempts"][-1].setdefault("stages", {})[name] = {
            "elapsed_seconds": elapsed,
            "cached": cached,
        }
        self.write_report()
        release_memory()
        self.log(f"Finished {name} in {time.perf_counter() - start:.1f}s")
        return result

    def step_logger(self, stage):
        previous = time.perf_counter()

        def step(index, total, sample):
            nonlocal previous
            now = time.perf_counter()
            self.log(
                f"{stage}: step {index}/{total}, {now - previous:.1f}s, active MLX {mx.get_active_memory() / 1e9:.2f}GB"
            )
            previous = now

        return step

    def noise(self, shape, stage_index):
        # Stage-specific streams make resuming independent of preceding RNG calls.
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, stage_index]))
        return mx.array(rng.standard_normal(shape, dtype=np.float32))

    def run(self, images, *, stop_after=None):
        paths = [Path(path).resolve() for path in images]
        self._cache_hits.clear()
        if not paths:
            raise ValueError("At least one input image is required")
        inputs = [{"path": str(p), "sha256": sha256(p)} for p in paths]
        identity = {
            "runtime_contract": 1,
            "inputs": inputs,
            "seed": self.seed,
            "config": self.config,
            "bundle_manifest_sha256": sha256(self.bundle / "bundle.json"),
        }
        fingerprint = hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode()
        ).hexdigest()
        report_path = self.output / "run.json"
        if report_path.exists():
            if not self.resume:
                raise FileExistsError(
                    f"Run already exists: {self.output}; select a new directory or --resume"
                )
            self.report = json.loads(report_path.read_text())
            if self.report["fingerprint"] != fingerprint:
                raise ValueError(
                    "Cannot resume with different inputs, settings or model bundle"
                )
            prior_precision = self.report.get("high_res_precision", "float32")
            if prior_precision != self.high_res_precision and any(
                (self.output / (name + ".npz")).exists()
                for name in ("shape1024", "texture")
            ):
                raise ValueError(
                    f"Completed high-resolution stages used {prior_precision}; "
                    "resume with that precision or select a new output directory"
                )
        else:
            self.report = {
                **identity,
                "fingerprint": fingerprint,
                "artifacts": {},
                "timings_seconds": {},
            }
        self.report.update(
            status="running",
            backend="mlx",
            device=mx.device_info()["device_name"],
            tf32=os.environ.get("MLX_ENABLE_TF32"),
            bundle=str(self.bundle),
            high_res_precision=self.high_res_precision,
        )
        runtime_root = Path(__file__).parent
        self.report.setdefault("attempts", []).append(
            {
                "started_unix": time.time(),
                "source_sha256": {
                    path.name: sha256(path)
                    for path in sorted(runtime_root.glob("*.py"))
                },
            }
        )
        for key in ("stopped_after", "failed_stage", "error"):
            self.report.pop(key, None)
        self.write_report()
        try:
            return self._run(paths, stop_after)
        finally:
            release_memory()

    def _run(self, images, stop_after):
        def prepare():
            cached = self.cached("preprocessed")
            if cached is None:
                batch = preprocess(
                    images,
                    self.config["image_preprocess"],
                    self.bundle,
                    preview_dir=self.output / "inputs",
                    log=self.log,
                )
                self.save(
                    "preprocessed",
                    dino512=batch.dino512,
                    dino1024=batch.dino1024,
                    vggt=batch.vggt,
                    masks=batch.masks,
                )
                return batch
            return InputBatch(
                mx.array(cached["dino512"]),
                mx.array(cached["dino1024"]),
                mx.array(cached["vggt"]),
                cached["masks"],
            )

        batch = self.stage("preprocess", prepare)

        def encode():
            cached = self.cached("dino")
            if cached is not None:
                return {key: mx.array(value) for key, value in cached.items()}
            model = DinoV3Encoder(self.bundle / "dinov3")
            result = {}
            for size in (512, 1024):
                # Encoding views separately is mathematically identical and bounds memory.
                self.log(f"DINOv3 at {size}px")
                pixels = getattr(batch, f"dino{size}")
                features = [
                    model(pixels[:, index : index + 1])
                    for index in range(pixels.shape[1])
                ]
                result[f"features{size}"] = mx.concatenate(features, axis=1)
                mx.eval(result[f"features{size}"])
            self.save("dino", **result)
            return result

        dino = self.stage("dino", encode)

        def structure():
            cached = self.cached("structure")
            if cached is not None:
                return cached
            config = self.config["sparse_structure"]
            model = JointStructureFlow(
                self.bundle, config["model"]["args"], batch.vggt, dino["features512"]
            )
            latent = sample_euler(
                self.noise((1, 8, 16, 16, 16), 0),
                model,
                config["sampler"],
                on_step=self.step_logger("structure"),
            )
            del model
            release_memory()
            c32, c64, logits = decode_structure(self.bundle, latent)
            result = {
                "coords32": c32,
                "coords64": c64,
                "latent": latent,
                "logits": logits,
            }
            self.save("structure", **result)
            self.log(f"Occupied voxels: {len(c32):,} at 32^3; {len(c64):,} at 64^3")
            return result

        structure = self.stage("structure", structure)
        if self.stop("structure", stop_after):
            return None

        shape512 = self.stage(
            "shape512",
            lambda: self.sample_shape(
                "shape512",
                "structured_latent_512",
                structure["coords32"],
                structure["coords64"],
                batch,
                dino["features512"],
                1,
            ),
        )
        if self.stop("shape512", stop_after):
            return None

        def handoff():
            cached = self.cached("handoff")
            if cached is not None:
                return cached["coords64"]
            cached = self.cached("shape512_fields")
            if cached is None:
                coords, fields, _ = decode_sparse(
                    self.bundle,
                    structure["coords32"],
                    shape512["vae_features"],
                    log=self.log,
                )
                self.save("shape512_fields", coordinates=coords, fields=fields)
            else:
                coords, fields = (
                    mx.array(cached["coordinates"]),
                    mx.array(cached["fields"]),
                )
            release_memory()
            coords64, stats = shape_handoff(
                coords,
                fields,
                self.config["mesh_postprocess"],
                self.config["structured_latent_handoff"],
                preview_path=self.output / "shape512.glb",
                log=self.log,
            )
            self.report["handoff"] = stats
            self.save("handoff", coords64=coords64)
            self.log(f"Handoff contains {len(coords64):,} surface voxels at 64^3")
            return coords64

        coords64 = self.stage("handoff", handoff)
        if self.stop("handoff", stop_after):
            return None
        shape1024 = self.stage(
            "shape1024",
            lambda: self.sample_shape(
                "shape1024",
                "structured_latent_1024",
                coords64,
                coords64,
                batch,
                dino["features1024"],
                2,
            ),
        )
        if self.stop("shape1024", stop_after):
            return None

        def texture():
            cached = self.cached("texture")
            if cached is not None:
                return cached
            config = self.config["texture"]
            bank, selected = self.condition(
                "texture", coords64, coords64, batch, dino["features1024"]
            )
            normalization = config["shape_slat_normalization"]
            # Preserve the source VAE->flow round trip used by texture conditioning.
            concat = (
                mx.array(shape1024["vae_features"]) - mx.array(normalization["mean"])
            ) / mx.array(normalization["std"])
            model = RoutedFlow(
                self.bundle / "texture",
                config["model"]["args"]["denoiser_args"],
                coords64,
                bank,
                selected,
                concat_features=concat,
                compute_dtype=self.high_res_precision,
            )
            features = sample_euler(
                self.noise((len(coords64), 32), 3),
                model,
                config["sampler"],
                on_step=self.step_logger("texture"),
                sparse=True,
            )
            normalization = config["texture_slat_normalization"]
            vae = features * mx.array(normalization["std"]) + mx.array(
                normalization["mean"]
            )
            self.save("texture", vae_features=vae, flow_features=features)
            return {"vae_features": vae, "flow_features": features}

        texture = self.stage("texture", texture)
        if self.stop("texture", stop_after):
            return None

        def decode():
            cached = self.cached("shape_decoder_fields")
            if cached is None:
                coords, fields, subdivisions = decode_sparse(
                    self.bundle,
                    coords64,
                    mx.array(shape1024["vae_features"]),
                    log=self.log,
                )
                self.save(
                    "shape_decoder_fields",
                    coordinates=coords,
                    fields=fields,
                    metadata_json=np.array(
                        json.dumps(
                            {
                                "source_model": "uniforge3d",
                                "actual_hr_resolution": 1024,
                                "source_coordinate_system": "trellis-z-up",
                            }
                        )
                    ),
                )
                self.save(
                    "subdivisions",
                    **{str(i): sub for i, sub in enumerate(subdivisions)},
                )
            else:
                coords = mx.array(cached["coordinates"])
                sub = self.cached("subdivisions")
                if sub is None:
                    raise ValueError(
                        "Missing matching shape subdivisions for texture decoding"
                    )
                subdivisions = tuple(mx.array(sub[str(i)]) for i in range(4))
            release_memory()
            cached = self.cached("texture_decoder_pbr")
            if cached is None:
                tcoords, attrs, _ = decode_sparse(
                    self.bundle,
                    coords64,
                    mx.array(texture["vae_features"]),
                    texture=True,
                    guide=subdivisions,
                    log=self.log,
                )
                if not np.array_equal(np.array(coords), np.array(tcoords)):
                    raise ValueError(
                        "Texture and shape decoder coordinate orders differ"
                    )
                self.save(
                    "texture_decoder_pbr",
                    coordinates=tcoords,
                    attributes=attrs,
                    spatial_shape=np.array(tcoords[:, 1:]).max(0) + 1,
                    batch_size=np.array(1),
                    decode_resolution=np.array(1024),
                    voxel_size=np.array(1 / 1024),
                    metadata_json=np.array(
                        json.dumps(
                            {
                                "source_model": "uniforge3d",
                                "decode_resolution": 1024,
                                "source_coordinate_system": "trellis-z-up",
                            }
                        )
                    ),
                )

        self.stage("decode", decode)
        if self.stop("decode", stop_after):
            return None

        # Release all neural activations before CPU remeshing / UVs / texture baking.
        batch = dino = structure = shape512 = shape1024 = texture = None
        release_memory()

        def export():
            from mlx_spatial.ovoxel_export import export_ovoxel_glb

            output = self.output / "model.glb"
            exported = export_ovoxel_glb(
                self.output,
                output,
                texture_size=self.config["texture"]["texture_size"],
                target_faces=self.config["texture"]["decimation_target"],
                grid_size=1024,
            )
            breakdown = export_timing_breakdown(exported.diagnostics)
            self.report["export_timings_seconds"] = breakdown
            for name, seconds in breakdown.items():
                self.log(f"export {name}: {seconds:.1f}s")
            self.report["glb"] = {
                "path": str(output),
                "sha256": sha256(output),
                "bytes": output.stat().st_size,
            }
            return output

        output = self.stage("export", export)
        self.report["status"] = "complete"
        self.write_report()
        return output

    def condition(self, stage, target_coords, structure_coords, batch, dino):
        cached_name = "vggt_shape512" if stage == "shape512" else "vggt_shape1024"
        cache = self.cached(cached_name)
        views = batch.vggt.shape[1]
        if cache is None:
            model = VGGT(self.bundle / "vggt_conditioner", conditioner=True)
            features, points, confidence = model.conditioned_features(
                batch.vggt,
                mx.array(structure_coords[:, 1:], dtype=mx.int32),
                return_points=views > 1,
            )
            cache = {f"layer_{key}": value for key, value in features.items()}
            if views > 1:
                cache.update(points=points, confidence=confidence)
            self.save(cached_name, **cache)
            del model, features, points, confidence
            release_memory()
        features = {i: mx.array(cache[f"layer_{i}"]) for i in (4, 11, 17, 23)}
        adapter = load_checkpoint(self.bundle / (stage + "_adapter"))
        delta = adapt_features(features, adapter, upsample=stage != "shape512")
        bank = mx.concatenate((dino[:, :, :5], dino[:, :, 5:] + delta), axis=2)
        mx.eval(bank)
        del adapter, features, delta
        release_memory()
        if views == 1:
            selected = np.zeros(len(target_coords), dtype=np.int32)
        else:
            selected = route_views(
                cache["points"],
                cache["confidence"],
                batch.masks,
                target_coords,
                grid=32 if stage == "shape512" else 64,
            )
        return bank, selected

    def sample_shape(
        self, stage, config_key, coords, structure_coords, batch, dino, index
    ):
        cached = self.cached(stage)
        if cached is not None:
            return cached
        config = self.config[config_key]
        bank, selected = self.condition(stage, coords, structure_coords, batch, dino)
        model = RoutedFlow(
            self.bundle / stage,
            config["model"]["args"]["denoiser_args"],
            coords,
            bank,
            selected,
            compute_dtype="float32" if stage == "shape512" else self.high_res_precision,
        )
        features = sample_euler(
            self.noise((len(coords), 32), index),
            model,
            config["sampler"],
            on_step=self.step_logger(stage),
            sparse=True,
        )
        normalization = config["decoder"]["normalization"]
        vae = features * mx.array(normalization["std"]) + mx.array(
            normalization["mean"]
        )
        self.save(
            stage, flow_features=features, vae_features=vae, selected_view=selected
        )
        return {
            "flow_features": features,
            "vae_features": vae,
            "selected_view": selected,
        }

    def stop(self, stage, requested):
        if stage != requested:
            return False
        self.report["status"] = "checkpointed"
        self.report["stopped_after"] = stage
        self.write_report()
        return True
