"""CPU mesh handoff, preserving the original triangle/box SAT voxelization."""

from __future__ import annotations

from typing import Any
import numpy as np


def shape_handoff(
    coordinates, fields, config, handoff, *, preview_path=None, log=print
):
    """Decode the 512 mesh, apply native narrow-band remeshing, voxelize at 64."""
    from mlx_spatial.ovoxel import flexi_dual_grid_fields_to_mesh
    from mlx_spatial.spatialkit.mesh import (
        fill_holes,
        remesh_narrow_band,
        simplify_mesh_mlx_parallel_qem,
    )
    import trimesh

    mesh = flexi_dual_grid_fields_to_mesh(coordinates, fields, grid_size=512)
    if not len(mesh.faces):
        raise ValueError("The shape decoder produced no mesh triangles")
    log(f"Shape512 mesh: {len(mesh.vertices):,} vertices, {len(mesh.faces):,} faces")
    if len(mesh.faces) > config["pre_simplify_target"]:
        mesh, _ = simplify_mesh_mlx_parallel_qem(
            mesh.vertices, mesh.faces, target_faces=config["pre_simplify_target"]
        )
    mesh, _ = fill_holes(mesh.vertices, mesh.faces, max_hole_perimeter=0.03)
    mesh, remesh_stats = remesh_narrow_band(
        mesh.vertices,
        mesh.faces,
        resolution=config["grid_size"],
        band=config["remesh_band"],
        project_back=config["remesh_project"],
    )
    if len(mesh.faces) > config["decimation_target"]:
        mesh, _ = simplify_mesh_mlx_parallel_qem(
            mesh.vertices, mesh.faces, target_faces=config["decimation_target"]
        )
    if preview_path is not None:
        vertices = np.array(mesh.vertices)[:, [0, 2, 1]] * [1.0, 1.0, -1.0]
        trimesh.Trimesh(vertices=vertices, faces=mesh.faces, process=False).export(
            preview_path
        )
    normalized, transform = normalize_to_cube(mesh.vertices, handoff["target_max_side"])
    coords = surface_voxelize(
        normalized, mesh.faces, handoff["surface_voxel_resolution"]
    )
    return np.pad(coords, ((0, 0), (1, 0))), {
        "normalization": transform,
        "remesh": remesh_stats,
        "vertices": len(mesh.vertices),
        "faces": len(mesh.faces),
    }


def normalize_to_cube(
    vertices: np.ndarray, target_max_side: float
) -> tuple[np.ndarray, dict[str, Any]]:
    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise ValueError("Expected nonempty [N,3] vertices")
    if not np.isfinite(vertices).all() or not 0 < target_max_side < 1:
        raise ValueError("Nonfinite mesh or invalid target side")
    lower, upper = vertices.min(axis=0), vertices.max(axis=0)
    center = (lower + upper) * 0.5
    longest_side = float((upper - lower).max())
    if longest_side <= 1e-12:
        raise ValueError("Degenerate mesh bounds")
    scale = target_max_side / longest_side
    normalized = (vertices - center) * scale
    bounds = np.stack((normalized.min(axis=0), normalized.max(axis=0)))
    if not np.allclose(bounds.mean(axis=0), 0.0, atol=1e-10):
        raise RuntimeError("Normalized mesh is not AABB centered")
    if not np.isclose(np.ptp(normalized, axis=0).max(), target_max_side, atol=1e-10):
        raise RuntimeError("Normalized mesh does not have the requested longest side")
    if np.abs(normalized).max() >= 0.5:
        raise RuntimeError("Normalized mesh is outside the open unit cube")
    transform = np.eye(4)
    transform[:3, :3] *= scale
    transform[:3, 3] = -center * scale
    return normalized, {
        "center_model": center.tolist(),
        "uniform_scale": scale,
        "target_max_side": target_max_side,
        "source_bounds_model": [lower.tolist(), upper.tolist()],
        "normalized_bounds_model": bounds.tolist(),
        "model_to_normalized_4x4": transform.tolist(),
    }


def surface_voxelize(
    vertices: np.ndarray,
    faces: np.ndarray,
    resolution: int,
    *,
    max_faces_per_chunk: int = 100_000,
) -> np.ndarray:
    """Voxelize triangle surfaces with an exact triangle-box SAT test."""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if max_faces_per_chunk <= 0:
        raise ValueError("max_faces_per_chunk must be positive")

    half_size = 0.5 / resolution
    coord_chunks = []
    for start in range(0, len(faces), max_faces_per_chunk):
        triangles = vertices[faces[start : start + max_faces_per_chunk]]
        grid_triangles = (triangles + 0.5) * resolution
        lower = np.floor(grid_triangles.min(axis=1)).astype(np.int32)
        upper = np.floor(grid_triangles.max(axis=1)).astype(np.int32)
        lower = np.clip(lower, 0, resolution - 1)
        upper = np.clip(upper, 0, resolution - 1)
        dimensions = upper - lower + 1

        for dimensions_key in np.unique(dimensions, axis=0):
            triangle_ids = np.flatnonzero(np.all(dimensions == dimensions_key, axis=1))
            ranges = [np.arange(int(size), dtype=np.int32) for size in dimensions_key]
            offsets = np.stack(np.meshgrid(*ranges, indexing="ij"), axis=-1).reshape(
                -1, 3
            )
            candidate_coords = lower[triangle_ids, None, :] + offsets[None, :, :]
            candidate_coords = candidate_coords.reshape(-1, 3)
            candidate_triangles = np.broadcast_to(
                triangles[triangle_ids, None, :, :],
                (len(triangle_ids), len(offsets), 3, 3),
            ).reshape(-1, 3, 3)
            centers = -0.5 + (candidate_coords.astype(np.float64) + 0.5) / resolution
            intersects = _triangles_intersect_axis_aligned_boxes(
                candidate_triangles,
                centers,
                half_size,
            )
            coord_chunks.append(
                candidate_coords[intersects].astype(np.int32, copy=False)
            )

    coords = (
        np.concatenate(coord_chunks, axis=0)
        if coord_chunks
        else np.empty((0, 3), dtype=np.int32)
    )
    if coords.size == 0:
        raise RuntimeError("Surface voxelization returned no voxels")
    coords = np.unique(coords, axis=0)
    if int(coords.min()) < 0 or int(coords.max()) >= resolution:
        raise RuntimeError("Surface voxels are outside the canonical grid")
    return coords


def _triangles_intersect_axis_aligned_boxes(
    triangles: np.ndarray,
    centers: np.ndarray,
    half_size: float,
) -> np.ndarray:
    """Vectorized Akenine-Moller triangle/AABB overlap test."""
    local = triangles - centers[:, None, :]
    edge0 = local[:, 1] - local[:, 0]
    edge1 = local[:, 2] - local[:, 1]
    edge2 = local[:, 0] - local[:, 2]
    keep = np.ones(len(local), dtype=bool)
    epsilon = np.finfo(np.float64).eps * 32

    keep &= np.all(local.min(axis=1) <= half_size + epsilon, axis=1)
    keep &= np.all(local.max(axis=1) >= -half_size - epsilon, axis=1)

    for edge in (edge0, edge1, edge2):
        axes = (
            np.stack((np.zeros(len(edge)), edge[:, 2], -edge[:, 1]), axis=1),
            np.stack((-edge[:, 2], np.zeros(len(edge)), edge[:, 0]), axis=1),
            np.stack((edge[:, 1], -edge[:, 0], np.zeros(len(edge))), axis=1),
        )
        for axis in axes:
            projection = np.einsum("nvc,nc->nv", local, axis)
            radius = half_size * np.abs(axis).sum(axis=1) + epsilon
            keep &= projection.min(axis=1) <= radius
            keep &= projection.max(axis=1) >= -radius

    normal = np.cross(edge0, local[:, 2] - local[:, 0])
    plane_distance = np.einsum("nc,nc->n", normal, local[:, 0])
    plane_radius = half_size * np.abs(normal).sum(axis=1) + epsilon
    keep &= np.abs(plane_distance) <= plane_radius
    return keep
