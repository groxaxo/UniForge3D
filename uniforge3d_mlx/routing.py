"""CPU geometry routing; the learned point maps are computed in MLX."""

from __future__ import annotations

import itertools
import math

import numpy as np


def route_views(
    points, confidence, masks, target_coords, *, grid, keep=0.95, iterations=64
):
    """Port of GPUVoxelViewRouter, including stable component/tie priorities.

    Operates on one case, returning one selected view for each input coordinate.
    Discrete routing has no model weights and does not allocate GPU memory.
    """
    points = np.asarray(points, dtype=np.float32)
    confidence = np.asarray(confidence, dtype=np.float32)
    masks = np.asarray(masks, dtype=bool)
    coords = np.asarray(target_coords, dtype=np.int64)
    if points.shape[0] != 1 or points.shape[2:4] != (448, 448) or grid not in (32, 64):
        raise ValueError("Router expects one case, 448px point maps, and grid 32 or 64")
    views = points.shape[1]
    if np.any(coords[:, 0] != 0) or np.any(
        (coords[:, 1:] < 0) | (coords[:, 1:] >= grid)
    ):
        raise ValueError("Target coordinates must lie in the selected grid, batch zero")
    if views == 1:
        return np.zeros(len(coords), dtype=np.int32)
    volume, patches = grid**3, grid**2
    p = points.reshape(views, -1, 3)
    conf = confidence.reshape(views, -1)
    valid = masks.reshape(views, -1) & np.isfinite(p).all(-1) & np.isfinite(conf)
    valid &= ((p >= -0.5) & (p < 0.5)).all(-1)
    for view in range(views):
        ids = np.flatnonzero(valid[view])
        chosen = ids[
            np.argsort(-conf[view, ids], kind="stable")[: math.ceil(len(ids) * keep)]
        ]
        valid[view] = False
        valid[view, chosen] = True
    view, pixel = np.nonzero(valid)
    if len(view) == 0:
        return np.zeros(len(coords), dtype=np.int32)
    xyz = np.floor((p[view, pixel] + 0.5) * grid).astype(np.int64)
    voxel = (xyz[:, 0] * grid + xyz[:, 1]) * grid + xyz[:, 2]
    patch_pixels = 448 // grid
    patch = (pixel // 448 // patch_pixels) * grid + pixel % 448 // patch_pixels
    keys = (view * volume + voxel) * patches + patch
    order = np.argsort(keys, kind="stable")
    keys, values = keys[order], conf[view, pixel][order]
    unique, starts, counts = np.unique(keys, return_index=True, return_counts=True)
    conf_sum = np.add.reduceat(values, starts)
    group, patch = unique // patches, unique % patches
    groups, group_ids = np.unique(group, return_inverse=True)
    length = len(unique)
    labels = np.arange(length)
    py, px = patch // grid, patch % grid
    neighbors = []
    for dy, dx in itertools.product((-1, 0, 1), repeat=2):
        if (dy, dx) == (0, 0):
            continue
        y, x = py + dy, px + dx
        inside = (y >= 0) & (y < grid) & (x >= 0) & (x < grid)
        key = group * patches + np.clip(y, 0, grid - 1) * grid + np.clip(x, 0, grid - 1)
        pos = np.searchsorted(unique, key)
        safe = np.minimum(pos, length - 1)
        found = inside & (pos < length) & (unique[safe] == key)
        neighbors.append(np.where(found, safe, np.arange(length)))
    for _ in range(iterations):
        old = labels
        labels = np.minimum.reduce([labels] + [labels[index] for index in neighbors])
        labels = labels[labels]
        if np.array_equal(old, labels):
            break
    support = np.bincount(labels, weights=counts, minlength=length).astype(np.int64)
    sums = np.zeros(length, dtype=np.float32)
    np.add.at(sums, labels, conf_sum)
    roots = np.flatnonzero(labels == np.arange(length))
    mean = sums[roots] / np.maximum(support[roots], 1)
    root_groups = group_ids[roots]
    # Per voxel/view component: support, confidence, then earliest component.
    order = np.lexsort((roots, -mean, -support[roots], root_groups))
    ordered = roots[order]
    first = np.r_[True, np.diff(root_groups[order]) != 0]
    chosen = ordered[first]
    g = groups
    component_support = support[chosen]
    component_mean = sums[chosen] / np.maximum(component_support, 1)
    voxels = g % volume
    views_for_group = g // volume
    anchor = (views_for_group == 0).astype(np.int32)
    # Between views: support, anchor, confidence, lowest view index.
    order = np.lexsort(
        (views_for_group, -component_mean, -anchor, -component_support, voxels)
    )
    first = np.r_[True, np.diff(voxels[order]) != 0]
    winner = order[first]
    direct = voxels[winner]
    selected = np.zeros(volume, dtype=np.int32)
    selected[direct] = views_for_group[winner]
    has = np.zeros(volume, dtype=bool)
    has[direct] = True
    support_grid = np.zeros(volume, dtype=np.int64)
    support_grid[direct] = component_support[winner]
    anchor_grid = np.full(volume, -1, dtype=np.int32)
    anchor_grid[direct] = anchor[winner]
    mean_grid = np.full(volume, -np.inf, dtype=np.float32)
    mean_grid[direct] = component_mean[winner]
    target = (coords[:, 1] * grid + coords[:, 2]) * grid + coords[:, 3]
    missing = np.flatnonzero(~has[target])
    if len(missing):
        offsets = np.array(
            [p for p in itertools.product((-1, 0, 1), repeat=3) if p != (0, 0, 0)]
        )
        neighbor = coords[missing, 1:, None].transpose(0, 2, 1) + offsets[None]
        inside = ((neighbor >= 0) & (neighbor < grid)).all(-1)
        neighbor = np.clip(neighbor, 0, grid - 1)
        ids = (neighbor[..., 0] * grid + neighbor[..., 1]) * grid + neighbor[..., 2]
        valid = inside & has[ids]
        support_values = np.where(valid, support_grid[ids], -1)
        winners = valid & (support_values == support_values.max(1, keepdims=True))
        anchors = np.where(winners, anchor_grid[ids], -1)
        winners &= anchors == anchors.max(1, keepdims=True)
        means = np.where(winners, mean_grid[ids], -np.inf)
        winners &= means == means.max(1, keepdims=True)
        choice = np.where(winners, selected[ids], views).min(1)
        good = choice < views
        selected[target[missing[good]]] = choice[good]
    return selected[target]
