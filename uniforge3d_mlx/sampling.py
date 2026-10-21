"""The source pipeline's Euler/CFG schedule, including x0 standard-deviation rescaling."""

from __future__ import annotations

import mlx.core as mx
import numpy as np


def sample_euler(noise, predict, config, *, on_step=None, sparse=False):
    params = config["params"]
    steps = int(params["steps"])
    rescale_t = float(params.get("rescale_t", 1.0))
    sigma_min = float(config["args"]["sigma_min"])
    times = np.linspace(1, 0, steps + 1)
    times = rescale_t * times / (1 + (rescale_t - 1) * times)
    x = noise.astype(mx.float32)
    interval = params.get("guidance_interval", (0, 1))
    for index, (current, previous) in enumerate(zip(times[:-1], times[1:])):
        t = float(current)
        strength = params["guidance_strength"] if interval[0] <= t <= interval[1] else 1
        if strength in (0, 1):
            prediction = predict(x, t, bool(strength))
        else:
            positive = predict(x, t, True)
            mx.eval(positive)
            negative = predict(x, t, False)
            if params.get("guidance_as_difference", False):
                prediction = negative + strength * (positive - negative)
            else:
                prediction = strength * positive + (1 - strength) * negative
            rescale = params.get("guidance_rescale", 0)
            if rescale > 0:
                sigma = sigma_min + (1 - sigma_min) * t
                x0_positive = (1 - sigma_min) * x - sigma * positive
                x0_guided = (1 - sigma_min) * x - sigma * prediction
                # SparseTensor.std groups all N*C features of each case, while
                # dense tensors reduce all dimensions except batch.
                axes = tuple(range(x.ndim)) if sparse else tuple(range(1, x.ndim))

                def std(value):
                    if sparse:
                        mean = mx.mean(value, axis=axes, keepdims=True)
                        return mx.sqrt(
                            mx.mean(value * value, axis=axes, keepdims=True)
                            - mean * mean
                        )
                    return mx.sqrt(mx.var(value, axis=axes, keepdims=True, ddof=1))

                positive_std = std(x0_positive)
                guided_std = mx.maximum(std(x0_guided), 1e-6)
                scaled = x0_guided * (positive_std / guided_std)
                x0 = rescale * scaled + (1 - rescale) * x0_guided
                prediction = ((1 - sigma_min) * x - x0) / sigma
        x = x - float(current - previous) * prediction
        mx.eval(x)
        if not bool(mx.all(mx.isfinite(x)).item()):
            raise FloatingPointError(f"Nonfinite flow sample at step {index + 1}")
        if on_step is not None:
            on_step(index + 1, steps, x)
    return x
