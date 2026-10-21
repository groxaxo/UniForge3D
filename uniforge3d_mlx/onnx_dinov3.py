"""Recover the inference weights from a pinned public FP32 DINOv3 export.

This is deliberately specific to the verified export, not a generic ONNX
converter. ONNX stores Linear matrices transposed and eliminates the zero
query/value biases. The training-only mask token does not affect this graph.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import re

import numpy as np

REPO = "onnx-community/dinov3-vitl16-pretrain-lvd1689m-ONNX"
REVISION = "9f74690bbbba9d76c4cce9062195fc7c95e23bb5"
FILES = {
    "onnx/model.onnx": (269737, "f81e34930ac50985b8ee7d71cbd93bf36cf07e94bb85e016cfbd5d157688e5b1"),
    "onnx/model.onnx_data": (1212317696, "cbb0241f84041fae7e4237110c88b42aff17e1e7cbb290cdbda4bbd73c60a5c8"),
}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_download(root):
    result = {}
    for name, (size, expected) in FILES.items():
        path = Path(root) / name
        actual = sha256(path)
        if path.stat().st_size != size or actual != expected:
            raise ValueError(f"Pinned ONNX source integrity failed: {name}")
        result[name] = {"bytes": size, "sha256": actual, "verified": True}
    return result


def transpose_linear(value, expected_shape):
    """ONNX MatMul stores [input, output]; HF/MLX store [output, input]."""
    if value.dtype != np.float32 or value.ndim != 2:
        raise ValueError("Expected an unquantized FP32 ONNX matrix")
    restored = np.ascontiguousarray(value.T)
    if restored.shape != tuple(expected_shape) or not np.isfinite(restored).all():
        raise ValueError("ONNX matrix shape or values do not match the model")
    return restored


def extract_state(path):
    import onnx
    from onnx import numpy_helper

    path = Path(path)
    if sha256(path) != FILES["onnx/model.onnx"][1]:
        raise ValueError("Unsupported ONNX graph revision")
    model = onnx.load(str(path), load_external_data=False)
    tensors = {t.name: t for t in model.graph.initializer}
    for tensor in tensors.values():
        external = {x.key: x.value for x in tensor.external_data}
        if external and external.get("location") != "model.onnx_data":
            raise ValueError("Unexpected external tensor file")
    named = re.compile(r"^(?:embeddings|layer\.\d+|norm)\.")
    state, mapping = {}, {}
    for name, tensor in tensors.items():
        if named.match(name):
            value = numpy_helper.to_array(tensor, base_dir=str(path.parent))
            if value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError(f"Invalid named tensor: {name}")
            state[name] = np.ascontiguousarray(value)
            mapping[name] = {"initializer": name, "operation": "identity"}
    matrices = [n for n in model.graph.node
                if n.op_type == "MatMul" and n.input[1] in tensors]
    if len(matrices) != 24 * 6:
        raise ValueError("Expected exactly six FP32 projection matrices per block")
    projections = ("attention.q_proj", "attention.k_proj", "attention.v_proj",
                   "attention.o_proj", "mlp.up_proj", "mlp.down_proj")
    shapes = ((1024, 1024),) * 4 + ((4096, 1024), (1024, 4096))
    consumers = {}
    for node in model.graph.node:
        for value in node.input:
            consumers.setdefault(value, []).append(node)
    for i in range(24):
        block = matrices[i * 6:(i + 1) * 6]
        norm1 = next(n for n in model.graph.node if n.op_type == "LayerNormalization"
                     and n.input[1] == f"layer.{i}.norm1.weight")
        norm2 = next(n for n in model.graph.node if n.op_type == "LayerNormalization"
                     and n.input[1] == f"layer.{i}.norm2.weight")
        if any(n.input[0] != norm1.output[0] for n in block[:3]):
            raise ValueError(f"Unexpected Q/K/V input mapping in block {i}")
        if block[4].input[0] != norm2.output[0]:
            raise ValueError(f"Unexpected MLP input mapping in block {i}")
        for j, (node, projection, shape) in enumerate(zip(block, projections, shapes)):
            name = f"layer.{i}.{projection}.weight"
            data = numpy_helper.to_array(tensors[node.input[1]], base_dir=str(path.parent))
            state[name] = transpose_linear(data, shape)
            mapping[name] = {"initializer": node.input[1], "node": node.name,
                             "operation": "transpose"}
            if j >= 3:
                bias = f"layer.{i}.{projection}.bias"
                if not any(n.op_type == "Add" and bias in n.input
                           for n in consumers[node.output[0]]):
                    raise ValueError(f"Projection bias mapping failed: {name}")
            elif any(n.op_type == "Add" for n in consumers[node.output[0]]):
                raise ValueError("Export contains an unexpected Q/K/V bias")
        for projection in ("q_proj", "v_proj"):
            name = f"layer.{i}.attention.{projection}.bias"
            state[name] = np.zeros(1024, dtype=np.float32)
            mapping[name] = {"operation": "zero bias absent from inference graph"}
    # HF expects this parameter even though the unmasked inference graph omits it.
    state["embeddings.mask_token"] = np.zeros((1, 1, 1024), dtype=np.float32)
    mapping["embeddings.mask_token"] = {"operation": "unused unmasked-inference placeholder"}
    if len(state) != 415:
        raise ValueError(f"Unexpected recovered tensor count: {len(state)}")
    used = {v["initializer"] for v in mapping.values() if "initializer" in v}
    external_names = {name for name, t in tensors.items() if t.external_data}
    if used != external_names:
        raise ValueError("Some external learned weights were omitted or misidentified")
    return state, mapping


def add_reference_output(path, destination):
    """Expose the exact pre-final-norm tensor, retaining both original outputs."""
    import onnx

    path, destination = Path(path), Path(destination)
    if destination.parent.resolve() != path.parent.resolve():
        raise ValueError("Reference graph must share the original external-data directory")
    model = onnx.load(str(path), load_external_data=False)
    final = next(n for n in model.graph.node
                 if n.op_type == "LayerNormalization" and "last_hidden_state" in n.output)
    if list(final.input[1:]) != ["norm.weight", "norm.bias"]:
        raise ValueError("Unexpected final normalization")
    raw_name = final.input[0]
    model.graph.output.append(onnx.helper.make_tensor_value_info(
        raw_name, onnx.TensorProto.FLOAT, ["batch", "tokens", 1024]))
    onnx.save(model, str(destination))
    return raw_name
