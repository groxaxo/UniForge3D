import numpy as np
import onnx
import pytest
from onnx import helper, TensorProto

from uniforge3d_mlx.onnx_dinov3 import add_reference_output, extract_state, transpose_linear


def test_recovered_matrix_matches_onnx_matmul_with_asymmetric_dimensions():
    rng = np.random.default_rng(24)
    onnx_weight = rng.normal(size=(7, 11)).astype(np.float32)
    inputs = rng.normal(size=(2, 3, 7)).astype(np.float32)
    recovered = transpose_linear(onnx_weight, (11, 7))
    np.testing.assert_array_equal(recovered.T, onnx_weight)
    # BLAS may accumulate C/F layouts differently; independently check both
    # FP32 products against FP64 instead of requiring identical rounding.
    expected = inputs.astype(np.float64) @ onnx_weight.astype(np.float64)
    np.testing.assert_allclose(inputs @ onnx_weight, expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(inputs @ recovered.T, expected, rtol=1e-6, atol=1e-6)
    with pytest.raises(ValueError, match="shape"):
        transpose_linear(onnx_weight, (7, 11))
    with pytest.raises(ValueError, match="FP32"):
        transpose_linear(onnx_weight.astype(np.int8), (11, 7))


def test_recovery_rejects_unknown_graph_instead_of_guessing_parameter_order(tmp_path):
    path = tmp_path / "model.onnx"
    path.write_bytes(b"unknown graph")
    with pytest.raises(ValueError, match="revision"):
        extract_state(path)


def test_reference_output_preserves_public_graph_and_external_data_location(tmp_path):
    raw = helper.make_tensor_value_info("raw", TensorProto.FLOAT, ["batch", "tokens", 1024])
    public = helper.make_tensor_value_info("last_hidden_state", TensorProto.FLOAT,
                                           ["batch", "tokens", 1024])
    norm = helper.make_node("LayerNormalization", ["raw", "norm.weight", "norm.bias"],
                            ["last_hidden_state"])
    graph = helper.make_graph([norm], "reference", [raw], [public])
    model = helper.make_model(graph)
    source, destination = tmp_path / "model.onnx", tmp_path / "reference.onnx"
    onnx.save(model, source)
    original = source.read_bytes()
    assert add_reference_output(source, destination) == "raw"
    restored = onnx.load(destination)
    assert [v.name for v in restored.graph.output] == ["last_hidden_state", "raw"]
    assert list(restored.graph.node[0].input) == ["raw", "norm.weight", "norm.bias"]
    assert source.read_bytes() == original
    with pytest.raises(ValueError, match="directory"):
        add_reference_output(source, tmp_path / "elsewhere/reference.onnx")


def test_selection_rejects_local_feature_damage_and_mismatched_reference():
    from copy import deepcopy
    from tools.report_onnx_dinov3 import choose_candidate
    identity = [{"case": "fixture", "split": "held_out", "size": 512, "image_sha256": "image"}]
    good = {"label": "accurate", "weight_bytes": 200, "parity": {
        "inputs": identity, "source_sha256": "source",
        "tests": [{**identity[0], "cosine": .99998, "minimum_token_cosine": .9998}],
        "multiview": {"cosine": .999999, "minimum_token_cosine": .99999},
    }}
    damaged = deepcopy(good)
    damaged.update(label="smaller", weight_bytes=100)
    damaged["parity"]["tests"][0]["minimum_token_cosine"] = .94
    assert choose_candidate([damaged, good], identity, "source")["label"] == "accurate"
    with pytest.raises(ValueError, match="No tested candidate"):
        choose_candidate([damaged], identity, "source")
    with pytest.raises(ValueError, match="different frozen inputs"):
        choose_candidate([good], identity, "different source")
    compact = deepcopy(good)
    compact.update(label="compact", weight_bytes=100)
    compact["parity"]["tests"][0]["cosine"] = .9994
    assert choose_candidate([compact, good], identity, "source")["label"] == "accurate"
    assert choose_candidate([compact, good], identity, "source", preference="compact")["label"] == "compact"
