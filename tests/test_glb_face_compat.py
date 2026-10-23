"""Regression tests for exact, same-material GLB primitive consolidation."""
import copy
import importlib.util
import json
from pathlib import Path
import struct
import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location('glb_face_compat', Path(__file__).parents[1] / 'tools/glb_face_compat.py')
compat = importlib.util.module_from_spec(SPEC)
if SPEC.loader and Path(SPEC.origin).exists():
    SPEC.loader.exec_module(compat)


def fixture():
    chunks = []
    doc = {'asset': {'version': '2.0'}, 'scene': 0, 'scenes': [{'nodes': [0]}],
           'nodes': [{'mesh': 0}], 'meshes': [{'primitives': []}],
           'materials': [{'pbrMetallicRoughness': {'baseColorTexture': {'index': 0}}}],
           'images': [{'bufferView': 0, 'mimeType': 'image/png'}],
           'textures': [{'source': 0}], 'buffers': [{'byteLength': 0}],
           'bufferViews': [], 'accessors': []}
    binary = bytearray(b'original image bytes\0')
    doc['bufferViews'].append({'buffer': 0, 'byteOffset': 0, 'byteLength': len(binary)})
    def add(array, kind, component):
        binary.extend(b'\0' * (-len(binary) % 4))
        view = len(doc['bufferViews'])
        doc['bufferViews'].append({'buffer': 0, 'byteOffset': len(binary), 'byteLength': array.nbytes})
        binary.extend(array.tobytes())
        doc['accessors'].append({'bufferView': view, 'componentType': component, 'type': kind, 'count': len(array)})
        return len(doc['accessors']) - 1
    for z in [0., 1.]:
        pos = np.array([[0, 0, z], [1, 0, z], [0, 1, z]], dtype='<f4')
        attrs = {'POSITION': add(pos, 'VEC3', 5126),
                 'NORMAL': add(np.array([[0, 0, 1]] * 3, dtype='<f4'), 'VEC3', 5126),
                 'TEXCOORD_0': add(np.array([[0, 0], [1, 0], [0, 1]], dtype='<f4'), 'VEC2', 5126)}
        primitive = {'attributes': attrs, 'indices': add(np.array([0, 1, 2], dtype='<u2'), 'SCALAR', 5123), 'material': 0}
        doc['meshes'][0]['primitives'].append(primitive)
    doc['buffers'][0]['byteLength'] = len(binary)
    return doc, bytes(binary)


def test_consolidates_all_triangles_and_exact_attributes_without_touching_original_binary():
    doc, binary = fixture()
    updated, emitted, proof = compat.consolidate(doc, binary)
    assert emitted[:len(binary)] == binary
    assert updated['materials'] == doc['materials'] and updated['images'] == doc['images']
    assert updated['textures'] == doc['textures'] and updated['nodes'] == doc['nodes']
    assert updated['accessors'][:len(doc['accessors'])] == doc['accessors']
    primitive, = updated['meshes'][0]['primitives']
    assert primitive['material'] == 0
    for name in ('POSITION', 'NORMAL', 'TEXCOORD_0'):
        expected = np.concatenate([compat.read_accessor(doc, binary, p['attributes'][name]) for p in doc['meshes'][0]['primitives']])
        actual = compat.read_accessor(updated, emitted, primitive['attributes'][name])
        assert actual.tobytes() == expected.tobytes()
    indices = compat.read_accessor(updated, emitted, primitive['indices'])
    np.testing.assert_array_equal(indices.reshape(-1), [0, 1, 2, 3, 4, 5])
    assert updated['accessors'][primitive['indices']]['componentType'] == 5125
    assert proof['geometry_exact'] and proof['original_binary_prefix_exact']
    assert proof['triangles'] == 2 and proof['vertices'] == 6
    assert len(doc['meshes'][0]['primitives']) == 2  # input document unmodified


@pytest.mark.parametrize('mutation', ['material', 'normal', 'morph', 'invalid-index', 'normalized', 'instances'])
def test_refuses_incompatible_or_invalid_inputs(mutation):
    doc, binary = fixture()
    p = doc['meshes'][0]['primitives'][1]
    if mutation == 'material':
        doc['materials'].append(copy.deepcopy(doc['materials'][0])); p['material'] = 1
    elif mutation == 'normal': del p['attributes']['NORMAL']
    elif mutation == 'morph': p['targets'] = [{}]
    elif mutation == 'invalid-index':
        acc = doc['accessors'][p['indices']]; view = doc['bufferViews'][acc['bufferView']]
        binary = bytearray(binary); struct.pack_into('<H', binary, view['byteOffset'], 55); binary = bytes(binary)
    elif mutation == 'normalized': doc['accessors'][p['attributes']['POSITION']]['normalized'] = True
    elif mutation == 'instances': doc['nodes'].append({'mesh': 0}); doc['scenes'][0]['nodes'].append(1)
    with pytest.raises(ValueError): compat.consolidate(doc, binary)


def test_glb_roundtrip_and_refusal_to_overwrite_source_or_existing_result(tmp_path):
    doc, binary = fixture()
    src = tmp_path / 'source.glb'; dest = tmp_path / 'compat.glb'; proof = tmp_path / 'proof.json'
    src.write_bytes(compat.pack_glb(doc, binary))
    original = src.read_bytes()
    compat.convert_file(src, dest, proof)
    assert src.read_bytes() == original
    parsed, emitted = compat.parse_glb(dest.read_bytes())
    assert len(parsed['meshes'][0]['primitives']) == 1
    assert json.loads(proof.read_text())['geometry_exact']
    with pytest.raises((ValueError, FileExistsError)): compat.convert_file(src, src, tmp_path/'other.json')
    with pytest.raises((ValueError, FileExistsError)): compat.convert_file(src, dest, proof)


def test_interleaved_attributes_preserve_raw_rows():
    doc, binary = fixture()
    index = doc['meshes'][0]['primitives'][0]['attributes']['POSITION']
    accessor = doc['accessors'][index]; view = doc['bufferViews'][accessor['bufferView']]
    rows = np.array([[0, 0, 0, 123], [1, 0, 0, 124], [0, 1, 0, 125]], dtype='<f4')
    binary += b'\0' * (-len(binary) % 4)
    offset = len(binary); binary += rows.tobytes()
    view.update(byteOffset=offset, byteLength=rows.nbytes, byteStride=16)
    doc['buffers'][0]['byteLength'] = len(binary)
    result = compat.read_accessor(doc, binary, index)
    np.testing.assert_array_equal(result, rows[:, :3])


def test_preserves_original_viewer_bounds_metadata_despite_decimal_rounding():
    doc, binary = fixture()
    for p in doc['meshes'][0]['primitives']:
        a = doc['accessors'][p['attributes']['POSITION']]
        values = compat.read_accessor(doc, binary, p['attributes']['POSITION'])
        a['min'] = values.min(axis=0).astype(float).tolist()
        a['max'] = values.max(axis=0).astype(float).tolist()
        a['min'][0] = -1e-10
        a['max'][0] = 1.0000000001
    updated, _, _ = compat.consolidate(doc, binary)
    index = updated['meshes'][0]['primitives'][0]['attributes']['POSITION']
    assert updated['accessors'][index]['min'] == [-1e-10, 0., 0.]
    assert updated['accessors'][index]['max'] == [1.0000000001, 1., 1.]
