#!/usr/bin/env python3
"""Consolidate static, same-material GLB chunks for a single-primitive face tool.

Concatenates vertex rows without welding, recomputing or changing their bytes.
Rebases triangle indices to uint32 and retains the entire original BIN prefix.
Different materials, differing attribute layouts and deformed assets are refused.
Requires NumPy; no model, GPU or upstream face-tool import is required.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import struct

import numpy as np

DTYPES = {5121: '<u1', 5123: '<u2', 5125: '<u4', 5126: '<f4'}
WIDTHS = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4}
ATTRIBUTES = {'POSITION': 'VEC3', 'NORMAL': 'VEC3', 'TEXCOORD_0': 'VEC2', 'TANGENT': 'VEC4'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def ref(doc, key, index):
    values = doc.get(key, [])
    require(type(index) is int and 0 <= index < len(values), f'Invalid {key} reference')
    return values[index]


def parse_glb(raw):
    require(len(raw) >= 28, 'Truncated GLB')
    magic, version, length = struct.unpack_from('<4sII', raw)
    require(magic == b'glTF' and version == 2 and length == len(raw), 'Expected complete GLB 2.0')
    chunks = []
    offset = 12
    while offset < length:
        require(offset + 8 <= length, 'Truncated chunk header')
        size, kind = struct.unpack_from('<II', raw, offset)
        require(size % 4 == 0 and offset + 8 + size <= length, 'Invalid chunk extent')
        chunks.append((kind, raw[offset + 8:offset + 8 + size]))
        offset += 8 + size
    require([kind for kind, _ in chunks] == [0x4E4F534A, 0x004E4942], 'Expected JSON then embedded BIN only')
    return json.loads(chunks[0][1]), chunks[1][1]


def pack_glb(doc, binary):
    payload = json.dumps(doc, separators=(',', ':'), allow_nan=False).encode()
    payload += b' ' * (-len(payload) % 4)
    binary = bytes(binary) + b'\0' * (-len(binary) % 4)
    return (struct.pack('<4sII', b'glTF', 2, 28 + len(payload) + len(binary))
            + struct.pack('<II', len(payload), 0x4E4F534A) + payload
            + struct.pack('<II', len(binary), 0x004E4942) + binary)


def read_accessor(doc, binary, index):
    accessor = ref(doc, 'accessors', index)
    require(not accessor.get('sparse') and not accessor.get('normalized') and not accessor.get('extensions'),
            'Sparse, normalized and extended accessors are unsupported')
    require(accessor.get('componentType') in DTYPES and accessor.get('type') in WIDTHS, 'Unsupported accessor layout')
    view = ref(doc, 'bufferViews', accessor.get('bufferView'))
    require(view.get('buffer', 0) == 0 and not view.get('extensions'), 'External/compressed buffer view')
    count = accessor.get('count')
    require(type(count) is int and count > 0, 'Empty or invalid accessor')
    dtype = np.dtype(DTYPES[accessor['componentType']])
    width = WIDTHS[accessor['type']]
    row_bytes = dtype.itemsize * width
    stride = view.get('byteStride', row_bytes)
    start = view.get('byteOffset', 0)
    relative = accessor.get('byteOffset', 0)
    length = view.get('byteLength')
    require(all(type(x) is int for x in (stride, start, relative, length)), 'Invalid accessor offsets')
    require(stride >= row_bytes and stride % dtype.itemsize == 0 and start >= 0 and relative >= 0
            and length >= 0 and start + length <= len(binary)
            and relative + (count - 1) * stride + row_bytes <= length, 'Accessor exceeds buffer view')
    return np.ndarray((count, width), dtype=dtype, buffer=binary,
                      offset=start + relative, strides=(stride, dtype.itemsize)).copy()


def consolidate(doc, binary):
    require(doc.get('asset', {}).get('version') == '2.0', 'Expected glTF 2.0')
    require(not doc.get('skins') and not doc.get('animations') and not doc.get('extensions'), 'Only static assets supported')
    buffers = doc.get('buffers', [])
    require(len(buffers) == 1 and 'uri' not in buffers[0] and 0 <= buffers[0]['byteLength'] <= len(binary), 'Expected one embedded buffer')
    require(len(doc.get('meshes', [])) == 1, 'Expected one mesh')
    mesh = doc['meshes'][0]
    require(not mesh.get('weights') and not mesh.get('extensions'), 'Deformed/extended mesh unsupported')
    nodes = doc.get('nodes', [])
    require(len(nodes) == 1 and nodes[0].get('mesh') == 0 and not nodes[0].get('children')
            and not any(k in nodes[0] for k in ('skin', 'weights', 'extensions')),
            'Expected one static mesh instance')
    require(doc.get('scene', 0) == 0 and len(doc.get('scenes', [])) == 1
            and doc['scenes'][0].get('nodes') == [0], 'Expected one explicit scene instance')
    primitives = mesh.get('primitives', [])
    require(bool(primitives), 'No primitives')
    material = primitives[0].get('material')
    ref(doc, 'materials', material)
    names = set(primitives[0].get('attributes', {}))
    require({'POSITION', 'TEXCOORD_0'} <= names <= set(ATTRIBUTES), 'Unsupported vertex attributes')
    arrays = {name: [] for name in sorted(names)}
    indices = []
    vertices = 0
    for primitive in primitives:
        require(set(primitive) <= {'attributes', 'indices', 'material', 'mode'}, 'Extended/deformed primitive unsupported')
        require(primitive.get('mode', 4) == 4 and primitive.get('material') == material, 'All chunks must share one material and triangle mode')
        require(set(primitive.get('attributes', {})) == names, 'Attribute layouts must match across chunks')
        rows = {}
        for name in arrays:
            index = primitive['attributes'][name]
            accessor = ref(doc, 'accessors', index)
            require(accessor.get('componentType') == 5126 and accessor.get('type') == ATTRIBUTES[name], 'Expected float32 vertex attributes')
            rows[name] = read_accessor(doc, binary, index)
            require(np.isfinite(rows[name]).all(), 'Nonfinite vertex attribute')
        count = len(rows['POSITION'])
        require(all(len(value) == count for value in rows.values()), 'Mismatched vertex counts')
        require((rows['TEXCOORD_0'] >= -1e-6).all() and (rows['TEXCOORD_0'] <= 1 + 1e-6).all(), 'Tiled UVs unsupported by face painter')
        index = primitive.get('indices')
        accessor = ref(doc, 'accessors', index)
        require(accessor.get('type') == 'SCALAR' and accessor.get('componentType') in (5121, 5123, 5125), 'Expected unsigned triangle indices')
        local = read_accessor(doc, binary, index).reshape(-1).astype(np.uint64)
        require(len(local) % 3 == 0 and int(local.max()) < count, 'Invalid triangle indices')
        require(vertices + count <= np.iinfo(np.uint32).max, 'Vertex table exceeds uint32')
        indices.append((local + vertices).astype('<u4').reshape(-1, 1))
        for name, value in rows.items():
            arrays[name].append(value)
        vertices += count
    expected = {name: np.concatenate(values) for name, values in arrays.items()}
    expected_indices = np.concatenate(indices)
    result = copy.deepcopy(doc)
    emitted = bytearray(binary)

    def append(array, kind, component, target):
        emitted.extend(b'\0' * (-len(emitted) % 4))
        view = len(result['bufferViews'])
        result['bufferViews'].append({'buffer': 0, 'byteOffset': len(emitted), 'byteLength': array.nbytes, 'target': target})
        emitted.extend(array.tobytes())
        result['accessors'].append({'bufferView': view, 'componentType': component, 'count': len(array), 'type': kind,
                                    'min': array.min(axis=0).tolist(), 'max': array.max(axis=0).tolist()})
        return len(result['accessors']) - 1

    attributes = {name: append(value, ATTRIBUTES[name], 5126, 34962) for name, value in expected.items()}
    # Preserve the viewer's original union bounds. Exporters may round JSON
    # bounds independently of float32 vertex rows; recomputing them can subtly
    # move auto-framed cameras even though the actual geometry is unchanged.
    position_sources = [ref(doc, 'accessors', p['attributes']['POSITION']) for p in primitives]
    if all('min' in a and 'max' in a for a in position_sources):
        for bound, reducer in (('min', np.min), ('max', np.max)):
            values = np.asarray([a[bound] for a in position_sources], dtype=np.float64)
            require(values.shape == (len(primitives), 3) and np.isfinite(values).all(), 'Invalid position bounds')
            for primitive, value in zip(primitives, values):
                rows = read_accessor(doc, binary, primitive['attributes']['POSITION'])
                require(np.allclose(value, reducer(rows, axis=0), rtol=1e-6, atol=1e-7), 'Inaccurate position bounds')
            result['accessors'][attributes['POSITION']][bound] = reducer(values, axis=0).tolist()
    merged_indices = append(expected_indices, 'SCALAR', 5125, 34963)
    result['meshes'][0]['primitives'] = [{'attributes': attributes, 'indices': merged_indices, 'material': material, 'mode': 4}]
    result['buffers'][0]['byteLength'] = len(emitted)
    emitted = bytes(emitted)
    # Read the actual emitted accessor bytes back, independently of append intent.
    for name, index in attributes.items():
        require(read_accessor(result, emitted, index).tobytes() == expected[name].tobytes(), 'Emitted attribute mismatch')
    require(read_accessor(result, emitted, merged_indices).tobytes() == expected_indices.tobytes(), 'Emitted triangle mismatch')
    require(emitted[:len(binary)] == binary, 'Original binary prefix changed')
    proof = {'format': 'uniforge3d-face-compat-v1', 'input_primitives': len(primitives), 'output_primitives': 1,
             'vertices': vertices, 'triangles': len(expected_indices) // 3,
             'geometry_exact': True, 'attribute_rows_byte_exact': True, 'triangle_order_and_winding_exact': True,
             'original_binary_prefix_exact': True, 'materials_images_textures_unchanged': True,
             'index_component_type': 5125,
             'original_viewer_bounds_preserved': all('min' in a and 'max' in a for a in position_sources),
             'attribute_sha256': {name: hashlib.sha256(value.tobytes()).hexdigest() for name, value in expected.items()},
             'rebased_index_sha256': hashlib.sha256(expected_indices.tobytes()).hexdigest(),
             'scope': 'Same-material static chunks only; no vertex welding, material atlas merge or face correction.'}
    return result, emitted, proof


def convert_file(source, output, proof_path):
    source, output, proof_path = map(Path, (source, output, proof_path))
    require(len({p.resolve() for p in (source, output, proof_path)}) == 3, 'Source/output/proof must be distinct')
    require(not output.exists() and not proof_path.exists(), 'Refusing to overwrite existing output or proof')
    raw = source.read_bytes()
    doc, binary = parse_glb(raw)
    result, emitted, proof = consolidate(doc, binary)
    packed = pack_glb(result, emitted)
    # Verify the serialized container, including preservation of original data.
    checked, checked_binary = parse_glb(packed)
    require(checked == result and checked_binary[:len(binary)] == binary, 'Serialized GLB verification failed')
    proof.update(input_sha256=hashlib.sha256(raw).hexdigest(), output_sha256=hashlib.sha256(packed).hexdigest(),
                 input_bytes=len(raw), output_bytes=len(packed))
    output.parent.mkdir(parents=True, exist_ok=True)
    proof_path.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as handle:
        handle.write(packed)
    try:
        with proof_path.open('x') as handle:
            handle.write(json.dumps(proof, indent=2) + '\n')
    except Exception:
        output.unlink()  # only the newly created file belongs to this invocation
        raise
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--proof', required=True, type=Path)
    args = parser.parse_args()
    proof = convert_file(args.source, args.output, args.proof)
    print(json.dumps(proof, indent=2))


if __name__ == '__main__':
    main()
