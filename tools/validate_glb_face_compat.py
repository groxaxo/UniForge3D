#!/usr/bin/env python3
"""Validate the adapter through pinned face-tool CPU vision on a public fixture.

Run with the face tool's private Python 3.12 interpreter. This verifies complete
multi-primitive -> compatible GLB -> photo retexture execution, separately from
whether a face detector recognizes a particular generated character.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

from glb_face_compat import parse_glb, pack_glb, convert_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--face-tool', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model-glb', type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.face_tool.resolve()))
    import face_core as core
    from tests.test_vision import textured_grid, PORTRAIT
    args.output.mkdir(parents=True, exist_ok=False)
    root = args.output
    original = root / 'portrait-original.glb'
    textured_grid(original, PORTRAIT.read_bytes())
    doc, binary = parse_glb(original.read_bytes())
    prim = doc['meshes'][0]['primitives'][0]
    indices = core.accessor_array(doc, binary, prim['indices']).reshape(-1)
    chunks = []
    for part in np.split(indices, 2):
        binary += b'\0' * (-len(binary) % 4)
        view = len(doc['bufferViews'])
        doc['bufferViews'].append({'buffer': 0, 'byteOffset': len(binary), 'byteLength': part.nbytes})
        binary += part.tobytes()
        doc['accessors'].append({'bufferView': view, 'componentType': 5125, 'count': len(part), 'type': 'SCALAR'})
        chunk = copy.deepcopy(prim)
        chunk['indices'] = len(doc['accessors']) - 1
        chunks.append(chunk)
    doc['meshes'][0]['primitives'] = chunks
    doc['buffers'][0]['byteLength'] = len(binary)
    multi = root / 'portrait-multi.glb'
    multi.write_bytes(pack_glb(doc, binary))
    try:
        core.read_source(multi)
    except core.FaceTextureError as error:
        assert 'one primitive' in str(error), str(error)
    else:
        raise AssertionError('Expected original multi-primitive input to be refused')
    compatible = root / 'portrait-compatible.glb'
    proof = convert_file(multi, compatible, root / 'compatibility.json')
    source = core.read_source(compatible)
    view = core.make_view(source['positions'], 256, 0)
    before = core.render_albedo(core.read_source(original), view)
    after = core.render_albedo(source, view)
    assert np.array_equal(before['rgb'], after['rgb']), 'Compatibility changed the original rendered albedo'
    with Image.open(PORTRAIT) as image:
        image.convert('RGB').save(root/'reference.png')
    core.run_retexture(compatible, root/'reference.png', root/'corrected.glb', root/'comparison.png',
                      root/'face-proof.json', args.face_tool/'models', render_resolution=256,
                      light_balance=1, tone_match=1, skin_recolor=1)
    face = json.loads((root/'face-proof.json').read_text())
    for key in ('geometry_unchanged', 'original_embedded_images_exact'):
        assert face[key], key
    assert face['output_triangle_count'] == face['original_triangle_count']
    assert face['protected_body_changed_render_pixels'] == 0
    assert face['changed_head_render_pixels'] > 0
    result = {'passed': True, 'fixture': 'Pinned upstream public NASA portrait textured-grid fixture, split into two chunks',
              'input_primitives': 2, 'compatible_primitives': 1,
              'albedo_pixel_exact_after_compatibility': True,
              'complete_photo_retexture_passed': True,
              'geometry_unchanged_after_face_fix': face['geometry_unchanged'],
              'protected_body_changed_render_pixels': face['protected_body_changed_render_pixels'],
              'changed_head_render_pixels': face['changed_head_render_pixels'],
              'triangles': face['output_triangle_count'], 'compatibility': proof}
    if args.model_glb:
        actual = core.read_source(args.model_glb)
        result['actual_model'] = {'face_tool_input_gate_passed': True,
            'sha256': hashlib.sha256(args.model_glb.read_bytes()).hexdigest(),
            'vertices': len(actual['positions']), 'triangles': len(actual['triangles']),
            'scope': 'Input-format acceptance of the saved model; no assertion that MediaPipe recognizes its stylized face.'}
    (root/'validation.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
