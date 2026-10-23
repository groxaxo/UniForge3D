"""API boundary and job lifecycle tests; the separate browser run uses the real GPU."""
import io
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import struct
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor

from fastapi.testclient import TestClient
from PIL import Image
import pytest

from studio.server.app import create_app, Settings, JobStore, RemoteBridge, normalize_image
from studio.server.remote_runner import glb_metadata

HEADERS = {'X-UniForge-Studio': '1', 'Origin': 'http://127.0.0.1:7861'}


def png():
    buffer = io.BytesIO()
    Image.new('RGBA', (80, 96), (123, 42, 25, 255)).save(buffer, format='PNG')
    return buffer.getvalue()


def uploads(count=4, data=None):
    return [('files', (f'view{i}.png', data if data is not None else png(), 'image/png')) for i in range(count)]


def verified_sam_checkpoint(models_dir, monkeypatch):
    """A complete tiny safetensors file avoids allocating multi-GB test weights."""
    from studio.server import optional_models
    header = json.dumps({'weight': {'dtype': 'BF16', 'shape': [2], 'data_offsets': [0, 4]}}).encode()
    header += b' ' * (-len(header) % 8)
    data = struct.pack('<Q', len(header)) + header + b'\0\0\0\0'
    path = models_dir / 'detection/sam_3d_body_dinov3_bf16.safetensors'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    specs = tuple({**item, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
                  if item['id'] == 'sam-3d-body-bf16' else item
                  for item in optional_models.OPTIONAL_MODELS)
    monkeypatch.setattr(optional_models, 'OPTIONAL_MODELS', specs)
    return path


class Bridge:
    def __init__(self):
        self.started = []
        self.cancelled = []
        self.done = threading.Event()

    def health(self):
        return {'connected': True, 'ready': True, 'local_mlx_ready': False}

    def start(self, job_id, directory, seed):
        self.started.append((job_id, directory, seed))

    def read_status(self, job_id):
        if job_id in self.cancelled:
            return {'status': 'cancelled', 'message': 'Generation cancelled', 'stage': None, 'completed_stages': []}
        return {'status': 'running', 'message': 'Running dino', 'stage': 'dino', 'completed_stages': []}

    def cancel(self, job_id):
        self.cancelled.append(job_id)


@pytest.fixture
def client(tmp_path):
    bridge = Bridge()
    app = create_app(Settings(data=tmp_path), bridge)
    with TestClient(app) as client:
        yield client, app.state.store, bridge


def test_upload_is_validated_before_a_job_can_start(client):
    web, store, bridge = client
    assert web.post('/api/jobs', headers=HEADERS, files=uploads(3)).status_code == 422
    assert web.post('/api/jobs', headers=HEADERS, files=uploads(data=b'not an image')).status_code == 422
    assert web.post('/api/jobs', headers=HEADERS, files=uploads(), data={'seed': '-1'}).status_code == 422
    assert web.post('/api/jobs', headers=HEADERS, files=uploads(), data={'name': '\n'}).status_code == 422
    assert bridge.started == []
    assert web.get('/api/jobs').json() == []


@pytest.mark.parametrize('count', [1, 2, 4])
def test_local_mlx_accepts_one_to_four_views(tmp_path, count):
    bridge = Bridge()
    bridge.engine = 'mlx'
    app = create_app(Settings(data=tmp_path), bridge)
    with TestClient(app) as web:
        result = web.post('/api/jobs', headers=HEADERS, files=uploads(count))
        assert result.status_code == 202
        job = result.json()
        assert job['engine'] == 'mlx'
        assert job['view_count'] == count
        assert job['encoder'] == 'Original DINOv3'
        web.post(f'/api/jobs/{job["id"]}/cancel', headers=HEADERS)


def test_local_mlx_rejects_more_than_four_views(tmp_path):
    bridge = Bridge()
    bridge.engine = 'mlx'
    app = create_app(Settings(data=tmp_path), bridge)
    with TestClient(app) as web:
        assert web.post('/api/jobs', headers=HEADERS, files=uploads(5)).status_code == 422
        assert web.get('/api/jobs').json() == []


@pytest.mark.parametrize('count', [2, 4])
def test_configured_cuda_engine_accepts_views_and_reports_its_encoder(tmp_path, count):
    bridge = Bridge()
    bridge.engine = 'cuda3060'
    bridge.min_views = 1
    bridge.encoder = 'Original DINOv3'
    app = create_app(Settings(data=tmp_path, port=7863), bridge)
    headers = {**HEADERS, 'Origin': 'http://127.0.0.1:7863'}
    with TestClient(app) as web:
        result = web.post('/api/jobs', headers=headers, files=uploads(count))
        assert result.status_code == 202
        job = result.json()
        assert job['view_count'] == count
        assert job['engine'] == 'cuda3060'
        assert job['encoder'] == 'Original DINOv3'
        assert web.post('/api/jobs', headers=headers, files=uploads(5)).status_code == 422
        web.post(f'/api/jobs/{job["id"]}/cancel', headers=headers)


@pytest.mark.parametrize('connected', [True, False])
def test_unavailable_engine_rejects_job_without_upload_or_queue(client, connected):
    web, store, bridge = client
    message = 'Private host is online, but its NVIDIA GPU is unavailable.'
    bridge.health = lambda: {'connected': connected, 'ready': False, 'message': message}

    response = web.post('/api/jobs', headers=HEADERS, files=uploads())

    assert response.status_code == 503
    assert response.json()['detail'] == message
    assert response.headers['retry-after'] == '15'
    assert web.get('/api/jobs').json() == []
    assert store.pending.empty()
    assert list(store.settings.data.glob('*/input')) == []
    assert bridge.started == []


def test_local_mutation_boundary_and_untrusted_host(client):
    web, _, _ = client
    assert web.post('/api/jobs', files=uploads()).status_code == 403
    assert web.post('/api/jobs', headers={**HEADERS, 'Origin': 'https://other.example'}, files=uploads()).status_code == 403
    assert web.get('/api/jobs', headers={'Host': 'attacker.example'}).status_code == 400
    assert web.post('/api/jobs', headers={**HEADERS, 'Content-Length': str(65 * 1024**2)}).status_code == 413


def test_job_files_use_generated_ids_and_cancel_own_worker(client):
    web, store, bridge = client
    result = web.post('/api/jobs', headers=HEADERS, files=uploads(), data={'name': '../../name;$(echo test)', 'seed': '91'})
    assert result.status_code == 202
    job = result.json()
    assert len(job['id']) == 32 and int(job['id'], 16) >= 0
    assert job['seed'] == 91
    assert web.get(f'/api/jobs/{job["id"]}/images/1').status_code == 200
    assert web.get(f'/api/jobs/{job["id"]}/images/0').status_code == 404
    assert web.get(f'/api/jobs/{job["id"]}/model').status_code == 404
    assert web.get('/api/jobs/invalid-id').status_code == 404
    assert web.get(f'/api/jobs/{job["id"]}/request.json').status_code == 404
    web.post(f'/api/jobs/{job["id"]}/cancel', headers=HEADERS)
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline and store.get(job['id'])['status'] != 'cancelled':
        time.sleep(0.05)
    assert store.get(job['id'])['status'] == 'cancelled'
    assert len(bridge.started) <= 1


def test_normalization_strips_metadata_and_rejects_tiny_images():
    assert Image.open(io.BytesIO(normalize_image(png()))).format == 'PNG'
    tiny = io.BytesIO()
    Image.new('RGB', (32, 32)).save(tiny, 'PNG')
    with pytest.raises(ValueError, match='64 pixels'):
        normalize_image(tiny.getvalue())


def test_download_only_after_verified_completion(client):
    web, store, _ = client
    result = web.post('/api/jobs', headers=HEADERS, files=uploads(), data={'name': 'A nice asset'})
    job_id = result.json()['id']
    web.post(f'/api/jobs/{job_id}/cancel', headers=HEADERS)
    folder = store.settings.data / job_id
    (folder / 'prediction.glb').write_bytes(b'test fixture')
    (folder / 'receipt.json').write_text('{"verified":true}')
    store.update(job_id, status='complete')
    response = web.get(f'/api/jobs/{job_id}/download')
    assert response.status_code == 200 and response.content == b'test fixture'
    assert 'A-nice-asset.glb' in response.headers['content-disposition']
    assert web.get(f'/api/jobs/{job_id}/receipt').json()['verified']


def test_partial_glb_is_rejected(tmp_path):
    path = tmp_path / 'partial.glb'
    path.write_bytes(b'glTF\x02\x00\x00\x00\xff\xff\xff\xff')
    with pytest.raises(ValueError, match='complete GLB'):
        glb_metadata(path)


def test_job_history_survives_restart(tmp_path):
    settings = Settings(data=tmp_path)
    store = JobStore(settings, Bridge())
    job = store.add('Saved asset', 42, [png()] * 4)
    store.update(job['id'], status='cancelled')
    restored = JobStore(settings, Bridge())
    assert restored.get(job['id'])['name'] == 'Saved asset'
    assert restored.get(job['id'])['status'] == 'cancelled'


def triangle_glb():
    positions = struct.pack('<9f', 0, 0, 0, 1, 0, 0, 0, 1, 0)
    binary = positions + struct.pack('<3H', 0, 1, 2) + b'\x00\x00'
    model = {'asset': {'version': '2.0'}, 'buffers': [{'byteLength': len(binary)}],
             'bufferViews': [{'buffer': 0, 'byteOffset': 0, 'byteLength': 36}, {'buffer': 0, 'byteOffset': 36, 'byteLength': 6}],
             'accessors': [{'bufferView': 0, 'componentType': 5126, 'count': 3, 'type': 'VEC3', 'min': [0, 0, 0], 'max': [1, 1, 0]},
                           {'bufferView': 1, 'componentType': 5123, 'count': 3, 'type': 'SCALAR'}],
             'meshes': [{'primitives': [{'attributes': {'POSITION': 0}, 'indices': 1}]}],
             'nodes': [{'mesh': 0}], 'scenes': [{'nodes': [0]}], 'scene': 0}
    encoded = json.dumps(model).encode()
    encoded += b' ' * (-len(encoded) % 4)
    return (struct.pack('<4sII', b'glTF', 2, 28 + len(encoded) + len(binary)) +
            struct.pack('<I4s', len(encoded), b'JSON') + encoded + struct.pack('<I4s', len(binary), b'BIN\x00') + binary)


def test_remote_output_hashing_supports_python310(tmp_path, monkeypatch):
    import hashlib
    monkeypatch.delattr(hashlib, 'file_digest', raising=False)
    path = tmp_path / 'model.glb'
    path.write_bytes(triangle_glb())
    result = glb_metadata(path)
    assert result['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert result['vertices'] == 3 and result['triangles'] == 1


def test_failed_cuda_job_resumes_its_checkpoints_before_newer_jobs(tmp_path):
    bridge = Bridge()
    bridge.engine = 'cuda3060'
    bridge.min_views = 1
    resumed = []
    bridge.prepare_resume = resumed.append
    app = create_app(Settings(data=tmp_path), bridge)
    store = app.state.store
    job = store.add('Saved failed asset', 42, [png()] * 2)
    store.pending.get_nowait(); store.pending.task_done()
    store.update(job['id'], status='failed', remote_started=True, error='CUDA OOM',
                 completed_stages=[{'phase': 'texture'}])
    newer = store.add('Newer asset', 7, [png()] * 2)
    web = TestClient(app)
    result = web.post(f'/api/jobs/{job["id"]}/resume', headers=HEADERS)
    assert result.status_code == 202
    saved = result.json()
    assert saved['status'] == 'queued' and not saved['remote_started']
    assert saved['completed_stages'] == [{'phase': 'texture'}]
    assert saved['last_error'] == 'CUDA OOM' and saved['error'] is None
    assert resumed == [job['id']]
    assert store.pending.get_nowait()[1] == job['id']
    assert store.pending.get_nowait()[1] == newer['id']
    assert web.post(f'/api/jobs/{job["id"]}/resume', headers=HEADERS).status_code == 409


def test_resume_respects_queue_capacity_before_touching_remote_checkpoints(tmp_path):
    bridge = Bridge()
    bridge.engine = 'cuda3060'
    resumed = []
    bridge.prepare_resume = resumed.append
    app = create_app(Settings(data=tmp_path), bridge)
    store = app.state.store
    failed = store.add('Failed asset', 42, [png()] * 2)
    store.update(failed['id'], status='failed', remote_started=True)
    for index in range(6):
        store.add(f'Queued {index}', index, [png()] * 2)
    response = TestClient(app).post(f'/api/jobs/{failed["id"]}/resume', headers=HEADERS)
    assert response.status_code == 429
    assert resumed == []
    assert store.get(failed['id'])['status'] == 'failed'


def test_restart_reconnects_active_worker_before_an_older_resumed_job(tmp_path):
    settings = Settings(data=tmp_path)
    initial = JobStore(settings, Bridge())
    older = initial.add('Older recovery', 42, [png()] * 2)
    active = initial.add('Already running', 43, [png()] * 2)
    initial.update(active['id'], remote_started=True, status='running')
    restored = JobStore(settings, Bridge())
    restored.run = lambda: None
    restored.start()
    try:
        assert restored.pending.get_nowait()[1] == active['id']
        assert restored.pending.get_nowait()[1] == older['id']
    finally:
        restored.close()


@pytest.mark.parametrize('count', [1, 2, 4])
def test_cuda_stage_input_views_are_ordered_and_not_duplicated(tmp_path, count):
    from tools.run_cuda_staged import input_images
    expected = [tmp_path / f'{i:03d}.png' for i in range(1, count + 1)]
    for path in expected:
        path.write_bytes(png())
    assert input_images(tmp_path) == expected
    expected[0].rename(tmp_path / '009.png')
    with pytest.raises(ValueError, match='ordered input'):
        input_images(tmp_path)


def test_corrupt_download_cannot_be_published(tmp_path):
    class CorruptBridge(RemoteBridge):
        def ssh(self, command, **kwargs):
            kwargs['stdout'].write(triangle_glb())
            return SimpleNamespace(returncode=0)
    bridge = CorruptBridge(Settings(data=tmp_path))
    with pytest.raises(RuntimeError, match='did not match'):
        bridge.fetch('a' * 32, tmp_path, {'sha256': 'wrong hash', 'bytes': 1})
    assert not (tmp_path / 'prediction.glb').exists()
    assert not (tmp_path / 'prediction.download').exists()


def test_completed_job_publishes_mesh_and_receipt_after_fetch(tmp_path):
    class CompletedBridge(Bridge):
        def read_status(self, job_id):
            return {'status': 'complete', 'message': 'Asset ready', 'stage': None,
                    'completed_stages': [], 'metadata': {'sha256': 'fixture'}}

        def fetch(self, job_id, directory, metadata):
            (directory / 'prediction.glb').write_bytes(triangle_glb())
            (directory / 'receipt.json').write_text('{"encoder":"SoftEq"}')
            return glb_metadata(directory / 'prediction.glb')
    app = create_app(Settings(data=tmp_path), CompletedBridge())
    with TestClient(app) as web:
        job = web.post('/api/jobs', headers=HEADERS, files=uploads()).json()
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            saved = web.get(f'/api/jobs/{job["id"]}').json()
            if saved['status'] == 'complete':
                break
            time.sleep(0.02)
        assert saved['status'] == 'complete'
        assert saved['metadata']['triangles'] == 1
        assert web.get(saved['model_url']).content == triangle_glb()
        assert web.get(saved['receipt_url']).json()['encoder'] == 'SoftEq'


@pytest.mark.parametrize('host', ['100.85.200.56', '172.20.10.2', 'op-2.local'])
@pytest.mark.parametrize('port', [7861, 5173])
def test_network_frontend_and_api_accept_configured_origin(client, host, port):
    web, store, _ = client
    headers = {'Host': f'{host}:{port}', 'Origin': f'http://{host}:{port}',
               'X-UniForge-Studio': '1'}
    assert web.get('/', headers={'Host': f'{host}:{port}'}).status_code == 200
    assert web.get('/api/jobs', headers=headers).status_code == 200
    # An empty submission must reach validation without starting generation.
    assert web.post('/api/jobs', headers=headers).status_code == 422
    assert store.pending.empty()


def test_optional_sam_model_is_listed_and_refused_until_the_file_is_complete(tmp_path):
    app = create_app(Settings(data=tmp_path / 'jobs', models_dir=tmp_path / 'models'), Bridge())
    with TestClient(app) as web:
        catalog = web.get('/api/health').json()['models']
        assert catalog[0] == {'id': 'uniforge3d', 'label': 'UniForge3D', 'optional': False, 'installed': True}
        optional = [item for item in catalog if item['optional']]
        assert [item['id'] for item in optional] == ['sam-3d-body-bf16', 'sam-3d-body-int8']
        assert [item['installed'] for item in optional] == [False, False]
        refused = web.post('/api/jobs', headers=HEADERS, files=uploads(), data={'optional_model': 'sam-3d-body-bf16'})
        assert refused.status_code == 422
        assert web.post('/api/jobs', headers=HEADERS, files=uploads(), data={'optional_model': 'uniforge3d'}).status_code == 422
        assert web.get('/api/jobs').json() == []
        job = web.post('/api/jobs', headers=HEADERS, files=uploads()).json()
        assert job['optional_model'] is None
        web.post(f'/api/jobs/{job["id"]}/cancel', headers=HEADERS)


def test_installed_sam_checkpoint_stays_optional_and_uniforge_still_generates(tmp_path, monkeypatch):
    verified_sam_checkpoint(tmp_path / 'models', monkeypatch)
    bridge = Bridge()
    app = create_app(Settings(data=tmp_path / 'jobs', models_dir=tmp_path / 'models'), bridge)
    with TestClient(app) as web:
        catalog = {item['id']: item for item in web.get('/api/health').json()['models']}
        assert catalog['sam-3d-body-bf16']['installed'] is True
        assert catalog['sam-3d-body-bf16']['optional'] is True
        assert catalog['sam-3d-body-bf16']['verification'] == 'verified'
        assert catalog['sam-3d-body-bf16']['usage'] == 'reference-only'
        assert catalog['sam-3d-body-int8']['installed'] is False
        result = web.post('/api/jobs', headers=HEADERS, files=uploads(), data={'optional_model': 'sam-3d-body-bf16'})
        assert result.status_code == 202
        job = result.json()
        assert job['optional_model'] == 'sam-3d-body-bf16'
        assert job['engine'] == 'cuda'
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline and not bridge.started:
            time.sleep(0.02)
        assert len(bridge.started) == 1
        missing = web.post('/api/jobs', headers=HEADERS, files=uploads(), data={'optional_model': 'sam-3d-body-int8'})
        assert missing.status_code == 422
        web.post(f'/api/jobs/{job["id"]}/cancel', headers=HEADERS)


def test_same_size_corrupt_sam_checkpoint_is_not_offered(tmp_path):
    from studio.server.app import model_catalog
    folder = tmp_path / 'detection'
    folder.mkdir()
    checkpoint = folder / 'sam_3d_body_dinov3_bf16.safetensors'
    checkpoint.touch()
    os.truncate(checkpoint, 2830737652)
    catalog = {item['id']: item for item in model_catalog(tmp_path)}
    assert catalog['sam-3d-body-bf16']['installed'] is False


def test_sam_verification_cache_invalidates_when_same_size_contents_change(tmp_path, monkeypatch):
    from studio.server.app import model_catalog
    checkpoint = verified_sam_checkpoint(tmp_path, monkeypatch)
    assert model_catalog(tmp_path)[1]['installed'] is True
    data = bytearray(checkpoint.read_bytes())
    data[-1] ^= 1
    checkpoint.write_bytes(data)
    item = model_catalog(tmp_path)[1]
    assert item['installed'] is False
    assert item['verification'] == 'invalid'


def test_sam_health_checks_do_not_rehash_unchanged_weights(tmp_path, monkeypatch):
    from studio.server.app import model_catalog
    checkpoint = verified_sam_checkpoint(tmp_path, monkeypatch)
    original = Path.open
    reads = []
    def open_file(path, *args, **kwargs):
        if path == checkpoint:
            reads.append(path)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', open_file)
    assert model_catalog(tmp_path)[1]['installed'] is True
    assert model_catalog(tmp_path)[1]['installed'] is True
    assert len(reads) == 1


def test_none_selection_does_not_scan_optional_checkpoints(client, monkeypatch):
    def forbidden(*args):
        raise AssertionError('A None selection must not scan optional model files')
    monkeypatch.setattr('studio.server.app.model_catalog', forbidden)
    web, _, _ = client
    response = web.post('/api/jobs', headers=HEADERS, files=uploads())
    assert response.status_code == 202
    web.post(f'/api/jobs/{response.json()["id"]}/cancel', headers=HEADERS)


def test_checkpoint_verification_does_not_block_job_polling(tmp_path, monkeypatch):
    entered = threading.Event()
    release = threading.Event()
    def slow_catalog(*args):
        entered.set()
        assert release.wait(5)
        return [{'id': 'sam-3d-body-bf16', 'optional': True, 'installed': True}]
    monkeypatch.setattr('studio.server.app.model_catalog', slow_catalog)
    app = create_app(Settings(data=tmp_path), Bridge())
    with TestClient(app) as web, ThreadPoolExecutor(max_workers=2) as pool:
        submission = pool.submit(web.post, '/api/jobs', headers=HEADERS, files=uploads(),
                                 data={'optional_model': 'sam-3d-body-bf16'})
        assert entered.wait(2)
        polling = pool.submit(web.get, '/api/jobs')
        try:
            assert polling.result(timeout=1.5).status_code == 200
        finally:
            release.set()
        response = submission.result(timeout=3)
        assert response.status_code == 202
        web.post(f'/api/jobs/{response.json()["id"]}/cancel', headers=HEADERS)


@pytest.mark.parametrize('with_face', [False, True])
def test_optional_checkpoint_reference_survives_in_downloadable_receipt(tmp_path, monkeypatch, with_face):
    from studio.server import app as server
    from studio.server.remote_runner import glb_metadata
    bridge = Bridge()
    bridge.engine = 'cuda3090'
    bridge.read_status = lambda _: {'status': 'complete', 'completed_stages': [], 'metadata': {}}
    def fetch(identifier, folder, metadata):
        (folder / 'prediction.glb').write_bytes(triangle_glb())
        (folder / 'receipt.json').write_text('{"engine":"cuda3090"}')
        return glb_metadata(folder / 'prediction.glb')
    bridge.fetch = fetch
    monkeypatch.setattr(server, 'preserve_face', lambda *args, **kwargs: {
        'status': 'skipped', 'message': 'Original texture kept'})
    store = JobStore(Settings(data=tmp_path), bridge)
    job = store.add('Optional reference', 42, [png()], 'sam-3d-body-bf16', preserve_face=with_face)
    store.execute(job['id'])
    receipt = json.loads((tmp_path / job['id'] / 'receipt.json').read_text())
    generation = receipt['generation_receipt'] if with_face else receipt
    assert generation['optional_model']['id'] == 'sam-3d-body-bf16'
    assert generation['optional_model']['executed'] is False
    assert generation['optional_model']['usage'] == 'reference-only'
    assert generation['engine'] == 'cuda3090'


def test_network_requests_reject_cross_origin_and_unknown_hosts(client):
    web, _, _ = client
    headers = {'Host': '100.85.200.56:7861', 'Origin': 'http://attacker.example:7861',
               'X-UniForge-Studio': '1'}
    assert web.post('/api/jobs', headers=headers).status_code == 403
    headers['Origin'] = 'http://100.85.200.56:9999'
    assert web.post('/api/jobs', headers=headers).status_code == 403
    headers.update(Host='attacker.example:7861', Origin='http://attacker.example:7861')
    assert web.get('/api/jobs', headers=headers).status_code == 400
