"""GPUQ protocol checks; no CUDA import or real GPU reservation is used."""
import hashlib
import io
import json
from pathlib import Path
import subprocess

from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image
import pytest

from studio.server.gpuq_common import atomic_json
from studio.server.gpuq_gateway import Gateway, register
from studio.server.gpuq_worker import admission

IDENTIFIER = 'a' * 32
QUEUE_ID = '123456789abc'
HEADERS = {'x-uniforge-studio': '1'}


class FakeGateway(Gateway):
    def __init__(self, home):
        super().__init__(home)
        self.calls = []
        self.jobs = {}
        self.uncertain_submit = False

    def health(self):
        return {'ready': True, 'engine_id': 'cuda3090', 'queue': 'gpuq'}

    def cli(self, *args):
        self.calls.append(args)
        if args[0] == 'list':
            return '\n'.join(f'{key} {job["status"]} GPUs=1 {job["name"]} Waiting'
                             for key, job in self.jobs.items())
        if args[0] == 'show':
            return json.dumps(self.jobs[args[1]])
        if args[0] == 'submit':
            assert args[args.index('--count') + 1] == '1'
            assert '--gpus' not in args and '--candidates' not in args
            self.jobs[QUEUE_ID] = dict(name=args[args.index('--name') + 1],
                command=json.dumps(args[args.index('--') + 1:]),
                cwd=str(self.repo), gpu_count=1, status='queued')
            if self.uncertain_submit:
                raise subprocess.TimeoutExpired('gpuq', 30)
            return QUEUE_ID
        if args[0] == 'cancel':
            self.jobs[args[1]]['status'] = 'cancelled'
            return 'Cancellation requested'
        raise AssertionError(args)


@pytest.fixture
def gateway(tmp_path):
    service = FakeGateway(tmp_path)
    app = FastAPI()
    register(app, service)
    with TestClient(app) as client:
        yield service, client


def reference(client):
    data = io.BytesIO()
    Image.new('RGB', (64, 64), 'red').save(data, format='PNG')
    result = client.put(f'/v1/3d/jobs/{IDENTIFIER}/images/1', content=data.getvalue(), headers=HEADERS)
    assert result.status_code == 200
    assert result.json()['sha256'] == hashlib.sha256(data.getvalue()).hexdigest()
    return {'seed': 42, 'inputs': {'001.png': result.json()['sha256']}}


def submit(client, payload):
    return client.post(f'/v1/3d/jobs/{IDENTIFIER}', json=payload, headers=HEADERS)


def test_submit_is_durable_idempotent_and_uses_flexible_one_gpu_admission(gateway):
    service, client = gateway
    payload = reference(client)
    first = submit(client, payload)
    assert first.status_code == 200 and first.json()['status'] == 'waiting'
    assert first.json()['queue_id'] == QUEUE_ID
    assert submit(client, payload).json() == first.json()
    # Recover the queue entry even if the HTTP process lost its queue-ID receipt.
    (service.base / IDENTIFIER / 'queue.json').unlink()
    assert submit(client, payload).json()['queue_id'] == QUEUE_ID
    assert sum(call[0] == 'submit' for call in service.calls) == 1
    assert service.command(IDENTIFIER)[2:4] == ['-m', 'studio.server.gpuq_worker']


def test_timeout_after_queue_commit_recovers_without_a_second_submission(gateway):
    service, client = gateway
    payload = reference(client)
    service.uncertain_submit = True
    assert submit(client, payload).json()['queue_id'] == QUEUE_ID
    assert sum(call[0] == 'submit' for call in service.calls) == 1


def test_transfer_hash_mismatch_never_submits_gpu_work(gateway):
    service, client = gateway
    payload = reference(client)
    payload['inputs']['001.png'] = '0' * 64
    assert submit(client, payload).status_code == 400
    assert not any(call[0] == 'submit' for call in service.calls)


def test_queued_job_cannot_be_replaced_by_different_inputs_or_seed(gateway):
    service, client = gateway
    payload = reference(client)
    assert submit(client, payload).status_code == 200
    payload['seed'] += 1
    assert submit(client, payload).status_code == 409
    other = io.BytesIO()
    Image.new('RGB', (64, 64), 'blue').save(other, format='PNG')
    assert client.put(f'/v1/3d/jobs/{IDENTIFIER}/images/1', content=other.getvalue(), headers=HEADERS).status_code == 409


def test_cancel_uses_only_the_verified_owned_gpuq_job(gateway):
    service, client = gateway
    submit(client, reference(client))
    result = client.post(f'/v1/3d/jobs/{IDENTIFIER}/cancel', json={}, headers=HEADERS)
    assert result.json()['status'] == 'cancelled'
    assert ('cancel', QUEUE_ID) in service.calls


def test_cancel_refuses_a_foreign_queue_command(gateway):
    service, client = gateway
    submit(client, reference(client))
    service.jobs[QUEUE_ID]['command'] = json.dumps(['/usr/bin/python3', 'other-work.py'])
    result = client.post(f'/v1/3d/jobs/{IDENTIFIER}/cancel', json={}, headers=HEADERS)
    assert result.status_code == 503
    assert not any(call[0] == 'cancel' for call in service.calls)


def test_output_is_not_published_until_gpuq_reports_command_completion(gateway):
    service, client = gateway
    submit(client, reference(client))
    service.jobs[QUEUE_ID]['status'] = 'running'
    atomic_json(service.base / IDENTIFIER / 'status.json',
                {'status': 'complete', 'metadata': {'sha256': 'x'}, 'completed_stages': []})
    assert client.get(f'/v1/3d/jobs/{IDENTIFIER}').json()['status'] == 'running'
    assert client.get(f'/v1/3d/jobs/{IDENTIFIER}/model').status_code == 409
    service.jobs[QUEUE_ID]['status'] = 'completed'
    assert client.get(f'/v1/3d/jobs/{IDENTIFIER}').json()['status'] == 'complete'


@pytest.mark.parametrize('state', ['failed', 'interrupted'])
def test_interrupted_queue_job_is_failed_instead_of_waiting_forever(gateway, state):
    service, client = gateway
    submit(client, reference(client))
    service.jobs[QUEUE_ID].update(status=state, reason='GPUQ owner stopped')
    result = client.get(f'/v1/3d/jobs/{IDENTIFIER}').json()
    assert result['status'] == 'failed' and result['error'] == 'GPUQ owner stopped'


def test_metadata_boundary_rejects_browser_writes_bad_ids_and_oversized_json(gateway):
    service, client = gateway
    assert client.post(f'/v1/3d/jobs/{IDENTIFIER}', json={}).status_code == 403
    assert client.post(f'/v1/3d/jobs/{IDENTIFIER}', json={}, headers={**HEADERS, 'origin': 'https://other.test'}).status_code == 403
    assert client.get('/v1/3d/jobs/not-a-job').status_code == 400
    assert client.post(f'/v1/3d/jobs/{IDENTIFIER}', content=b'x' * 4097, headers=HEADERS).status_code == 413
    assert client.get(f'/v1/3d/jobs/{IDENTIFIER}/unrelated-file').status_code == 404
    assert not any(call[0] == 'submit' for call in service.calls)


def test_worker_requires_queue_ownership_before_any_cuda_import():
    with pytest.raises(RuntimeError, match='admitted'):
        admission({})
    with pytest.raises(RuntimeError, match='admitted'):
        admission({'GPU_QUEUE_JOB_ID': QUEUE_ID, 'CUDA_VISIBLE_DEVICES': '0,1', 'GPU_QUEUE_LOCK_FDS': '4'})
    assert admission({'GPU_QUEUE_JOB_ID': QUEUE_ID, 'CUDA_VISIBLE_DEVICES': 'GPU-assigned',
                      'GPU_QUEUE_LOCK_FDS': '4,5'}) == (QUEUE_ID, 'GPU-assigned')


def test_proxy_configuration_preserves_other_families_permissions_and_is_idempotent(tmp_path):
    from tools.configure_uniforge_proxy import configure
    source = ('app = FastAPI(title="qwen36-multi-variant-proxy")\n'
              'def list_models() -> dict[str, Any]:\n    data = []\n'
              '    for prefix, config in _available_families().items():\n        pass\n')
    proxy = tmp_path / 'qwen36_variant_proxy.py'
    registry = tmp_path / 'qwen36_model_families.json'
    proxy.write_text(source)
    registry.write_text(json.dumps({'existing': {'user_setting': 'preserve this'}}) + '\n')
    registry.chmod(0o600)
    first = configure(tmp_path, Path('/home/op/uniforge3d-work/UniForge3D'))
    assert len(first['changed']) == 2
    current = json.loads(registry.read_text())
    assert current['existing'] == {'user_setting': 'preserve this'}
    assert current['uniforge3d']['queue_mode'] == 'per-job'
    assert registry.stat().st_mode & 0o777 == 0o600
    assert configure(tmp_path, Path('/home/op/uniforge3d-work/UniForge3D'))['changed'] == []


def test_legacy_direct_cuda_launch_fails_before_any_ssh_or_gpu_work(tmp_path):
    from studio.server.app import RemoteBridge, Settings
    bridge = RemoteBridge(Settings(data=tmp_path, engine='cuda'))
    def forbidden(*args, **kwargs):
        raise AssertionError('The retired launcher must never reach SSH')
    bridge.ssh = forbidden
    assert bridge.health()['ready'] is False
    with pytest.raises(RuntimeError, match='GPUQ'):
        bridge.start(IDENTIFIER, tmp_path, 42)
