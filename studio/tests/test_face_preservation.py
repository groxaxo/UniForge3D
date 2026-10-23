"""Face-preservation opt-in persistence, isolation and safe publication gates."""
import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from studio.server.app import Settings, create_app, JobStore
from studio.tests.test_api import Bridge, HEADERS, uploads, png
from studio.server import face_preservation as face


def test_checkbox_defaults_off_and_persists_explicit_reference(tmp_path, monkeypatch):
    monkeypatch.setattr('studio.server.app.face_readiness', lambda: {'available': True})
    bridge = Bridge(); bridge.engine = 'mlx'
    app = create_app(Settings(data=tmp_path), bridge)
    with TestClient(app) as web:
        first = web.post('/api/jobs', headers=HEADERS, files=uploads(1)).json()
        assert first['preserve_face'] is False
        web.post(f'/api/jobs/{first["id"]}/cancel', headers=HEADERS)
        second = web.post('/api/jobs', headers=HEADERS, files=uploads(1)+[
            ('face_reference', ('face.png', png(), 'image/png'))], data={'preserve_face':'true'}).json()
        assert second['preserve_face'] is True
        assert web.get(second['face_reference_url']).status_code == 200
        web.post(f'/api/jobs/{second["id"]}/cancel', headers=HEADERS)
    restored = JobStore(Settings(data=tmp_path), bridge)
    assert restored.get(second['id'])['preserve_face'] is True


def test_missing_face_worker_rejects_opt_in_before_queue(tmp_path, monkeypatch):
    monkeypatch.setattr('studio.server.app.face_readiness', lambda: {'available': False})
    bridge = Bridge(); bridge.engine = 'mlx'
    app = create_app(Settings(data=tmp_path), bridge)
    with TestClient(app) as web:
        result = web.post('/api/jobs', headers=HEADERS, files=uploads(1), data={'preserve_face':'true'})
        assert result.status_code == 422
        assert web.get('/api/jobs').json() == []


def test_failed_face_match_keeps_source_exact(tmp_path, monkeypatch):
    source = tmp_path/'original.glb';source.write_bytes(b'original immutable geometry')
    reference=tmp_path/'face.png';reference.write_bytes(png())
    monkeypatch.setattr(face, 'face_readiness', lambda: {'available':True})
    monkeypatch.setattr(face, 'run_process', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('No matching face')))
    result = face.preserve_face(source, reference, tmp_path/'face-work')
    assert result['status'] == 'skipped'
    assert source.read_bytes() == b'original immutable geometry'
    assert 'output' not in result


def test_cancellation_is_not_reported_as_a_skipped_match(tmp_path, monkeypatch):
    source=tmp_path/'source.glb';source.write_bytes(b'original')
    monkeypatch.setattr(face, 'face_readiness', lambda: {'available':True})
    monkeypatch.setattr(face, 'run_process', lambda *a, **k: (_ for _ in ()).throw(face.FaceCancelled()))
    with pytest.raises(face.FaceCancelled):
        face.preserve_face(source, tmp_path/'face.png', tmp_path/'attempt')


def test_worker_success_with_bad_preservation_proof_is_not_published(tmp_path, monkeypatch):
    source=tmp_path/'source.glb';source.write_bytes(b'original')
    monkeypatch.setattr(face, 'face_readiness', lambda: {'available':True})
    def fake_process(command, log, cancelled):
        if '--proof-json' in command:
            p=Path(command[command.index('--proof-json')+1])
            p.write_text(json.dumps({'geometry_unchanged':False}))
    monkeypatch.setattr(face,'run_process',fake_process)
    result=face.preserve_face(source,tmp_path/'photo.png',tmp_path/'face-work')
    assert result['status']=='skipped' and 'output' not in result
    assert source.read_bytes()==b'original'


@pytest.mark.parametrize('outcome', ['applied', 'skipped'])
def test_queue_runs_opt_in_postprocess_and_records_the_published_hash(tmp_path, monkeypatch, outcome):
    from studio.tests.test_api import triangle_glb
    from studio.server.remote_runner import glb_metadata
    bridge=Bridge();bridge.engine='mlx'
    bridge.read_status=lambda job_id: {'status':'complete','message':'Asset ready','stage':None,'completed_stages':[],'metadata':{}}
    def fetch(job_id, folder, metadata):
        (folder/'prediction.glb').write_bytes(triangle_glb())
        (folder/'receipt.json').write_text('{"engine":"mlx","original_generation":true}')
        return glb_metadata(folder/'prediction.glb')
    bridge.fetch=fetch
    observed=[]
    def correct(source, reference, workspace, **kwargs):
        observed.append((source,reference))
        assert reference.read_bytes()==png()
        if outcome=='skipped':
            return {'status':'skipped','requested':True,'message':'Original texture kept'}
        candidate=tmp_path/'corrected.glb';candidate.write_bytes(triangle_glb())
        return {'status':'applied','requested':True,'output':str(candidate),'metadata':glb_metadata(candidate),'message':'Face preserved'}
    monkeypatch.setattr('studio.server.app.preserve_face',correct)
    store=JobStore(Settings(data=tmp_path),bridge)
    job=store.add('Face test',42,[png()],preserve_face=True)
    store.execute(job['id'])
    done=store.get(job['id']);folder=tmp_path/job['id']
    assert done['status']=='complete' and done['face_preservation']['status']==outcome
    assert len(observed)==1
    assert (folder/'generation.glb').read_bytes()==triangle_glb()
    receipt=json.loads((folder/'receipt.json').read_text())
    assert receipt['generation_receipt']['original_generation']
    assert receipt['sha256']==done['metadata']['sha256']==glb_metadata(folder/'prediction.glb')['sha256']
