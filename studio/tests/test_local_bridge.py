"""Local worker ownership, crash handling and output publication boundaries."""
import json
from types import SimpleNamespace

import pytest

from studio.server.app import Settings
from studio.server.local_bridge import LocalBridge
from studio.server.remote_runner import glb_metadata
from studio.tests.test_api import triangle_glb


def test_cancellation_does_not_signal_a_reused_unrelated_pid(tmp_path, monkeypatch):
    folder = tmp_path / ('a' * 32)
    folder.mkdir()
    (folder / 'controller.json').write_text(json.dumps({'pid': 12345, 'token': 'owned-worker'}))
    monkeypatch.setattr('studio.server.local_bridge.subprocess.run',
                        lambda *args, **kwargs: SimpleNamespace(stdout='another application'))
    signals = []
    monkeypatch.setattr('studio.server.local_bridge.os.killpg', lambda *args: signals.append(args))

    LocalBridge(Settings(data=tmp_path)).cancel(folder.name)

    assert (folder / 'cancel').exists()
    assert signals == []


@pytest.mark.parametrize('cancelled', [False, True])
def test_a_stopped_local_worker_reaches_a_terminal_state(tmp_path, cancelled):
    folder = tmp_path / ('a' * 32)
    folder.mkdir()
    (folder / 'status.json').write_text('{"status":"running","stage":"structure"}')
    (folder / 'worker.log').write_text('Worker diagnostic')
    if cancelled:
        (folder / 'cancel').touch()
    state = LocalBridge(Settings(data=tmp_path)).read_status(folder.name)
    assert state['status'] == ('cancelled' if cancelled else 'failed')
    assert state['stage'] is None
    assert not cancelled or 'error' not in state


def test_local_glb_is_published_only_after_matching_worker_hash(tmp_path):
    (tmp_path / 'work').mkdir()
    source = tmp_path / 'work/model.glb'
    source.write_bytes(triangle_glb())
    bridge = LocalBridge(Settings(data=tmp_path))
    expected = glb_metadata(source)
    with pytest.raises(RuntimeError, match='changed'):
        bridge.fetch('a' * 32, tmp_path, {**expected, 'sha256': 'changed'})
    assert not (tmp_path / 'prediction.glb').exists()
    assert bridge.fetch('a' * 32, tmp_path, expected) == expected
    assert (tmp_path / 'prediction.glb').read_bytes() == triangle_glb()


def test_native_uv_progress_remains_visible_while_worker_is_busy(tmp_path, monkeypatch):
    folder = tmp_path / ('b' * 32)
    folder.mkdir()
    (folder / 'status.json').write_text('{"status":"running","stage":"decode","message":"Bake & export"}')
    (folder / 'export-events.jsonl').write_text(
        json.dumps({'stage': 'uv', 'event': 'start', 'monotonic_sec': 40}) + '\n')
    bridge = LocalBridge(Settings(data=tmp_path))
    monkeypatch.setattr(bridge, 'alive', lambda job_id: True)
    monkeypatch.setattr('time.monotonic', lambda: 100)

    state = bridge.read_status(folder.name)

    assert state['status'] == 'running'
    assert 'texture atlas' in state['message']
    assert '1m 00s' in state['message']


def test_partial_native_event_does_not_hide_the_last_complete_progress(tmp_path, monkeypatch):
    folder = tmp_path / ('b' * 32)
    folder.mkdir()
    (folder / 'status.json').write_text('{"status":"running","stage":"decode","message":"Bake & export"}')
    (folder / 'export-events.jsonl').write_text(
        json.dumps({'stage': 'texture_bake', 'event': 'start', 'monotonic_sec': 40}) + '\n{"stage":')
    bridge = LocalBridge(Settings(data=tmp_path))
    monkeypatch.setattr(bridge, 'alive', lambda job_id: True)
    monkeypatch.setattr('time.monotonic', lambda: 100)

    state = bridge.read_status(folder.name)

    assert 'Baking color and material maps' in state['message']


def test_nested_uv_events_keep_the_parent_atlas_stage_and_elapsed_time(tmp_path, monkeypatch):
    folder = tmp_path / ('b' * 32)
    folder.mkdir()
    (folder / 'status.json').write_text('{"status":"running","stage":"decode","message":"Bake & export"}')
    events = [
        {'stage': 'uv', 'event': 'start', 'monotonic_sec': 40},
        {'stage': 'uv.compute_charts', 'event': 'start', 'monotonic_sec': 41},
        {'stage': 'uv.compute_charts', 'event': 'end', 'monotonic_sec': 50},
        {'stage': 'uv.grow_charts', 'event': 'start', 'monotonic_sec': 60},
    ]
    (folder / 'export-events.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in events))
    bridge = LocalBridge(Settings(data=tmp_path))
    monkeypatch.setattr(bridge, 'alive', lambda job_id: True)
    monkeypatch.setattr('time.monotonic', lambda: 100)

    message = bridge.read_status(folder.name)['message']

    assert 'texture atlas' in message
    assert '1m 00s' in message
