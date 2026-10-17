"""CPU-only ownership mutations; no live process or device queries."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/evaluation/moss'))
import ownership as o


@pytest.fixture
def lease():
    return dict(gpu_uuid=o.RUNNER_GPU_UUID,
                desktop_baseline=[dict(pid=4, name='System', created_epoch=0.0),
                                  dict(pid=25, name='Desktop.exe', created_epoch=123.0)],
                baseline_max_memory_mib=1024, baseline_max_utilization_percent=2)


@pytest.fixture
def live(monkeypatch):
    identities = {4: ('System', 0.0), 25: ('Desktop.exe', 123.0)}

    def process(pid):
        value = identities[pid]
        if isinstance(value, Exception):
            raise value
        return SimpleNamespace(name=lambda: value[0], create_time=lambda: value[1])

    monkeypatch.setitem(sys.modules, 'psutil', SimpleNamespace(Process=process))
    monkeypatch.setattr(o, 'compute_pids', lambda uuid: {4, 25})
    monkeypatch.setattr(o, 'gpu_rows', lambda: [dict(index='1', uuid=o.RUNNER_GPU_UUID,
                                                  used_mib='933', utilization_percent='0')])
    return identities


def test_exact_reviewed_system_zero_and_ordinary_identity_pass(lease, live):
    o.validate_baseline(lease)
    assert o.desktop_identities(lease) == {4, 25}
    assert o.idle_device(lease)['gpu']['used_mib'] == '933'


@pytest.mark.parametrize('change', [dict(pid=5), dict(name='system'), dict(name='Other'),
                                   dict(created_epoch=-1), dict(created_epoch=float('nan')),
                                   dict(created_epoch=float('inf')), dict(created_epoch=-float('inf'))])
def test_zero_exception_is_exact_and_finite(lease, change):
    lease['desktop_baseline'][0].update(change)
    with pytest.raises(ValueError, match='known process identity'):
        o.validate_baseline(lease)


@pytest.mark.parametrize('epoch', [0, -1, float('nan'), float('inf'), -float('inf')])
def test_other_process_nonpositive_or_nonfinite_epoch_rejected(lease, epoch):
    lease['desktop_baseline'][1]['created_epoch'] = epoch
    with pytest.raises(ValueError, match='known process identity'):
        o.validate_baseline(lease)


@pytest.mark.parametrize('epoch', [1792039003.543363, .001])
def test_nonzero_system_timestamp_cannot_spoof_actual_psutil_zero(lease, live, epoch):
    lease['desktop_baseline'][0]['created_epoch'] = epoch
    o.validate_baseline(lease)
    with pytest.raises(ValueError, match='identity changed for PID4'):
        o.desktop_identities(lease)


@pytest.mark.parametrize('identity', [('system', 0), ('Other', 0), ('System', .001),
                                     ('System', float('nan')), ('System', float('inf'))])
def test_live_system_identity_must_match_exactly(lease, live, identity):
    live[4] = identity
    with pytest.raises(ValueError, match='identity changed for PID4'):
        o.desktop_identities(lease)


def test_reviewed_live_system_may_have_no_current_gpu_context(lease, live, monkeypatch):
    monkeypatch.setattr(o, 'compute_pids', lambda uuid: {25})
    assert o.idle_device(lease)['desktop_baseline'] == lease['desktop_baseline']


def test_foreign_owner_rejected_even_with_low_device_usage(lease, live, monkeypatch):
    monkeypatch.setattr(o, 'compute_pids', lambda uuid: {25, 999})
    with pytest.raises(ValueError, match='owners differ'):
        o.idle_device(lease)


@pytest.mark.parametrize('error', [KeyError(4), PermissionError('unresolved')])
def test_missing_or_inaccessible_reviewed_process_still_blocks(lease, live, monkeypatch, error):
    live[4] = error
    monkeypatch.setattr(o, 'compute_pids', lambda uuid: {25})
    with pytest.raises(type(error)):
        o.idle_device(lease)


def test_reused_ordinary_pid_still_blocks(lease, live):
    live[25] = ('Desktop.exe', 124)
    with pytest.raises(ValueError, match='identity changed for PID25'):
        o.idle_device(lease)


@pytest.mark.parametrize('field,value', [('baseline_max_memory_mib', -1),
                                        ('baseline_max_memory_mib', 4097),
                                        ('baseline_max_utilization_percent', -1),
                                        ('baseline_max_utilization_percent', 6)])
def test_ceiling_bounds_preserved(lease, field, value):
    lease[field] = value
    with pytest.raises(ValueError, match='ceiling'):
        o.validate_baseline(lease)


@pytest.mark.parametrize('used,utilization', [('1025', '0'), ('933', '3')])
def test_live_idle_ceilings_preserved(lease, live, monkeypatch, used, utilization):
    monkeypatch.setattr(o, 'gpu_rows', lambda: [dict(index='1', uuid=o.RUNNER_GPU_UUID,
                                                  used_mib=used, utilization_percent=utilization)])
    with pytest.raises(ValueError, match='idle memory/utilization ceiling'):
        o.idle_device(lease)


@pytest.mark.parametrize('field,value', [('index', '0'), ('uuid', 'GPU-foreign')])
def test_physical_device_checks_preserved(lease, live, monkeypatch, field, value):
    row = dict(index='1', uuid=o.RUNNER_GPU_UUID, used_mib='0', utilization_percent='0')
    row[field] = value
    monkeypatch.setattr(o, 'gpu_rows', lambda: [row])
    with pytest.raises(ValueError, match='physical GPU1 identity changed|leased device absent'):
        o.idle_device(lease)
