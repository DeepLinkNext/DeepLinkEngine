import hashlib
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third-party-test-kit/proc_capture"))
from m01_output_cache import ProfilerCache, release


def test_real_readonly_hint_preserves_recorded_bytes_and_claims_no_reclamation(tmp_path):
    path = tmp_path / "evidence.bin"; path.write_bytes(b"owned evidence" * 4096)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    record = release(path, tmp_path)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    assert not record["file_content_written"] and not record["global_drop_caches"]
    assert record["reclaimed_bytes_claimed"] is False


def test_symlink_and_outside_files_cannot_be_evicted(tmp_path):
    owned = tmp_path / "owned"; owned.mkdir()
    other = tmp_path / "other"; other.write_bytes(b"other workload")
    link = owned / "link"; link.symlink_to(other)
    for path in (link, other):
        with pytest.raises(ValueError, match="escaped"):
            release(path, owned)


def test_profiler_sweep_avoids_route_population_and_revisits_live_writers(tmp_path, monkeypatch):
    raw = tmp_path / "rank-0-raw-profiler" / "framework"; raw.mkdir(parents=True)
    data = raw / "events"; data.write_bytes(b"x" * 2**20)
    (tmp_path / "rank-0-wave-0-routes.npz").write_bytes(b"unchanged route evidence")
    probe = ProfilerCache(tmp_path)
    first = probe.sweep(); assert len(first) == 1
    assert probe.sweep() == []
    with data.open("ab") as stream: stream.write(b"more events")
    monkeypatch.setattr(probe, "active", lambda owners: iter([data]))
    second = probe.sweep([123]); assert len(second) == 1
    assert second[0]["bytes_requested"] > first[0]["bytes_requested"]


def test_profiler_tick_never_scans_an_unbounded_closed_packet_population(tmp_path):
    raw = tmp_path / "rank-0-raw-profiler"; raw.mkdir()
    for i in range(20): (raw / str(i)).write_bytes(b"packet")
    probe = ProfilerCache(tmp_path)
    assert len(probe.sweep(limit=3)) == 3
    assert len(probe.previous) == 3
    assert len(probe.sweep(limit=3)) == 3
    assert len(probe.previous) == 6
