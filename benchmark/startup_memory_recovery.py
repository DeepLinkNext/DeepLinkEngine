"""Return optional controller pages before driver host allocation runs out of room."""


def release_startup_headroom(reserve, check):
    """Preserve the original guard; never allocate pressure against a live model."""
    if reserve.record['released']:
        return
    value = check()
    required = 24 * 2**30
    if value['limit'] - value['current'] >= required:
        return
    event = dict(before=value, required_raw_bytes=required, released_bytes=0,
                 reason='Return optional startup reserve before driver pinned allocation',
                 allocated_bytes=0, global_drop_caches=False, memory_limit_changed=False)
    reserve.record.setdefault('raw_headroom_releases', []).append(event)
    try:
        while reserve.buffers and value['limit'] - value['current'] < required:
            count = min(128, len(reserve.buffers))
            reserve.release_chunks(count, 'release-owned-buffer-for-driver-raw-headroom')
            event['released_bytes'] += count * 64 * 2**20
            # New OOMs and genuine model pressure remain fatal after release.
            value = check()
        event['after'] = value
        event['reserve_exhausted'] = not reserve.buffers
    finally:
        reserve.save()
