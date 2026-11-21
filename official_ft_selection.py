"""Keep the source candidate identity separate from official OCI identities."""
import json
from pathlib import Path


VERSION = '0.28.0rc1-source'
BASELINE = 'upstream-vllm-0.28.0-ascend-0.28.0rc1-source'


def version(identity):
    if identity.get('manifest_kind') == 'source-build-derivation':
        if (identity.get('version') != VERSION or identity.get('id') != BASELINE
                or identity.get('candidate') is not True
                or identity.get('official_release_image') is not False):
            raise ValueError('Unmapped source candidate arithmetic identity')
        return VERSION
    return identity['upstream_version']


def selection(identity, cache, root):
    """Resolve the existing public preparer, without importing an SDK."""
    selected = version(identity)
    if selected == VERSION:
        from official_source_candidate import identity as declared, key
        expected = declared()
        expected.update(manifest_digest='sha256:' + key(), manifest_kind='source-build-derivation')
        if identity != expected:
            raise ValueError('Original source candidate differs from its public derivation')
        return selected, expected, Path(cache) / 'source-candidate-sdks' / (BASELINE + '-' + key()[:16])
    from official_image import read_lock
    expected = read_lock(Path(root) / 'engine/official-baselines' / (selected + '.json'))
    if expected != identity:
        raise ValueError('Original official image differs from its public lock')
    return selected, expected, Path(cache) / 'official-sdks' / (selected + '-' + expected['manifest_digest'].split(':')[1][:16])


def declaration_for_rejection(identity, root):
    """Match pinned source facts for a JSON counterexample, never a pass."""
    from official_ft_audit import SOURCE_DOMAINS_SHA256, sha
    path = Path(root) / 'engine/official-ft-source-domains.json'
    if sha(path) != SOURCE_DOMAINS_SHA256 or version(identity) != VERSION:
        raise ValueError('Candidate source declaration policy differs')
    domain = json.loads(path.read_bytes())['domains'][VERSION]
    actual = identity.get('sources', {})
    if (identity.get('base_image_manifest') != domain['base_image_manifest']
            or identity.get('upstream_source_patched') is not False
            or any(actual.get(name.replace('_', '-'), {}).get(field) != pin[field]
                   for name, pin in domain['source_archives'].items() for field in ('version', 'sha256'))):
        raise ValueError('Original candidate source archives/base declaration differs')
    return identity


def admit_candidate(image, identity, out):
    # Reuse the same whole-filemap + build-receipt + live metadata admission
    # used by public cold preparation. A base OCI digest is never substituted
    # for this derivation digest. No new build is started by continuation.
    from official_source_candidate import reuse
    from official_sdk_state import sdk_preparation
    from official_baselines import save
    with sdk_preparation(image):
        installed = reuse(image)
    if (json.loads((Path(image) / 'image-lock.json').read_bytes()) != identity
            or installed.get('schema') != 'deeplinkengine-source-candidate-sdk/v1'
            or installed.get('image_manifest') != identity['manifest_digest']
            or installed.get('official_release_image') is not False):
        raise ValueError('Completed source candidate SDK admission differs')
    result = dict(manifest_kind='source-build-derivation', candidate=True,
        official_release_image=False, source_derivation=identity,
        installed_files_sha256=installed['installed_files_sha256'],
        build_receipt_sha256=installed['build_receipt_sha256'],
        cached_payload_verified=True, actual_metadata=installed['metadata'], sdk_reinstalled=False)
    save(Path(out) / 'sdk-admission.json', result)
    return result
