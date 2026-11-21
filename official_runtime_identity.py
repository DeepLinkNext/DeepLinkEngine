"""Keep released images and source derivations distinct in all result records."""
import hashlib
import json
from pathlib import Path


def runtime_identity(sdk):
    sdk = Path(sdk)
    lock = json.loads((sdk / 'image-lock.json').read_bytes())
    if lock.get('manifest_kind') != 'source-build-derivation':
        return dict(runtime_kind='official-release-image', official_runtime=True,
                    official_image_manifest=lock['manifest_digest'], source_code_patched=False)
    receipt = sdk / 'source-build/receipt.json'
    receipt_raw = receipt.read_bytes()
    value = json.loads(receipt_raw)
    installed = json.loads((sdk / 'installation.json').read_bytes())
    receipt_sha256 = hashlib.sha256(receipt_raw).hexdigest()
    if (value.get('status') != 'prepared' or value.get('candidate') != json.loads((sdk / 'source-candidate-lock.json').read_bytes())
            or value['candidate']['base_image_manifest'] != lock['base_image_manifest']
            or installed.get('build_receipt_sha256') != receipt_sha256):
        raise ValueError('Source candidate build receipt is incomplete or differs')
    return dict(runtime_kind='source-build-candidate', official_runtime=False,
                official_release_image=False, base_image_manifest=lock['base_image_manifest'],
                source_derivation_sha256=lock['manifest_digest'].removeprefix('sha256:'),
                build_receipt=str(receipt.resolve()), build_receipt_sha256=receipt_sha256,
                upstream_source_patched=False, published_sdists=True,
                git_tag_references=lock['git_tag_references'],
                sdist_git_file_equivalence_certified=lock['sdist_git_file_equivalence_certified'])


def observer_basis(identity):
    if identity.get('runtime_kind') == 'source-build-candidate':
        return 'Published upstream source distributions built in a private SDK; no optimized-engine MODs or per-step counting hooks'
    return 'Unmodified official image; no optimized-engine MODs or per-step counting hooks'
