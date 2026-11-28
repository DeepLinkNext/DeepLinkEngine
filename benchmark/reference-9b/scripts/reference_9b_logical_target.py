"""Admit the original six target-KV logical ledgers without borrowing companion addresses."""

from gzip import open as open_gzip
from hashlib import sha256
from pathlib import Path

from dataset_contract import sha
from reference_9b_archive import WORKLOAD_SEAL, member, read


PINS = {
    "B0": "ac006f30e1beeb7c8af37f0bbd6ad32360e9c28c96a44b59a398bc8f33dc035b",
    "B1": "c971ed1ab2b4f92dadf957e39b3c09cfd3e53e069d37f18a52ad81814e6b0f52",
}


def uncompressed_sha(path):
    dig = sha256()
    with open_gzip(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            dig.update(block)
    return dig.hexdigest()


def admit(path, role, clock):
    if path.is_symlink() or sha(path) != PINS[role]:
        raise ValueError(f"{role} original target-KV proof SHA-256 differs")
    proof = read(path)
    if role == "B0":
        model_seal, rounds = proof["original_seal_sha256"], proof["original_rounds"]
        fields = ("round", "source_wall_sha256", "raw_frames_sha256", "geometry_sha256")
        artifacts = [dict(filename=row["artifact"], sha256=row["artifact_sha256"]) for row in rounds]
    else:
        model_seal, rounds = proof["official_native_sha256"], proof["original_windows"]
        fields = ("repetition", "native_wall_sha256", "native_public_frames_sha256", "native_geometry_sha256")
        artifacts = [row["logical_interval_ledger"] for row in rounds]
    if (model_seal != clock["checksums_sha256"] or proof["workload_sha256"] != WORKLOAD_SEAL
            or len(rounds) != len(clock["rounds"]) or len(rounds) != 3):
        raise ValueError(f"{role} target-KV proof is from a different engine, workload or repetition set")
    base = Path(clock["archive"])
    mapped = []
    for row, archived, ledger in zip(rounds, clock["rounds"], artifacts, strict=True):
        rep, wall_pin, public_pin, geo_pin = [row[name] for name in fields]
        if rep != archived["repetition"] or wall_pin != archived["source_sha256"]:
            raise ValueError(f"{role} target-KV ledger is paired to the wrong original wall at round {rep}")
        folder = f"client/formal-{rep}"
        member(base, f"{folder}/raw-public-output-frames.json", public_pin)
        member(base, f"{folder}/public-output-geometry.json", geo_pin)
        counts = row["counts"]
        if (counts["public_target_logits"] != 51200
                or counts["necessary_target_kv_positions_per_full_attention_layer"] != 333265
                or counts["necessary_target_kv_positions_for_all_eight_layers"] != 8 * 333265):
            raise ValueError(f"{role} logical target-KV inventory differs at round {rep}")
        ledger_file = member(path.parent, ledger["filename"], ledger["sha256"])
        if uncompressed_sha(ledger_file) != row["logical_producer_intervals_sha256"]:
            raise ValueError(f"{role} logical target-KV rows differ at round {rep}")
        mapped.append(dict(repetition=rep, original_wall_seconds=archived["wall_seconds"], wall_sha256=wall_pin,
            public_frames_sha256=public_pin, logical_ledger=str(ledger_file.resolve()),
            logical_ledger_sha256=ledger["sha256"], **counts))
    return dict(evidence_path=str(path.resolve()), evidence_sha256=PINS[role], original_clean_archive=model_seal,
        original_windows=mapped, original_absolute_physical_slots_admitted_by_this_proof=False,
        added_reference_duration_seconds=None)
