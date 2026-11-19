"""Recompute published current F/T against the independently verified full reference."""

from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from one_click import reference_evidence, verify_kit


def main():
    import os
    os.environ["VLLM_HUST_PROJECT_EVIDENCE"] = str(reference_evidence(verify_kit()))
    sys.path.insert(0, str(ROOT / "benchmark/one-click-kit/scripts"))
    from full_reference_reuse import arm_statistics, prepare

    inputs = json.loads((HERE / "recomputation-inputs.json").read_bytes())
    archive_path = HERE / "current-work-and-timing.zip"
    if hashlib.sha256(archive_path.read_bytes()).hexdigest() != inputs["zip_sha256"]:
        raise ValueError("Current work/timing archive differs")
    pair_path = HERE / "compute-reference/matched-reference-pair.json"
    if hashlib.sha256(pair_path.read_bytes()).hexdigest() != inputs["certificate_source_sha256"]:
        raise ValueError("Published certificate differs")
    published = json.loads(pair_path.read_bytes())
    with tempfile.TemporaryDirectory(prefix="deeplink-reference-recompute-") as temp:
        reference = prepare(Path(temp))
        with zipfile.ZipFile(archive_path) as archive:
            if (len(archive.namelist()) != len(set(archive.namelist()))
                    or set(archive.namelist()) != set(inputs["members"]) | {"SHA256.json"}
                    or json.loads(archive.read("SHA256.json")) != inputs["members"]):
                raise ValueError("Work/timing archive inventory differs")
            raw = {name: archive.read(name) for name in inputs["members"]}
            for name, value in raw.items():
                if hashlib.sha256(value).hexdigest() != inputs["members"][name]:
                    raise ValueError("Work/timing member differs: " + name)
            for role in ("B0", "B1"):
                rows = []
                for rep in range(1, 4):
                    prefix = role if rep == 1 else f"{role}-r{rep:03d}"
                    work_bytes = raw[prefix + "-necessary-work.json"]
                    timing_bytes = raw[prefix + "-timing.json"]
                    work, timing = json.loads(work_bytes), json.loads(timing_bytes)
                    sealed = published["arms"][role]["repetitions"][rep-1]
                    if (hashlib.sha256(work_bytes).hexdigest() != sealed["work_sha256"]
                            or hashlib.sha256(timing_bytes).hexdigest() != sealed["timing_sha256"]):
                        raise ValueError("Current work/timing is not certificate-bound")
                    for endpoint, position in (("lower", 0), ("upper", 1)):
                        value = work["endpoints"][endpoint]
                        if sum(value["components"].values()) != value["total_basic"] or value["total_basic"] != work["F_interval"][position]:
                            raise ValueError("Necessary-work component arithmetic differs")
                    seconds = Decimal(timing["end_ns"] - timing["start_ns"]) / 10**9
                    if (str(seconds) != sealed["T_seconds"] or timing["output_tokens"] != 51200
                            or timing["repetition"] != rep or work["requests"] != 200):
                        raise ValueError("Native clock population or interval differs")
                    rows.append(dict(repetition=rep, F_interval=work["F_interval"], T_seconds=str(seconds)))
                value = arm_statistics(rows, reference["arms"][role])
                for key in ("F_interval", "T_seconds", "reference_F_interval", "reference_seconds_lower",
                            "reference_tflops_per_card_upper", "efficiency_lower_percent"):
                    if value[key] != published["arms"][role][key]:
                        raise ValueError("Published current reference statistic differs: " + role + "/" + key)
                print(role, "efficiency_lower_percent=" + value["efficiency_lower_percent"])
    print("PASS: six necessary-work records, six native clocks and the original complete-reference denominator")


if __name__ == "__main__":
    main()
