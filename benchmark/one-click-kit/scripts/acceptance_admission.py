"""Admit the separate complete-context/tenant workload before model loading."""

import json
from pathlib import Path
import shutil

from acceptance_client import load
from agent_trace import sha


def prepare(contract, root):
    contract = Path(contract).resolve()
    data, _ = load(contract)
    folder = Path(root) / "acceptance-inputs"
    folder.mkdir(exist_ok=False)
    for name in (contract.name, data["input_file"]):
        source = contract.parent / name
        if source.is_symlink() or source.resolve().parent != contract.parent:
            raise ValueError("Acceptance inputs must be local immutable files")
        shutil.copyfile(source, folder / name)
    result = dict(
        contract=str(folder / contract.name),
        contract_sha256=sha(folder / contract.name),
        inputs_sha256=data["input_sha256"],
        five_fixed_datasets_unchanged=True,
    )
    (Path(root) / "acceptance-admission.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result


def extend_warmup(inputs, contract):
    path = Path(inputs["path"])
    data = json.loads(path.read_bytes())
    _, real = load(contract)
    if set(data) != {"4096", "16384"}:
        raise ValueError("The historical C8 warmup inputs changed before extension")
    data["32768"] = real["32768"]
    output = path.with_name("acceptance-warmup-inputs-v1.json")
    with output.open("x") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return dict(
        inputs,
        path=str(output),
        sha256=sha(output),
        scope="Original unmodified synthetic4K/16K warmup plus five unchanged published32K requests with eight warmup lanes; separate acceptance traffic",
    )


def apply(plan, command, contract, baseline=None):
    data, _ = load(contract)
    if command.count("--max-model-len") != 1:
        raise ValueError("Require an explicit graph-mode context limit")
    if (
        int(command[command.index("--max-model-len") + 1])
        < data["max_model_len_required"]
    ):
        raise ValueError(
            "Fixed model context limit cannot cover the complete32K request band"
        )
    if "--enforce-eager" in command:
        raise ValueError("No eager fallback for acceptance")
    plan["acceptance_suite"] = dict(
        contract=str(Path(contract).resolve()),
        contract_sha256=sha(contract),
        input_sha256=data["input_sha256"],
        baseline_run=str(Path(baseline).resolve()) if baseline else None,
    )
