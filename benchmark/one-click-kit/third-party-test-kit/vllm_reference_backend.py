"""vLLM offline adapter for live M01, using the installed Ascend workers."""

from hashlib import sha256
from importlib import import_module, metadata
import json
import os
from pathlib import Path
from uuid import uuid4

from reference_capture_worker import canonical_sha, load_backend, write


def isolated_inputs(workload, nonce):
    """Give every warmup/measured request its own prefix-cache namespace."""
    if not isinstance(nonce, str) or not nonce:
        raise ValueError("A fresh capture namespace is required")
    populations = []
    for field in ("warmup_token_ids", "request_token_ids"):
        rows = workload[field]
        if not isinstance(rows, list) or not rows:
            raise ValueError("Freeze nonempty warm-up and measurement token IDs")
        if any(
            not isinstance(row, list)
            or not row
            or any(type(token) is not int or token < 0 for token in row)
            for row in rows
        ):
            raise ValueError("Expected nonnegative integer token-ID populations")
        populations.append(
            [
                {
                    "prompt_token_ids": list(row),
                    "cache_salt": f"m01:{nonce}:{field}:{index}",
                }
                for index, row in enumerate(rows)
            ]
        )
    return populations


def synchronize_worker(worker):
    import torch
    import torch_npu  # noqa: F401 -- register Ascend device operations

    device = getattr(worker, "device", None)
    if getattr(device, "type", None) != "npu":
        raise ValueError("M01 fence did not run on an Ascend worker")
    torch.npu.synchronize(device)
    return {"rank": torch.distributed.get_rank(), "logical_device": device.index}


class NativeInference:
    def __init__(self, config, workload, output, phase, cards):
        self.phase, self.cards, self.output = phase, cards, output
        source = Path(os.environ["VLLM_HUST_M01_ACCOUNTING"])
        self.accounting = load_backend(source)
        if hasattr(self.accounting, "prepare"):
            self.accounting.prepare(config, phase)
        from vllm import LLM, SamplingParams

        self.workload = json.loads(workload.read_bytes())
        kwargs = config["llm"]
        if kwargs.get("tensor_parallel_size") != 2 or kwargs.get(
            "enforce_eager", False
        ):
            raise ValueError("Live M01 requires the frozen TP2 graph-mode deployment")
        self.sampling = SamplingParams(**self.workload["sampling"])
        if self.sampling.n != 1:
            raise ValueError("Live M01 expects one completion per frozen request")
        self.warmup_inputs, self.inputs = isolated_inputs(self.workload, uuid4().hex)
        write(
            output / "prefix-cache-isolation.json",
            {
                "policy": "fresh namespace per capture and distinct salt per request; no warmup or cross-request prefix reuse",
                "warmup": [item["cache_salt"] for item in self.warmup_inputs],
                "measurement": [item["cache_salt"] for item in self.inputs],
                "token_populations_sha256": canonical_sha(
                    {
                        name: self.workload[name]
                        for name in ("warmup_token_ids", "request_token_ids")
                    }
                ),
                "created_outside_measurement": True,
            },
        )
        self.llm = LLM(**kwargs)
        self.config = config
        self.host_counted_timing = hasattr(self.accounting, "measurement_method")
        self.collector = None
        if phase != "timing" or self.host_counted_timing:
            self.collector = self.accounting.bind(self.llm, config, phase, output)

    def identity(self):
        packages = {}
        modules = {}
        for distribution, module_name in (
            ("vllm", "vllm"),
            ("vllm-ascend", "vllm_ascend"),
            ("torch", "torch"),
            ("torch-npu", "torch_npu"),
        ):
            packages[distribution] = metadata.version(distribution)
            module = import_module(module_name)
            modules[distribution] = sha256(
                Path(module.__file__).read_bytes()
            ).hexdigest()
        return {
            "packages": packages,
            "implementation_sha256": self.accounting.implementation_digest(
                self.llm, self.config, modules
            ),
        }

    def fence(self):
        receipts = self.llm.collective_rpc(synchronize_worker, timeout=60)
        if {row["rank"] for row in receipts} != {0, 1} or len(receipts) != 2:
            raise ValueError("Missing unique TP2 worker fences")
        if {row["logical_device"] for row in receipts} != {0, 1}:
            raise ValueError(
                "TP2 worker fences disagree with the two visible logical devices"
            )
        return self.cards

    def warmup(self):
        self.llm.generate(self.warmup_inputs, self.sampling, use_tqdm=False)
        self.fence()
        if not self.llm.reset_prefix_cache():
            raise ValueError("Failed to reset prefix cache after warm-up")
        if self.collector is not None:
            self.collector.begin_capture()

    def run_batch(self):
        return self.llm.generate(self.inputs, self.sampling, use_tqdm=False)

    def request_identity(self, outputs):
        if len(outputs) != len(self.inputs):
            raise ValueError("Incomplete live inference population")
        requests = []
        for expected, result in zip(self.inputs, outputs, strict=True):
            if (
                not result.finished
                or len(result.outputs) != 1
                or list(result.prompt_token_ids) != expected["prompt_token_ids"]
            ):
                raise ValueError(
                    "Live response differs from the frozen token-ID population"
                )
            choice = result.outputs[0]
            if not choice.finish_reason or not choice.token_ids:
                raise ValueError("Live response is incomplete or has no output tokens")
            requests.append(
                {
                    "id": result.request_id,
                    "input_tokens": len(result.prompt_token_ids),
                    "output_tokens": len(choice.token_ids),
                    "finish_reason": choice.finish_reason,
                    "output_ids_sha256": canonical_sha(list(choice.token_ids)),
                }
            )
        write(self.output / "request-counts.json", requests)
        return canonical_sha(
            [
                {key: value for key, value in row.items() if key != "id"}
                for row in requests
            ]
        )

    def account(self, outputs):
        return self.collector.effective_work(outputs)

    def account_timing(self, outputs):
        if self.host_counted_timing:
            return self.collector.effective_work(outputs)
        return self.accounting.account_timing(
            self.llm, self.config, outputs, self.output
        )

    def measurement_method(self):
        if self.host_counted_timing:
            return self.accounting.measurement_method()
        return "Host monotonic time with native two-card fences, separate from path observation"

    def run_operation(self, name):
        return self.collector.run_operation(name)

    def dependency_graph(self, outputs):
        return self.collector.dependency_graph(outputs)

    def close(self):
        try:
            if self.collector is not None:
                self.collector.close()
        finally:
            self.llm.llm_engine.engine_core.shutdown()


def open_backend(config, workload, output, phase, cards):
    return NativeInference(config, workload, output, phase, cards)
