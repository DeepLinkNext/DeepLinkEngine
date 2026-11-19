"""Loaded configuration uses metadata only, without transferring weights/state."""

import enum
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third-party-test-kit/proc_capture"))
from m01_model_inventory import collect


def test_live_module_metadata_does_not_read_tensor_values():
    class Activation(enum.Enum):
        SILU = "silu"
    class Tensor:
        shape = (256, 2, 2048)
        dtype = "torch.bfloat16"
        device = SimpleNamespace(type="npu")
        def cpu(self):
            raise AssertionError("Inventory must never read device tensor contents")
    class Module:
        top_k = 8
        activation = Activation.SILU
        scoring_func = "softmax"
        arbitrary_private_value = "excluded"
        def forward(self):
            pass
        def named_modules(self):
            return [("", self)]
        def named_parameters(self, recurse):
            assert recurse is False
            return [("weight", Tensor())]
        def named_buffers(self, recurse):
            return []
    result = collect(Module())
    row, = result["modules"]
    assert row["parameters"]["weight"]["shape"] == [256, 2, 2048]
    assert row["scalars"]["activation"]["name"] == "SILU"
    assert row["scalars"]["top_k"] == 8 and "arbitrary_private_value" not in row["scalars"]
    assert result["tensor_values_observed"] is False
    assert result["source_sha256"] and row["methods"]["forward"]["file"] == str(Path(__file__).resolve())


def test_quantized_dispatch_methods_bind_real_implementations_and_instance_overrides():
    class QuantMethod:
        def apply(self):
            pass
        def apply_gmm1(self):
            pass
        def apply_gmm2(self):
            pass
    class Module:
        quant_method = QuantMethod()
        def named_modules(self):
            return [("", self)]
        def named_parameters(self, recurse):
            return []
        def named_buffers(self, recurse):
            return []
    module = Module()
    module.quant_method.apply_gmm2 = lambda: None
    result = collect(module)
    methods = result["modules"][0]["objects"]["quant_method"]["methods"]
    assert set(methods) == {"apply", "apply_gmm1", "apply_gmm2"}
    for method in methods.values():
        assert method["implementation"]["file"] in result["source_sha256"]
    assert methods["apply"]["instance_override"] is False
    assert methods["apply_gmm2"]["instance_override"] is True


def test_slotted_quant_method_metadata_needs_no_instance_dictionary():
    class Method:
        __slots__ = ()
        def apply(self):
            pass
    class Module:
        quant_method = Method()
        def named_modules(self):
            return [("", self)]
        def named_parameters(self, recurse):
            return []
        def named_buffers(self, recurse):
            return []
    result = collect(Module())
    actual = result["modules"][0]["objects"]["quant_method"]["methods"]["apply"]
    assert actual["instance_override"] is False
    assert actual["implementation"]["file"] in result["source_sha256"]
