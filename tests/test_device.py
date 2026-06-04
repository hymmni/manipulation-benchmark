import torch
from utils.device import get_device, to_device


def test_get_device_returns_torch_device():
    d = get_device()
    assert isinstance(d, torch.device)


def test_get_device_cpu_forced():
    d = get_device("cpu")
    assert d == torch.device("cpu")


def test_get_device_auto_returns_valid():
    d = get_device("auto")
    assert d.type in ("cuda", "mps", "cpu")


def test_to_device_tensor():
    t = torch.zeros(3)
    moved = to_device(t, torch.device("cpu"))
    assert isinstance(moved, torch.Tensor)
    assert moved.device.type == "cpu"


def test_to_device_dict():
    d = {"a": torch.zeros(2), "b": torch.ones(2)}
    moved = to_device(d, torch.device("cpu"))
    assert all(v.device.type == "cpu" for v in moved.values())


def test_to_device_passthrough():
    assert to_device(42, torch.device("cpu")) == 42
    assert to_device("hello", torch.device("cpu")) == "hello"
