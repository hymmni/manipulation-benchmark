import torch


def get_device(prefer: str = "auto") -> torch.device:
    """Return best available device: cuda → mps → cpu. 'cpu' forces cpu."""
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def to_device(obj, device):
    """Recursively move tensors/modules/dicts to device."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, torch.nn.Module):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        moved = [to_device(v, device) for v in obj]
        return type(obj)(moved)
    return obj
