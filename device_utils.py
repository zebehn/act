import platform
import torch


def _mps_is_available():
    return (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_built()
        and torch.backends.mps.is_available()
    )


def resolve_device(preferred="auto"):
    """
    Resolve the runtime device.

    preferred:
      - "auto": prefer MPS on Apple Silicon, then CUDA, then CPU
      - "mps" | "cuda" | "cpu": force a specific backend
    """
    if preferred is None:
        preferred = "auto"
    preferred = preferred.lower()

    if preferred == "auto":
        # Prioritize MPS on Apple Silicon; otherwise fall back to CUDA/CPU.
        if _mps_is_available() and platform.system() == "Darwin":
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        if _mps_is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if preferred == "mps":
        if not _mps_is_available():
            raise RuntimeError("MPS backend requested but not available in this PyTorch build.")
        return torch.device("mps")

    if preferred == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA backend requested but no CUDA device is available.")
        return torch.device("cuda")

    if preferred == "cpu":
        return torch.device("cpu")

    raise ValueError(f"Unsupported device preference: {preferred}")


def dataloader_pin_memory(device):
    return device.type == "cuda"
