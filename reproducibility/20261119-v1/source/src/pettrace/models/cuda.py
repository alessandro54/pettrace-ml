"""Pinned ONNX Runtime CUDA provider used by experiment embedding extraction."""

from copy import copy
import json
from pathlib import Path
import platform
import shutil
import sys
import tarfile
import tempfile
from urllib.request import urlopen

import onnxruntime as ort

from pettrace.data.verification import sha256


PLUGIN_VERSION = "0.3.0"
PLUGIN_BASE = "https://github.com/microsoft/onnxruntime/releases/download/plugin-ep-cuda/v0.3.0/"
PLUGIN_SPECS = {
    "12": {"minimum": (12, 8), "file": "cuda_ep_cuda12_0.3.0_linux-x64.tar.gz",
           "sha256": "ef6864a088b8c2a74e46b18a016eac8dd47950cd3a2a224603998c4fbb09d332"},
    "13": {"minimum": (13, 0), "file": "cuda_ep_cuda13_0.3.0_linux-x64.tar.gz",
           "sha256": "0818121f81fc4461f9ba2b588dbcd03d62efb039a1d87169dfe74e03c37acd44"},
}
OPTIONS = {"use_tf32": "0", "cudnn_conv_algo_search": "HEURISTIC"}
_REGISTERED = False


def _plugin_spec(cuda_version):
    try:
        version = tuple(map(int, cuda_version.split(".")[:2]))
    except (AttributeError, ValueError):
        version = ()
    spec = PLUGIN_SPECS.get(str(version[0])) if version else None
    if spec is None or version < spec["minimum"]:
        raise RuntimeError(f"Unsupported PyTorch CUDA {cuda_version!r}; requires CUDA 12.8+ or CUDA 13.x")
    return str(version[0]), spec


def _plugin(cache, cuda_version):
    """Download and verify the official provider binary once per CUDA major."""
    major, spec = _plugin_spec(cuda_version)
    cache = Path(cache) / f"cuda{major}"
    library, receipt = cache / "libonnxruntime_providers_cuda.so", cache / "plugin.json"
    if library.is_file() and receipt.is_file():
        saved = json.loads(receipt.read_text())
        if saved.get("archive_sha256") != spec["sha256"] or saved.get("library_sha256") != sha256(library):
            raise ValueError("CUDA plugin checksum mismatch")
        return library
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=cache, prefix=".download-") as folder:
        archive_path = Path(folder) / "plugin.tar.gz"
        with urlopen(PLUGIN_BASE + spec["file"], timeout=120) as incoming, archive_path.open("wb") as outgoing:
            shutil.copyfileobj(incoming, outgoing)
        if sha256(archive_path) != spec["sha256"]:
            raise ValueError("CUDA plugin archive checksum mismatch")
        with tarfile.open(archive_path) as archive:
            members = [member for member in archive if Path(member.name).name == library.name]
            if len(members) != 1 or not members[0].isfile():
                raise ValueError("CUDA archive must contain exactly one regular provider library")
            partial = Path(folder) / library.name
            with archive.extractfile(members[0]) as incoming, partial.open("wb") as outgoing:
                shutil.copyfileobj(incoming, outgoing)
        digest = sha256(partial)
        partial.replace(library)
        receipt.write_text(json.dumps({"archive_sha256": spec["sha256"], "library_sha256": digest}))
    return library


def check_cuda_environment():
    if ort.__version__ != "1.31.0" or sys.platform != "linux" or platform.machine() != "x86_64":
        raise RuntimeError("CUDA experiments require Linux x86_64 and ONNX Runtime 1.31.0")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA device. Select a GPU runtime before starting the experiment.")
    _plugin_spec(torch.version.cuda)
    return torch


def cuda_session(model, model_file, cache):
    """Copy a pipeline component and replace its session with verified CUDA execution."""
    global _REGISTERED
    torch = check_cuda_environment()
    _, spec = _plugin_spec(torch.version.cuda)
    library = _plugin(cache, torch.version.cuda)
    if not _REGISTERED:
        ort.register_execution_provider_library("CUDAExecutionProvider", str(library))
        _REGISTERED = True
    devices = [device for device in ort.get_ep_devices() if device.ep_name == "CUDAExecutionProvider"]
    if not devices:
        raise RuntimeError("CUDA plugin registered no device; CPU fallback is not allowed")
    options = ort.SessionOptions()
    options.add_session_config_entry("session.record_ep_graph_assignment_info", "1")
    options.add_provider_for_devices([devices[0]], OPTIONS)
    accelerated = copy(model)
    accelerated.session = ort.InferenceSession(str(model_file), sess_options=options)
    accelerated.session.disable_fallback()
    assignments = list(accelerated.session.get_provider_graph_assignment_info())
    convolutions = [group.ep_name for group in assignments for node in group.get_nodes() if "Conv" in node.op_type]
    if not convolutions or any(provider != "CUDAExecutionProvider" for provider in convolutions):
        raise RuntimeError("Model convolutions are not all assigned to CUDA")
    evidence = {
        "provider": "CUDAExecutionProvider", "plugin_version": PLUGIN_VERSION,
        "plugin_archive_sha256": spec["sha256"], "library_sha256": sha256(library),
        "options": OPTIONS, "gpu": torch.cuda.get_device_name(0), "torch_cuda": torch.version.cuda,
        "graph_nodes": {group.ep_name: sum(len(item.get_nodes()) for item in assignments
                                           if item.ep_name == group.ep_name) for group in assignments},
    }
    return accelerated, evidence
