"""Explicit pilot hardware contracts, not automatic device fallback."""

PROFILES = {
    "H200": {
        "partition": "gpu-h200", "gres": "gpu:h200:1",
        "name": "H200", "architecture": "compute_90", "budget_gib": 120,
    },
    "RTX_PRO_BLACKWELL": {
        "partition": "gpu-rtxpro-blackwell", "gres": "gpu:rtx_pro_6000_blackwell:1",
        "name": "RTX PRO 6000 Blackwell", "architecture": "compute_120", "budget_gib": 80,
    },
}


def validate_device(family, name, architecture, memory_mib, budget_bytes):
    profile = PROFILES[family]
    if profile["name"] not in name or architecture != profile["architecture"]:
        raise RuntimeError("Actual CUDA/NVML device is not the requested " + family)
    if budget_bytes > memory_mib * 1024**2:
        raise RuntimeError("Declared device budget exceeds physical GPU memory")
