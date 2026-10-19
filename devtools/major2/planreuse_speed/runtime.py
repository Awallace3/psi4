"""Checked CUDA synchronization through the pinned, already-required cudart."""
import ctypes
from pathlib import Path
import config as s
import policy as p
import probe


class Cuda:
    def __init__(self):
        pin = p.RUNTIME_LIBS["libcudart.so"]
        s.require(str(Path(pin["alias"]).resolve(strict=True)) == pin["realpath"] and
                  s.sha(pin["realpath"]) == pin["sha256"], "pinned cudart identity")
        self.lib = ctypes.CDLL(pin["realpath"])
        self.function = self.lib.cudaDeviceSynchronize
        self.function.restype = ctypes.c_int
        self.function.argtypes = []
        self.lib.cudaRuntimeGetVersion.restype = ctypes.c_int
        self.lib.cudaRuntimeGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
        self.lib.cudaGetDeviceCount.restype = ctypes.c_int
        self.lib.cudaGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
        self.lib.cudaGetDevice.restype = ctypes.c_int
        self.lib.cudaGetDevice.argtypes = [ctypes.POINTER(ctypes.c_int)]
        values = []
        for fn in (self.lib.cudaRuntimeGetVersion, self.lib.cudaGetDeviceCount,
                   self.lib.cudaGetDevice):
            v = ctypes.c_int(-1)
            s.require(fn(ctypes.byref(v)) == 0, "CUDA runtime identity status")
            values.append(v.value)
        s.require(values == [13030, 1, 0], "CUDA13 single logical device 0")
        self.pin = pin
        self.maps = validate_maps()  # AFTER CDLL; never inferred from link aliases

    def synchronize(self):
        # Nonzero CUDA status propagates through Timer/dispatch/finally/controller.
        s.require(self.function() == 0, "cudaDeviceSynchronize failed")


def validate_maps():
    loaded = probe.maps()
    p.check_runtime_maps(loaded)
    p.check_required_runtime_maps(loaded)
    return loaded
