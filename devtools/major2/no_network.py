"""Fail-closed build-only Linux/x86_64 seccomp: no network source/dependency fetch.

No namespaces/privilege escalation/site changes. All descendants inherit the
filter. Runtime chemistry does NOT inherit it. Missing kernel support stops build.
"""
import ctypes
import errno
import os
import platform
import sys


class Filter(ctypes.Structure):
    _fields_ = [("code", ctypes.c_ushort), ("jt", ctypes.c_ubyte),
                ("jf", ctypes.c_ubyte), ("k", ctypes.c_uint)]


class Program(ctypes.Structure):
    _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.POINTER(Filter))]


def install():
    if sys.platform != "linux" or platform.machine() != "x86_64":
        raise RuntimeError("review seccomp syscall ABI for this platform")
    # seccomp_data: nr at offset0, arch at offset4. Require x86_64, reject
    # x32 syscall ABI too, then deny all socket/connect/sendto/sendmsg calls.
    # Socketpair (local pipe-like IPC) remains permitted.
    rows = [
        (0x20, 0, 0, 4), (0x15, 1, 0, 0xC000003E), (0x06, 0, 0, 0x80000000),
        (0x20, 0, 0, 0), (0x35, 0, 1, 0x40000000), (0x06, 0, 0, 0x80000000),
    ]
    for syscall in (41, 42, 44, 46):  # socket, connect, sendto, sendmsg
        rows += [(0x15, 0, 1, syscall), (0x06, 0, 0, 0x00050000 | errno.EPERM)]
    rows += [(0x06, 0, 0, 0x7FFF0000)]
    filters = (Filter * len(rows))(*(Filter(*row) for row in rows))
    prog = Program(len(rows), filters)
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(38, 1, 0, 0, 0) or libc.prctl(22, 2, ctypes.byref(prog), 0, 0):
        raise OSError(ctypes.get_errno(), "no-network seccomp setup failed")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise RuntimeError("command argv required")
    install()
    os.execvp(sys.argv[1], sys.argv[1:])
