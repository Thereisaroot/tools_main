from __future__ import annotations

import ctypes
import os
import sys
from dataclasses import dataclass

try:
    import resource
except Exception:  # pragma: no cover - not available on Windows
    resource = None

try:
    import psutil
except Exception:  # pragma: no cover - optional runtime import
    psutil = None


@dataclass(slots=True)
class MemorySnapshot:
    rss_mb: float
    tree_rss_mb: float
    phys_footprint_mb: float | None
    child_count: int
    backend: str


def _ru_maxrss_mb() -> float:
    if resource is None:
        return 0.0
    rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return rss / (1024.0 * 1024.0)
    return rss / 1024.0


def _darwin_phys_footprint_mb(pid: int) -> float | None:
    if sys.platform != "darwin":
        return None
    try:
        lib = ctypes.CDLL("/usr/lib/libproc.dylib")
    except Exception:
        return None

    class _RUsageInfoV2(ctypes.Structure):
        _fields_ = [
            ("ri_uuid", ctypes.c_uint8 * 16),
            ("ri_user_time", ctypes.c_uint64),
            ("ri_system_time", ctypes.c_uint64),
            ("ri_pkg_idle_wkups", ctypes.c_uint64),
            ("ri_interrupt_wkups", ctypes.c_uint64),
            ("ri_pageins", ctypes.c_uint64),
            ("ri_wired_size", ctypes.c_uint64),
            ("ri_resident_size", ctypes.c_uint64),
            ("ri_phys_footprint", ctypes.c_uint64),
            ("ri_proc_start_abstime", ctypes.c_uint64),
            ("ri_proc_exit_abstime", ctypes.c_uint64),
            ("ri_child_user_time", ctypes.c_uint64),
            ("ri_child_system_time", ctypes.c_uint64),
            ("ri_child_pkg_idle_wkups", ctypes.c_uint64),
            ("ri_child_interrupt_wkups", ctypes.c_uint64),
            ("ri_child_pageins", ctypes.c_uint64),
            ("ri_child_elapsed_abstime", ctypes.c_uint64),
            ("ri_diskio_bytesread", ctypes.c_uint64),
            ("ri_diskio_byteswritten", ctypes.c_uint64),
        ]

    proc_pid_rusage = lib.proc_pid_rusage
    proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    proc_pid_rusage.restype = ctypes.c_int

    info = _RUsageInfoV2()
    # RUSAGE_INFO_V2 = 2
    ret = proc_pid_rusage(int(pid), 2, ctypes.byref(info))
    if ret != 0:
        return None
    return float(info.ri_phys_footprint) / (1024.0 * 1024.0)


def get_memory_snapshot() -> MemorySnapshot:
    pid = os.getpid()
    footprint = _darwin_phys_footprint_mb(pid)
    if psutil is None:
        value = _ru_maxrss_mb()
        return MemorySnapshot(
            rss_mb=value,
            tree_rss_mb=value,
            phys_footprint_mb=footprint,
            child_count=0,
            backend="resource" if resource is not None else "none",
        )

    proc = psutil.Process(pid)
    rss = float(proc.memory_info().rss) / (1024.0 * 1024.0)
    total = rss
    children = 0
    for child in proc.children(recursive=True):
        children += 1
        try:
            total += float(child.memory_info().rss) / (1024.0 * 1024.0)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    return MemorySnapshot(
        rss_mb=rss,
        tree_rss_mb=total,
        phys_footprint_mb=footprint,
        child_count=children,
        backend="psutil",
    )
