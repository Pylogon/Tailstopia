"""Read-only process counters; child CPU includes the actual Tesseract process."""
import ctypes
from ctypes import wintypes
import os
import platform
import subprocess


def process_memory_cpu(handle=None):
    if os.name != "nt":
        return None
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    if handle is None:
        handle = kernel.GetCurrentProcess()
    else:
        handle = wintypes.HANDLE(int(handle))
    times = [wintypes.FILETIME() for _ in range(4)]
    kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    cpu = None
    if kernel.GetProcessTimes(handle, *(ctypes.byref(t) for t in times)):
        cpu = sum((t.dwHighDateTime << 32) + t.dwLowDateTime for t in times[2:]) / 10000

    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in ("peak_rss", "rss", "peak_paged", "paged",
            "peak_nonpaged", "nonpaged", "pagefile", "peak_pagefile")]
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    api = ctypes.WinDLL("psapi", use_last_error=True)
    api.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    ok = api.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
    return {"cpu_ms": cpu, "rss_bytes": counters.rss if ok else None,
            "peak_rss_bytes": counters.peak_rss if ok else None}


class ChildMeter:
    """Instrument only children already started by pytesseract; spawn nothing extra."""
    def __enter__(self):
        self.records = []
        original = subprocess.Popen
        self.original = original
        meter = self

        class TrackedPopen(original):
            def communicate(self, *args, **kwargs):
                try:
                    return super().communicate(*args, **kwargs)
                finally:
                    if self.poll() is not None and not getattr(self, "_stat_metered", False):
                        self._stat_metered = True
                        meter.records.append(process_memory_cpu(self._handle) if os.name == "nt" else None)
        subprocess.Popen = TrackedPopen
        return self

    def __exit__(self, *args):
        subprocess.Popen = self.original


def host_details():
    result = {"platform": platform.platform(), "python": platform.python_version(),
              "logical_cpus": os.cpu_count(), "processor": platform.processor()}
    if os.name == "nt":
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            result["processor"] = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    return result
