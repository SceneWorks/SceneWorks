"""Windows Job Object ownership and memory probes for guarded media children.

The root process is created suspended, assigned to a kill-on-close Job, and only
then resumed. An incompatible enclosing CI Job is an explicit refusal.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import time
from ctypes import wintypes


class WindowsJobError(RuntimeError):
    pass


class _BasicLimits(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class _IoCounters(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint64) for name in (
        "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
        "ReadTransferCount", "WriteTransferCount", "OtherTransferCount",
    )]


class _ExtendedLimits(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _BasicLimits), ("IoInfo", _IoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class _ThreadEntry(ctypes.Structure):
    _fields_ = [
        ("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD),
        ("th32ThreadID", wintypes.DWORD), ("th32OwnerProcessID", wintypes.DWORD),
        ("tpBasePri", wintypes.LONG), ("tpDeltaPri", wintypes.LONG),
        ("dwFlags", wintypes.DWORD),
    ]


class _ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]


class _MemoryStatus(ctypes.Structure):
    _fields_ = [
        ("dwLength", wintypes.DWORD), ("dwMemoryLoad", wintypes.DWORD),
        ("ullTotalPhys", ctypes.c_uint64), ("ullAvailPhys", ctypes.c_uint64),
        ("ullTotalPageFile", ctypes.c_uint64), ("ullAvailPageFile", ctypes.c_uint64),
        ("ullTotalVirtual", ctypes.c_uint64), ("ullAvailVirtual", ctypes.c_uint64),
        ("ullAvailExtendedVirtual", ctypes.c_uint64),
    ]


def _apis():
    if os.name != "nt":
        raise WindowsJobError("Windows Job Object APIs require Windows")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel.CreateJobObjectW.restype = wintypes.HANDLE
    kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int,
                                               ctypes.c_void_p, wintypes.DWORD]
    kernel.SetInformationJobObject.restype = wintypes.BOOL
    kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel.QueryInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int,
                                                 ctypes.c_void_p, wintypes.DWORD,
                                                 ctypes.POINTER(wintypes.DWORD)]
    kernel.QueryInformationJobObject.restype = wintypes.BOOL
    kernel.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
    kernel.TerminateJobObject.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    kernel.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    kernel.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel.Thread32First.argtypes = [wintypes.HANDLE, ctypes.POINTER(_ThreadEntry)]
    kernel.Thread32First.restype = wintypes.BOOL
    kernel.Thread32Next.argtypes = [wintypes.HANDLE, ctypes.POINTER(_ThreadEntry)]
    kernel.Thread32Next.restype = wintypes.BOOL
    kernel.OpenThread.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenThread.restype = wintypes.HANDLE
    kernel.ResumeThread.argtypes = [wintypes.HANDLE]
    kernel.ResumeThread.restype = wintypes.DWORD
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(_MemoryStatus)]
    kernel.GlobalMemoryStatusEx.restype = wintypes.BOOL
    kernel.GetSystemDirectoryW.argtypes = [wintypes.LPWSTR, wintypes.UINT]
    kernel.GetSystemDirectoryW.restype = wintypes.UINT
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE,
                                           ctypes.POINTER(_ProcessMemoryCounters), wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    return kernel, psapi


def _failed(operation: str) -> WindowsJobError:
    return WindowsJobError(f"{operation}: {ctypes.WinError(ctypes.get_last_error())}")


def host_free_bytes() -> int:
    kernel, _ = _apis()
    status = _MemoryStatus()
    status.dwLength = ctypes.sizeof(status)
    if not kernel.GlobalMemoryStatusEx(ctypes.byref(status)) or not status.ullAvailPhys:
        raise _failed("GlobalMemoryStatusEx")
    return int(status.ullAvailPhys)


def trusted_nvidia_smi() -> str:
    kernel, _ = _apis()
    buffer = ctypes.create_unicode_buffer(32768)
    length = kernel.GetSystemDirectoryW(buffer, len(buffer))
    if not 0 < length < len(buffer):
        raise _failed("GetSystemDirectoryW")
    path = os.path.join(buffer.value, "nvidia-smi.exe")
    if not os.path.isfile(path):
        raise WindowsJobError("trusted System32 nvidia-smi.exe is unavailable")
    return path


class WindowsJob:
    """Own one suspended root and every descendant through a non-breakaway Job."""

    def __init__(self):
        self.kernel, self.psapi = _apis()
        self.handle = self.kernel.CreateJobObjectW(None, None)
        if not self.handle:
            raise _failed("CreateJobObjectW")
        limits = _ExtendedLimits()
        limits.BasicLimitInformation.LimitFlags = 0x2000  # KILL_ON_JOB_CLOSE
        if not self.kernel.SetInformationJobObject(self.handle, 9, ctypes.byref(limits),
                                                   ctypes.sizeof(limits)):
            self.close()
            raise _failed("SetInformationJobObject")

    def assign_and_resume(self, child: subprocess.Popen[bytes]) -> None:
        if not self.kernel.AssignProcessToJobObject(self.handle, child._handle):
            raise _failed("AssignProcessToJobObject")
        snapshot = self.kernel.CreateToolhelp32Snapshot(0x00000004, 0)  # SNAPTHREAD
        if not snapshot or snapshot == ctypes.c_void_p(-1).value:
            raise _failed("CreateToolhelp32Snapshot")
        try:
            entry = _ThreadEntry()
            entry.dwSize = ctypes.sizeof(entry)
            if not self.kernel.Thread32First(snapshot, ctypes.byref(entry)):
                raise _failed("Thread32First")
            threads = []
            while True:
                if entry.th32OwnerProcessID == child.pid:
                    threads.append(entry.th32ThreadID)
                if not self.kernel.Thread32Next(snapshot, ctypes.byref(entry)):
                    if ctypes.get_last_error() != 18:  # ERROR_NO_MORE_FILES
                        raise _failed("Thread32Next")
                    break
            if len(threads) != 1:
                raise WindowsJobError("suspended child must have exactly one primary thread")
            thread = self.kernel.OpenThread(0x0002, False, threads[0])  # SUSPEND_RESUME
            if not thread:
                raise _failed("OpenThread")
            try:
                if self.kernel.ResumeThread(thread) != 1:
                    raise WindowsJobError("primary thread was not suspended exactly once")
            finally:
                self.kernel.CloseHandle(thread)
        finally:
            self.kernel.CloseHandle(snapshot)

    def members(self) -> set[int]:
        for capacity in (16, 64, 256, 1024):
            class ProcessList(ctypes.Structure):
                _fields_ = [("NumberOfAssignedProcesses", wintypes.DWORD),
                            ("NumberOfProcessIdsInList", wintypes.DWORD),
                            ("ProcessIdList", ctypes.c_size_t * capacity)]
            listing = ProcessList()
            returned = wintypes.DWORD()
            if self.kernel.QueryInformationJobObject(self.handle, 3, ctypes.byref(listing),
                                                      ctypes.sizeof(listing), ctypes.byref(returned)):
                if listing.NumberOfAssignedProcesses > listing.NumberOfProcessIdsInList:
                    continue
                return set(listing.ProcessIdList[:listing.NumberOfProcessIdsInList])
            if ctypes.get_last_error() != 234:  # ERROR_MORE_DATA
                raise _failed("QueryInformationJobObject/process-list")
        raise WindowsJobError("owned Job has more than 1024 processes")

    def footprint_bytes(self) -> int:
        pids = self.members()
        if not pids:
            raise WindowsJobError("owned Job disappeared during memory sample")
        total = 0
        for pid in pids:
            process = self.kernel.OpenProcess(0x0400 | 0x0010, False, pid)  # QUERY_INFO|VM_READ
            if not process:
                if pid not in self.members():
                    continue  # Exited between the Job list and OpenProcess.
                raise _failed(f"OpenProcess({pid})")
            try:
                counters = _ProcessMemoryCounters()
                counters.cb = ctypes.sizeof(counters)
                if not self.psapi.GetProcessMemoryInfo(process, ctypes.byref(counters),
                                                        counters.cb):
                    if pid not in self.members():
                        continue  # Exited during the sample.
                    raise _failed(f"GetProcessMemoryInfo({pid})")
                total += max(counters.WorkingSetSize, counters.PrivateUsage)
            finally:
                self.kernel.CloseHandle(process)
        limits = _ExtendedLimits()
        if not self.kernel.QueryInformationJobObject(self.handle, 9, ctypes.byref(limits),
                                                      ctypes.sizeof(limits), None):
            raise _failed("QueryInformationJobObject/peak-memory")
        return max(total, limits.PeakJobMemoryUsed)

    def terminate_and_reap(self, child: subprocess.Popen[bytes], grace: float) -> None:
        # TerminateJobObject is atomic for every member, including descendants that outlive root.
        if not self.kernel.TerminateJobObject(self.handle, 1):
            raise _failed("TerminateJobObject")
        child.wait(timeout=grace)
        until = time.monotonic() + grace
        while time.monotonic() < until:
            if not self.members():
                return
            time.sleep(0.01)
        raise WindowsJobError("owned Job still has live processes after termination")

    def close(self) -> None:
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None
