"""Read the dedicated-memory counter for the owned Krea worker on Windows.

CUDA adapter mapping and exact PDH array reads come from SceneWorks/inference
e3adbd36b0eab3cfc872936a85d05e93a20afa1a:scripts/ci/wddm_process_memory.py.
"""
import argparse
import ctypes as C
import json
import ntpath
import os
import re
import queue
import sys
import threading
import time
import uuid
from contextlib import contextmanager


COUNTER = r"\GPU Process Memory(*)\Dedicated Usage"
NO_INSTANCE = 0x800007D1
NO_DATA = 0x800007D5
MORE_DATA = 0x800007D2
LARGE_RAWCOUNT = 0x10100
UUID = re.compile(r"GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", re.I)
DWORD = C.c_uint32
HANDLE = C.c_void_p


def _function(dll, name, args):
    fn = getattr(dll, name)
    fn.argtypes = args
    fn.restype = DWORD
    return fn


def _check(status, operation):
    if status:
        raise OSError(f"{operation} returned 0x{status:08x}")


def _adapter(cuda_uuid):
    """CUDA driver metadata only. No device context, uploads, kernels or device-mode writes."""
    if not isinstance(cuda_uuid, str) or not UUID.fullmatch(cuda_uuid):
        raise ValueError("expected a complete CUDA UUID")
    cuda = C.WinDLL("nvcuda.dll", winmode=0x800)  # LOAD_LIBRARY_SEARCH_SYSTEM32
    _check(_function(cuda, "cuInit", [DWORD])(0), "cuInit")
    count = C.c_int()
    _check(_function(cuda, "cuDeviceGetCount", [C.POINTER(C.c_int)])(C.byref(count)), "cuDeviceGetCount")
    device_get = _function(cuda, "cuDeviceGet", [C.POINTER(C.c_int), C.c_int])
    uuid_get = _function(cuda, "cuDeviceGetUuid_v2", [HANDLE, C.c_int])
    if count.value != 1:
        raise ValueError("expected exactly one visible CUDA device")
    device, raw = C.c_int(), (C.c_ubyte * 16)()
    _check(device_get(C.byref(device), 0), "cuDeviceGet")
    _check(uuid_get(raw, device), "cuDeviceGetUuid_v2")
    if uuid.UUID(bytes=bytes(raw)) != uuid.UUID(cuda_uuid[4:]):
        raise ValueError("masked CUDA ordinal zero does not match the leased GPU UUID")
    luid, mask = (C.c_ubyte * 8)(), DWORD()
    fn = _function(cuda, "cuDeviceGetLuid", [HANDLE, C.POINTER(DWORD), C.c_int])
    _check(fn(luid, C.byref(mask), device), "cuDeviceGetLuid")
    if not any(luid) or mask.value == 0 or mask.value & (mask.value - 1):
        raise ValueError("CUDA device does not identify exactly one adapter node")
    high, low = int.from_bytes(luid[4:], "little"), int.from_bytes(luid[:4], "little")
    return {"cuda_uuid": cuda_uuid, "luid": f"luid_0x{high:08x}_0x{low:08x}",
            "node_mask": mask.value, "physical_node": mask.value.bit_length() - 1, "counter": COUNTER}


class _Raw(C.Structure):
    _fields_ = [("status", DWORD), ("stamp_low", DWORD), ("stamp_high", DWORD),
                ("first", C.c_int64), ("second", C.c_int64), ("multi", DWORD)]


class _RawItem(C.Structure):
    _fields_ = [("name", C.c_wchar_p), ("raw", _Raw)]


def _sample(raw):
    if raw.status == NO_INSTANCE:
        return None
    if raw.status not in (0, 1) or raw.first < 0:
        raise OSError(f"invalid PDH byte gauge status/value 0x{raw.status:x}/{raw.first}")
    stamp = raw.stamp_high << 32 | raw.stamp_low
    if not stamp:
        raise OSError("PDH byte gauge has no provider timestamp")
    return [stamp, raw.first]


class _Pdh:
    def __init__(self):
        dll = C.WinDLL("pdh.dll", winmode=0x800)
        self.open = _function(dll, "PdhOpenQueryW", [C.c_wchar_p, C.c_size_t, C.POINTER(HANDLE)])
        self.add = _function(dll, "PdhAddEnglishCounterW", [HANDLE, C.c_wchar_p, C.c_size_t, C.POINTER(HANDLE)])
        self.collect = _function(dll, "PdhCollectQueryData", [HANDLE])
        self.raw = _function(dll, "PdhGetRawCounterValue", [HANDLE, C.POINTER(DWORD), C.POINTER(_Raw)])
        self.array = _function(dll, "PdhGetRawCounterArrayW", [HANDLE, C.POINTER(DWORD), C.POINTER(DWORD), HANDLE])
        self.close = _function(dll, "PdhCloseQuery", [HANDLE])

    def read(self, handle):
        kind, raw = DWORD(), _Raw()
        _check(self.raw(handle, C.byref(kind), C.byref(raw)), "PdhGetRawCounterValue")
        if raw.status != NO_INSTANCE and kind.value != LARGE_RAWCOUNT:
            raise OSError(f"invalid PDH byte gauge type 0x{kind.value:x}")
        return _sample(raw)

    def selected_rows(self, counter, instance):
        # PdhGetRawCounterArrayW returns the actual instance name with each raw
        # sample. Do not depend on the value obtained by requesting an index
        # the provider may alias. No foreign process identities/bytes are retained.
        for _ in range(3):
            size, count = DWORD(), DWORD()
            status = self.array(counter, C.byref(size), C.byref(count), None)
            if status == NO_DATA or (status == 0 and not size.value and not count.value):
                return []
            if status != MORE_DATA:
                _check(status, "PdhGetRawCounterArrayW sizing")
            if not 0 < size.value <= 16 * 1024**2:
                raise OSError("invalid PDH counter array size")
            buffer = C.create_string_buffer(size.value)
            status = self.array(counter, C.byref(size), C.byref(count), buffer)
            if status == MORE_DATA:
                continue  # Windows requires sizing again, not trusting this call's length
            if status == NO_DATA:
                return []
            _check(status, "PdhGetRawCounterArrayW")
            if count.value * C.sizeof(_RawItem) > size.value or size.value > len(buffer):
                raise OSError("invalid PDH counter array length")
            items = (_RawItem * count.value).from_buffer(buffer)
            selected = []
            for item in items:
                name = (item.name or "").casefold()
                base = name.partition("#")[0]
                if base == instance:
                    selected.append((name, _sample(item.raw)))
            return selected
        raise OSError("PDH counter array size kept changing")


@contextmanager
def _query(adapter, pid):
    instance = f"pid_{pid}_{adapter['luid']}_phys_{adapter['physical_node']}"
    api, query = _Pdh(), HANDLE()
    _check(api.open(None, 0, C.byref(query)), "PdhOpenQueryW")
    try:
        def counter(name):
            handle = HANDLE()
            path = rf"\GPU Process Memory({name})\Dedicated Usage"
            _check(api.add(query, path, 0, C.byref(handle)), "PdhAddEnglishCounterW")
            return handle

        wildcard = counter("*")
        exact = counter(instance)

        def read():
            status = api.collect(query)
            if status != NO_DATA:
                _check(status, "PdhCollectQueryData")
            rows = api.selected_rows(wildcard, instance)
            if len(rows) > 1 or (rows and rows[0][0] not in (instance, instance + "#0")):
                raise OSError("ambiguous enumerated PDH dedicated-memory instance")
            if not rows or rows[0][1] is None:
                return None
            # The scalar call supplies lpdwType. Source the number and instance
            # identity from the array. Both must be live in this collection.
            if status == NO_DATA or api.read(exact) is None:
                raise OSError("inconsistent PDH instance/type observation")
            return rows[0][1]

        yield read
    finally:
        _check(api.close(query), "PdhCloseQuery")

INTERVAL_MS = 200
WINDOWS_EPOCH_FILETIME = 116444736000000000
WAIT_TIMEOUT = 0x102


class _FileTime(C.Structure):
    _fields_ = [("low", DWORD), ("high", DWORD)]


@contextmanager
def _worker(pid, executable):
    # Hold the original process handle until the collector exits, even if Windows
    # reuses its numeric PID. A successful path comparison alone is insufficient.
    kernel = C.WinDLL("kernel32.dll", use_last_error=True)
    open_process = kernel.OpenProcess
    open_process.argtypes = (DWORD, C.c_int, DWORD)
    open_process.restype = HANDLE
    handle = open_process(0x1000 | 0x100000, 0, pid)  # QUERY_LIMITED_INFORMATION | SYNCHRONIZE
    if not handle:
        raise C.WinError(C.get_last_error())
    close = _function(kernel, "CloseHandle", [HANDLE])
    wait = _function(kernel, "WaitForSingleObject", [HANDLE, DWORD])
    times = _function(kernel, "GetProcessTimes", [HANDLE] + [C.POINTER(_FileTime)] * 4)
    image = _function(kernel, "QueryFullProcessImageNameW", [HANDLE, DWORD, C.c_wchar_p, C.POINTER(DWORD)])
    try:
        path = C.create_unicode_buffer(32768)
        size = DWORD(len(path))
        created, exited, system, user = (_FileTime() for _ in range(4))
        if not image(handle, 0, path, C.byref(size)) or not times(
                handle, C.byref(created), C.byref(exited), C.byref(system), C.byref(user)):
            raise C.WinError(C.get_last_error())
        if ntpath.normcase(ntpath.normpath(path.value)) != ntpath.normcase(ntpath.normpath(executable)):
            raise ValueError("owned Krea worker executable differs from the launched executable")
        created_filetime = created.high << 32 | created.low
        if not created_filetime:
            raise ValueError("owned Krea worker creation time is unavailable")

        def verify_live():
            if wait(handle, 0) != WAIT_TIMEOUT:
                raise ValueError("original owned Krea worker exited during capture")

        verify_live()
        yield {"pid": pid, "created_filetime": str(created_filetime), "executable": path.value}, verify_live
    finally:
        close(handle)


def _local_filetime_clock():
    # PDH_RAW_COUNTER.TimeStamp is a local FILETIME, while GetProcessTimes
    # and the Python wall clock are UTC. The two domains must stay distinct.
    kernel = C.WinDLL("kernel32.dll", use_last_error=True)
    convert = _function(kernel, "FileTimeToLocalFileTime", [C.POINTER(_FileTime)] * 2)

    def read():
        stamp = WINDOWS_EPOCH_FILETIME + time.time_ns() // 100
        utc, local = _FileTime(stamp & 0xffffffff, stamp >> 32), _FileTime()
        if not convert(C.byref(utc), C.byref(local)):
            raise C.WinError(C.get_last_error())
        return local.high << 32 | local.low

    return read


def _commands(stream, mailbox):
    for line in stream:
        mailbox.put(line.strip())
    mailbox.put(None)


def _emit(row):
    print(json.dumps(row, separators=(",", ":")), flush=True)


def _capture(args):
    adapter = _adapter(args.cuda_uuid)
    local_filetime = _local_filetime_clock()
    mailbox = queue.SimpleQueue()
    threading.Thread(target=_commands, args=(sys.stdin, mailbox), daemon=True).start()
    with _worker(args.pid, args.executable) as (identity, verify_live), _query(adapter, args.pid) as read:
        start = time.monotonic()
        _emit({"event": "ready", "identity": identity, "adapter": adapter, "interval_ms": INTERVAL_MS})
        terminal = None
        active = False
        last = None
        next_poll = start
        while True:
            try:
                command = mailbox.get(timeout=max(0, next_poll - time.monotonic()))
                if command is None:
                    return
                if command == "active" and not active:
                    active = True
                    _emit({"event": "active", "elapsed_ms": int((time.monotonic() - start) * 1000)})
                    continue
                if command != "terminal" or terminal is not None:
                    raise ValueError("unexpected Krea memory observer command")
                terminal = local_filetime()
                _emit({"event": "terminal_requested", "elapsed_ms": int((time.monotonic() - start) * 1000),
                       "not_before_filetime": str(terminal)})
                continue  # the terminal read uses the next normal sample, not a second cadence
            except queue.Empty:
                # A mailbox timeout starts the next scheduled sample.
                pass
            read_started_ms = int((time.monotonic() - start) * 1000)
            verify_live()
            read_started_filetime = local_filetime()
            sample = read()
            read_ended_filetime = local_filetime()
            verify_live()
            now = time.monotonic()
            elapsed_ms = int((now - start) * 1000)
            next_poll = now + INTERVAL_MS / 1000
            timing = {"elapsed_ms": elapsed_ms, "read_started_ms": read_started_ms,
                      "read_started_filetime": str(read_started_filetime),
                      "read_ended_filetime": str(read_ended_filetime)}
            if sample is None:
                _emit({"event": "missing", **timing})
                if active or last is not None or terminal is not None:
                    raise ValueError("owned PDH counter absent after attachment or terminal request")
                continue
            if sample == last:
                _emit({"event": "cached", **timing, "filetime": str(sample[0])})
                continue
            if last is not None and sample[0] <= last[0]:
                raise ValueError("PDH provider timestamp regressed or was reused with different bytes")
            last = sample
            measured_terminal = terminal is not None and sample[0] >= terminal
            _emit({"event": "terminal" if measured_terminal else "sample", **timing,
                   "filetime": str(sample[0]), "dedicated_bytes": sample[1]})
            if measured_terminal:
                return


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--executable", required=True)
    parser.add_argument("--cuda-uuid", required=True)
    args = parser.parse_args()
    if os.name != "nt" or not 0 < args.pid < 2**32 or not ntpath.isabs(args.executable):
        parser.error("expected Windows, a valid worker PID and an absolute executable")
    try:
        _capture(args)
    except (OSError, ValueError) as exc:
        _emit({"event": "error", "message": str(exc)})
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
