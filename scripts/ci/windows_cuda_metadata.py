"""Context-free CUDA ordinal/visibility -> UUID metadata, never NVML-index guessing.

ABI: app-pinned cudarc0.19.7 (Cargo.lock checksum1cea5f10...): driver/sys
CUdevice=c_int (407-408), CUuuid repr(C)c_char[16] (6912-6916), CUresult
repr(u32); extern-C cuInit(11036), cuDeviceGetCount(8318), cuDeviceGet(8270),
cuDeviceGetUuid(8546). result.rs88-89,109-123,167-178 uses these metadata APIs
(the CUDA12.9 path uses unsuffixed UUID). NO context/memory/kernel APIs.
"""
from __future__ import annotations
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import uuid

ABI_CRATE_CHECKSUM = "1cea5f10a99e025c1b44ae2354c2d8326b25ddbd0baf76bde8e55cfd4018a2cc"


class CUuuid(ctypes.Structure):
    _fields_ = [("bytes", ctypes.c_char * 16)]


def visible_devices(library=None):
    if library is None:
        if os.name != "nt":
            raise RuntimeError("CUDA metadata probe requires Windows")
        # Same authoritative System32 resolution ABI used by pinned WindowsJob.
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetSystemDirectoryW.argtypes = [wintypes.LPWSTR, wintypes.UINT]
        kernel.GetSystemDirectoryW.restype = wintypes.UINT
        directory = ctypes.create_unicode_buffer(32768)
        length = kernel.GetSystemDirectoryW(directory, len(directory))
        if not 0 < length < len(directory):
            raise RuntimeError("cannot resolve trusted System32 CUDA driver")
        driver = Path(directory.value) / "nvcuda.dll"
        if not driver.is_file():
            raise RuntimeError("trusted System32 CUDA driver is unavailable")
        library = ctypes.CDLL(str(driver))  # exact extern-C ABI; Windows runner is X64
    signatures = {
        "cuInit": [ctypes.c_uint],
        "cuDeviceGetCount": [ctypes.POINTER(ctypes.c_int)],
        "cuDeviceGet": [ctypes.POINTER(ctypes.c_int), ctypes.c_int],
        "cuDeviceGetUuid": [ctypes.POINTER(CUuuid), ctypes.c_int],
    }
    for name, arguments in signatures.items():
        function = getattr(library, name)
        function.argtypes = arguments
        function.restype = ctypes.c_uint32
    def checked(name, *args):
        status = int(getattr(library, name)(*args))
        if status != 0:
            raise RuntimeError(f"metadata-only {name} failed with CUDA status {status}")
    checked("cuInit", 0)
    count = ctypes.c_int()
    checked("cuDeviceGetCount", ctypes.byref(count))
    if count.value <= 0:
        raise RuntimeError("no CUDA-visible device mapping; cannot accept a GPU execution")
    result = []
    for ordinal in range(count.value):
        device = ctypes.c_int()
        checked("cuDeviceGet", ctypes.byref(device), ordinal)
        identifier = CUuuid()
        checked("cuDeviceGetUuid", ctypes.byref(identifier), device.value)
        # c_char-array attribute conversion can truncate at NUL. Read all16 ABI bytes.
        raw = ctypes.string_at(ctypes.addressof(identifier), ctypes.sizeof(identifier))
        result.append({"ordinal": ordinal, "uuid": "GPU-" + str(uuid.UUID(bytes=raw))})
    if len({row["uuid"] for row in result}) != len(result):
        raise RuntimeError("CUDA-visible UUID mapping is ambiguous")
    return {"schema_version": 1, "context_created": False,
            "abi_source": {"crate": "cudarc", "version": "0.19.7", "checksum": ABI_CRATE_CHECKSUM},
            "visibility": {"CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                           "CUDA_DEVICE_ORDER": os.environ.get("CUDA_DEVICE_ORDER")},
            "devices": result}


if __name__ == "__main__":
    print(json.dumps(visible_devices()))
