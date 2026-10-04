"""Print the NVRTC version actually loaded by PyTorch in this process."""

import ctypes

import torch

lib = ctypes.CDLL("libnvrtc.so.12")
major = ctypes.c_int()
minor = ctypes.c_int()
result = lib.nvrtcVersion(ctypes.byref(major), ctypes.byref(minor))
print("torch", torch.__version__, "CUDA", torch.version.cuda)
print("NVRTC", major.value, minor.value, "status", result)
for line in open("/proc/self/maps", encoding="utf-8"):
    if "libnvrtc.so" in line and "/" in line:
        print(line.strip())
