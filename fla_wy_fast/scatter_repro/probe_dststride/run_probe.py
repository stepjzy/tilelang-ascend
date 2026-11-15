import os
import ctypes
import subprocess
import sys

import torch
import torch_npu

CPP = "probe_dststride.cpp"
SO = CPP.replace(".cpp", ".so")
ASCEND = "/home/developer/Ascend/ascend-toolkit/latest"
UL = "/mnt/workspace/tilelang-ascend"

cmd = [
    "bisheng",
    "--npu-arch=dav-2201",
    "-std=c++17",
    "-xasc",
    f"-I{ASCEND}/include",
    f"-I{ASCEND}/include/experiment/msprof",
    f"-I{ASCEND}/include/experiment/runtime",
    f"-I{ASCEND}/pkg_inc",
    f"-I{ASCEND}/pkg_inc/runtime",
    f"-I{ASCEND}/pkg_inc/profiling",
    f"-I{UL}/3rdparty/catlass/include",
    f"-I{UL}/3rdparty/shmem/include",
    f"-I{UL}/3rdparty/shmem/src/device",
    "-DBACKEND_HYBM",
    f"-I{UL}/src",
    f"-L{ASCEND}/lib64",
    "-Wno-macro-redefined",
    "-Wno-ignored-attributes",
    "-Wno-non-c-typedef-for-linkage",
    "-lruntime",
    "-lascendcl",
    "-lm",
    "-ltiling_api",
    "-lplatform",
    "-lc_sec",
    "-ldl",
    "-fPIC",
    "--shared",
    "-o",
    SO,
    CPP,
]
r = subprocess.run(cmd, capture_output=True, text=True)
if r.returncode != 0:
    print("COMPILE FAILED")
    print(r.stderr[-4000:])
    sys.exit(1)
print("COMPILE OK")

lib = ctypes.CDLL(os.path.abspath(SO))
lib.call.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
lib.call.restype = None

x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], device="npu", dtype=torch.float32)
out = torch.full((160,), -777.0, device="npu", dtype=torch.float32)
stream = torch_npu.npu.current_stream().npu_stream
lib.call(ctypes.c_void_p(x.data_ptr()), ctypes.c_void_p(out.data_ptr()), ctypes.c_void_p(stream))
torch.npu.synchronize()

o = out.cpu().numpy()


def seg(a, b):
    return [f"{v:>7.1f}" for v in o[a:b]]


print("== case1: Ext(2,4,0,0,0) -> buf[0] (window Out[0..15]) ==")
print("  ", seg(0, 16))
print("== case2: Ext(2,4,0,1,0) -> buf[32] (window Out[32..55]) ==")
print("  ", seg(32, 56))
print("== case3: Ext(1,32,0,0,0) -> buf[64] (window Out[64..71]) ==")
print("  ", seg(64, 72))
print("== arm4: UB->GM Ext(2,4,1,0,0) <- buf[0]  (window Out[96..103]) ==")
print("  ", seg(96, 104))
print("== arm5: UB->GM Ext(2,4,0,0,0) <- buf[0]  (window Out[112..119]) ==")
print("  ", seg(112, 120))
print("== arm6: UB->GM Ext(1,32,0,0,0) <- buf[0]  (window Out[128..135]) ==")
print("  ", seg(128, 136))

# 判定助手
print()
print("判定提示：")
print("  case1 Out[1]==2.0 -> 相邻burst紧凑4B； Out[8]==2.0 -> 相邻burst 32B槽")
print("  case2 Out[40]==2.0 -> dstStride单位=32B块； Out[33]附近出现2.0 -> dstStride单位=字节")
print("  case3 应精确为 1..8 （验证读回路径本身）")
print("  arm4 Out[96]=1.0； Out[97]==2.0 -> srcStride字节； Out[100]==2.0 -> srcStride=32B块")
print("     （arm4 burst1 源步进 = ceil(4/32)+srcStride 块：srcStride=1块 -> 跳2块=64B = buf[16] 槽???）")
print("  arm5 Out[112]=1.0； Out[113]==2.0 -> 相邻burst源紧凑4B； Out[116]==2.0 -> 相邻burst源步进32B槽(槽1首)")
print("  arm6 Out[128..135] 应为 1.0x8 （槽0整槽读回）")