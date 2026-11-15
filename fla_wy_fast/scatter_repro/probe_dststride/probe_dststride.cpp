// Probe: 直接测 CANN DataCopyPad GM->UB 的 UB 侧地址计算规则。
// 三个 case 写入 UB 不同区域，然后单 burst 连续读回 GM，观察元素间距。
#include "tl_templates/ascend/common.h"
#include "acl/acl.h"
#include <runtime/rt_ffts.h>
using namespace Catlass;
using uint = unsigned int;
using uchar = unsigned char;
using ushort = unsigned short;

extern "C" __global__ __aicore__ void main_kernel(GM_ADDR In_handle, GM_ADDR Out_handle, uint64_t fftsAddr) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
  AscendC::TPipe pipe;

  AscendC::GlobalTensor<float> In;
  In.SetGlobalBuffer((__gm__ float*)In_handle);
  AscendC::GlobalTensor<float> Out;
  Out.SetGlobalBuffer((__gm__ float*)Out_handle);

  AscendC::TBuf<AscendC::TPosition::VECCALC> ascend_ub;
  pipe.InitBuffer(ascend_ub, 1024);
  pipe.Destroy();
  // UTUB total = 1024B = 256 fp32
  auto buf = ascend_ub.GetWithOffset<float>(256, 0);
  auto vid = AscendC::GetSubBlockIdx();
  if ASCEND_IS_AIV {
    // 清零整个 1KB，保证未写区域读回为 0
    AscendC::Duplicate<float>(buf, 0.0f, 256);

    // ---- case1: Ext(blockCount=2, blockLen=4B, srcStride=0, dstStride=0) ----
    // 写 buf[0..15] 区域（64B）
    AscendC::DataCopyExtParams p1(2, 4, 0, 0, 0);
    AscendC::DataCopyPadExtParams<float> pd1(false, 0, 0, 0.0f);
    AscendC::DataCopyPad(buf[0], In, p1, pd1);  // MTE2 GM->UB

    // ---- case2: Ext(2, 4, 0, dstStride=1, 0) ----
    // 写 buf[32..55] 区域（96B 窗口）
    AscendC::DataCopyExtParams p2(2, 4, 0, 1, 0);
    AscendC::DataCopyPad(buf[32], In, p2, pd1);  // MTE2 GM->UB

    // ---- case3: Ext(1, 32B, 0, 0, 0) 对照 ----
    // 写 buf[64..71]（32B，整块搬 8 个 fp32）
    AscendC::DataCopyExtParams p3(1, 32, 0, 0, 0);
    AscendC::DataCopyPad(buf[64], In, p3, pd1);  // MTE2 GM->UB

    // MTE2(写UB) -> MTE3(读UB) 同步
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(0);

    // ---- 读回：单 burst 384B 连续读 buf[0..95] -> Out[0..95] ----
    AscendC::DataCopyExtParams q(1, 384, 0, 0, 0);
    AscendC::DataCopyPad(Out, buf[0], q);  // MTE3 UB->GM

    // ---- arm4: UB->GM Ext(2, 4, srcStride=1, dstStride=0, 0) ----
    // 从 buf[0]（槽0=1.0 槽1=2.0）读 2 bursts x 4B，srcStride=1
    // 写 Out[96..]：burst0 读槽0首4B；burst1 源步进 ? -> 读哪里由硬件定
    AscendC::DataCopyExtParams s1(2, 4, 1, 0, 0);
    AscendC::DataCopyPad(Out[96], buf[0], s1);  // MTE3 UB->GM

    // ---- arm5: UB->GM Ext(2, 4, srcStride=0, dstStride=0, 0) ----
    // srcStride=0 时相邻 burst 的源自然步进：槽首读验证
    AscendC::DataCopyExtParams s2(2, 4, 0, 0, 0);
    AscendC::DataCopyPad(Out[112], buf[0], s2);  // MTE3 UB->GM

    // ---- arm6: UB->GM Ext(1, 32, 0, 0, 0) 整槽读对照 ----
    AscendC::DataCopyExtParams s3(1, 32, 0, 0, 0);
    AscendC::DataCopyPad(Out[128], buf[0], s3);  // MTE3 UB->GM
  }
}

void main_kernel_tiling() {
}

extern "C" void call(uint8_t* In_handle, uint8_t* Out_handle, aclrtStream stream) {
  uint32_t fftsLen{0};
  uint64_t fftsAddr{0};
  rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
  main_kernel_tiling();
  main_kernel<<<1, nullptr, stream>>>(In_handle, Out_handle, fftsAddr);
}