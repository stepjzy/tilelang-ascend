#include "tl_templates/ascend/common.h"
#include "acl/acl.h"
#include <runtime/rt_ffts.h>
using namespace Catlass;
using uint = unsigned int;
using uchar = unsigned char;
using ushort = unsigned short;

extern "C" __global__ __aicore__ void main_kernel( GM_ADDR In_handle,  GM_ADDR OutFlat_handle, uint64_t fftsAddr) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
  AscendC::TPipe pipe;

  AscendC::GlobalTensor<float> In;
  In.SetGlobalBuffer((__gm__ float*)In_handle);
  AscendC::GlobalTensor<float> OutFlat;
  OutFlat.SetGlobalBuffer((__gm__ float*)OutFlat_handle);

  AscendC::TBuf<AscendC::TPosition::A2> ascend_l0a;
  pipe.InitBuffer(ascend_l0a, 65536);
  AscendC::TBuf<AscendC::TPosition::B2> ascend_l0b;
  pipe.InitBuffer(ascend_l0b, 65536);
  AscendC::TBuf<AscendC::TPosition::A1> ascend_l1; pipe.InitBuffer(ascend_l1, 524032);
  AscendC::TBuf<AscendC::TPosition::CO1> ascend_l0c; pipe.InitBuffer(ascend_l0c, 131072);
  AscendC::TBuf<AscendC::TPosition::VECCALC> ascend_ub; pipe.InitBuffer(ascend_ub, 196352);
  pipe.Destroy();
  auto cid = AscendC::GetBlockIdx();
  if ASCEND_IS_AIV {
    }
  auto in_ub = ascend_ub.GetWithOffset<float>(64, 0);
  auto acc = ascend_ub.GetWithOffset<float>(64, 256);
  auto vid = AscendC::GetSubBlockIdx();
  if ASCEND_IS_AIV {
    tl::ascend::copy_gm_to_ub<float, 64>(in_ub[0], In[(cid * 64)], 64, 1, 64, 0.000000e+00f);
    AscendC::SetMaskNorm();
    AscendC::SetVectorMask<uint8_t>((uint64_t)0, 0xffffffffffffffffULL);
    AscendC::Add<float, false>(acc[0], in_ub[0], in_ub[0], AscendC::MASK_PLACEHOLDER, 1, AscendC::BinaryRepeatParams());
    tl::ascend::copy_ub_to_gm<float, 64>(OutFlat[(cid * 64)], acc[0], 64, 1, 64);
  }
}

void main_kernel_tiling() {
}

extern "C" void call(uint8_t* In_handle, uint8_t* OutFlat_handle, aclrtStream stream) {
  uint32_t fftsLen{0};
  uint64_t fftsAddr{0};
  rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
  main_kernel_tiling();
  main_kernel<<<8, nullptr, stream>>>(In_handle, OutFlat_handle, fftsAddr);
}
