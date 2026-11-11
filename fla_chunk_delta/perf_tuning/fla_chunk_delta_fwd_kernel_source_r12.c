#include "tl_templates/ascend/common.h"
#include "acl/acl.h"
#include <runtime/rt_ffts.h>
using namespace Catlass;
using uint = unsigned int;
using uchar = unsigned char;
using ushort = unsigned short;

extern "C" __global__ __aicore__ void main_kernel( GM_ADDR K3_handle,  GM_ADDR W3_handle,  GM_ADDR U3_handle,  GM_ADDR GT_handle,  GM_ADDR GD_handle,  GM_ADDR H0_handle,  GM_ADDR Hout_handle,  GM_ADDR Vnew_handle,  GM_ADDR Ht_handle,  GM_ADDR workspace_1_handle,  GM_ADDR workspace_2_handle,  GM_ADDR workspace_3_handle,  GM_ADDR workspace_4_handle, uint64_t fftsAddr) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
  AscendC::TPipe pipe;

  AscendC::GlobalTensor<half> K3;
  K3.SetGlobalBuffer((__gm__ half*)K3_handle);
  AscendC::GlobalTensor<half> W3;
  W3.SetGlobalBuffer((__gm__ half*)W3_handle);
  AscendC::GlobalTensor<half> U3;
  U3.SetGlobalBuffer((__gm__ half*)U3_handle);
  AscendC::GlobalTensor<float> GT;
  GT.SetGlobalBuffer((__gm__ float*)GT_handle);
  AscendC::GlobalTensor<float> GD;
  GD.SetGlobalBuffer((__gm__ float*)GD_handle);
  AscendC::GlobalTensor<float> H0;
  H0.SetGlobalBuffer((__gm__ float*)H0_handle);
  AscendC::GlobalTensor<half> Hout;
  Hout.SetGlobalBuffer((__gm__ half*)Hout_handle);
  AscendC::GlobalTensor<half> Vnew;
  Vnew.SetGlobalBuffer((__gm__ half*)Vnew_handle);
  AscendC::GlobalTensor<float> Ht;
  Ht.SetGlobalBuffer((__gm__ float*)Ht_handle);
  AscendC::GlobalTensor<half> workspace_1;
  workspace_1.SetGlobalBuffer((__gm__ half*)workspace_1_handle);
  AscendC::GlobalTensor<float> workspace_2;
  workspace_2.SetGlobalBuffer((__gm__ float*)workspace_2_handle);
  AscendC::GlobalTensor<half> workspace_3;
  workspace_3.SetGlobalBuffer((__gm__ half*)workspace_3_handle);
  AscendC::GlobalTensor<float> workspace_4;
  workspace_4.SetGlobalBuffer((__gm__ float*)workspace_4_handle);

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
    cid = cid / 2;
  }
  auto w_l1 = ascend_l1.GetWithOffset<half>(8192, 0);
  auto h16_l1 = ascend_l1.GetWithOffset<half>(16384, 16384);
  auto bv_frag = ascend_l0c.GetWithOffset<float>(8192, 0);
  auto k_l1 = ascend_l1.GetWithOffset<half>(8192, 49152);
  auto v16_l1 = ascend_l1.GetWithOffset<half>(8192, 65536);
  auto kh_frag = ascend_l0c.GetWithOffset<float>(16384, 32768);
  auto h_ub = ascend_ub.GetWithOffset<float>(8192, 0);
  auto u16_ub = ascend_ub.GetWithOffset<half>(4096, 32768);
  auto gate_ub = ascend_ub.GetWithOffset<float>(32, 40960);
  auto dsrc_ub = ascend_ub.GetWithOffset<float>(64, 41088);
  auto h16_ub = ascend_ub.GetWithOffset<half>(8192, 41344);
  auto u_ub = ascend_ub.GetWithOffset<float>(4096, 57728);
  auto bv_ub = ascend_ub.GetWithOffset<float>(4096, 74112);
  auto vn_ub = ascend_ub.GetWithOffset<float>(4096, 90496);
  auto vn_g_ub = ascend_ub.GetWithOffset<float>(4096, 107904);
  auto tmp_gate_a = ascend_ub.GetWithOffset<float>(256, 106880);
  auto tmp_gate_b = ascend_ub.GetWithOffset<float>(256, 124288);
  auto vn16_ub = ascend_ub.GetWithOffset<half>(4096, 125312);
  auto vn16v_ub = ascend_ub.GetWithOffset<half>(4096, 133504);
  auto kh_ub = ascend_ub.GetWithOffset<float>(8192, 141696);
  auto tmp_decay_a = ascend_ub.GetWithOffset<float>(512, 174464);
  auto tmp_decay_b = ascend_ub.GetWithOffset<float>(512, 176512);
  auto vid = AscendC::GetSubBlockIdx();
  if ASCEND_IS_AIC {
    for (int32_t k_task = 0; k_task < 4; ++k_task) {
      AscendC::PipeBarrier<PIPE_ALL>();
      if (((k_task * 5) + (cid / 4)) < 16) {
        for (int32_t i_t = 0; i_t < 128; ++i_t) {
          AscendC::SetFlag<AscendC::HardEvent::M_MTE2>(5);
          AscendC::WaitFlag<AscendC::HardEvent::M_MTE2>(5);
          AscendC::PipeBarrier<PIPE_MTE2>();
          tl::ascend::copy_gm_to_l1<half, 64, 128>(w_l1[0], W3[(((i_t * 524288) + (k_task * 2560)) + (cid * 128))], 8192, 64, 128, true);
          AscendC::CrossCoreWaitFlag(0);
          tl::ascend::copy_gm_to_l1<half, 128, 128>(h16_l1[0], workspace_1[((cid * 262144) + ((i_t % 16) * 16384))], 128, 128, 128, true);
          AscendC::SetFlag<AscendC::HardEvent::FIX_M>(6);
          AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(6);
          AscendC::SetFlag<AscendC::HardEvent::MTE2_M>(1);
          AscendC::WaitFlag<AscendC::HardEvent::MTE2_M>(1);
          AscendC::PipeBarrier<PIPE_M>();
          tl::ascend::gemm_v0<half, float, 64, 128, 128, false, false, 128>(w_l1[0], h16_l1[0], bv_frag[0], ascend_l0a, ascend_l0b, (bool)1, 128);
          AscendC::SetFlag<AscendC::HardEvent::M_FIX>(2);
          AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(2);
          AscendC::PipeBarrier<PIPE_FIX>();
          tl::ascend::copy_l0c_to_gm<float, float, layout::RowMajor, 64, 128, 0>(workspace_2[((cid * 131072) + ((i_t % 16) * 8192))], bv_frag[0], 128, 64, 128, 0);
          AscendC::CrossCoreSetFlag<2, PIPE_FIX>(1);
          tl::ascend::copy_gm_to_l1<half, 64, 128>(k_l1[0], K3[(((i_t * 524288) + (k_task * 2560)) + (cid * 128))], 8192, 64, 128, true);
          AscendC::CrossCoreWaitFlag(2);
          tl::ascend::copy_gm_to_l1<half, 64, 128>(v16_l1[0], workspace_3[((cid * 131072) + ((i_t % 16) * 8192))], 128, 64, 128, true);
          AscendC::SetFlag<AscendC::HardEvent::MTE2_M>(3);
          AscendC::WaitFlag<AscendC::HardEvent::MTE2_M>(3);
          tl::ascend::gemm_v0<half, float, 128, 128, 64, true, false, 128>(k_l1[0], v16_l1[0], kh_frag[0], ascend_l0a, ascend_l0b, (bool)1, 128);
          AscendC::SetFlag<AscendC::HardEvent::M_FIX>(4);
          AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(4);
          tl::ascend::copy_l0c_to_gm<float, float, layout::RowMajor, 128, 128, 0>(workspace_4[((cid * 262144) + ((i_t % 16) * 16384))], kh_frag[0], 128, 128, 128, 0);
          AscendC::CrossCoreSetFlag<2, PIPE_FIX>(3);
        }
      }
      AscendC::PipeBarrier<PIPE_ALL>();
    }
  }
  if ASCEND_IS_AIV {
    for (int32_t k_task_1 = 0; k_task_1 < 4; ++k_task_1) {
      AscendC::PipeBarrier<PIPE_ALL>();
      if (((k_task_1 * 5) + (cid / 4)) < 16) {
        AscendC::SetMaskCount();
        AscendC::SetVectorMask<uint8_t>((uint64_t)0, 8192);
        AscendC::Duplicate<float, false>(h_ub[0], 0.000000e+00f, AscendC::MASK_PLACEHOLDER, 1, 1, 8);
        for (int32_t i_t_1 = 0; i_t_1 < 128; ++i_t_1) {
          AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(3);
          AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(3);
          AscendC::PipeBarrier<PIPE_MTE2>();
          tl::ascend::copy_gm_to_ub<half, 128, 32>(u16_ub[0], U3[((((i_t_1 * 524288) + (vid * 262144)) + (k_task_1 * 2560)) + (cid * 128))], 8192, 32, 128, half(0.000000e+00f));
          tl::ascend::copy_gm_to_ub<float, 32>(gate_ub[0], GT[((((i_t_1 * 4096) + (k_task_1 * 1280)) + (cid * 64)) + (vid * 32))], 64, 1, 32, 0.000000e+00f);
          tl::ascend::copy_gm_to_ub<float, 64>(dsrc_ub[0], GD[((((i_t_1 * 8192) + (k_task_1 * 2560)) + (cid * 128)) + (vid * 64))], 128, 1, 64, 0.000000e+00f);
          AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(4);
          AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(4);
          AscendC::PipeBarrier<PIPE_V>();
          tl::ascend::copy_ub_to_ub<half, float, 8192>(h16_ub[0], h_ub[0], 64, 128, 128, 64, 128, 128);
          AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(5);
          AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(5);
          AscendC::PipeBarrier<PIPE_MTE3>();
          tl::ascend::copy_ub_to_gm<half, 128, 64>(workspace_1[(((cid * 262144) + ((i_t_1 % 16) * 16384)) + (vid * 8192))], h16_ub[0], 128, 64, 128);
          AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(0);
          tl::ascend::copy_ub_to_gm<half, 128, 64>(Hout[((((i_t_1 * 1048576) + (k_task_1 * 327680)) + (cid * 16384)) + (vid * 8192))], h16_ub[0], 128, 64, 128);
          AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(6);
          AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(6);
          tl::ascend::copy_ub_to_ub<float, half, 4096>(u_ub[0], u16_ub[0], 32, 128, 128, 32, 128, 128);
          AscendC::CrossCoreWaitFlag(1);
          tl::ascend::copy_gm_to_ub<float, 128, 32>(bv_ub[0], workspace_2[(((cid * 131072) + ((i_t_1 % 16) * 8192)) + (vid * 4096))], 128, 32, 128, 0.000000e+00f);
          AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(7);
          AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(7);
          AscendC::PipeBarrier<PIPE_V>();
          AscendC::Sub<float, false>(vn_ub[0], u_ub[0], bv_ub[0], AscendC::MASK_PLACEHOLDER, 64, AscendC::BinaryRepeatParams());
          AscendC::PipeBarrier<PIPE_V>();
          tl::ascend::brcb<float>(tmp_gate_a[0], gate_ub[0], 4, 1, 8);
          AscendC::PipeBarrier<PIPE_V>();
          tl::ascend::mul_mask<float>(vn_g_ub[0], vn_ub[0], tmp_gate_a[0], 0xFFFFFFFFFFFFFFFF, 0, 32, 1, 1, 0, 16, 16, 1);
          AscendC::PipeBarrier<PIPE_V>();
          tl::ascend::brcb<float>(tmp_gate_b[0], gate_ub[0], 8, 1, 8);
          AscendC::PipeBarrier<PIPE_V>();
          tl::ascend::mul_mask<float>(vn_g_ub[64], vn_ub[64], tmp_gate_b[0], 0xFFFFFFFFFFFFFFFF, 0, 64, 1, 1, 0, 16, 16, 1);
          AscendC::PipeBarrier<PIPE_V>();
          tl::ascend::copy_ub_to_ub<half, float, 4096>(vn16_ub[0], vn_g_ub[0], 32, 128, 128, 32, 128, 128);
          AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
          AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
          tl::ascend::copy_ub_to_gm<half, 128, 32>(workspace_3[(((cid * 131072) + ((i_t_1 % 16) * 8192)) + (vid * 4096))], vn16_ub[0], 128, 32, 128);
          AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(2);
          tl::ascend::copy_ub_to_ub<half, float, 4096>(vn16v_ub[0], vn_ub[0], 32, 128, 128, 32, 128, 128);
          AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(1);
          AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(1);
          tl::ascend::copy_ub_to_gm<half, 128, 32>(Vnew[((((i_t_1 * 524288) + (vid * 262144)) + (k_task_1 * 2560)) + (cid * 128))], vn16v_ub[0], 8192, 32, 128);
          AscendC::CrossCoreWaitFlag(3);
          tl::ascend::copy_gm_to_ub<float, 128, 64>(kh_ub[0], workspace_4[(((cid * 262144) + ((i_t_1 % 16) * 16384)) + (vid * 8192))], 128, 64, 128, 0.000000e+00f);
          tl::ascend::brcb<float>(tmp_decay_a[0], dsrc_ub[0], 8, 1, 8);
          AscendC::PipeBarrier<PIPE_V>();
          tl::ascend::mul_mask<float>(h_ub[0], h_ub[0], tmp_decay_a[0], 0xFFFFFFFFFFFFFFFF, 0, 64, 1, 1, 0, 16, 16, 1);
          AscendC::PipeBarrier<PIPE_V>();
          tl::ascend::brcb<float>(tmp_decay_b[0], dsrc_ub[0], 16, 1, 8);
          AscendC::PipeBarrier<PIPE_V>();
          tl::ascend::mul_mask<float>(h_ub[64], h_ub[64], tmp_decay_b[0], 0xFFFFFFFFFFFFFFFF, 0, 128, 1, 1, 0, 16, 16, 1);
          AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(2);
          AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(2);
          AscendC::PipeBarrier<PIPE_V>();
          AscendC::Add<float, false>(h_ub[0], h_ub[0], kh_ub[0], AscendC::MASK_PLACEHOLDER, 128, AscendC::BinaryRepeatParams());
        }
      }
      AscendC::PipeBarrier<PIPE_ALL>();
    }
  }
}

void main_kernel_tiling() {
}

extern "C" void call(uint8_t* K3_handle, uint8_t* W3_handle, uint8_t* U3_handle, uint8_t* GT_handle, uint8_t* GD_handle, uint8_t* H0_handle, uint8_t* Hout_handle, uint8_t* Vnew_handle, uint8_t* Ht_handle, uint8_t* workspace_1_handle, uint8_t* workspace_2_handle, uint8_t* workspace_3_handle, uint8_t* workspace_4_handle, aclrtStream stream) {
  uint32_t fftsLen{0};
  uint64_t fftsAddr{0};
  rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
  main_kernel_tiling();
  main_kernel<<<20, nullptr, stream>>>(K3_handle, W3_handle, U3_handle, GT_handle, GD_handle, H0_handle, Hout_handle, Vnew_handle, Ht_handle, workspace_1_handle, workspace_2_handle, workspace_3_handle, workspace_4_handle, fftsAddr);
}
