#include "tl_templates/ascend/common.h"
#include "acl/acl.h"
#include <runtime/rt_ffts.h>
using namespace Catlass;
using uint = unsigned int;
using uchar = unsigned char;
using ushort = unsigned short;

extern "C" __global__ __aicore__ void main_kernel( GM_ADDR Q_handle,  GM_ADDR V_handle,  GM_ADDR G_handle,  GM_ADDR A_handle,  GM_ADDR Hstate_handle,  GM_ADDR Msk_handle,  GM_ADDR workspace_1_handle,  GM_ADDR workspace_2_handle,  GM_ADDR workspace_3_handle,  GM_ADDR workspace_4_handle,  GM_ADDR O_handle, uint64_t fftsAddr) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
  AscendC::TPipe pipe;

  AscendC::GlobalTensor<bfloat16_t> Q;
  Q.SetGlobalBuffer((__gm__ bfloat16_t*)Q_handle);
  AscendC::GlobalTensor<bfloat16_t> V;
  V.SetGlobalBuffer((__gm__ bfloat16_t*)V_handle);
  AscendC::GlobalTensor<float> G;
  G.SetGlobalBuffer((__gm__ float*)G_handle);
  AscendC::GlobalTensor<bfloat16_t> A;
  A.SetGlobalBuffer((__gm__ bfloat16_t*)A_handle);
  AscendC::GlobalTensor<bfloat16_t> Hstate;
  Hstate.SetGlobalBuffer((__gm__ bfloat16_t*)Hstate_handle);
  AscendC::GlobalTensor<float> Msk;
  Msk.SetGlobalBuffer((__gm__ float*)Msk_handle);
  AscendC::GlobalTensor<bfloat16_t> workspace_1;
  workspace_1.SetGlobalBuffer((__gm__ bfloat16_t*)workspace_1_handle);
  AscendC::GlobalTensor<bfloat16_t> workspace_2;
  workspace_2.SetGlobalBuffer((__gm__ bfloat16_t*)workspace_2_handle);
  AscendC::GlobalTensor<bfloat16_t> workspace_3;
  workspace_3.SetGlobalBuffer((__gm__ bfloat16_t*)workspace_3_handle);
  AscendC::GlobalTensor<float> workspace_4;
  workspace_4.SetGlobalBuffer((__gm__ float*)workspace_4_handle);
  AscendC::GlobalTensor<bfloat16_t> O;
  O.SetGlobalBuffer((__gm__ bfloat16_t*)O_handle);

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
  auto H_l1 = ascend_l1.GetWithOffset<bfloat16_t>(16384, 0);
  auto V_l1 = ascend_l1.GetWithOffset<bfloat16_t>(8192, 32768);
  auto qg_hi_l1 = ascend_l1.GetWithOffset<bfloat16_t>(8192, 49152);
  auto o_frag = ascend_l0c.GetWithOffset<float>(8192, 0);
  auto qg_lo_l1 = ascend_l1.GetWithOffset<bfloat16_t>(8192, 65536);
  auto A_l1 = ascend_l1.GetWithOffset<bfloat16_t>(4096, 81920);
  auto msk_ub = ascend_ub.GetWithOffset<float>(2048, 0);
  auto o_ub = ascend_ub.GetWithOffset<float>(4096, 8192);
  auto o_half = ascend_ub.GetWithOffset<bfloat16_t>(4096, 24576);
  auto a_half = ascend_ub.GetWithOffset<bfloat16_t>(2048, 32768);
  auto q_half = ascend_ub.GetWithOffset<bfloat16_t>(4096, 36864);
  auto g_ub = ascend_ub.GetWithOffset<float>(4096, 45056);
  auto a_f32 = ascend_ub.GetWithOffset<float>(2048, 61440);
  auto a_mask = ascend_ub.GetWithOffset<bfloat16_t>(2048, 69632);
  auto q_ub = ascend_ub.GetWithOffset<float>(4096, 73728);
  auto gL_ub = ascend_ub.GetWithOffset<float>(4096, 90112);
  auto gexp_ub = ascend_ub.GetWithOffset<float>(4096, 106496);
  auto qg_ub = ascend_ub.GetWithOffset<float>(4096, 122880);
  auto qg_hi_ub = ascend_ub.GetWithOffset<bfloat16_t>(4096, 139264);
  auto qg_hi_f32 = ascend_ub.GetWithOffset<float>(4096, 147456);
  auto qg_lo_f32 = ascend_ub.GetWithOffset<float>(4096, 163840);
  auto qg_lo_ub = ascend_ub.GetWithOffset<bfloat16_t>(4096, 180224);
  auto vid = AscendC::GetSubBlockIdx();
  if ASCEND_IS_AIC {
    for (int32_t t = 0; t < 411; ++t) {
      AscendC::PipeBarrier<PIPE_ALL>();
      if (t < 410) {
        AscendC::PipeBarrier<PIPE_ALL>();
        if (((cid * 205) + (t / 2)) < 4096) {
          tl::ascend::copy_gm_to_l1<bfloat16_t, 128, 128>(H_l1[0], Hstate[(((((cid * 410) + t) % 128) * 1048576) + ((((cid * 205) + (t / 2)) / 64) * 16384))], 128, 128, 128, true);
          tl::ascend::copy_gm_to_l1<bfloat16_t, 64, 128>(V_l1[0], V[(((((cid * 410) + t) % 128) * 524288) + ((((cid * 205) + (t / 2)) / 64) * 128))], 8192, 64, 128, true);
          AscendC::CrossCoreWaitFlag(0);
          tl::ascend::copy_gm_to_l1<bfloat16_t, 64, 128>(qg_hi_l1[0], workspace_1[((cid * 16384) + ((t % 2) * 8192))], 128, 64, 128, true);
          AscendC::SetFlag<AscendC::HardEvent::MTE2_M>(1);
          AscendC::WaitFlag<AscendC::HardEvent::MTE2_M>(1);
          tl::ascend::gemm_v0<bfloat16_t, float, 64, 128, 128, false, false, 128>(qg_hi_l1[0], H_l1[0], o_frag[0], ascend_l0a, ascend_l0b, (bool)1, 128);
          AscendC::CrossCoreWaitFlag(1);
          tl::ascend::copy_gm_to_l1<bfloat16_t, 64, 128>(qg_lo_l1[0], workspace_2[((cid * 16384) + ((t % 2) * 8192))], 128, 64, 128, true);
          AscendC::SetFlag<AscendC::HardEvent::MTE2_M>(2);
          AscendC::WaitFlag<AscendC::HardEvent::MTE2_M>(2);
          AscendC::PipeBarrier<PIPE_M>();
          tl::ascend::gemm_v0<bfloat16_t, float, 64, 128, 128, false, false, 128>(qg_lo_l1[0], H_l1[0], o_frag[0], ascend_l0a, ascend_l0b, (bool)0, 128);
          AscendC::CrossCoreWaitFlag(2);
          tl::ascend::copy_gm_to_l1<bfloat16_t, 64, 64>(A_l1[0], workspace_3[((cid * 8192) + ((t % 2) * 4096))], 64, 64, 64, true);
          AscendC::SetFlag<AscendC::HardEvent::MTE2_M>(3);
          AscendC::WaitFlag<AscendC::HardEvent::MTE2_M>(3);
          AscendC::PipeBarrier<PIPE_M>();
          tl::ascend::gemm_v0<bfloat16_t, float, 64, 128, 64, false, false, 128>(A_l1[0], V_l1[0], o_frag[0], ascend_l0a, ascend_l0b, (bool)0, 128);
          AscendC::SetFlag<AscendC::HardEvent::M_FIX>(4);
          AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(4);
          tl::ascend::copy_l0c_to_gm<float, float, layout::RowMajor, 64, 128, 0>(workspace_4[((cid * 16384) + ((t % 2) * 8192))], o_frag[0], 128, 64, 128, 0);
          AscendC::CrossCoreSetFlag<2, PIPE_FIX>(3);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
      }
      AscendC::PipeBarrier<PIPE_ALL>();
    }
  }
  if ASCEND_IS_AIV {
    tl::ascend::copy_gm_to_ub<float, 64, 32>(msk_ub[0], Msk[(vid * 2048)], 64, 32, 64, 0.000000e+00f);
    for (int32_t t_1 = 0; t_1 < 411; ++t_1) {
      AscendC::PipeBarrier<PIPE_ALL>();
      if (1 <= t_1) {
        AscendC::PipeBarrier<PIPE_ALL>();
        if (((cid * 410) + t_1) < 8193) {
          AscendC::CrossCoreWaitFlag(3);
          tl::ascend::copy_gm_to_ub<float, 128, 32>(o_ub[0], workspace_4[(((cid * 16384) + (((t_1 + 1) % 2) * 8192)) + (vid * 4096))], 128, 32, 128, 0.000000e+00f);
          AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(1);
          AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(1);
          tl::ascend::copy_ub_to_ub<bfloat16_t, float, 4096>(o_half[0], o_ub[0], 32, 128, 128, 32, 128, 128);
          AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(2);
          AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(2);
          tl::ascend::copy_ub_to_gm<bfloat16_t, 128, 32>(O[(((((((cid * 410) + t_1) + 127) % 128) * 524288) + (vid * 262144)) + (((((cid * 410) + t_1) - 1) / 128) * 128))], o_half[0], 8192, 32, 128);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
      }
      AscendC::PipeBarrier<PIPE_ALL>();
      AscendC::PipeBarrier<PIPE_ALL>();
      if (t_1 < 410) {
        AscendC::PipeBarrier<PIPE_ALL>();
        if (((cid * 205) + (t_1 / 2)) < 4096) {
          tl::ascend::copy_gm_to_ub<bfloat16_t, 64, 32>(a_half[0], A[((((((cid * 410) + t_1) % 128) * 262144) + (vid * 131072)) + ((((cid * 205) + (t_1 / 2)) / 64) * 64))], 4096, 32, 64, bfloat16_t(0.000000e+00f));
          tl::ascend::copy_gm_to_ub<bfloat16_t, 128, 32>(q_half[0], Q[((((((cid * 410) + t_1) % 128) * 524288) + (vid * 262144)) + ((((cid * 205) + (t_1 / 2)) / 64) * 128))], 8192, 32, 128, bfloat16_t(0.000000e+00f));
          tl::ascend::copy_gm_to_ub<float, 128, 32>(g_ub[0], G[((((((cid * 410) + t_1) % 128) * 524288) + (vid * 262144)) + ((((cid * 205) + (t_1 / 2)) / 64) * 128))], 8192, 32, 128, 0.000000e+00f);
          AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(3);
          AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(3);
          tl::ascend::copy_ub_to_ub<float, bfloat16_t, 2048>(a_f32[0], a_half[0], 32, 64, 64, 32, 64, 64);
          AscendC::PipeBarrier<PIPE_V>();
          AscendC::Mul<float, false>(a_f32[0], a_f32[0], msk_ub[0], AscendC::MASK_PLACEHOLDER, 32, AscendC::BinaryRepeatParams());
          AscendC::PipeBarrier<PIPE_V>();
          tl::ascend::copy_ub_to_ub<bfloat16_t, float, 2048>(a_mask[0], a_f32[0], 32, 64, 64, 32, 64, 64);
          tl::ascend::copy_ub_to_ub<float, bfloat16_t, 4096>(q_ub[0], q_half[0], 32, 128, 128, 32, 128, 128);
          {
          AscendC::Muls<float, false>(gL_ub[0], g_ub[0], 6.931472e-01f, AscendC::MASK_PLACEHOLDER, 64, AscendC::UnaryRepeatParams());
          }
          AscendC::PipeBarrier<PIPE_V>();
          AscendC::Exp<float, false>(gexp_ub[0], gL_ub[0], AscendC::MASK_PLACEHOLDER, 64, AscendC::UnaryRepeatParams());
          {
          AscendC::Muls<float, false>(q_ub[0], q_ub[0], 8.838835e-02f, AscendC::MASK_PLACEHOLDER, 64, AscendC::UnaryRepeatParams());
          }
          AscendC::PipeBarrier<PIPE_V>();
          AscendC::Mul<float, false>(qg_ub[0], q_ub[0], gexp_ub[0], AscendC::MASK_PLACEHOLDER, 64, AscendC::BinaryRepeatParams());
          AscendC::PipeBarrier<PIPE_V>();
          tl::ascend::copy_ub_to_ub<bfloat16_t, float, 4096>(qg_hi_ub[0], qg_ub[0], 32, 128, 128, 32, 128, 128);
          AscendC::PipeBarrier<PIPE_V>();
          tl::ascend::copy_ub_to_ub<float, bfloat16_t, 4096>(qg_hi_f32[0], qg_hi_ub[0], 32, 128, 128, 32, 128, 128);
          AscendC::PipeBarrier<PIPE_V>();
          AscendC::Sub<float, false>(qg_lo_f32[0], qg_ub[0], qg_hi_f32[0], AscendC::MASK_PLACEHOLDER, 64, AscendC::BinaryRepeatParams());
          AscendC::PipeBarrier<PIPE_V>();
          tl::ascend::copy_ub_to_ub<bfloat16_t, float, 4096>(qg_lo_ub[0], qg_lo_f32[0], 32, 128, 128, 32, 128, 128);
          AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(4);
          AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(4);
          tl::ascend::copy_ub_to_gm<bfloat16_t, 128, 32>(workspace_1[(((cid * 16384) + ((t_1 % 2) * 8192)) + (vid * 4096))], qg_hi_ub[0], 128, 32, 128);
          AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(0);
          tl::ascend::copy_ub_to_gm<bfloat16_t, 128, 32>(workspace_2[(((cid * 16384) + ((t_1 % 2) * 8192)) + (vid * 4096))], qg_lo_ub[0], 128, 32, 128);
          AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(1);
          tl::ascend::copy_ub_to_gm<bfloat16_t, 64, 32>(workspace_3[(((cid * 8192) + ((t_1 % 2) * 4096)) + (vid * 2048))], a_mask[0], 64, 32, 64);
          AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(2);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
      }
      AscendC::PipeBarrier<PIPE_ALL>();
    }
  }
}

void main_kernel_tiling() {
}

extern "C" void call(uint8_t* Q_handle, uint8_t* V_handle, uint8_t* G_handle, uint8_t* A_handle, uint8_t* Hstate_handle, uint8_t* Msk_handle, uint8_t* workspace_1_handle, uint8_t* workspace_2_handle, uint8_t* workspace_3_handle, uint8_t* workspace_4_handle, uint8_t* O_handle, aclrtStream stream) {
  uint32_t fftsLen{0};
  uint64_t fftsAddr{0};
  rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
  main_kernel_tiling();
  main_kernel<<<20, nullptr, stream>>>(Q_handle, V_handle, G_handle, A_handle, Hstate_handle, Msk_handle, workspace_1_handle, workspace_2_handle, workspace_3_handle, workspace_4_handle, O_handle, fftsAddr);
}
