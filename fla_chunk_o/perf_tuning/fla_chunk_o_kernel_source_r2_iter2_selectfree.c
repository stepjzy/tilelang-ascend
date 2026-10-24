#include "tl_templates/ascend/common.h"
#include "acl/acl.h"
#include <runtime/rt_ffts.h>
using namespace Catlass;
using uint = unsigned int;
using uchar = unsigned char;
using ushort = unsigned short;

extern "C" __global__ __aicore__ void main_kernel( GM_ADDR Q_handle,  GM_ADDR V_handle,  GM_ADDR G_handle,  GM_ADDR A_handle,  GM_ADDR Hstate_handle,  GM_ADDR Msk_handle,  GM_ADDR workspace_1_handle,  GM_ADDR workspace_2_handle,  GM_ADDR workspace_3_handle,  GM_ADDR O_handle, uint64_t fftsAddr) {
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
  auto A_l1 = ascend_l1.GetWithOffset<bfloat16_t>(4096, 49152);
  auto o_frag = ascend_l0c.GetWithOffset<float>(8192, 0);
  auto qg_hi_l1 = ascend_l1.GetWithOffset<bfloat16_t>(8192, 57344);
  auto qg_lo_l1 = ascend_l1.GetWithOffset<bfloat16_t>(8192, 73728);
  auto msk_ub = ascend_ub.GetWithOffset<float>(2048, 0);
  auto a_half = ascend_ub.GetWithOffset<bfloat16_t>(2048, 8192);
  auto q_half = ascend_ub.GetWithOffset<bfloat16_t>(4096, 12288);
  auto g_ub = ascend_ub.GetWithOffset<float>(4096, 20480);
  auto a_f32 = ascend_ub.GetWithOffset<float>(2048, 36864);
  auto a_mask = ascend_ub.GetWithOffset<bfloat16_t>(2048, 45056);
  auto q_ub = ascend_ub.GetWithOffset<float>(4096, 49152);
  auto gL_ub = ascend_ub.GetWithOffset<float>(4096, 65536);
  auto gexp_ub = ascend_ub.GetWithOffset<float>(4096, 81920);
  auto qg_ub = ascend_ub.GetWithOffset<float>(4096, 98304);
  auto qg_hi_ub = ascend_ub.GetWithOffset<bfloat16_t>(4096, 114688);
  auto qg_hi_f32 = ascend_ub.GetWithOffset<float>(4096, 122880);
  auto qg_lo_f32 = ascend_ub.GetWithOffset<float>(4096, 139264);
  auto qg_lo_ub = ascend_ub.GetWithOffset<bfloat16_t>(4096, 155648);
  auto vid = AscendC::GetSubBlockIdx();
  if ASCEND_IS_AIC {
    tl::ascend::copy_gm_to_l1<bfloat16_t, 128, 128>(H_l1[0], Hstate[(cid * 1048576)], 128, 128, 128, true);
    tl::ascend::copy_gm_to_l1<bfloat16_t, 64, 128>(V_l1[0], V[(cid * 524288)], 8192, 64, 128, true);
    AscendC::CrossCoreWaitFlag(8);
    tl::ascend::copy_gm_to_l1<bfloat16_t, 64, 64>(A_l1[0], workspace_3[(cid * 16384)], 64, 64, 64, true);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_M>(1);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_M>(1);
    tl::ascend::gemm_v0<bfloat16_t, float, 64, 128, 64, false, false, 128>(A_l1[0], V_l1[0], o_frag[0], ascend_l0a, ascend_l0b, (bool)1, 128);
    AscendC::CrossCoreWaitFlag(0);
    tl::ascend::copy_gm_to_l1<bfloat16_t, 64, 128>(qg_hi_l1[0], workspace_1[(cid * 32768)], 128, 64, 128, true);
    AscendC::CrossCoreWaitFlag(4);
    tl::ascend::copy_gm_to_l1<bfloat16_t, 64, 128>(qg_lo_l1[0], workspace_2[(cid * 32768)], 128, 64, 128, true);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_M>(2);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_M>(2);
    AscendC::PipeBarrier<PIPE_M>();
    tl::ascend::gemm_v0<bfloat16_t, float, 64, 128, 128, false, false, 128>(qg_hi_l1[0], H_l1[0], o_frag[0], ascend_l0a, ascend_l0b, (bool)0, 128);
    AscendC::PipeBarrier<PIPE_M>();
    tl::ascend::gemm_v0<bfloat16_t, float, 64, 128, 128, false, false, 128>(qg_lo_l1[0], H_l1[0], o_frag[0], ascend_l0a, ascend_l0b, (bool)0, 128);
    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(3);
    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(3);
    tl::ascend::copy_l0c_to_gm<float, bfloat16_t, layout::RowMajor, 64, 128, 0>(O[(cid * 524288)], o_frag[0], 8192, 64, 128, 0);
    AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(12);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE2>(4);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE2>(4);
    AscendC::PipeBarrier<PIPE_MTE2>();
    tl::ascend::copy_gm_to_l1<bfloat16_t, 128, 128>(H_l1[0], Hstate[((cid * 1048576) + 20971520)], 128, 128, 128, true);
    tl::ascend::copy_gm_to_l1<bfloat16_t, 64, 128>(V_l1[0], V[((cid * 524288) + 10485760)], 8192, 64, 128, true);
    AscendC::CrossCoreWaitFlag(9);
    tl::ascend::copy_gm_to_l1<bfloat16_t, 64, 64>(A_l1[0], workspace_3[((cid * 16384) + 4096)], 64, 64, 64, true);
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(5);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(5);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_M>(6);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_M>(6);
    AscendC::PipeBarrier<PIPE_M>();
    tl::ascend::gemm_v0<bfloat16_t, float, 64, 128, 64, false, false, 128>(A_l1[0], V_l1[0], o_frag[0], ascend_l0a, ascend_l0b, (bool)1, 128);
    AscendC::CrossCoreWaitFlag(1);
    tl::ascend::copy_gm_to_l1<bfloat16_t, 64, 128>(qg_hi_l1[0], workspace_1[((cid * 32768) + 8192)], 128, 64, 128, true);
    AscendC::CrossCoreWaitFlag(5);
    tl::ascend::copy_gm_to_l1<bfloat16_t, 64, 128>(qg_lo_l1[0], workspace_2[((cid * 32768) + 8192)], 128, 64, 128, true);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_M>(7);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_M>(7);
    AscendC::PipeBarrier<PIPE_M>();
    tl::ascend::gemm_v0<bfloat16_t, float, 64, 128, 128, false, false, 128>(qg_hi_l1[0], H_l1[0], o_frag[0], ascend_l0a, ascend_l0b, (bool)0, 128);
    AscendC::PipeBarrier<PIPE_M>();
    tl::ascend::gemm_v0<bfloat16_t, float, 64, 128, 128, false, false, 128>(qg_lo_l1[0], H_l1[0], o_frag[0], ascend_l0a, ascend_l0b, (bool)0, 128);
    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(0);
    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(0);
    AscendC::PipeBarrier<PIPE_FIX>();
    tl::ascend::copy_l0c_to_gm<float, bfloat16_t, layout::RowMajor, 64, 128, 0>(O[((cid * 524288) + 10485760)], o_frag[0], 8192, 64, 128, 0);
    AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(13);
    for (int32_t tt = 0; tt < 406; ++tt) {
      AscendC::SetFlag<AscendC::HardEvent::M_MTE2>(1);
      AscendC::WaitFlag<AscendC::HardEvent::M_MTE2>(1);
      AscendC::PipeBarrier<PIPE_MTE2>();
      tl::ascend::copy_gm_to_l1<bfloat16_t, 128, 128>(H_l1[0], Hstate[((((((tt * 20) + cid) + 40) % 128) * 1048576) + (((((tt * 5) + (cid / 4)) + 10) / 32) * 16384))], 128, 128, 128, true);
      tl::ascend::copy_gm_to_l1<bfloat16_t, 64, 128>(V_l1[0], V[((((((tt * 20) + cid) + 40) % 128) * 524288) + (((((tt * 5) + (cid / 4)) + 10) / 32) * 128))], 8192, 64, 128, true);
      AscendC::CrossCoreWaitFlag(10);
      tl::ascend::copy_gm_to_l1<bfloat16_t, 64, 64>(A_l1[0], workspace_3[((cid * 16384) + (((tt + 2) % 4) * 4096))], 64, 64, 64, true);
      AscendC::SetFlag<AscendC::HardEvent::FIX_M>(2);
      AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(2);
      AscendC::SetFlag<AscendC::HardEvent::MTE2_M>(3);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_M>(3);
      AscendC::PipeBarrier<PIPE_M>();
      tl::ascend::gemm_v0<bfloat16_t, float, 64, 128, 64, false, false, 128>(A_l1[0], V_l1[0], o_frag[0], ascend_l0a, ascend_l0b, (bool)1, 128);
      AscendC::CrossCoreWaitFlag(2);
      tl::ascend::copy_gm_to_l1<bfloat16_t, 64, 128>(qg_hi_l1[0], workspace_1[((cid * 32768) + (((tt + 2) % 4) * 8192))], 128, 64, 128, true);
      AscendC::CrossCoreWaitFlag(6);
      tl::ascend::copy_gm_to_l1<bfloat16_t, 64, 128>(qg_lo_l1[0], workspace_2[((cid * 32768) + (((tt + 2) % 4) * 8192))], 128, 64, 128, true);
      AscendC::SetFlag<AscendC::HardEvent::MTE2_M>(4);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_M>(4);
      AscendC::PipeBarrier<PIPE_M>();
      tl::ascend::gemm_v0<bfloat16_t, float, 64, 128, 128, false, false, 128>(qg_hi_l1[0], H_l1[0], o_frag[0], ascend_l0a, ascend_l0b, (bool)0, 128);
      AscendC::PipeBarrier<PIPE_M>();
      tl::ascend::gemm_v0<bfloat16_t, float, 64, 128, 128, false, false, 128>(qg_lo_l1[0], H_l1[0], o_frag[0], ascend_l0a, ascend_l0b, (bool)0, 128);
      AscendC::SetFlag<AscendC::HardEvent::M_FIX>(5);
      AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(5);
      AscendC::PipeBarrier<PIPE_FIX>();
      tl::ascend::copy_l0c_to_gm<float, bfloat16_t, layout::RowMajor, 64, 128, 0>(O[((((((tt * 20) + cid) + 40) % 128) * 524288) + (((((tt * 5) + (cid / 4)) + 10) / 32) * 128))], o_frag[0], 8192, 64, 128, 0);
      AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(((tt % 2) + 12));
    }
    for (int32_t tt2 = 0; tt2 < 4; ++tt2) {
      AscendC::SetFlag<AscendC::HardEvent::M_MTE2>(3);
      AscendC::WaitFlag<AscendC::HardEvent::M_MTE2>(3);
      AscendC::PipeBarrier<PIPE_MTE2>();
      int32_t condval;
      if (((tt2 * 5) + (cid / 4)) < 8) {
        condval = (((tt2 * 20) + cid) + 8160);
      } else {
        condval = cid;
      }
      int32_t condval_1;
      if (((tt2 * 5) + (cid / 4)) < 8) {
        condval_1 = (((tt2 * 20) + cid) + 8160);
      } else {
        condval_1 = cid;
      }
      int32_t condval_2;
      if (((tt2 * 5) + (cid / 4)) < 8) {
        condval_2 = (((tt2 * 20) + cid) + 8160);
      } else {
        condval_2 = cid;
      }
      int32_t condval_3;
      if (((tt2 * 5) + (cid / 4)) < 8) {
        condval_3 = (((tt2 * 20) + cid) + 8160);
      } else {
        condval_3 = cid;
      }
      int32_t condval_4;
      if (((tt2 * 5) + (cid / 4)) < 8) {
        condval_4 = (((tt2 * 20) + cid) + 8160);
      } else {
        condval_4 = cid;
      }
      int32_t condval_5;
      if (((tt2 * 5) + (cid / 4)) < 8) {
        condval_5 = (((tt2 * 20) + cid) + 8160);
      } else {
        condval_5 = cid;
      }
      tl::ascend::copy_gm_to_l1<bfloat16_t, 128, 128>(H_l1[0], Hstate[((((condval_3 / 8192) * 134217728) + ((condval_4 % 128) * 1048576)) + (((condval_5 % 8192) / 128) * 16384))], 128, 128, 128, true);
      int32_t condval_6;
      if (((tt2 * 5) + (cid / 4)) < 8) {
        condval_6 = (((tt2 * 20) + cid) + 8160);
      } else {
        condval_6 = cid;
      }
      int32_t condval_7;
      if (((tt2 * 5) + (cid / 4)) < 8) {
        condval_7 = (((tt2 * 20) + cid) + 8160);
      } else {
        condval_7 = cid;
      }
      int32_t condval_8;
      if (((tt2 * 5) + (cid / 4)) < 8) {
        condval_8 = (((tt2 * 20) + cid) + 8160);
      } else {
        condval_8 = cid;
      }
      int32_t condval_9;
      if (((tt2 * 5) + (cid / 4)) < 8) {
        condval_9 = (((tt2 * 20) + cid) + 8160);
      } else {
        condval_9 = cid;
      }
      int32_t condval_10;
      if (((tt2 * 5) + (cid / 4)) < 8) {
        condval_10 = (((tt2 * 20) + cid) + 8160);
      } else {
        condval_10 = cid;
      }
      int32_t condval_11;
      if (((tt2 * 5) + (cid / 4)) < 8) {
        condval_11 = (((tt2 * 20) + cid) + 8160);
      } else {
        condval_11 = cid;
      }
      tl::ascend::copy_gm_to_l1<bfloat16_t, 64, 128>(V_l1[0], V[((((condval_9 / 8192) * 67108864) + ((condval_10 % 128) * 524288)) + (((condval_11 % 8192) / 128) * 128))], 8192, 64, 128, true);
      AscendC::CrossCoreWaitFlag(11);
      tl::ascend::copy_gm_to_l1<bfloat16_t, 64, 64>(A_l1[0], workspace_3[((cid * 16384) + (tt2 * 4096))], 64, 64, 64, true);
      AscendC::SetFlag<AscendC::HardEvent::FIX_M>(4);
      AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(4);
      AscendC::SetFlag<AscendC::HardEvent::MTE2_M>(5);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_M>(5);
      AscendC::PipeBarrier<PIPE_M>();
      tl::ascend::gemm_v0<bfloat16_t, float, 64, 128, 64, false, false, 128>(A_l1[0], V_l1[0], o_frag[0], ascend_l0a, ascend_l0b, (bool)1, 128);
      AscendC::CrossCoreWaitFlag(3);
      tl::ascend::copy_gm_to_l1<bfloat16_t, 64, 128>(qg_hi_l1[0], workspace_1[((cid * 32768) + (tt2 * 8192))], 128, 64, 128, true);
      AscendC::CrossCoreWaitFlag(7);
      tl::ascend::copy_gm_to_l1<bfloat16_t, 64, 128>(qg_lo_l1[0], workspace_2[((cid * 32768) + (tt2 * 8192))], 128, 64, 128, true);
      AscendC::SetFlag<AscendC::HardEvent::MTE2_M>(6);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_M>(6);
      AscendC::PipeBarrier<PIPE_M>();
      tl::ascend::gemm_v0<bfloat16_t, float, 64, 128, 128, false, false, 128>(qg_hi_l1[0], H_l1[0], o_frag[0], ascend_l0a, ascend_l0b, (bool)0, 128);
      AscendC::PipeBarrier<PIPE_M>();
      tl::ascend::gemm_v0<bfloat16_t, float, 64, 128, 128, false, false, 128>(qg_lo_l1[0], H_l1[0], o_frag[0], ascend_l0a, ascend_l0b, (bool)0, 128);
      AscendC::SetFlag<AscendC::HardEvent::M_FIX>(7);
      AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(7);
      AscendC::PipeBarrier<PIPE_FIX>();
      int32_t condval_12;
      if (((tt2 * 5) + (cid / 4)) < 8) {
        condval_12 = (((tt2 * 20) + cid) + 8160);
      } else {
        condval_12 = cid;
      }
      int32_t condval_13;
      if (((tt2 * 5) + (cid / 4)) < 8) {
        condval_13 = (((tt2 * 20) + cid) + 8160);
      } else {
        condval_13 = cid;
      }
      int32_t condval_14;
      if (((tt2 * 5) + (cid / 4)) < 8) {
        condval_14 = (((tt2 * 20) + cid) + 8160);
      } else {
        condval_14 = cid;
      }
      int32_t condval_15;
      if (((tt2 * 5) + (cid / 4)) < 8) {
        condval_15 = (((tt2 * 20) + cid) + 8160);
      } else {
        condval_15 = cid;
      }
      int32_t condval_16;
      if (((tt2 * 5) + (cid / 4)) < 8) {
        condval_16 = (((tt2 * 20) + cid) + 8160);
      } else {
        condval_16 = cid;
      }
      int32_t condval_17;
      if (((tt2 * 5) + (cid / 4)) < 8) {
        condval_17 = (((tt2 * 20) + cid) + 8160);
      } else {
        condval_17 = cid;
      }
      tl::ascend::copy_l0c_to_gm<float, bfloat16_t, layout::RowMajor, 64, 128, 0>(O[((((condval_15 / 8192) * 67108864) + ((condval_16 % 128) * 524288)) + (((condval_17 % 8192) / 128) * 128))], o_frag[0], 8192, 64, 128, 0);
      AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(((tt2 % 2) + 12));
    }
  }
  if ASCEND_IS_AIV {
    tl::ascend::copy_gm_to_ub<float, 64, 32>(msk_ub[0], Msk[(vid * 2048)], 64, 32, 64, 0.000000e+00f);
    tl::ascend::copy_gm_to_ub<bfloat16_t, 64, 32>(a_half[0], A[((cid * 262144) + (vid * 131072))], 4096, 32, 64, bfloat16_t(0.000000e+00f));
    tl::ascend::copy_gm_to_ub<bfloat16_t, 128, 32>(q_half[0], Q[((cid * 524288) + (vid * 262144))], 8192, 32, 128, bfloat16_t(0.000000e+00f));
    tl::ascend::copy_gm_to_ub<float, 128, 32>(g_ub[0], G[((cid * 524288) + (vid * 262144))], 8192, 32, 128, 0.000000e+00f);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(5);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(5);
    tl::ascend::copy_ub_to_ub<float, bfloat16_t, 2048>(a_f32[0], a_half[0], 32, 64, 64, 32, 64, 64);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mul<float, false>(a_f32[0], a_f32[0], msk_ub[0], AscendC::MASK_PLACEHOLDER, 32, AscendC::BinaryRepeatParams());
    AscendC::PipeBarrier<PIPE_V>();
    tl::ascend::copy_ub_to_ub<bfloat16_t, float, 2048>(a_mask[0], a_f32[0], 32, 64, 64, 32, 64, 64);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(6);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(6);
    tl::ascend::copy_ub_to_gm<bfloat16_t, 64, 32>(workspace_3[((cid * 16384) + (vid * 2048))], a_mask[0], 64, 32, 64);
    AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(8);
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
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(7);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(7);
    tl::ascend::copy_ub_to_gm<bfloat16_t, 128, 32>(workspace_1[((cid * 32768) + (vid * 4096))], qg_hi_ub[0], 128, 32, 128);
    AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(0);
    tl::ascend::copy_ub_to_gm<bfloat16_t, 128, 32>(workspace_2[((cid * 32768) + (vid * 4096))], qg_lo_ub[0], 128, 32, 128);
    AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(4);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
    AscendC::PipeBarrier<PIPE_MTE2>();
    tl::ascend::copy_gm_to_ub<bfloat16_t, 64, 32>(a_half[0], A[(((cid * 262144) + (vid * 131072)) + 5242880)], 4096, 32, 64, bfloat16_t(0.000000e+00f));
    tl::ascend::copy_gm_to_ub<bfloat16_t, 128, 32>(q_half[0], Q[(((cid * 524288) + (vid * 262144)) + 10485760)], 8192, 32, 128, bfloat16_t(0.000000e+00f));
    tl::ascend::copy_gm_to_ub<float, 128, 32>(g_ub[0], G[(((cid * 524288) + (vid * 262144)) + 10485760)], 8192, 32, 128, 0.000000e+00f);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(1);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(1);
    tl::ascend::copy_ub_to_ub<float, bfloat16_t, 2048>(a_f32[0], a_half[0], 32, 64, 64, 32, 64, 64);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mul<float, false>(a_f32[0], a_f32[0], msk_ub[0], AscendC::MASK_PLACEHOLDER, 32, AscendC::BinaryRepeatParams());
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(2);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(2);
    AscendC::PipeBarrier<PIPE_V>();
    tl::ascend::copy_ub_to_ub<bfloat16_t, float, 2048>(a_mask[0], a_f32[0], 32, 64, 64, 32, 64, 64);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(3);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(3);
    AscendC::PipeBarrier<PIPE_MTE3>();
    tl::ascend::copy_ub_to_gm<bfloat16_t, 64, 32>(workspace_3[(((cid * 16384) + (vid * 2048)) + 4096)], a_mask[0], 64, 32, 64);
    AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(9);
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
    tl::ascend::copy_ub_to_gm<bfloat16_t, 128, 32>(workspace_1[(((cid * 32768) + (vid * 4096)) + 8192)], qg_hi_ub[0], 128, 32, 128);
    AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(1);
    tl::ascend::copy_ub_to_gm<bfloat16_t, 128, 32>(workspace_2[(((cid * 32768) + (vid * 4096)) + 8192)], qg_lo_ub[0], 128, 32, 128);
    AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(5);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(5);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(5);
    AscendC::PipeBarrier<PIPE_MTE2>();
    tl::ascend::copy_gm_to_ub<bfloat16_t, 64, 32>(a_half[0], A[(((cid * 262144) + (vid * 131072)) + 10485760)], 4096, 32, 64, bfloat16_t(0.000000e+00f));
    tl::ascend::copy_gm_to_ub<bfloat16_t, 128, 32>(q_half[0], Q[(((cid * 524288) + (vid * 262144)) + 20971520)], 8192, 32, 128, bfloat16_t(0.000000e+00f));
    tl::ascend::copy_gm_to_ub<float, 128, 32>(g_ub[0], G[(((cid * 524288) + (vid * 262144)) + 20971520)], 8192, 32, 128, 0.000000e+00f);
    for (int32_t tt_1 = 0; tt_1 < 406; ++tt_1) {
      AscendC::CrossCoreWaitFlag(((tt_1 % 2) + 12));
      AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(6);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(6);
      tl::ascend::copy_ub_to_ub<float, bfloat16_t, 2048>(a_f32[0], a_half[0], 32, 64, 64, 32, 64, 64);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Mul<float, false>(a_f32[0], a_f32[0], msk_ub[0], AscendC::MASK_PLACEHOLDER, 32, AscendC::BinaryRepeatParams());
      AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(7);
      AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(7);
      AscendC::PipeBarrier<PIPE_V>();
      tl::ascend::copy_ub_to_ub<bfloat16_t, float, 2048>(a_mask[0], a_f32[0], 32, 64, 64, 32, 64, 64);
      AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
      AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
      AscendC::PipeBarrier<PIPE_MTE3>();
      tl::ascend::copy_ub_to_gm<bfloat16_t, 64, 32>(workspace_3[(((cid * 16384) + (((tt_1 + 2) % 4) * 4096)) + (vid * 2048))], a_mask[0], 64, 32, 64);
      AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(10);
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
      AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(1);
      AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(1);
      tl::ascend::copy_ub_to_gm<bfloat16_t, 128, 32>(workspace_1[(((cid * 32768) + (((tt_1 + 2) % 4) * 8192)) + (vid * 4096))], qg_hi_ub[0], 128, 32, 128);
      AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(2);
      tl::ascend::copy_ub_to_gm<bfloat16_t, 128, 32>(workspace_2[(((cid * 32768) + (((tt_1 + 2) % 4) * 8192)) + (vid * 4096))], qg_lo_ub[0], 128, 32, 128);
      AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(6);
      AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(2);
      AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(2);
      AscendC::PipeBarrier<PIPE_MTE2>();
      tl::ascend::copy_gm_to_ub<bfloat16_t, 64, 32>(a_half[0], A[(((((((tt_1 * 20) + cid) + 60) % 128) * 262144) + (vid * 131072)) + (((((tt_1 * 5) + (cid / 4)) + 15) / 32) * 64))], 4096, 32, 64, bfloat16_t(0.000000e+00f));
      tl::ascend::copy_gm_to_ub<bfloat16_t, 128, 32>(q_half[0], Q[(((((((tt_1 * 20) + cid) + 60) % 128) * 524288) + (vid * 262144)) + (((((tt_1 * 5) + (cid / 4)) + 15) / 32) * 128))], 8192, 32, 128, bfloat16_t(0.000000e+00f));
      tl::ascend::copy_gm_to_ub<float, 128, 32>(g_ub[0], G[(((((((tt_1 * 20) + cid) + 60) % 128) * 524288) + (vid * 262144)) + (((((tt_1 * 5) + (cid / 4)) + 15) / 32) * 128))], 8192, 32, 128, 0.000000e+00f);
    }
    for (int32_t tt2_1 = 0; tt2_1 < 4; ++tt2_1) {
      AscendC::CrossCoreWaitFlag(((tt2_1 % 2) + 12));
      AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
      tl::ascend::copy_ub_to_ub<float, bfloat16_t, 2048>(a_f32[0], a_half[0], 32, 64, 64, 32, 64, 64);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Mul<float, false>(a_f32[0], a_f32[0], msk_ub[0], AscendC::MASK_PLACEHOLDER, 32, AscendC::BinaryRepeatParams());
      AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(1);
      AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(1);
      AscendC::PipeBarrier<PIPE_V>();
      tl::ascend::copy_ub_to_ub<bfloat16_t, float, 2048>(a_mask[0], a_f32[0], 32, 64, 64, 32, 64, 64);
      AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(2);
      AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(2);
      AscendC::PipeBarrier<PIPE_MTE3>();
      tl::ascend::copy_ub_to_gm<bfloat16_t, 64, 32>(workspace_3[(((cid * 16384) + (tt2_1 * 4096)) + (vid * 2048))], a_mask[0], 64, 32, 64);
      AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(11);
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
      AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(3);
      AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(3);
      tl::ascend::copy_ub_to_gm<bfloat16_t, 128, 32>(workspace_1[(((cid * 32768) + (tt2_1 * 8192)) + (vid * 4096))], qg_hi_ub[0], 128, 32, 128);
      AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(3);
      tl::ascend::copy_ub_to_gm<bfloat16_t, 128, 32>(workspace_2[(((cid * 32768) + (tt2_1 * 8192)) + (vid * 4096))], qg_lo_ub[0], 128, 32, 128);
      AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(7);
      AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(4);
      AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(4);
      AscendC::PipeBarrier<PIPE_MTE2>();
      int32_t condval_18;
      if (((tt2_1 * 5) + (cid / 4)) < 3) {
        condval_18 = (((tt2_1 * 20) + cid) + 8180);
      } else {
        condval_18 = cid;
      }
      int32_t condval_19;
      if (((tt2_1 * 5) + (cid / 4)) < 3) {
        condval_19 = (((tt2_1 * 20) + cid) + 8180);
      } else {
        condval_19 = cid;
      }
      int32_t condval_20;
      if (((tt2_1 * 5) + (cid / 4)) < 3) {
        condval_20 = (((tt2_1 * 20) + cid) + 8180);
      } else {
        condval_20 = cid;
      }
      int32_t condval_21;
      if (((tt2_1 * 5) + (cid / 4)) < 3) {
        condval_21 = (((tt2_1 * 20) + cid) + 8180);
      } else {
        condval_21 = cid;
      }
      int32_t condval_22;
      if (((tt2_1 * 5) + (cid / 4)) < 3) {
        condval_22 = (((tt2_1 * 20) + cid) + 8180);
      } else {
        condval_22 = cid;
      }
      int32_t condval_23;
      if (((tt2_1 * 5) + (cid / 4)) < 3) {
        condval_23 = (((tt2_1 * 20) + cid) + 8180);
      } else {
        condval_23 = cid;
      }
      tl::ascend::copy_gm_to_ub<bfloat16_t, 64, 32>(a_half[0], A[(((((condval_21 / 8192) * 33554432) + ((condval_22 % 128) * 262144)) + (vid * 131072)) + (((condval_23 % 8192) / 128) * 64))], 4096, 32, 64, bfloat16_t(0.000000e+00f));
      int32_t condval_24;
      if (((tt2_1 * 5) + (cid / 4)) < 3) {
        condval_24 = (((tt2_1 * 20) + cid) + 8180);
      } else {
        condval_24 = cid;
      }
      int32_t condval_25;
      if (((tt2_1 * 5) + (cid / 4)) < 3) {
        condval_25 = (((tt2_1 * 20) + cid) + 8180);
      } else {
        condval_25 = cid;
      }
      int32_t condval_26;
      if (((tt2_1 * 5) + (cid / 4)) < 3) {
        condval_26 = (((tt2_1 * 20) + cid) + 8180);
      } else {
        condval_26 = cid;
      }
      int32_t condval_27;
      if (((tt2_1 * 5) + (cid / 4)) < 3) {
        condval_27 = (((tt2_1 * 20) + cid) + 8180);
      } else {
        condval_27 = cid;
      }
      int32_t condval_28;
      if (((tt2_1 * 5) + (cid / 4)) < 3) {
        condval_28 = (((tt2_1 * 20) + cid) + 8180);
      } else {
        condval_28 = cid;
      }
      int32_t condval_29;
      if (((tt2_1 * 5) + (cid / 4)) < 3) {
        condval_29 = (((tt2_1 * 20) + cid) + 8180);
      } else {
        condval_29 = cid;
      }
      tl::ascend::copy_gm_to_ub<bfloat16_t, 128, 32>(q_half[0], Q[(((((condval_27 / 8192) * 67108864) + ((condval_28 % 128) * 524288)) + (vid * 262144)) + (((condval_29 % 8192) / 128) * 128))], 8192, 32, 128, bfloat16_t(0.000000e+00f));
      int32_t condval_30;
      if (((tt2_1 * 5) + (cid / 4)) < 3) {
        condval_30 = (((tt2_1 * 20) + cid) + 8180);
      } else {
        condval_30 = cid;
      }
      int32_t condval_31;
      if (((tt2_1 * 5) + (cid / 4)) < 3) {
        condval_31 = (((tt2_1 * 20) + cid) + 8180);
      } else {
        condval_31 = cid;
      }
      int32_t condval_32;
      if (((tt2_1 * 5) + (cid / 4)) < 3) {
        condval_32 = (((tt2_1 * 20) + cid) + 8180);
      } else {
        condval_32 = cid;
      }
      int32_t condval_33;
      if (((tt2_1 * 5) + (cid / 4)) < 3) {
        condval_33 = (((tt2_1 * 20) + cid) + 8180);
      } else {
        condval_33 = cid;
      }
      int32_t condval_34;
      if (((tt2_1 * 5) + (cid / 4)) < 3) {
        condval_34 = (((tt2_1 * 20) + cid) + 8180);
      } else {
        condval_34 = cid;
      }
      int32_t condval_35;
      if (((tt2_1 * 5) + (cid / 4)) < 3) {
        condval_35 = (((tt2_1 * 20) + cid) + 8180);
      } else {
        condval_35 = cid;
      }
      tl::ascend::copy_gm_to_ub<float, 128, 32>(g_ub[0], G[(((((condval_33 / 8192) * 67108864) + ((condval_34 % 128) * 524288)) + (vid * 262144)) + (((condval_35 % 8192) / 128) * 128))], 8192, 32, 128, 0.000000e+00f);
    }
  }
}

void main_kernel_tiling() {
}

extern "C" void call(uint8_t* Q_handle, uint8_t* V_handle, uint8_t* G_handle, uint8_t* A_handle, uint8_t* Hstate_handle, uint8_t* Msk_handle, uint8_t* workspace_1_handle, uint8_t* workspace_2_handle, uint8_t* workspace_3_handle, uint8_t* O_handle, aclrtStream stream) {
  uint32_t fftsLen{0};
  uint64_t fftsAddr{0};
  rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
  main_kernel_tiling();
  main_kernel<<<20, nullptr, stream>>>(Q_handle, V_handle, G_handle, A_handle, Hstate_handle, Msk_handle, workspace_1_handle, workspace_2_handle, workspace_3_handle, O_handle, fftsAddr);
}
