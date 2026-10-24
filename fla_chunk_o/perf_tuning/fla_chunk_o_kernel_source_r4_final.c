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
  AscendC::GlobalTensor<bfloat16_t> workspace_4;
  workspace_4.SetGlobalBuffer((__gm__ bfloat16_t*)workspace_4_handle);
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
  auto H_l1_a = ascend_l1.GetWithOffset<bfloat16_t>(16384, 0);
  auto H_l1_b = ascend_l1.GetWithOffset<bfloat16_t>(16384, 32768);
  auto H_l1_c = ascend_l1.GetWithOffset<bfloat16_t>(16384, 65536);
  auto V_l1_a = ascend_l1.GetWithOffset<bfloat16_t>(8192, 98304);
  auto V_l1_b = ascend_l1.GetWithOffset<bfloat16_t>(8192, 114688);
  auto V_l1_c = ascend_l1.GetWithOffset<bfloat16_t>(8192, 131072);
  auto A_l1_a = ascend_l1.GetWithOffset<bfloat16_t>(4096, 147456);
  auto A_l1_b = ascend_l1.GetWithOffset<bfloat16_t>(4096, 155648);
  auto A_l1_c = ascend_l1.GetWithOffset<bfloat16_t>(4096, 163840);
  auto qg_hi_l1_a = ascend_l1.GetWithOffset<bfloat16_t>(8192, 172032);
  auto qg_hi_l1_b = ascend_l1.GetWithOffset<bfloat16_t>(8192, 188416);
  auto qg_hi_l1_c = ascend_l1.GetWithOffset<bfloat16_t>(8192, 204800);
  auto qg_lo_l1_a = ascend_l1.GetWithOffset<bfloat16_t>(8192, 221184);
  auto qg_lo_l1_b = ascend_l1.GetWithOffset<bfloat16_t>(8192, 237568);
  auto qg_lo_l1_c = ascend_l1.GetWithOffset<bfloat16_t>(8192, 253952);
  auto o_frag_a = ascend_l0c.GetWithOffset<float>(8192, 0);
  auto o_frag_b = ascend_l0c.GetWithOffset<float>(8192, 32768);
  auto msk_ub = ascend_ub.GetWithOffset<float>(2048, 0);
  auto a_half = ascend_ub.GetWithOffset<bfloat16_t>(2048, 8192);
  auto q_half = ascend_ub.GetWithOffset<bfloat16_t>(4096, 12288);
  auto g_ub = ascend_ub.GetWithOffset<float>(4096, 20480);
  auto v_half = ascend_ub.GetWithOffset<bfloat16_t>(4096, 36864);
  auto a_half2 = ascend_ub.GetWithOffset<bfloat16_t>(2048, 45056);
  auto q_half2 = ascend_ub.GetWithOffset<bfloat16_t>(4096, 49152);
  auto g_ub2 = ascend_ub.GetWithOffset<float>(4096, 57344);
  auto v_half2 = ascend_ub.GetWithOffset<bfloat16_t>(4096, 73728);
  auto a_half3 = ascend_ub.GetWithOffset<bfloat16_t>(2048, 81920);
  auto q_half3 = ascend_ub.GetWithOffset<bfloat16_t>(4096, 86016);
  auto g_ub3 = ascend_ub.GetWithOffset<float>(4096, 94208);
  auto v_half3 = ascend_ub.GetWithOffset<bfloat16_t>(4096, 110592);
  auto a_f32 = ascend_ub.GetWithOffset<float>(2048, 118784);
  auto a_mask = ascend_ub.GetWithOffset<bfloat16_t>(2048, 126976);
  auto q_ub = ascend_ub.GetWithOffset<float>(4096, 131072);
  auto qg_hi_ub = ascend_ub.GetWithOffset<bfloat16_t>(4096, 147456);
  auto qg_hi_f32 = ascend_ub.GetWithOffset<float>(4096, 155648);
  auto qg_lo_ub = ascend_ub.GetWithOffset<bfloat16_t>(4096, 172032);
  auto vid = AscendC::GetSubBlockIdx();
  if ASCEND_IS_AIC {
    tl::ascend::copy_gm_to_l1<bfloat16_t, 128, 128>(H_l1_a[0], Hstate[(cid * 1048576)], 128, 128, 128, true);
    tl::ascend::copy_gm_to_l1<bfloat16_t, 128, 128>(H_l1_b[0], Hstate[((cid * 1048576) + 20971520)], 128, 128, 128, true);
    tl::ascend::copy_gm_to_l1<bfloat16_t, 128, 128>(H_l1_c[0], Hstate[((cid * 1048576) + 41943040)], 128, 128, 128, true);
    AscendC::CrossCoreWaitFlag(36);
    tl::ascend::copy_gm_to_l1<bfloat16_t, 64, 128>(V_l1_a[0], workspace_4[(cid * 73728)], 128, 64, 128, true);
    AscendC::CrossCoreWaitFlag(37);
    tl::ascend::copy_gm_to_l1<bfloat16_t, 64, 128>(V_l1_b[0], workspace_4[((cid * 73728) + 8192)], 128, 64, 128, true);
    AscendC::CrossCoreWaitFlag(38);
    tl::ascend::copy_gm_to_l1<bfloat16_t, 64, 128>(V_l1_c[0], workspace_4[((cid * 73728) + 16384)], 128, 64, 128, true);
    AscendC::CrossCoreWaitFlag(24);
    tl::ascend::copy_gm_to_l1<bfloat16_t, 64, 64>(A_l1_a[0], workspace_3[(cid * 36864)], 64, 64, 64, true);
    AscendC::CrossCoreWaitFlag(25);
    tl::ascend::copy_gm_to_l1<bfloat16_t, 64, 64>(A_l1_b[0], workspace_3[((cid * 36864) + 4096)], 64, 64, 64, true);
    AscendC::CrossCoreWaitFlag(26);
    tl::ascend::copy_gm_to_l1<bfloat16_t, 64, 64>(A_l1_c[0], workspace_3[((cid * 36864) + 8192)], 64, 64, 64, true);
    AscendC::CrossCoreWaitFlag(0);
    tl::ascend::copy_gm_to_l1<bfloat16_t, 64, 128>(qg_hi_l1_a[0], workspace_1[(cid * 73728)], 128, 64, 128, true);
    AscendC::CrossCoreWaitFlag(1);
    tl::ascend::copy_gm_to_l1<bfloat16_t, 64, 128>(qg_hi_l1_b[0], workspace_1[((cid * 73728) + 8192)], 128, 64, 128, true);
    AscendC::CrossCoreWaitFlag(2);
    tl::ascend::copy_gm_to_l1<bfloat16_t, 64, 128>(qg_hi_l1_c[0], workspace_1[((cid * 73728) + 16384)], 128, 64, 128, true);
    AscendC::CrossCoreWaitFlag(12);
    tl::ascend::copy_gm_to_l1<bfloat16_t, 64, 128>(qg_lo_l1_a[0], workspace_2[(cid * 73728)], 128, 64, 128, true);
    AscendC::CrossCoreWaitFlag(13);
    tl::ascend::copy_gm_to_l1<bfloat16_t, 64, 128>(qg_lo_l1_b[0], workspace_2[((cid * 73728) + 8192)], 128, 64, 128, true);
    AscendC::CrossCoreWaitFlag(14);
    tl::ascend::copy_gm_to_l1<bfloat16_t, 64, 128>(qg_lo_l1_c[0], workspace_2[((cid * 73728) + 16384)], 128, 64, 128, true);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_M>(1);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_M>(1);
    tl::ascend::gemm_v0<bfloat16_t, float, 64, 128, 64, false, false, 128>(A_l1_a[0], V_l1_a[0], o_frag_a[0], ascend_l0a, ascend_l0b, (bool)1, 128);
    AscendC::PipeBarrier<PIPE_M>();
    tl::ascend::gemm_v0<bfloat16_t, float, 64, 128, 128, false, false, 128>(qg_hi_l1_a[0], H_l1_a[0], o_frag_a[0], ascend_l0a, ascend_l0b, (bool)0, 128);
    AscendC::PipeBarrier<PIPE_M>();
    tl::ascend::gemm_v0<bfloat16_t, float, 64, 128, 128, false, false, 128>(qg_lo_l1_a[0], H_l1_a[0], o_frag_a[0], ascend_l0a, ascend_l0b, (bool)0, 128);
    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(2);
    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(2);
    tl::ascend::copy_l0c_to_gm<float, bfloat16_t, layout::RowMajor, 64, 128, 0>(O[(cid * 524288)], o_frag_a[0], 8192, 64, 128, 0);
    AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(48);
    tl::ascend::gemm_v0<bfloat16_t, float, 64, 128, 64, false, false, 128>(A_l1_b[0], V_l1_b[0], o_frag_b[0], ascend_l0a, ascend_l0b, (bool)1, 128);
    AscendC::PipeBarrier<PIPE_M>();
    tl::ascend::gemm_v0<bfloat16_t, float, 64, 128, 128, false, false, 128>(qg_hi_l1_b[0], H_l1_b[0], o_frag_b[0], ascend_l0a, ascend_l0b, (bool)0, 128);
    AscendC::PipeBarrier<PIPE_M>();
    tl::ascend::gemm_v0<bfloat16_t, float, 64, 128, 128, false, false, 128>(qg_lo_l1_b[0], H_l1_b[0], o_frag_b[0], ascend_l0a, ascend_l0b, (bool)0, 128);
    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(3);
    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(3);
    AscendC::PipeBarrier<PIPE_FIX>();
    tl::ascend::copy_l0c_to_gm<float, bfloat16_t, layout::RowMajor, 64, 128, 0>(O[((cid * 524288) + 10485760)], o_frag_b[0], 8192, 64, 128, 0);
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(4);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(4);
    tl::ascend::gemm_v0<bfloat16_t, float, 64, 128, 64, false, false, 128>(A_l1_c[0], V_l1_c[0], o_frag_a[0], ascend_l0a, ascend_l0b, (bool)1, 128);
    AscendC::PipeBarrier<PIPE_M>();
    tl::ascend::gemm_v0<bfloat16_t, float, 64, 128, 128, false, false, 128>(qg_hi_l1_c[0], H_l1_c[0], o_frag_a[0], ascend_l0a, ascend_l0b, (bool)0, 128);
    AscendC::PipeBarrier<PIPE_M>();
    tl::ascend::gemm_v0<bfloat16_t, float, 64, 128, 128, false, false, 128>(qg_lo_l1_c[0], H_l1_c[0], o_frag_a[0], ascend_l0a, ascend_l0b, (bool)0, 128);
    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(5);
    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(5);
    AscendC::PipeBarrier<PIPE_FIX>();
    tl::ascend::copy_l0c_to_gm<float, bfloat16_t, layout::RowMajor, 64, 128, 0>(O[((cid * 524288) + 20971520)], o_frag_a[0], 8192, 64, 128, 0);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE2>(6);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE2>(6);
    AscendC::PipeBarrier<PIPE_MTE2>();
    tl::ascend::copy_gm_to_l1<bfloat16_t, 128, 128>(H_l1_a[0], Hstate[((cid * 1048576) + 62914560)], 128, 128, 128, true);
    tl::ascend::copy_gm_to_l1<bfloat16_t, 128, 128>(H_l1_b[0], Hstate[((cid * 1048576) + 83886080)], 128, 128, 128, true);
    tl::ascend::copy_gm_to_l1<bfloat16_t, 128, 128>(H_l1_c[0], Hstate[((cid * 1048576) + 104857600)], 128, 128, 128, true);
    AscendC::CrossCoreWaitFlag(39);
    tl::ascend::copy_gm_to_l1<bfloat16_t, 64, 128>(V_l1_a[0], workspace_4[((cid * 73728) + 24576)], 128, 64, 128, true);
    AscendC::CrossCoreWaitFlag(40);
    tl::ascend::copy_gm_to_l1<bfloat16_t, 64, 128>(V_l1_b[0], workspace_4[((cid * 73728) + 32768)], 128, 64, 128, true);
    AscendC::CrossCoreWaitFlag(41);
    tl::ascend::copy_gm_to_l1<bfloat16_t, 64, 128>(V_l1_c[0], workspace_4[((cid * 73728) + 40960)], 128, 64, 128, true);
    AscendC::CrossCoreWaitFlag(27);
    tl::ascend::copy_gm_to_l1<bfloat16_t, 64, 64>(A_l1_a[0], workspace_3[((cid * 36864) + 12288)], 64, 64, 64, true);
    AscendC::CrossCoreWaitFlag(28);
    tl::ascend::copy_gm_to_l1<bfloat16_t, 64, 64>(A_l1_b[0], workspace_3[((cid * 36864) + 16384)], 64, 64, 64, true);
    AscendC::CrossCoreWaitFlag(29);
    tl::ascend::copy_gm_to_l1<bfloat16_t, 64, 64>(A_l1_c[0], workspace_3[((cid * 36864) + 20480)], 64, 64, 64, true);
    AscendC::CrossCoreWaitFlag(3);
    tl::ascend::copy_gm_to_l1<bfloat16_t, 64, 128>(qg_hi_l1_a[0], workspace_1[((cid * 73728) + 24576)], 128, 64, 128, true);
    AscendC::CrossCoreWaitFlag(4);
    tl::ascend::copy_gm_to_l1<bfloat16_t, 64, 128>(qg_hi_l1_b[0], workspace_1[((cid * 73728) + 32768)], 128, 64, 128, true);
    AscendC::CrossCoreWaitFlag(5);
    tl::ascend::copy_gm_to_l1<bfloat16_t, 64, 128>(qg_hi_l1_c[0], workspace_1[((cid * 73728) + 40960)], 128, 64, 128, true);
    AscendC::CrossCoreWaitFlag(15);
    tl::ascend::copy_gm_to_l1<bfloat16_t, 64, 128>(qg_lo_l1_a[0], workspace_2[((cid * 73728) + 24576)], 128, 64, 128, true);
    AscendC::CrossCoreWaitFlag(16);
    tl::ascend::copy_gm_to_l1<bfloat16_t, 64, 128>(qg_lo_l1_b[0], workspace_2[((cid * 73728) + 32768)], 128, 64, 128, true);
    AscendC::CrossCoreWaitFlag(17);
    tl::ascend::copy_gm_to_l1<bfloat16_t, 64, 128>(qg_lo_l1_c[0], workspace_2[((cid * 73728) + 40960)], 128, 64, 128, true);
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(7);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(7);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_M>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_M>(0);
    AscendC::PipeBarrier<PIPE_M>();
    tl::ascend::gemm_v0<bfloat16_t, float, 64, 128, 64, false, false, 128>(A_l1_a[0], V_l1_a[0], o_frag_a[0], ascend_l0a, ascend_l0b, (bool)1, 128);
    AscendC::PipeBarrier<PIPE_M>();
    tl::ascend::gemm_v0<bfloat16_t, float, 64, 128, 128, false, false, 128>(qg_hi_l1_a[0], H_l1_a[0], o_frag_a[0], ascend_l0a, ascend_l0b, (bool)0, 128);
    AscendC::PipeBarrier<PIPE_M>();
    tl::ascend::gemm_v0<bfloat16_t, float, 64, 128, 128, false, false, 128>(qg_lo_l1_a[0], H_l1_a[0], o_frag_a[0], ascend_l0a, ascend_l0b, (bool)0, 128);
    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(1);
    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(1);
    AscendC::PipeBarrier<PIPE_FIX>();
    tl::ascend::copy_l0c_to_gm<float, bfloat16_t, layout::RowMajor, 64, 128, 0>(O[((cid * 524288) + 31457280)], o_frag_a[0], 8192, 64, 128, 0);
    AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(48);
    tl::ascend::gemm_v0<bfloat16_t, float, 64, 128, 64, false, false, 128>(A_l1_b[0], V_l1_b[0], o_frag_b[0], ascend_l0a, ascend_l0b, (bool)1, 128);
    AscendC::PipeBarrier<PIPE_M>();
    tl::ascend::gemm_v0<bfloat16_t, float, 64, 128, 128, false, false, 128>(qg_hi_l1_b[0], H_l1_b[0], o_frag_b[0], ascend_l0a, ascend_l0b, (bool)0, 128);
    AscendC::PipeBarrier<PIPE_M>();
    tl::ascend::gemm_v0<bfloat16_t, float, 64, 128, 128, false, false, 128>(qg_lo_l1_b[0], H_l1_b[0], o_frag_b[0], ascend_l0a, ascend_l0b, (bool)0, 128);
    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(2);
    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(2);
    AscendC::PipeBarrier<PIPE_FIX>();
    tl::ascend::copy_l0c_to_gm<float, bfloat16_t, layout::RowMajor, 64, 128, 0>(O[((cid * 524288) + 41943040)], o_frag_b[0], 8192, 64, 128, 0);
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(3);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(3);
    tl::ascend::gemm_v0<bfloat16_t, float, 64, 128, 64, false, false, 128>(A_l1_c[0], V_l1_c[0], o_frag_a[0], ascend_l0a, ascend_l0b, (bool)1, 128);
    AscendC::PipeBarrier<PIPE_M>();
    tl::ascend::gemm_v0<bfloat16_t, float, 64, 128, 128, false, false, 128>(qg_hi_l1_c[0], H_l1_c[0], o_frag_a[0], ascend_l0a, ascend_l0b, (bool)0, 128);
    AscendC::PipeBarrier<PIPE_M>();
    tl::ascend::gemm_v0<bfloat16_t, float, 64, 128, 128, false, false, 128>(qg_lo_l1_c[0], H_l1_c[0], o_frag_a[0], ascend_l0a, ascend_l0b, (bool)0, 128);
    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(4);
    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(4);
    AscendC::PipeBarrier<PIPE_FIX>();
    tl::ascend::copy_l0c_to_gm<float, bfloat16_t, layout::RowMajor, 64, 128, 0>(O[((cid * 524288) + 52428800)], o_frag_a[0], 8192, 64, 128, 0);
    for (int32_t kk = 0; kk < 133; ++kk) {
      AscendC::SetFlag<AscendC::HardEvent::M_MTE2>(5);
      AscendC::WaitFlag<AscendC::HardEvent::M_MTE2>(5);
      AscendC::PipeBarrier<PIPE_MTE2>();
      tl::ascend::copy_gm_to_l1<bfloat16_t, 128, 128>(H_l1_a[0], Hstate[((((((kk * 60) + cid) + 120) % 128) * 1048576) + (((((kk * 15) + (cid / 4)) + 30) / 32) * 16384))], 128, 128, 128, true);
      tl::ascend::copy_gm_to_l1<bfloat16_t, 128, 128>(H_l1_b[0], Hstate[((((((kk * 60) + cid) + 12) % 128) * 1048576) + (((((kk * 15) + (cid / 4)) + 35) / 32) * 16384))], 128, 128, 128, true);
      tl::ascend::copy_gm_to_l1<bfloat16_t, 128, 128>(H_l1_c[0], Hstate[((((((kk * 60) + cid) + 32) % 128) * 1048576) + (((((kk * 15) + (cid / 4)) + 40) / 32) * 16384))], 128, 128, 128, true);
      AscendC::CrossCoreWaitFlag(42);
      tl::ascend::copy_gm_to_l1<bfloat16_t, 64, 128>(V_l1_a[0], workspace_4[((cid * 73728) + ((((kk * 3) + 6) % 9) * 8192))], 128, 64, 128, true);
      AscendC::CrossCoreWaitFlag(43);
      tl::ascend::copy_gm_to_l1<bfloat16_t, 64, 128>(V_l1_b[0], workspace_4[((cid * 73728) + ((((kk * 3) + 7) % 9) * 8192))], 128, 64, 128, true);
      AscendC::CrossCoreWaitFlag(44);
      tl::ascend::copy_gm_to_l1<bfloat16_t, 64, 128>(V_l1_c[0], workspace_4[((cid * 73728) + ((((kk * 3) + 8) % 9) * 8192))], 128, 64, 128, true);
      AscendC::CrossCoreWaitFlag(30);
      tl::ascend::copy_gm_to_l1<bfloat16_t, 64, 64>(A_l1_a[0], workspace_3[((cid * 36864) + ((((kk * 3) + 6) % 9) * 4096))], 64, 64, 64, true);
      AscendC::CrossCoreWaitFlag(31);
      tl::ascend::copy_gm_to_l1<bfloat16_t, 64, 64>(A_l1_b[0], workspace_3[((cid * 36864) + ((((kk * 3) + 7) % 9) * 4096))], 64, 64, 64, true);
      AscendC::CrossCoreWaitFlag(32);
      tl::ascend::copy_gm_to_l1<bfloat16_t, 64, 64>(A_l1_c[0], workspace_3[((cid * 36864) + ((((kk * 3) + 8) % 9) * 4096))], 64, 64, 64, true);
      AscendC::CrossCoreWaitFlag(6);
      tl::ascend::copy_gm_to_l1<bfloat16_t, 64, 128>(qg_hi_l1_a[0], workspace_1[((cid * 73728) + ((((kk * 3) + 6) % 9) * 8192))], 128, 64, 128, true);
      AscendC::CrossCoreWaitFlag(7);
      tl::ascend::copy_gm_to_l1<bfloat16_t, 64, 128>(qg_hi_l1_b[0], workspace_1[((cid * 73728) + ((((kk * 3) + 7) % 9) * 8192))], 128, 64, 128, true);
      AscendC::CrossCoreWaitFlag(8);
      tl::ascend::copy_gm_to_l1<bfloat16_t, 64, 128>(qg_hi_l1_c[0], workspace_1[((cid * 73728) + ((((kk * 3) + 8) % 9) * 8192))], 128, 64, 128, true);
      AscendC::CrossCoreWaitFlag(18);
      tl::ascend::copy_gm_to_l1<bfloat16_t, 64, 128>(qg_lo_l1_a[0], workspace_2[((cid * 73728) + ((((kk * 3) + 6) % 9) * 8192))], 128, 64, 128, true);
      AscendC::CrossCoreWaitFlag(19);
      tl::ascend::copy_gm_to_l1<bfloat16_t, 64, 128>(qg_lo_l1_b[0], workspace_2[((cid * 73728) + ((((kk * 3) + 7) % 9) * 8192))], 128, 64, 128, true);
      AscendC::CrossCoreWaitFlag(20);
      tl::ascend::copy_gm_to_l1<bfloat16_t, 64, 128>(qg_lo_l1_c[0], workspace_2[((cid * 73728) + ((((kk * 3) + 8) % 9) * 8192))], 128, 64, 128, true);
      AscendC::SetFlag<AscendC::HardEvent::FIX_M>(6);
      AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(6);
      AscendC::SetFlag<AscendC::HardEvent::MTE2_M>(7);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_M>(7);
      AscendC::PipeBarrier<PIPE_M>();
      tl::ascend::gemm_v0<bfloat16_t, float, 64, 128, 64, false, false, 128>(A_l1_a[0], V_l1_a[0], o_frag_a[0], ascend_l0a, ascend_l0b, (bool)1, 128);
      AscendC::PipeBarrier<PIPE_M>();
      tl::ascend::gemm_v0<bfloat16_t, float, 64, 128, 128, false, false, 128>(qg_hi_l1_a[0], H_l1_a[0], o_frag_a[0], ascend_l0a, ascend_l0b, (bool)0, 128);
      AscendC::PipeBarrier<PIPE_M>();
      tl::ascend::gemm_v0<bfloat16_t, float, 64, 128, 128, false, false, 128>(qg_lo_l1_a[0], H_l1_a[0], o_frag_a[0], ascend_l0a, ascend_l0b, (bool)0, 128);
      AscendC::SetFlag<AscendC::HardEvent::M_FIX>(0);
      AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(0);
      AscendC::PipeBarrier<PIPE_FIX>();
      tl::ascend::copy_l0c_to_gm<float, bfloat16_t, layout::RowMajor, 64, 128, 0>(O[((((((kk * 60) + cid) + 120) % 128) * 524288) + (((((kk * 15) + (cid / 4)) + 30) / 32) * 128))], o_frag_a[0], 8192, 64, 128, 0);
      AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(48);
      tl::ascend::gemm_v0<bfloat16_t, float, 64, 128, 64, false, false, 128>(A_l1_b[0], V_l1_b[0], o_frag_b[0], ascend_l0a, ascend_l0b, (bool)1, 128);
      AscendC::PipeBarrier<PIPE_M>();
      tl::ascend::gemm_v0<bfloat16_t, float, 64, 128, 128, false, false, 128>(qg_hi_l1_b[0], H_l1_b[0], o_frag_b[0], ascend_l0a, ascend_l0b, (bool)0, 128);
      AscendC::PipeBarrier<PIPE_M>();
      tl::ascend::gemm_v0<bfloat16_t, float, 64, 128, 128, false, false, 128>(qg_lo_l1_b[0], H_l1_b[0], o_frag_b[0], ascend_l0a, ascend_l0b, (bool)0, 128);
      AscendC::SetFlag<AscendC::HardEvent::M_FIX>(1);
      AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(1);
      AscendC::PipeBarrier<PIPE_FIX>();
      tl::ascend::copy_l0c_to_gm<float, bfloat16_t, layout::RowMajor, 64, 128, 0>(O[((((((kk * 60) + cid) + 12) % 128) * 524288) + (((((kk * 15) + (cid / 4)) + 35) / 32) * 128))], o_frag_b[0], 8192, 64, 128, 0);
      AscendC::SetFlag<AscendC::HardEvent::FIX_M>(2);
      AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(2);
      tl::ascend::gemm_v0<bfloat16_t, float, 64, 128, 64, false, false, 128>(A_l1_c[0], V_l1_c[0], o_frag_a[0], ascend_l0a, ascend_l0b, (bool)1, 128);
      AscendC::PipeBarrier<PIPE_M>();
      tl::ascend::gemm_v0<bfloat16_t, float, 64, 128, 128, false, false, 128>(qg_hi_l1_c[0], H_l1_c[0], o_frag_a[0], ascend_l0a, ascend_l0b, (bool)0, 128);
      AscendC::PipeBarrier<PIPE_M>();
      tl::ascend::gemm_v0<bfloat16_t, float, 64, 128, 128, false, false, 128>(qg_lo_l1_c[0], H_l1_c[0], o_frag_a[0], ascend_l0a, ascend_l0b, (bool)0, 128);
      AscendC::SetFlag<AscendC::HardEvent::M_FIX>(3);
      AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(3);
      AscendC::PipeBarrier<PIPE_FIX>();
      tl::ascend::copy_l0c_to_gm<float, bfloat16_t, layout::RowMajor, 64, 128, 0>(O[((((((kk * 60) + cid) + 32) % 128) * 524288) + (((((kk * 15) + (cid / 4)) + 40) / 32) * 128))], o_frag_a[0], 8192, 64, 128, 0);
    }
    for (int32_t kk2 = 0; kk2 < 3; ++kk2) {
      AscendC::SetFlag<AscendC::HardEvent::M_MTE2>(3);
      AscendC::WaitFlag<AscendC::HardEvent::M_MTE2>(3);
      AscendC::PipeBarrier<PIPE_MTE2>();
      int32_t condval;
      if (((kk2 * 15) + (cid / 4)) < 23) {
        condval = (((kk2 * 60) + cid) + 8100);
      } else {
        condval = cid;
      }
      int32_t condval_1;
      if (((kk2 * 15) + (cid / 4)) < 23) {
        condval_1 = (((kk2 * 60) + cid) + 8100);
      } else {
        condval_1 = cid;
      }
      int32_t condval_2;
      if (((kk2 * 15) + (cid / 4)) < 23) {
        condval_2 = (((kk2 * 60) + cid) + 8100);
      } else {
        condval_2 = cid;
      }
      int32_t condval_3;
      if (((kk2 * 15) + (cid / 4)) < 23) {
        condval_3 = (((kk2 * 60) + cid) + 8100);
      } else {
        condval_3 = cid;
      }
      int32_t condval_4;
      if (((kk2 * 15) + (cid / 4)) < 23) {
        condval_4 = (((kk2 * 60) + cid) + 8100);
      } else {
        condval_4 = cid;
      }
      int32_t condval_5;
      if (((kk2 * 15) + (cid / 4)) < 23) {
        condval_5 = (((kk2 * 60) + cid) + 8100);
      } else {
        condval_5 = cid;
      }
      tl::ascend::copy_gm_to_l1<bfloat16_t, 128, 128>(H_l1_a[0], Hstate[((((condval_3 / 8192) * 134217728) + ((condval_4 % 128) * 1048576)) + (((condval_5 % 8192) / 128) * 16384))], 128, 128, 128, true);
      int32_t condval_6;
      if (((kk2 * 5) + (cid / 12)) < 6) {
        condval_6 = (((kk2 * 60) + cid) + 8120);
      } else {
        condval_6 = cid;
      }
      int32_t condval_7;
      if (((kk2 * 5) + (cid / 12)) < 6) {
        condval_7 = (((kk2 * 60) + cid) + 8120);
      } else {
        condval_7 = cid;
      }
      int32_t condval_8;
      if (((kk2 * 5) + (cid / 12)) < 6) {
        condval_8 = (((kk2 * 60) + cid) + 8120);
      } else {
        condval_8 = cid;
      }
      int32_t condval_9;
      if (((kk2 * 5) + (cid / 12)) < 6) {
        condval_9 = (((kk2 * 60) + cid) + 8120);
      } else {
        condval_9 = cid;
      }
      int32_t condval_10;
      if (((kk2 * 5) + (cid / 12)) < 6) {
        condval_10 = (((kk2 * 60) + cid) + 8120);
      } else {
        condval_10 = cid;
      }
      int32_t condval_11;
      if (((kk2 * 5) + (cid / 12)) < 6) {
        condval_11 = (((kk2 * 60) + cid) + 8120);
      } else {
        condval_11 = cid;
      }
      tl::ascend::copy_gm_to_l1<bfloat16_t, 128, 128>(H_l1_b[0], Hstate[((((condval_9 / 8192) * 134217728) + ((condval_10 % 128) * 1048576)) + (((condval_11 % 8192) / 128) * 16384))], 128, 128, 128, true);
      int32_t condval_12;
      if (((kk2 * 15) + (cid / 4)) < 13) {
        condval_12 = (((kk2 * 60) + cid) + 8140);
      } else {
        condval_12 = cid;
      }
      int32_t condval_13;
      if (((kk2 * 15) + (cid / 4)) < 13) {
        condval_13 = (((kk2 * 60) + cid) + 8140);
      } else {
        condval_13 = cid;
      }
      int32_t condval_14;
      if (((kk2 * 15) + (cid / 4)) < 13) {
        condval_14 = (((kk2 * 60) + cid) + 8140);
      } else {
        condval_14 = cid;
      }
      int32_t condval_15;
      if (((kk2 * 15) + (cid / 4)) < 13) {
        condval_15 = (((kk2 * 60) + cid) + 8140);
      } else {
        condval_15 = cid;
      }
      int32_t condval_16;
      if (((kk2 * 15) + (cid / 4)) < 13) {
        condval_16 = (((kk2 * 60) + cid) + 8140);
      } else {
        condval_16 = cid;
      }
      int32_t condval_17;
      if (((kk2 * 15) + (cid / 4)) < 13) {
        condval_17 = (((kk2 * 60) + cid) + 8140);
      } else {
        condval_17 = cid;
      }
      tl::ascend::copy_gm_to_l1<bfloat16_t, 128, 128>(H_l1_c[0], Hstate[((((condval_15 / 8192) * 134217728) + ((condval_16 % 128) * 1048576)) + (((condval_17 % 8192) / 128) * 16384))], 128, 128, 128, true);
      AscendC::CrossCoreWaitFlag(45);
      tl::ascend::copy_gm_to_l1<bfloat16_t, 64, 128>(V_l1_a[0], workspace_4[((cid * 73728) + (kk2 * 24576))], 128, 64, 128, true);
      AscendC::CrossCoreWaitFlag(46);
      tl::ascend::copy_gm_to_l1<bfloat16_t, 64, 128>(V_l1_b[0], workspace_4[(((cid * 73728) + (kk2 * 24576)) + 8192)], 128, 64, 128, true);
      AscendC::CrossCoreWaitFlag(47);
      tl::ascend::copy_gm_to_l1<bfloat16_t, 64, 128>(V_l1_c[0], workspace_4[(((cid * 73728) + (kk2 * 24576)) + 16384)], 128, 64, 128, true);
      AscendC::CrossCoreWaitFlag(33);
      tl::ascend::copy_gm_to_l1<bfloat16_t, 64, 64>(A_l1_a[0], workspace_3[((cid * 36864) + (kk2 * 12288))], 64, 64, 64, true);
      AscendC::CrossCoreWaitFlag(34);
      tl::ascend::copy_gm_to_l1<bfloat16_t, 64, 64>(A_l1_b[0], workspace_3[(((cid * 36864) + (kk2 * 12288)) + 4096)], 64, 64, 64, true);
      AscendC::CrossCoreWaitFlag(35);
      tl::ascend::copy_gm_to_l1<bfloat16_t, 64, 64>(A_l1_c[0], workspace_3[(((cid * 36864) + (kk2 * 12288)) + 8192)], 64, 64, 64, true);
      AscendC::CrossCoreWaitFlag(9);
      tl::ascend::copy_gm_to_l1<bfloat16_t, 64, 128>(qg_hi_l1_a[0], workspace_1[((cid * 73728) + (kk2 * 24576))], 128, 64, 128, true);
      AscendC::CrossCoreWaitFlag(10);
      tl::ascend::copy_gm_to_l1<bfloat16_t, 64, 128>(qg_hi_l1_b[0], workspace_1[(((cid * 73728) + (kk2 * 24576)) + 8192)], 128, 64, 128, true);
      AscendC::CrossCoreWaitFlag(11);
      tl::ascend::copy_gm_to_l1<bfloat16_t, 64, 128>(qg_hi_l1_c[0], workspace_1[(((cid * 73728) + (kk2 * 24576)) + 16384)], 128, 64, 128, true);
      AscendC::CrossCoreWaitFlag(21);
      tl::ascend::copy_gm_to_l1<bfloat16_t, 64, 128>(qg_lo_l1_a[0], workspace_2[((cid * 73728) + (kk2 * 24576))], 128, 64, 128, true);
      AscendC::CrossCoreWaitFlag(22);
      tl::ascend::copy_gm_to_l1<bfloat16_t, 64, 128>(qg_lo_l1_b[0], workspace_2[(((cid * 73728) + (kk2 * 24576)) + 8192)], 128, 64, 128, true);
      AscendC::CrossCoreWaitFlag(23);
      tl::ascend::copy_gm_to_l1<bfloat16_t, 64, 128>(qg_lo_l1_c[0], workspace_2[(((cid * 73728) + (kk2 * 24576)) + 16384)], 128, 64, 128, true);
      AscendC::SetFlag<AscendC::HardEvent::FIX_M>(4);
      AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(4);
      AscendC::SetFlag<AscendC::HardEvent::MTE2_M>(5);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_M>(5);
      AscendC::PipeBarrier<PIPE_M>();
      tl::ascend::gemm_v0<bfloat16_t, float, 64, 128, 64, false, false, 128>(A_l1_a[0], V_l1_a[0], o_frag_a[0], ascend_l0a, ascend_l0b, (bool)1, 128);
      AscendC::PipeBarrier<PIPE_M>();
      tl::ascend::gemm_v0<bfloat16_t, float, 64, 128, 128, false, false, 128>(qg_hi_l1_a[0], H_l1_a[0], o_frag_a[0], ascend_l0a, ascend_l0b, (bool)0, 128);
      AscendC::PipeBarrier<PIPE_M>();
      tl::ascend::gemm_v0<bfloat16_t, float, 64, 128, 128, false, false, 128>(qg_lo_l1_a[0], H_l1_a[0], o_frag_a[0], ascend_l0a, ascend_l0b, (bool)0, 128);
      AscendC::SetFlag<AscendC::HardEvent::M_FIX>(6);
      AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(6);
      AscendC::PipeBarrier<PIPE_FIX>();
      int32_t condval_18;
      if (((kk2 * 15) + (cid / 4)) < 23) {
        condval_18 = (((kk2 * 60) + cid) + 8100);
      } else {
        condval_18 = cid;
      }
      int32_t condval_19;
      if (((kk2 * 15) + (cid / 4)) < 23) {
        condval_19 = (((kk2 * 60) + cid) + 8100);
      } else {
        condval_19 = cid;
      }
      int32_t condval_20;
      if (((kk2 * 15) + (cid / 4)) < 23) {
        condval_20 = (((kk2 * 60) + cid) + 8100);
      } else {
        condval_20 = cid;
      }
      int32_t condval_21;
      if (((kk2 * 15) + (cid / 4)) < 23) {
        condval_21 = (((kk2 * 60) + cid) + 8100);
      } else {
        condval_21 = cid;
      }
      int32_t condval_22;
      if (((kk2 * 15) + (cid / 4)) < 23) {
        condval_22 = (((kk2 * 60) + cid) + 8100);
      } else {
        condval_22 = cid;
      }
      int32_t condval_23;
      if (((kk2 * 15) + (cid / 4)) < 23) {
        condval_23 = (((kk2 * 60) + cid) + 8100);
      } else {
        condval_23 = cid;
      }
      tl::ascend::copy_l0c_to_gm<float, bfloat16_t, layout::RowMajor, 64, 128, 0>(O[((((condval_21 / 8192) * 67108864) + ((condval_22 % 128) * 524288)) + (((condval_23 % 8192) / 128) * 128))], o_frag_a[0], 8192, 64, 128, 0);
      AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(48);
      tl::ascend::gemm_v0<bfloat16_t, float, 64, 128, 64, false, false, 128>(A_l1_b[0], V_l1_b[0], o_frag_b[0], ascend_l0a, ascend_l0b, (bool)1, 128);
      AscendC::PipeBarrier<PIPE_M>();
      tl::ascend::gemm_v0<bfloat16_t, float, 64, 128, 128, false, false, 128>(qg_hi_l1_b[0], H_l1_b[0], o_frag_b[0], ascend_l0a, ascend_l0b, (bool)0, 128);
      AscendC::PipeBarrier<PIPE_M>();
      tl::ascend::gemm_v0<bfloat16_t, float, 64, 128, 128, false, false, 128>(qg_lo_l1_b[0], H_l1_b[0], o_frag_b[0], ascend_l0a, ascend_l0b, (bool)0, 128);
      AscendC::SetFlag<AscendC::HardEvent::M_FIX>(7);
      AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(7);
      AscendC::PipeBarrier<PIPE_FIX>();
      int32_t condval_24;
      if (((kk2 * 5) + (cid / 12)) < 6) {
        condval_24 = (((kk2 * 60) + cid) + 8120);
      } else {
        condval_24 = cid;
      }
      int32_t condval_25;
      if (((kk2 * 5) + (cid / 12)) < 6) {
        condval_25 = (((kk2 * 60) + cid) + 8120);
      } else {
        condval_25 = cid;
      }
      int32_t condval_26;
      if (((kk2 * 5) + (cid / 12)) < 6) {
        condval_26 = (((kk2 * 60) + cid) + 8120);
      } else {
        condval_26 = cid;
      }
      int32_t condval_27;
      if (((kk2 * 5) + (cid / 12)) < 6) {
        condval_27 = (((kk2 * 60) + cid) + 8120);
      } else {
        condval_27 = cid;
      }
      int32_t condval_28;
      if (((kk2 * 5) + (cid / 12)) < 6) {
        condval_28 = (((kk2 * 60) + cid) + 8120);
      } else {
        condval_28 = cid;
      }
      int32_t condval_29;
      if (((kk2 * 5) + (cid / 12)) < 6) {
        condval_29 = (((kk2 * 60) + cid) + 8120);
      } else {
        condval_29 = cid;
      }
      tl::ascend::copy_l0c_to_gm<float, bfloat16_t, layout::RowMajor, 64, 128, 0>(O[((((condval_27 / 8192) * 67108864) + ((condval_28 % 128) * 524288)) + (((condval_29 % 8192) / 128) * 128))], o_frag_b[0], 8192, 64, 128, 0);
      AscendC::SetFlag<AscendC::HardEvent::FIX_M>(0);
      AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(0);
      tl::ascend::gemm_v0<bfloat16_t, float, 64, 128, 64, false, false, 128>(A_l1_c[0], V_l1_c[0], o_frag_a[0], ascend_l0a, ascend_l0b, (bool)1, 128);
      AscendC::PipeBarrier<PIPE_M>();
      tl::ascend::gemm_v0<bfloat16_t, float, 64, 128, 128, false, false, 128>(qg_hi_l1_c[0], H_l1_c[0], o_frag_a[0], ascend_l0a, ascend_l0b, (bool)0, 128);
      AscendC::PipeBarrier<PIPE_M>();
      tl::ascend::gemm_v0<bfloat16_t, float, 64, 128, 128, false, false, 128>(qg_lo_l1_c[0], H_l1_c[0], o_frag_a[0], ascend_l0a, ascend_l0b, (bool)0, 128);
      AscendC::SetFlag<AscendC::HardEvent::M_FIX>(1);
      AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(1);
      AscendC::PipeBarrier<PIPE_FIX>();
      int32_t condval_30;
      if (((kk2 * 15) + (cid / 4)) < 13) {
        condval_30 = (((kk2 * 60) + cid) + 8140);
      } else {
        condval_30 = cid;
      }
      int32_t condval_31;
      if (((kk2 * 15) + (cid / 4)) < 13) {
        condval_31 = (((kk2 * 60) + cid) + 8140);
      } else {
        condval_31 = cid;
      }
      int32_t condval_32;
      if (((kk2 * 15) + (cid / 4)) < 13) {
        condval_32 = (((kk2 * 60) + cid) + 8140);
      } else {
        condval_32 = cid;
      }
      int32_t condval_33;
      if (((kk2 * 15) + (cid / 4)) < 13) {
        condval_33 = (((kk2 * 60) + cid) + 8140);
      } else {
        condval_33 = cid;
      }
      int32_t condval_34;
      if (((kk2 * 15) + (cid / 4)) < 13) {
        condval_34 = (((kk2 * 60) + cid) + 8140);
      } else {
        condval_34 = cid;
      }
      int32_t condval_35;
      if (((kk2 * 15) + (cid / 4)) < 13) {
        condval_35 = (((kk2 * 60) + cid) + 8140);
      } else {
        condval_35 = cid;
      }
      tl::ascend::copy_l0c_to_gm<float, bfloat16_t, layout::RowMajor, 64, 128, 0>(O[((((condval_33 / 8192) * 67108864) + ((condval_34 % 128) * 524288)) + (((condval_35 % 8192) / 128) * 128))], o_frag_a[0], 8192, 64, 128, 0);
    }
  }
  if ASCEND_IS_AIV {
    tl::ascend::copy_gm_to_ub<float, 64, 32>(msk_ub[0], Msk[(vid * 2048)], 64, 32, 64, 0.000000e+00f);
    tl::ascend::copy_gm_to_ub<bfloat16_t, 64, 32>(a_half[0], A[((cid * 262144) + (vid * 131072))], 4096, 32, 64, bfloat16_t(0.000000e+00f));
    tl::ascend::copy_gm_to_ub<bfloat16_t, 128, 32>(q_half[0], Q[((cid * 524288) + (vid * 262144))], 8192, 32, 128, bfloat16_t(0.000000e+00f));
    tl::ascend::copy_gm_to_ub<float, 128, 32>(g_ub[0], G[((cid * 524288) + (vid * 262144))], 8192, 32, 128, 0.000000e+00f);
    tl::ascend::copy_gm_to_ub<bfloat16_t, 128, 32>(v_half[0], V[((cid * 524288) + (vid * 262144))], 8192, 32, 128, bfloat16_t(0.000000e+00f));
    tl::ascend::copy_gm_to_ub<bfloat16_t, 64, 32>(a_half2[0], A[(((cid * 262144) + (vid * 131072)) + 5242880)], 4096, 32, 64, bfloat16_t(0.000000e+00f));
    tl::ascend::copy_gm_to_ub<bfloat16_t, 128, 32>(q_half2[0], Q[(((cid * 524288) + (vid * 262144)) + 10485760)], 8192, 32, 128, bfloat16_t(0.000000e+00f));
    tl::ascend::copy_gm_to_ub<float, 128, 32>(g_ub2[0], G[(((cid * 524288) + (vid * 262144)) + 10485760)], 8192, 32, 128, 0.000000e+00f);
    tl::ascend::copy_gm_to_ub<bfloat16_t, 128, 32>(v_half2[0], V[(((cid * 524288) + (vid * 262144)) + 10485760)], 8192, 32, 128, bfloat16_t(0.000000e+00f));
    tl::ascend::copy_gm_to_ub<bfloat16_t, 64, 32>(a_half3[0], A[(((cid * 262144) + (vid * 131072)) + 10485760)], 4096, 32, 64, bfloat16_t(0.000000e+00f));
    tl::ascend::copy_gm_to_ub<bfloat16_t, 128, 32>(q_half3[0], Q[(((cid * 524288) + (vid * 262144)) + 20971520)], 8192, 32, 128, bfloat16_t(0.000000e+00f));
    tl::ascend::copy_gm_to_ub<float, 128, 32>(g_ub3[0], G[(((cid * 524288) + (vid * 262144)) + 20971520)], 8192, 32, 128, 0.000000e+00f);
    tl::ascend::copy_gm_to_ub<bfloat16_t, 128, 32>(v_half3[0], V[(((cid * 524288) + (vid * 262144)) + 20971520)], 8192, 32, 128, bfloat16_t(0.000000e+00f));
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(1);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(1);
    tl::ascend::copy_ub_to_gm<bfloat16_t, 128, 32>(workspace_4[((cid * 73728) + (vid * 4096))], v_half[0], 128, 32, 128);
    AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(36);
    AscendC::PipeBarrier<PIPE_MTE3>();
    tl::ascend::copy_ub_to_gm<bfloat16_t, 128, 32>(workspace_4[(((cid * 73728) + (vid * 4096)) + 8192)], v_half2[0], 128, 32, 128);
    AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(37);
    AscendC::PipeBarrier<PIPE_MTE3>();
    tl::ascend::copy_ub_to_gm<bfloat16_t, 128, 32>(workspace_4[(((cid * 73728) + (vid * 4096)) + 16384)], v_half3[0], 128, 32, 128);
    AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(38);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(2);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(2);
    AscendC::PipeBarrier<PIPE_MTE2>();
    tl::ascend::copy_gm_to_ub<bfloat16_t, 128, 32>(v_half[0], V[(((cid * 524288) + (vid * 262144)) + 31457280)], 8192, 32, 128, bfloat16_t(0.000000e+00f));
    tl::ascend::copy_gm_to_ub<bfloat16_t, 128, 32>(v_half2[0], V[(((cid * 524288) + (vid * 262144)) + 41943040)], 8192, 32, 128, bfloat16_t(0.000000e+00f));
    tl::ascend::copy_gm_to_ub<bfloat16_t, 128, 32>(v_half3[0], V[(((cid * 524288) + (vid * 262144)) + 52428800)], 8192, 32, 128, bfloat16_t(0.000000e+00f));
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(3);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(3);
    tl::ascend::copy_ub_to_ub<float, bfloat16_t, 2048>(a_f32[0], a_half[0], 32, 64, 64, 32, 64, 64);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mul<float, false>(a_f32[0], a_f32[0], msk_ub[0], AscendC::MASK_PLACEHOLDER, 32, AscendC::BinaryRepeatParams());
    AscendC::PipeBarrier<PIPE_V>();
    tl::ascend::copy_ub_to_ub<bfloat16_t, float, 2048>(a_mask[0], a_f32[0], 32, 64, 64, 32, 64, 64);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(4);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(4);
    tl::ascend::copy_ub_to_gm<bfloat16_t, 64, 32>(workspace_3[((cid * 36864) + (vid * 2048))], a_mask[0], 64, 32, 64);
    AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(24);
    tl::ascend::copy_ub_to_ub<float, bfloat16_t, 4096>(q_ub[0], q_half[0], 32, 128, 128, 32, 128, 128);
    {
    AscendC::Muls<float, false>(g_ub[0], g_ub[0], 6.931472e-01f, AscendC::MASK_PLACEHOLDER, 64, AscendC::UnaryRepeatParams());
    }
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Exp<float, false>(g_ub[0], g_ub[0], AscendC::MASK_PLACEHOLDER, 64, AscendC::UnaryRepeatParams());
    {
    AscendC::Muls<float, false>(q_ub[0], q_ub[0], 8.838835e-02f, AscendC::MASK_PLACEHOLDER, 64, AscendC::UnaryRepeatParams());
    }
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mul<float, false>(q_ub[0], q_ub[0], g_ub[0], AscendC::MASK_PLACEHOLDER, 64, AscendC::BinaryRepeatParams());
    AscendC::PipeBarrier<PIPE_V>();
    tl::ascend::copy_ub_to_ub<bfloat16_t, float, 4096>(qg_hi_ub[0], q_ub[0], 32, 128, 128, 32, 128, 128);
    AscendC::PipeBarrier<PIPE_V>();
    tl::ascend::copy_ub_to_ub<float, bfloat16_t, 4096>(qg_hi_f32[0], qg_hi_ub[0], 32, 128, 128, 32, 128, 128);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Sub<float, false>(q_ub[0], q_ub[0], qg_hi_f32[0], AscendC::MASK_PLACEHOLDER, 64, AscendC::BinaryRepeatParams());
    AscendC::PipeBarrier<PIPE_V>();
    tl::ascend::copy_ub_to_ub<bfloat16_t, float, 4096>(qg_lo_ub[0], q_ub[0], 32, 128, 128, 32, 128, 128);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(5);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(5);
    tl::ascend::copy_ub_to_gm<bfloat16_t, 128, 32>(workspace_1[((cid * 73728) + (vid * 4096))], qg_hi_ub[0], 128, 32, 128);
    AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(0);
    tl::ascend::copy_ub_to_gm<bfloat16_t, 128, 32>(workspace_2[((cid * 73728) + (vid * 4096))], qg_lo_ub[0], 128, 32, 128);
    AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(12);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(6);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(6);
    tl::ascend::copy_gm_to_ub<bfloat16_t, 64, 32>(a_half[0], A[(((cid * 262144) + (vid * 131072)) + 15728640)], 4096, 32, 64, bfloat16_t(0.000000e+00f));
    tl::ascend::copy_gm_to_ub<bfloat16_t, 128, 32>(q_half[0], Q[(((cid * 524288) + (vid * 262144)) + 31457280)], 8192, 32, 128, bfloat16_t(0.000000e+00f));
    tl::ascend::copy_gm_to_ub<float, 128, 32>(g_ub[0], G[(((cid * 524288) + (vid * 262144)) + 31457280)], 8192, 32, 128, 0.000000e+00f);
    tl::ascend::copy_ub_to_ub<float, bfloat16_t, 2048>(a_f32[0], a_half2[0], 32, 64, 64, 32, 64, 64);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mul<float, false>(a_f32[0], a_f32[0], msk_ub[0], AscendC::MASK_PLACEHOLDER, 32, AscendC::BinaryRepeatParams());
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(7);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(7);
    AscendC::PipeBarrier<PIPE_V>();
    tl::ascend::copy_ub_to_ub<bfloat16_t, float, 2048>(a_mask[0], a_f32[0], 32, 64, 64, 32, 64, 64);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
    AscendC::PipeBarrier<PIPE_MTE3>();
    tl::ascend::copy_ub_to_gm<bfloat16_t, 64, 32>(workspace_3[(((cid * 36864) + (vid * 2048)) + 4096)], a_mask[0], 64, 32, 64);
    AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(25);
    tl::ascend::copy_ub_to_ub<float, bfloat16_t, 4096>(q_ub[0], q_half2[0], 32, 128, 128, 32, 128, 128);
    {
    AscendC::Muls<float, false>(g_ub2[0], g_ub2[0], 6.931472e-01f, AscendC::MASK_PLACEHOLDER, 64, AscendC::UnaryRepeatParams());
    }
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Exp<float, false>(g_ub2[0], g_ub2[0], AscendC::MASK_PLACEHOLDER, 64, AscendC::UnaryRepeatParams());
    {
    AscendC::Muls<float, false>(q_ub[0], q_ub[0], 8.838835e-02f, AscendC::MASK_PLACEHOLDER, 64, AscendC::UnaryRepeatParams());
    }
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mul<float, false>(q_ub[0], q_ub[0], g_ub2[0], AscendC::MASK_PLACEHOLDER, 64, AscendC::BinaryRepeatParams());
    AscendC::PipeBarrier<PIPE_V>();
    tl::ascend::copy_ub_to_ub<bfloat16_t, float, 4096>(qg_hi_ub[0], q_ub[0], 32, 128, 128, 32, 128, 128);
    AscendC::PipeBarrier<PIPE_V>();
    tl::ascend::copy_ub_to_ub<float, bfloat16_t, 4096>(qg_hi_f32[0], qg_hi_ub[0], 32, 128, 128, 32, 128, 128);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Sub<float, false>(q_ub[0], q_ub[0], qg_hi_f32[0], AscendC::MASK_PLACEHOLDER, 64, AscendC::BinaryRepeatParams());
    AscendC::PipeBarrier<PIPE_V>();
    tl::ascend::copy_ub_to_ub<bfloat16_t, float, 4096>(qg_lo_ub[0], q_ub[0], 32, 128, 128, 32, 128, 128);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(1);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(1);
    tl::ascend::copy_ub_to_gm<bfloat16_t, 128, 32>(workspace_1[(((cid * 73728) + (vid * 4096)) + 8192)], qg_hi_ub[0], 128, 32, 128);
    AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(1);
    tl::ascend::copy_ub_to_gm<bfloat16_t, 128, 32>(workspace_2[(((cid * 73728) + (vid * 4096)) + 8192)], qg_lo_ub[0], 128, 32, 128);
    AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(13);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(2);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(2);
    tl::ascend::copy_gm_to_ub<bfloat16_t, 64, 32>(a_half2[0], A[(((cid * 262144) + (vid * 131072)) + 20971520)], 4096, 32, 64, bfloat16_t(0.000000e+00f));
    tl::ascend::copy_gm_to_ub<bfloat16_t, 128, 32>(q_half2[0], Q[(((cid * 524288) + (vid * 262144)) + 41943040)], 8192, 32, 128, bfloat16_t(0.000000e+00f));
    tl::ascend::copy_gm_to_ub<float, 128, 32>(g_ub2[0], G[(((cid * 524288) + (vid * 262144)) + 41943040)], 8192, 32, 128, 0.000000e+00f);
    tl::ascend::copy_ub_to_ub<float, bfloat16_t, 2048>(a_f32[0], a_half3[0], 32, 64, 64, 32, 64, 64);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mul<float, false>(a_f32[0], a_f32[0], msk_ub[0], AscendC::MASK_PLACEHOLDER, 32, AscendC::BinaryRepeatParams());
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(3);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(3);
    AscendC::PipeBarrier<PIPE_V>();
    tl::ascend::copy_ub_to_ub<bfloat16_t, float, 2048>(a_mask[0], a_f32[0], 32, 64, 64, 32, 64, 64);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(4);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(4);
    AscendC::PipeBarrier<PIPE_MTE3>();
    tl::ascend::copy_ub_to_gm<bfloat16_t, 64, 32>(workspace_3[(((cid * 36864) + (vid * 2048)) + 8192)], a_mask[0], 64, 32, 64);
    AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(26);
    tl::ascend::copy_ub_to_ub<float, bfloat16_t, 4096>(q_ub[0], q_half3[0], 32, 128, 128, 32, 128, 128);
    {
    AscendC::Muls<float, false>(g_ub3[0], g_ub3[0], 6.931472e-01f, AscendC::MASK_PLACEHOLDER, 64, AscendC::UnaryRepeatParams());
    }
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Exp<float, false>(g_ub3[0], g_ub3[0], AscendC::MASK_PLACEHOLDER, 64, AscendC::UnaryRepeatParams());
    {
    AscendC::Muls<float, false>(q_ub[0], q_ub[0], 8.838835e-02f, AscendC::MASK_PLACEHOLDER, 64, AscendC::UnaryRepeatParams());
    }
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mul<float, false>(q_ub[0], q_ub[0], g_ub3[0], AscendC::MASK_PLACEHOLDER, 64, AscendC::BinaryRepeatParams());
    AscendC::PipeBarrier<PIPE_V>();
    tl::ascend::copy_ub_to_ub<bfloat16_t, float, 4096>(qg_hi_ub[0], q_ub[0], 32, 128, 128, 32, 128, 128);
    AscendC::PipeBarrier<PIPE_V>();
    tl::ascend::copy_ub_to_ub<float, bfloat16_t, 4096>(qg_hi_f32[0], qg_hi_ub[0], 32, 128, 128, 32, 128, 128);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Sub<float, false>(q_ub[0], q_ub[0], qg_hi_f32[0], AscendC::MASK_PLACEHOLDER, 64, AscendC::BinaryRepeatParams());
    AscendC::PipeBarrier<PIPE_V>();
    tl::ascend::copy_ub_to_ub<bfloat16_t, float, 4096>(qg_lo_ub[0], q_ub[0], 32, 128, 128, 32, 128, 128);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(5);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(5);
    tl::ascend::copy_ub_to_gm<bfloat16_t, 128, 32>(workspace_1[(((cid * 73728) + (vid * 4096)) + 16384)], qg_hi_ub[0], 128, 32, 128);
    AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(2);
    tl::ascend::copy_ub_to_gm<bfloat16_t, 128, 32>(workspace_2[(((cid * 73728) + (vid * 4096)) + 16384)], qg_lo_ub[0], 128, 32, 128);
    AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(14);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(6);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(6);
    tl::ascend::copy_gm_to_ub<bfloat16_t, 64, 32>(a_half3[0], A[(((cid * 262144) + (vid * 131072)) + 26214400)], 4096, 32, 64, bfloat16_t(0.000000e+00f));
    tl::ascend::copy_gm_to_ub<bfloat16_t, 128, 32>(q_half3[0], Q[(((cid * 524288) + (vid * 262144)) + 52428800)], 8192, 32, 128, bfloat16_t(0.000000e+00f));
    tl::ascend::copy_gm_to_ub<float, 128, 32>(g_ub3[0], G[(((cid * 524288) + (vid * 262144)) + 52428800)], 8192, 32, 128, 0.000000e+00f);
    tl::ascend::copy_ub_to_gm<bfloat16_t, 128, 32>(workspace_4[(((cid * 73728) + (vid * 4096)) + 24576)], v_half[0], 128, 32, 128);
    AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(39);
    AscendC::PipeBarrier<PIPE_MTE3>();
    tl::ascend::copy_ub_to_gm<bfloat16_t, 128, 32>(workspace_4[(((cid * 73728) + (vid * 4096)) + 32768)], v_half2[0], 128, 32, 128);
    AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(40);
    AscendC::PipeBarrier<PIPE_MTE3>();
    tl::ascend::copy_ub_to_gm<bfloat16_t, 128, 32>(workspace_4[(((cid * 73728) + (vid * 4096)) + 40960)], v_half3[0], 128, 32, 128);
    AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(41);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(7);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(7);
    AscendC::PipeBarrier<PIPE_MTE2>();
    tl::ascend::copy_gm_to_ub<bfloat16_t, 128, 32>(v_half[0], V[(((((cid + 120) % 128) * 524288) + (vid * 262144)) + (((cid + 120) / 128) * 128))], 8192, 32, 128, bfloat16_t(0.000000e+00f));
    tl::ascend::copy_gm_to_ub<bfloat16_t, 128, 32>(v_half2[0], V[((((cid * 524288) + (vid * 262144)) + (((cid + 140) / 128) * 128)) + 6291456)], 8192, 32, 128, bfloat16_t(0.000000e+00f));
    tl::ascend::copy_gm_to_ub<bfloat16_t, 128, 32>(v_half3[0], V[((((cid * 524288) + (vid * 262144)) + (((cid + 160) / 128) * 128)) + 16777216)], 8192, 32, 128, bfloat16_t(0.000000e+00f));
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
    tl::ascend::copy_ub_to_ub<float, bfloat16_t, 2048>(a_f32[0], a_half[0], 32, 64, 64, 32, 64, 64);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mul<float, false>(a_f32[0], a_f32[0], msk_ub[0], AscendC::MASK_PLACEHOLDER, 32, AscendC::BinaryRepeatParams());
    AscendC::PipeBarrier<PIPE_V>();
    tl::ascend::copy_ub_to_ub<bfloat16_t, float, 2048>(a_mask[0], a_f32[0], 32, 64, 64, 32, 64, 64);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(1);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(1);
    tl::ascend::copy_ub_to_gm<bfloat16_t, 64, 32>(workspace_3[(((cid * 36864) + (vid * 2048)) + 12288)], a_mask[0], 64, 32, 64);
    AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(27);
    tl::ascend::copy_ub_to_ub<float, bfloat16_t, 4096>(q_ub[0], q_half[0], 32, 128, 128, 32, 128, 128);
    {
    AscendC::Muls<float, false>(g_ub[0], g_ub[0], 6.931472e-01f, AscendC::MASK_PLACEHOLDER, 64, AscendC::UnaryRepeatParams());
    }
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Exp<float, false>(g_ub[0], g_ub[0], AscendC::MASK_PLACEHOLDER, 64, AscendC::UnaryRepeatParams());
    {
    AscendC::Muls<float, false>(q_ub[0], q_ub[0], 8.838835e-02f, AscendC::MASK_PLACEHOLDER, 64, AscendC::UnaryRepeatParams());
    }
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mul<float, false>(q_ub[0], q_ub[0], g_ub[0], AscendC::MASK_PLACEHOLDER, 64, AscendC::BinaryRepeatParams());
    AscendC::PipeBarrier<PIPE_V>();
    tl::ascend::copy_ub_to_ub<bfloat16_t, float, 4096>(qg_hi_ub[0], q_ub[0], 32, 128, 128, 32, 128, 128);
    AscendC::PipeBarrier<PIPE_V>();
    tl::ascend::copy_ub_to_ub<float, bfloat16_t, 4096>(qg_hi_f32[0], qg_hi_ub[0], 32, 128, 128, 32, 128, 128);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Sub<float, false>(q_ub[0], q_ub[0], qg_hi_f32[0], AscendC::MASK_PLACEHOLDER, 64, AscendC::BinaryRepeatParams());
    AscendC::PipeBarrier<PIPE_V>();
    tl::ascend::copy_ub_to_ub<bfloat16_t, float, 4096>(qg_lo_ub[0], q_ub[0], 32, 128, 128, 32, 128, 128);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(2);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(2);
    tl::ascend::copy_ub_to_gm<bfloat16_t, 128, 32>(workspace_1[(((cid * 73728) + (vid * 4096)) + 24576)], qg_hi_ub[0], 128, 32, 128);
    AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(3);
    tl::ascend::copy_ub_to_gm<bfloat16_t, 128, 32>(workspace_2[(((cid * 73728) + (vid * 4096)) + 24576)], qg_lo_ub[0], 128, 32, 128);
    AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(15);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(3);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(3);
    tl::ascend::copy_gm_to_ub<bfloat16_t, 64, 32>(a_half[0], A[(((((cid + 120) % 128) * 262144) + (vid * 131072)) + (((cid + 120) / 128) * 64))], 4096, 32, 64, bfloat16_t(0.000000e+00f));
    tl::ascend::copy_gm_to_ub<bfloat16_t, 128, 32>(q_half[0], Q[(((((cid + 120) % 128) * 524288) + (vid * 262144)) + (((cid + 120) / 128) * 128))], 8192, 32, 128, bfloat16_t(0.000000e+00f));
    tl::ascend::copy_gm_to_ub<float, 128, 32>(g_ub[0], G[(((((cid + 120) % 128) * 524288) + (vid * 262144)) + (((cid + 120) / 128) * 128))], 8192, 32, 128, 0.000000e+00f);
    tl::ascend::copy_ub_to_ub<float, bfloat16_t, 2048>(a_f32[0], a_half2[0], 32, 64, 64, 32, 64, 64);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mul<float, false>(a_f32[0], a_f32[0], msk_ub[0], AscendC::MASK_PLACEHOLDER, 32, AscendC::BinaryRepeatParams());
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(4);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(4);
    AscendC::PipeBarrier<PIPE_V>();
    tl::ascend::copy_ub_to_ub<bfloat16_t, float, 2048>(a_mask[0], a_f32[0], 32, 64, 64, 32, 64, 64);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(5);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(5);
    AscendC::PipeBarrier<PIPE_MTE3>();
    tl::ascend::copy_ub_to_gm<bfloat16_t, 64, 32>(workspace_3[(((cid * 36864) + (vid * 2048)) + 16384)], a_mask[0], 64, 32, 64);
    AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(28);
    tl::ascend::copy_ub_to_ub<float, bfloat16_t, 4096>(q_ub[0], q_half2[0], 32, 128, 128, 32, 128, 128);
    {
    AscendC::Muls<float, false>(g_ub2[0], g_ub2[0], 6.931472e-01f, AscendC::MASK_PLACEHOLDER, 64, AscendC::UnaryRepeatParams());
    }
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Exp<float, false>(g_ub2[0], g_ub2[0], AscendC::MASK_PLACEHOLDER, 64, AscendC::UnaryRepeatParams());
    {
    AscendC::Muls<float, false>(q_ub[0], q_ub[0], 8.838835e-02f, AscendC::MASK_PLACEHOLDER, 64, AscendC::UnaryRepeatParams());
    }
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mul<float, false>(q_ub[0], q_ub[0], g_ub2[0], AscendC::MASK_PLACEHOLDER, 64, AscendC::BinaryRepeatParams());
    AscendC::PipeBarrier<PIPE_V>();
    tl::ascend::copy_ub_to_ub<bfloat16_t, float, 4096>(qg_hi_ub[0], q_ub[0], 32, 128, 128, 32, 128, 128);
    AscendC::PipeBarrier<PIPE_V>();
    tl::ascend::copy_ub_to_ub<float, bfloat16_t, 4096>(qg_hi_f32[0], qg_hi_ub[0], 32, 128, 128, 32, 128, 128);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Sub<float, false>(q_ub[0], q_ub[0], qg_hi_f32[0], AscendC::MASK_PLACEHOLDER, 64, AscendC::BinaryRepeatParams());
    AscendC::PipeBarrier<PIPE_V>();
    tl::ascend::copy_ub_to_ub<bfloat16_t, float, 4096>(qg_lo_ub[0], q_ub[0], 32, 128, 128, 32, 128, 128);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(6);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(6);
    tl::ascend::copy_ub_to_gm<bfloat16_t, 128, 32>(workspace_1[(((cid * 73728) + (vid * 4096)) + 32768)], qg_hi_ub[0], 128, 32, 128);
    AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(4);
    tl::ascend::copy_ub_to_gm<bfloat16_t, 128, 32>(workspace_2[(((cid * 73728) + (vid * 4096)) + 32768)], qg_lo_ub[0], 128, 32, 128);
    AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(16);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(7);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(7);
    tl::ascend::copy_gm_to_ub<bfloat16_t, 64, 32>(a_half2[0], A[((((cid * 262144) + (vid * 131072)) + (((cid + 140) / 128) * 64)) + 3145728)], 4096, 32, 64, bfloat16_t(0.000000e+00f));
    tl::ascend::copy_gm_to_ub<bfloat16_t, 128, 32>(q_half2[0], Q[((((cid * 524288) + (vid * 262144)) + (((cid + 140) / 128) * 128)) + 6291456)], 8192, 32, 128, bfloat16_t(0.000000e+00f));
    tl::ascend::copy_gm_to_ub<float, 128, 32>(g_ub2[0], G[((((cid * 524288) + (vid * 262144)) + (((cid + 140) / 128) * 128)) + 6291456)], 8192, 32, 128, 0.000000e+00f);
    tl::ascend::copy_ub_to_ub<float, bfloat16_t, 2048>(a_f32[0], a_half3[0], 32, 64, 64, 32, 64, 64);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mul<float, false>(a_f32[0], a_f32[0], msk_ub[0], AscendC::MASK_PLACEHOLDER, 32, AscendC::BinaryRepeatParams());
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
    AscendC::PipeBarrier<PIPE_V>();
    tl::ascend::copy_ub_to_ub<bfloat16_t, float, 2048>(a_mask[0], a_f32[0], 32, 64, 64, 32, 64, 64);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(1);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(1);
    AscendC::PipeBarrier<PIPE_MTE3>();
    tl::ascend::copy_ub_to_gm<bfloat16_t, 64, 32>(workspace_3[(((cid * 36864) + (vid * 2048)) + 20480)], a_mask[0], 64, 32, 64);
    AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(29);
    tl::ascend::copy_ub_to_ub<float, bfloat16_t, 4096>(q_ub[0], q_half3[0], 32, 128, 128, 32, 128, 128);
    {
    AscendC::Muls<float, false>(g_ub3[0], g_ub3[0], 6.931472e-01f, AscendC::MASK_PLACEHOLDER, 64, AscendC::UnaryRepeatParams());
    }
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Exp<float, false>(g_ub3[0], g_ub3[0], AscendC::MASK_PLACEHOLDER, 64, AscendC::UnaryRepeatParams());
    {
    AscendC::Muls<float, false>(q_ub[0], q_ub[0], 8.838835e-02f, AscendC::MASK_PLACEHOLDER, 64, AscendC::UnaryRepeatParams());
    }
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mul<float, false>(q_ub[0], q_ub[0], g_ub3[0], AscendC::MASK_PLACEHOLDER, 64, AscendC::BinaryRepeatParams());
    AscendC::PipeBarrier<PIPE_V>();
    tl::ascend::copy_ub_to_ub<bfloat16_t, float, 4096>(qg_hi_ub[0], q_ub[0], 32, 128, 128, 32, 128, 128);
    AscendC::PipeBarrier<PIPE_V>();
    tl::ascend::copy_ub_to_ub<float, bfloat16_t, 4096>(qg_hi_f32[0], qg_hi_ub[0], 32, 128, 128, 32, 128, 128);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Sub<float, false>(q_ub[0], q_ub[0], qg_hi_f32[0], AscendC::MASK_PLACEHOLDER, 64, AscendC::BinaryRepeatParams());
    AscendC::PipeBarrier<PIPE_V>();
    tl::ascend::copy_ub_to_ub<bfloat16_t, float, 4096>(qg_lo_ub[0], q_ub[0], 32, 128, 128, 32, 128, 128);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(2);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(2);
    tl::ascend::copy_ub_to_gm<bfloat16_t, 128, 32>(workspace_1[(((cid * 73728) + (vid * 4096)) + 40960)], qg_hi_ub[0], 128, 32, 128);
    AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(5);
    tl::ascend::copy_ub_to_gm<bfloat16_t, 128, 32>(workspace_2[(((cid * 73728) + (vid * 4096)) + 40960)], qg_lo_ub[0], 128, 32, 128);
    AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(17);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(3);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(3);
    tl::ascend::copy_gm_to_ub<bfloat16_t, 64, 32>(a_half3[0], A[((((cid * 262144) + (vid * 131072)) + (((cid + 160) / 128) * 64)) + 8388608)], 4096, 32, 64, bfloat16_t(0.000000e+00f));
    tl::ascend::copy_gm_to_ub<bfloat16_t, 128, 32>(q_half3[0], Q[((((cid * 524288) + (vid * 262144)) + (((cid + 160) / 128) * 128)) + 16777216)], 8192, 32, 128, bfloat16_t(0.000000e+00f));
    tl::ascend::copy_gm_to_ub<float, 128, 32>(g_ub3[0], G[((((cid * 524288) + (vid * 262144)) + (((cid + 160) / 128) * 128)) + 16777216)], 8192, 32, 128, 0.000000e+00f);
    for (int32_t kk_1 = 0; kk_1 < 133; ++kk_1) {
      AscendC::CrossCoreWaitFlag(48);
      tl::ascend::copy_ub_to_gm<bfloat16_t, 128, 32>(workspace_4[(((cid * 73728) + ((((kk_1 * 3) + 6) % 9) * 8192)) + (vid * 4096))], v_half[0], 128, 32, 128);
      AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(42);
      AscendC::PipeBarrier<PIPE_MTE3>();
      tl::ascend::copy_ub_to_gm<bfloat16_t, 128, 32>(workspace_4[(((cid * 73728) + ((((kk_1 * 3) + 7) % 9) * 8192)) + (vid * 4096))], v_half2[0], 128, 32, 128);
      AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(43);
      AscendC::PipeBarrier<PIPE_MTE3>();
      tl::ascend::copy_ub_to_gm<bfloat16_t, 128, 32>(workspace_4[(((cid * 73728) + ((((kk_1 * 3) + 8) % 9) * 8192)) + (vid * 4096))], v_half3[0], 128, 32, 128);
      AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(44);
      AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(4);
      AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(4);
      AscendC::PipeBarrier<PIPE_MTE2>();
      tl::ascend::copy_gm_to_ub<bfloat16_t, 128, 32>(v_half[0], V[(((((((kk_1 * 60) + cid) + 52) % 128) * 524288) + (vid * 262144)) + (((((kk_1 * 15) + (cid / 4)) + 45) / 32) * 128))], 8192, 32, 128, bfloat16_t(0.000000e+00f));
      tl::ascend::copy_gm_to_ub<bfloat16_t, 128, 32>(v_half2[0], V[(((((((kk_1 * 60) + cid) + 72) % 128) * 524288) + (vid * 262144)) + (((((kk_1 * 15) + (cid / 4)) + 50) / 32) * 128))], 8192, 32, 128, bfloat16_t(0.000000e+00f));
      tl::ascend::copy_gm_to_ub<bfloat16_t, 128, 32>(v_half3[0], V[(((((((kk_1 * 60) + cid) + 92) % 128) * 524288) + (vid * 262144)) + (((((kk_1 * 15) + (cid / 4)) + 55) / 32) * 128))], 8192, 32, 128, bfloat16_t(0.000000e+00f));
      AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(5);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(5);
      tl::ascend::copy_ub_to_ub<float, bfloat16_t, 2048>(a_f32[0], a_half[0], 32, 64, 64, 32, 64, 64);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Mul<float, false>(a_f32[0], a_f32[0], msk_ub[0], AscendC::MASK_PLACEHOLDER, 32, AscendC::BinaryRepeatParams());
      AscendC::PipeBarrier<PIPE_V>();
      tl::ascend::copy_ub_to_ub<bfloat16_t, float, 2048>(a_mask[0], a_f32[0], 32, 64, 64, 32, 64, 64);
      AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(6);
      AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(6);
      tl::ascend::copy_ub_to_gm<bfloat16_t, 64, 32>(workspace_3[(((cid * 36864) + ((((kk_1 * 3) + 6) % 9) * 4096)) + (vid * 2048))], a_mask[0], 64, 32, 64);
      AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(30);
      tl::ascend::copy_ub_to_ub<float, bfloat16_t, 4096>(q_ub[0], q_half[0], 32, 128, 128, 32, 128, 128);
      {
      AscendC::Muls<float, false>(g_ub[0], g_ub[0], 6.931472e-01f, AscendC::MASK_PLACEHOLDER, 64, AscendC::UnaryRepeatParams());
      }
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Exp<float, false>(g_ub[0], g_ub[0], AscendC::MASK_PLACEHOLDER, 64, AscendC::UnaryRepeatParams());
      {
      AscendC::Muls<float, false>(q_ub[0], q_ub[0], 8.838835e-02f, AscendC::MASK_PLACEHOLDER, 64, AscendC::UnaryRepeatParams());
      }
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Mul<float, false>(q_ub[0], q_ub[0], g_ub[0], AscendC::MASK_PLACEHOLDER, 64, AscendC::BinaryRepeatParams());
      AscendC::PipeBarrier<PIPE_V>();
      tl::ascend::copy_ub_to_ub<bfloat16_t, float, 4096>(qg_hi_ub[0], q_ub[0], 32, 128, 128, 32, 128, 128);
      AscendC::PipeBarrier<PIPE_V>();
      tl::ascend::copy_ub_to_ub<float, bfloat16_t, 4096>(qg_hi_f32[0], qg_hi_ub[0], 32, 128, 128, 32, 128, 128);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Sub<float, false>(q_ub[0], q_ub[0], qg_hi_f32[0], AscendC::MASK_PLACEHOLDER, 64, AscendC::BinaryRepeatParams());
      AscendC::PipeBarrier<PIPE_V>();
      tl::ascend::copy_ub_to_ub<bfloat16_t, float, 4096>(qg_lo_ub[0], q_ub[0], 32, 128, 128, 32, 128, 128);
      AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(7);
      AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(7);
      tl::ascend::copy_ub_to_gm<bfloat16_t, 128, 32>(workspace_1[(((cid * 73728) + ((((kk_1 * 3) + 6) % 9) * 8192)) + (vid * 4096))], qg_hi_ub[0], 128, 32, 128);
      AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(6);
      tl::ascend::copy_ub_to_gm<bfloat16_t, 128, 32>(workspace_2[(((cid * 73728) + ((((kk_1 * 3) + 6) % 9) * 8192)) + (vid * 4096))], qg_lo_ub[0], 128, 32, 128);
      AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(18);
      AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
      AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
      tl::ascend::copy_gm_to_ub<bfloat16_t, 64, 32>(a_half[0], A[(((((((kk_1 * 60) + cid) + 52) % 128) * 262144) + (vid * 131072)) + (((((kk_1 * 15) + (cid / 4)) + 45) / 32) * 64))], 4096, 32, 64, bfloat16_t(0.000000e+00f));
      tl::ascend::copy_gm_to_ub<bfloat16_t, 128, 32>(q_half[0], Q[(((((((kk_1 * 60) + cid) + 52) % 128) * 524288) + (vid * 262144)) + (((((kk_1 * 15) + (cid / 4)) + 45) / 32) * 128))], 8192, 32, 128, bfloat16_t(0.000000e+00f));
      tl::ascend::copy_gm_to_ub<float, 128, 32>(g_ub[0], G[(((((((kk_1 * 60) + cid) + 52) % 128) * 524288) + (vid * 262144)) + (((((kk_1 * 15) + (cid / 4)) + 45) / 32) * 128))], 8192, 32, 128, 0.000000e+00f);
      tl::ascend::copy_ub_to_ub<float, bfloat16_t, 2048>(a_f32[0], a_half2[0], 32, 64, 64, 32, 64, 64);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Mul<float, false>(a_f32[0], a_f32[0], msk_ub[0], AscendC::MASK_PLACEHOLDER, 32, AscendC::BinaryRepeatParams());
      AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(1);
      AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(1);
      AscendC::PipeBarrier<PIPE_V>();
      tl::ascend::copy_ub_to_ub<bfloat16_t, float, 2048>(a_mask[0], a_f32[0], 32, 64, 64, 32, 64, 64);
      AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(2);
      AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(2);
      AscendC::PipeBarrier<PIPE_MTE3>();
      tl::ascend::copy_ub_to_gm<bfloat16_t, 64, 32>(workspace_3[(((cid * 36864) + ((((kk_1 * 3) + 7) % 9) * 4096)) + (vid * 2048))], a_mask[0], 64, 32, 64);
      AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(31);
      tl::ascend::copy_ub_to_ub<float, bfloat16_t, 4096>(q_ub[0], q_half2[0], 32, 128, 128, 32, 128, 128);
      {
      AscendC::Muls<float, false>(g_ub2[0], g_ub2[0], 6.931472e-01f, AscendC::MASK_PLACEHOLDER, 64, AscendC::UnaryRepeatParams());
      }
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Exp<float, false>(g_ub2[0], g_ub2[0], AscendC::MASK_PLACEHOLDER, 64, AscendC::UnaryRepeatParams());
      {
      AscendC::Muls<float, false>(q_ub[0], q_ub[0], 8.838835e-02f, AscendC::MASK_PLACEHOLDER, 64, AscendC::UnaryRepeatParams());
      }
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Mul<float, false>(q_ub[0], q_ub[0], g_ub2[0], AscendC::MASK_PLACEHOLDER, 64, AscendC::BinaryRepeatParams());
      AscendC::PipeBarrier<PIPE_V>();
      tl::ascend::copy_ub_to_ub<bfloat16_t, float, 4096>(qg_hi_ub[0], q_ub[0], 32, 128, 128, 32, 128, 128);
      AscendC::PipeBarrier<PIPE_V>();
      tl::ascend::copy_ub_to_ub<float, bfloat16_t, 4096>(qg_hi_f32[0], qg_hi_ub[0], 32, 128, 128, 32, 128, 128);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Sub<float, false>(q_ub[0], q_ub[0], qg_hi_f32[0], AscendC::MASK_PLACEHOLDER, 64, AscendC::BinaryRepeatParams());
      AscendC::PipeBarrier<PIPE_V>();
      tl::ascend::copy_ub_to_ub<bfloat16_t, float, 4096>(qg_lo_ub[0], q_ub[0], 32, 128, 128, 32, 128, 128);
      AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(3);
      AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(3);
      tl::ascend::copy_ub_to_gm<bfloat16_t, 128, 32>(workspace_1[(((cid * 73728) + ((((kk_1 * 3) + 7) % 9) * 8192)) + (vid * 4096))], qg_hi_ub[0], 128, 32, 128);
      AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(7);
      tl::ascend::copy_ub_to_gm<bfloat16_t, 128, 32>(workspace_2[(((cid * 73728) + ((((kk_1 * 3) + 7) % 9) * 8192)) + (vid * 4096))], qg_lo_ub[0], 128, 32, 128);
      AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(19);
      AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(4);
      AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(4);
      tl::ascend::copy_gm_to_ub<bfloat16_t, 64, 32>(a_half2[0], A[(((((((kk_1 * 60) + cid) + 72) % 128) * 262144) + (vid * 131072)) + (((((kk_1 * 15) + (cid / 4)) + 50) / 32) * 64))], 4096, 32, 64, bfloat16_t(0.000000e+00f));
      tl::ascend::copy_gm_to_ub<bfloat16_t, 128, 32>(q_half2[0], Q[(((((((kk_1 * 60) + cid) + 72) % 128) * 524288) + (vid * 262144)) + (((((kk_1 * 15) + (cid / 4)) + 50) / 32) * 128))], 8192, 32, 128, bfloat16_t(0.000000e+00f));
      tl::ascend::copy_gm_to_ub<float, 128, 32>(g_ub2[0], G[(((((((kk_1 * 60) + cid) + 72) % 128) * 524288) + (vid * 262144)) + (((((kk_1 * 15) + (cid / 4)) + 50) / 32) * 128))], 8192, 32, 128, 0.000000e+00f);
      tl::ascend::copy_ub_to_ub<float, bfloat16_t, 2048>(a_f32[0], a_half3[0], 32, 64, 64, 32, 64, 64);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Mul<float, false>(a_f32[0], a_f32[0], msk_ub[0], AscendC::MASK_PLACEHOLDER, 32, AscendC::BinaryRepeatParams());
      AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(5);
      AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(5);
      AscendC::PipeBarrier<PIPE_V>();
      tl::ascend::copy_ub_to_ub<bfloat16_t, float, 2048>(a_mask[0], a_f32[0], 32, 64, 64, 32, 64, 64);
      AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(6);
      AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(6);
      AscendC::PipeBarrier<PIPE_MTE3>();
      tl::ascend::copy_ub_to_gm<bfloat16_t, 64, 32>(workspace_3[(((cid * 36864) + ((((kk_1 * 3) + 8) % 9) * 4096)) + (vid * 2048))], a_mask[0], 64, 32, 64);
      AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(32);
      tl::ascend::copy_ub_to_ub<float, bfloat16_t, 4096>(q_ub[0], q_half3[0], 32, 128, 128, 32, 128, 128);
      {
      AscendC::Muls<float, false>(g_ub3[0], g_ub3[0], 6.931472e-01f, AscendC::MASK_PLACEHOLDER, 64, AscendC::UnaryRepeatParams());
      }
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Exp<float, false>(g_ub3[0], g_ub3[0], AscendC::MASK_PLACEHOLDER, 64, AscendC::UnaryRepeatParams());
      {
      AscendC::Muls<float, false>(q_ub[0], q_ub[0], 8.838835e-02f, AscendC::MASK_PLACEHOLDER, 64, AscendC::UnaryRepeatParams());
      }
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Mul<float, false>(q_ub[0], q_ub[0], g_ub3[0], AscendC::MASK_PLACEHOLDER, 64, AscendC::BinaryRepeatParams());
      AscendC::PipeBarrier<PIPE_V>();
      tl::ascend::copy_ub_to_ub<bfloat16_t, float, 4096>(qg_hi_ub[0], q_ub[0], 32, 128, 128, 32, 128, 128);
      AscendC::PipeBarrier<PIPE_V>();
      tl::ascend::copy_ub_to_ub<float, bfloat16_t, 4096>(qg_hi_f32[0], qg_hi_ub[0], 32, 128, 128, 32, 128, 128);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Sub<float, false>(q_ub[0], q_ub[0], qg_hi_f32[0], AscendC::MASK_PLACEHOLDER, 64, AscendC::BinaryRepeatParams());
      AscendC::PipeBarrier<PIPE_V>();
      tl::ascend::copy_ub_to_ub<bfloat16_t, float, 4096>(qg_lo_ub[0], q_ub[0], 32, 128, 128, 32, 128, 128);
      AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(7);
      AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(7);
      tl::ascend::copy_ub_to_gm<bfloat16_t, 128, 32>(workspace_1[(((cid * 73728) + ((((kk_1 * 3) + 8) % 9) * 8192)) + (vid * 4096))], qg_hi_ub[0], 128, 32, 128);
      AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(8);
      tl::ascend::copy_ub_to_gm<bfloat16_t, 128, 32>(workspace_2[(((cid * 73728) + ((((kk_1 * 3) + 8) % 9) * 8192)) + (vid * 4096))], qg_lo_ub[0], 128, 32, 128);
      AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(20);
      AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
      AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
      tl::ascend::copy_gm_to_ub<bfloat16_t, 64, 32>(a_half3[0], A[(((((((kk_1 * 60) + cid) + 92) % 128) * 262144) + (vid * 131072)) + (((((kk_1 * 15) + (cid / 4)) + 55) / 32) * 64))], 4096, 32, 64, bfloat16_t(0.000000e+00f));
      tl::ascend::copy_gm_to_ub<bfloat16_t, 128, 32>(q_half3[0], Q[(((((((kk_1 * 60) + cid) + 92) % 128) * 524288) + (vid * 262144)) + (((((kk_1 * 15) + (cid / 4)) + 55) / 32) * 128))], 8192, 32, 128, bfloat16_t(0.000000e+00f));
      tl::ascend::copy_gm_to_ub<float, 128, 32>(g_ub3[0], G[(((((((kk_1 * 60) + cid) + 92) % 128) * 524288) + (vid * 262144)) + (((((kk_1 * 15) + (cid / 4)) + 55) / 32) * 128))], 8192, 32, 128, 0.000000e+00f);
    }
    for (int32_t kk2_1 = 0; kk2_1 < 3; ++kk2_1) {
      AscendC::CrossCoreWaitFlag(48);
      tl::ascend::copy_ub_to_gm<bfloat16_t, 128, 32>(workspace_4[(((cid * 73728) + (kk2_1 * 24576)) + (vid * 4096))], v_half[0], 128, 32, 128);
      AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(45);
      AscendC::PipeBarrier<PIPE_MTE3>();
      tl::ascend::copy_ub_to_gm<bfloat16_t, 128, 32>(workspace_4[((((cid * 73728) + (kk2_1 * 24576)) + (vid * 4096)) + 8192)], v_half2[0], 128, 32, 128);
      AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(46);
      AscendC::PipeBarrier<PIPE_MTE3>();
      tl::ascend::copy_ub_to_gm<bfloat16_t, 128, 32>(workspace_4[((((cid * 73728) + (kk2_1 * 24576)) + (vid * 4096)) + 16384)], v_half3[0], 128, 32, 128);
      AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(47);
      AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(6);
      AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(6);
      AscendC::PipeBarrier<PIPE_MTE2>();
      int32_t condval_36;
      if (((kk2_1 * 15) + (cid / 4)) < 8) {
        condval_36 = (((kk2_1 * 60) + cid) + 8160);
      } else {
        condval_36 = cid;
      }
      int32_t condval_37;
      if (((kk2_1 * 15) + (cid / 4)) < 8) {
        condval_37 = (((kk2_1 * 60) + cid) + 8160);
      } else {
        condval_37 = cid;
      }
      int32_t condval_38;
      if (((kk2_1 * 15) + (cid / 4)) < 8) {
        condval_38 = (((kk2_1 * 60) + cid) + 8160);
      } else {
        condval_38 = cid;
      }
      int32_t condval_39;
      if (((kk2_1 * 15) + (cid / 4)) < 8) {
        condval_39 = (((kk2_1 * 60) + cid) + 8160);
      } else {
        condval_39 = cid;
      }
      int32_t condval_40;
      if (((kk2_1 * 15) + (cid / 4)) < 8) {
        condval_40 = (((kk2_1 * 60) + cid) + 8160);
      } else {
        condval_40 = cid;
      }
      int32_t condval_41;
      if (((kk2_1 * 15) + (cid / 4)) < 8) {
        condval_41 = (((kk2_1 * 60) + cid) + 8160);
      } else {
        condval_41 = cid;
      }
      tl::ascend::copy_gm_to_ub<bfloat16_t, 128, 32>(v_half[0], V[(((((condval_39 / 8192) * 67108864) + ((condval_40 % 128) * 524288)) + (vid * 262144)) + (((condval_41 % 8192) / 128) * 128))], 8192, 32, 128, bfloat16_t(0.000000e+00f));
      int32_t condval_42;
      if (((kk2_1 * 5) + (cid / 12)) < 1) {
        condval_42 = (((kk2_1 * 60) + cid) + 8180);
      } else {
        condval_42 = cid;
      }
      int32_t condval_43;
      if (((kk2_1 * 5) + (cid / 12)) < 1) {
        condval_43 = (((kk2_1 * 60) + cid) + 8180);
      } else {
        condval_43 = cid;
      }
      int32_t condval_44;
      if (((kk2_1 * 5) + (cid / 12)) < 1) {
        condval_44 = (((kk2_1 * 60) + cid) + 8180);
      } else {
        condval_44 = cid;
      }
      int32_t condval_45;
      if (((kk2_1 * 5) + (cid / 12)) < 1) {
        condval_45 = (((kk2_1 * 60) + cid) + 8180);
      } else {
        condval_45 = cid;
      }
      int32_t condval_46;
      if (((kk2_1 * 5) + (cid / 12)) < 1) {
        condval_46 = (((kk2_1 * 60) + cid) + 8180);
      } else {
        condval_46 = cid;
      }
      int32_t condval_47;
      if (((kk2_1 * 5) + (cid / 12)) < 1) {
        condval_47 = (((kk2_1 * 60) + cid) + 8180);
      } else {
        condval_47 = cid;
      }
      tl::ascend::copy_gm_to_ub<bfloat16_t, 128, 32>(v_half2[0], V[(((((condval_45 / 8192) * 67108864) + ((condval_46 % 128) * 524288)) + (vid * 262144)) + (((condval_47 % 8192) / 128) * 128))], 8192, 32, 128, bfloat16_t(0.000000e+00f));
      tl::ascend::copy_gm_to_ub<bfloat16_t, 128, 32>(v_half3[0], V[((cid * 524288) + (vid * 262144))], 8192, 32, 128, bfloat16_t(0.000000e+00f));
      AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(7);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(7);
      tl::ascend::copy_ub_to_ub<float, bfloat16_t, 2048>(a_f32[0], a_half[0], 32, 64, 64, 32, 64, 64);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Mul<float, false>(a_f32[0], a_f32[0], msk_ub[0], AscendC::MASK_PLACEHOLDER, 32, AscendC::BinaryRepeatParams());
      AscendC::PipeBarrier<PIPE_V>();
      tl::ascend::copy_ub_to_ub<bfloat16_t, float, 2048>(a_mask[0], a_f32[0], 32, 64, 64, 32, 64, 64);
      AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
      AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
      tl::ascend::copy_ub_to_gm<bfloat16_t, 64, 32>(workspace_3[(((cid * 36864) + (kk2_1 * 12288)) + (vid * 2048))], a_mask[0], 64, 32, 64);
      AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(33);
      tl::ascend::copy_ub_to_ub<float, bfloat16_t, 4096>(q_ub[0], q_half[0], 32, 128, 128, 32, 128, 128);
      {
      AscendC::Muls<float, false>(g_ub[0], g_ub[0], 6.931472e-01f, AscendC::MASK_PLACEHOLDER, 64, AscendC::UnaryRepeatParams());
      }
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Exp<float, false>(g_ub[0], g_ub[0], AscendC::MASK_PLACEHOLDER, 64, AscendC::UnaryRepeatParams());
      {
      AscendC::Muls<float, false>(q_ub[0], q_ub[0], 8.838835e-02f, AscendC::MASK_PLACEHOLDER, 64, AscendC::UnaryRepeatParams());
      }
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Mul<float, false>(q_ub[0], q_ub[0], g_ub[0], AscendC::MASK_PLACEHOLDER, 64, AscendC::BinaryRepeatParams());
      AscendC::PipeBarrier<PIPE_V>();
      tl::ascend::copy_ub_to_ub<bfloat16_t, float, 4096>(qg_hi_ub[0], q_ub[0], 32, 128, 128, 32, 128, 128);
      AscendC::PipeBarrier<PIPE_V>();
      tl::ascend::copy_ub_to_ub<float, bfloat16_t, 4096>(qg_hi_f32[0], qg_hi_ub[0], 32, 128, 128, 32, 128, 128);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Sub<float, false>(q_ub[0], q_ub[0], qg_hi_f32[0], AscendC::MASK_PLACEHOLDER, 64, AscendC::BinaryRepeatParams());
      AscendC::PipeBarrier<PIPE_V>();
      tl::ascend::copy_ub_to_ub<bfloat16_t, float, 4096>(qg_lo_ub[0], q_ub[0], 32, 128, 128, 32, 128, 128);
      AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(1);
      AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(1);
      tl::ascend::copy_ub_to_gm<bfloat16_t, 128, 32>(workspace_1[(((cid * 73728) + (kk2_1 * 24576)) + (vid * 4096))], qg_hi_ub[0], 128, 32, 128);
      AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(9);
      tl::ascend::copy_ub_to_gm<bfloat16_t, 128, 32>(workspace_2[(((cid * 73728) + (kk2_1 * 24576)) + (vid * 4096))], qg_lo_ub[0], 128, 32, 128);
      AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(21);
      AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(2);
      AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(2);
      int32_t condval_48;
      if (((kk2_1 * 15) + (cid / 4)) < 8) {
        condval_48 = (((kk2_1 * 60) + cid) + 8160);
      } else {
        condval_48 = cid;
      }
      int32_t condval_49;
      if (((kk2_1 * 15) + (cid / 4)) < 8) {
        condval_49 = (((kk2_1 * 60) + cid) + 8160);
      } else {
        condval_49 = cid;
      }
      int32_t condval_50;
      if (((kk2_1 * 15) + (cid / 4)) < 8) {
        condval_50 = (((kk2_1 * 60) + cid) + 8160);
      } else {
        condval_50 = cid;
      }
      int32_t condval_51;
      if (((kk2_1 * 15) + (cid / 4)) < 8) {
        condval_51 = (((kk2_1 * 60) + cid) + 8160);
      } else {
        condval_51 = cid;
      }
      int32_t condval_52;
      if (((kk2_1 * 15) + (cid / 4)) < 8) {
        condval_52 = (((kk2_1 * 60) + cid) + 8160);
      } else {
        condval_52 = cid;
      }
      int32_t condval_53;
      if (((kk2_1 * 15) + (cid / 4)) < 8) {
        condval_53 = (((kk2_1 * 60) + cid) + 8160);
      } else {
        condval_53 = cid;
      }
      tl::ascend::copy_gm_to_ub<bfloat16_t, 64, 32>(a_half[0], A[(((((condval_51 / 8192) * 33554432) + ((condval_52 % 128) * 262144)) + (vid * 131072)) + (((condval_53 % 8192) / 128) * 64))], 4096, 32, 64, bfloat16_t(0.000000e+00f));
      int32_t condval_54;
      if (((kk2_1 * 15) + (cid / 4)) < 8) {
        condval_54 = (((kk2_1 * 60) + cid) + 8160);
      } else {
        condval_54 = cid;
      }
      int32_t condval_55;
      if (((kk2_1 * 15) + (cid / 4)) < 8) {
        condval_55 = (((kk2_1 * 60) + cid) + 8160);
      } else {
        condval_55 = cid;
      }
      int32_t condval_56;
      if (((kk2_1 * 15) + (cid / 4)) < 8) {
        condval_56 = (((kk2_1 * 60) + cid) + 8160);
      } else {
        condval_56 = cid;
      }
      int32_t condval_57;
      if (((kk2_1 * 15) + (cid / 4)) < 8) {
        condval_57 = (((kk2_1 * 60) + cid) + 8160);
      } else {
        condval_57 = cid;
      }
      int32_t condval_58;
      if (((kk2_1 * 15) + (cid / 4)) < 8) {
        condval_58 = (((kk2_1 * 60) + cid) + 8160);
      } else {
        condval_58 = cid;
      }
      int32_t condval_59;
      if (((kk2_1 * 15) + (cid / 4)) < 8) {
        condval_59 = (((kk2_1 * 60) + cid) + 8160);
      } else {
        condval_59 = cid;
      }
      tl::ascend::copy_gm_to_ub<bfloat16_t, 128, 32>(q_half[0], Q[(((((condval_57 / 8192) * 67108864) + ((condval_58 % 128) * 524288)) + (vid * 262144)) + (((condval_59 % 8192) / 128) * 128))], 8192, 32, 128, bfloat16_t(0.000000e+00f));
      int32_t condval_60;
      if (((kk2_1 * 15) + (cid / 4)) < 8) {
        condval_60 = (((kk2_1 * 60) + cid) + 8160);
      } else {
        condval_60 = cid;
      }
      int32_t condval_61;
      if (((kk2_1 * 15) + (cid / 4)) < 8) {
        condval_61 = (((kk2_1 * 60) + cid) + 8160);
      } else {
        condval_61 = cid;
      }
      int32_t condval_62;
      if (((kk2_1 * 15) + (cid / 4)) < 8) {
        condval_62 = (((kk2_1 * 60) + cid) + 8160);
      } else {
        condval_62 = cid;
      }
      int32_t condval_63;
      if (((kk2_1 * 15) + (cid / 4)) < 8) {
        condval_63 = (((kk2_1 * 60) + cid) + 8160);
      } else {
        condval_63 = cid;
      }
      int32_t condval_64;
      if (((kk2_1 * 15) + (cid / 4)) < 8) {
        condval_64 = (((kk2_1 * 60) + cid) + 8160);
      } else {
        condval_64 = cid;
      }
      int32_t condval_65;
      if (((kk2_1 * 15) + (cid / 4)) < 8) {
        condval_65 = (((kk2_1 * 60) + cid) + 8160);
      } else {
        condval_65 = cid;
      }
      tl::ascend::copy_gm_to_ub<float, 128, 32>(g_ub[0], G[(((((condval_63 / 8192) * 67108864) + ((condval_64 % 128) * 524288)) + (vid * 262144)) + (((condval_65 % 8192) / 128) * 128))], 8192, 32, 128, 0.000000e+00f);
      tl::ascend::copy_ub_to_ub<float, bfloat16_t, 2048>(a_f32[0], a_half2[0], 32, 64, 64, 32, 64, 64);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Mul<float, false>(a_f32[0], a_f32[0], msk_ub[0], AscendC::MASK_PLACEHOLDER, 32, AscendC::BinaryRepeatParams());
      AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(3);
      AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(3);
      AscendC::PipeBarrier<PIPE_V>();
      tl::ascend::copy_ub_to_ub<bfloat16_t, float, 2048>(a_mask[0], a_f32[0], 32, 64, 64, 32, 64, 64);
      AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(4);
      AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(4);
      AscendC::PipeBarrier<PIPE_MTE3>();
      tl::ascend::copy_ub_to_gm<bfloat16_t, 64, 32>(workspace_3[((((cid * 36864) + (kk2_1 * 12288)) + (vid * 2048)) + 4096)], a_mask[0], 64, 32, 64);
      AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(34);
      tl::ascend::copy_ub_to_ub<float, bfloat16_t, 4096>(q_ub[0], q_half2[0], 32, 128, 128, 32, 128, 128);
      {
      AscendC::Muls<float, false>(g_ub2[0], g_ub2[0], 6.931472e-01f, AscendC::MASK_PLACEHOLDER, 64, AscendC::UnaryRepeatParams());
      }
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Exp<float, false>(g_ub2[0], g_ub2[0], AscendC::MASK_PLACEHOLDER, 64, AscendC::UnaryRepeatParams());
      {
      AscendC::Muls<float, false>(q_ub[0], q_ub[0], 8.838835e-02f, AscendC::MASK_PLACEHOLDER, 64, AscendC::UnaryRepeatParams());
      }
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Mul<float, false>(q_ub[0], q_ub[0], g_ub2[0], AscendC::MASK_PLACEHOLDER, 64, AscendC::BinaryRepeatParams());
      AscendC::PipeBarrier<PIPE_V>();
      tl::ascend::copy_ub_to_ub<bfloat16_t, float, 4096>(qg_hi_ub[0], q_ub[0], 32, 128, 128, 32, 128, 128);
      AscendC::PipeBarrier<PIPE_V>();
      tl::ascend::copy_ub_to_ub<float, bfloat16_t, 4096>(qg_hi_f32[0], qg_hi_ub[0], 32, 128, 128, 32, 128, 128);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Sub<float, false>(q_ub[0], q_ub[0], qg_hi_f32[0], AscendC::MASK_PLACEHOLDER, 64, AscendC::BinaryRepeatParams());
      AscendC::PipeBarrier<PIPE_V>();
      tl::ascend::copy_ub_to_ub<bfloat16_t, float, 4096>(qg_lo_ub[0], q_ub[0], 32, 128, 128, 32, 128, 128);
      AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(5);
      AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(5);
      tl::ascend::copy_ub_to_gm<bfloat16_t, 128, 32>(workspace_1[((((cid * 73728) + (kk2_1 * 24576)) + (vid * 4096)) + 8192)], qg_hi_ub[0], 128, 32, 128);
      AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(10);
      tl::ascend::copy_ub_to_gm<bfloat16_t, 128, 32>(workspace_2[((((cid * 73728) + (kk2_1 * 24576)) + (vid * 4096)) + 8192)], qg_lo_ub[0], 128, 32, 128);
      AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(22);
      AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(6);
      AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(6);
      int32_t condval_66;
      if (((kk2_1 * 5) + (cid / 12)) < 1) {
        condval_66 = (((kk2_1 * 60) + cid) + 8180);
      } else {
        condval_66 = cid;
      }
      int32_t condval_67;
      if (((kk2_1 * 5) + (cid / 12)) < 1) {
        condval_67 = (((kk2_1 * 60) + cid) + 8180);
      } else {
        condval_67 = cid;
      }
      int32_t condval_68;
      if (((kk2_1 * 5) + (cid / 12)) < 1) {
        condval_68 = (((kk2_1 * 60) + cid) + 8180);
      } else {
        condval_68 = cid;
      }
      int32_t condval_69;
      if (((kk2_1 * 5) + (cid / 12)) < 1) {
        condval_69 = (((kk2_1 * 60) + cid) + 8180);
      } else {
        condval_69 = cid;
      }
      int32_t condval_70;
      if (((kk2_1 * 5) + (cid / 12)) < 1) {
        condval_70 = (((kk2_1 * 60) + cid) + 8180);
      } else {
        condval_70 = cid;
      }
      int32_t condval_71;
      if (((kk2_1 * 5) + (cid / 12)) < 1) {
        condval_71 = (((kk2_1 * 60) + cid) + 8180);
      } else {
        condval_71 = cid;
      }
      tl::ascend::copy_gm_to_ub<bfloat16_t, 64, 32>(a_half2[0], A[(((((condval_69 / 8192) * 33554432) + ((condval_70 % 128) * 262144)) + (vid * 131072)) + (((condval_71 % 8192) / 128) * 64))], 4096, 32, 64, bfloat16_t(0.000000e+00f));
      int32_t condval_72;
      if (((kk2_1 * 5) + (cid / 12)) < 1) {
        condval_72 = (((kk2_1 * 60) + cid) + 8180);
      } else {
        condval_72 = cid;
      }
      int32_t condval_73;
      if (((kk2_1 * 5) + (cid / 12)) < 1) {
        condval_73 = (((kk2_1 * 60) + cid) + 8180);
      } else {
        condval_73 = cid;
      }
      int32_t condval_74;
      if (((kk2_1 * 5) + (cid / 12)) < 1) {
        condval_74 = (((kk2_1 * 60) + cid) + 8180);
      } else {
        condval_74 = cid;
      }
      int32_t condval_75;
      if (((kk2_1 * 5) + (cid / 12)) < 1) {
        condval_75 = (((kk2_1 * 60) + cid) + 8180);
      } else {
        condval_75 = cid;
      }
      int32_t condval_76;
      if (((kk2_1 * 5) + (cid / 12)) < 1) {
        condval_76 = (((kk2_1 * 60) + cid) + 8180);
      } else {
        condval_76 = cid;
      }
      int32_t condval_77;
      if (((kk2_1 * 5) + (cid / 12)) < 1) {
        condval_77 = (((kk2_1 * 60) + cid) + 8180);
      } else {
        condval_77 = cid;
      }
      tl::ascend::copy_gm_to_ub<bfloat16_t, 128, 32>(q_half2[0], Q[(((((condval_75 / 8192) * 67108864) + ((condval_76 % 128) * 524288)) + (vid * 262144)) + (((condval_77 % 8192) / 128) * 128))], 8192, 32, 128, bfloat16_t(0.000000e+00f));
      int32_t condval_78;
      if (((kk2_1 * 5) + (cid / 12)) < 1) {
        condval_78 = (((kk2_1 * 60) + cid) + 8180);
      } else {
        condval_78 = cid;
      }
      int32_t condval_79;
      if (((kk2_1 * 5) + (cid / 12)) < 1) {
        condval_79 = (((kk2_1 * 60) + cid) + 8180);
      } else {
        condval_79 = cid;
      }
      int32_t condval_80;
      if (((kk2_1 * 5) + (cid / 12)) < 1) {
        condval_80 = (((kk2_1 * 60) + cid) + 8180);
      } else {
        condval_80 = cid;
      }
      int32_t condval_81;
      if (((kk2_1 * 5) + (cid / 12)) < 1) {
        condval_81 = (((kk2_1 * 60) + cid) + 8180);
      } else {
        condval_81 = cid;
      }
      int32_t condval_82;
      if (((kk2_1 * 5) + (cid / 12)) < 1) {
        condval_82 = (((kk2_1 * 60) + cid) + 8180);
      } else {
        condval_82 = cid;
      }
      int32_t condval_83;
      if (((kk2_1 * 5) + (cid / 12)) < 1) {
        condval_83 = (((kk2_1 * 60) + cid) + 8180);
      } else {
        condval_83 = cid;
      }
      tl::ascend::copy_gm_to_ub<float, 128, 32>(g_ub2[0], G[(((((condval_81 / 8192) * 67108864) + ((condval_82 % 128) * 524288)) + (vid * 262144)) + (((condval_83 % 8192) / 128) * 128))], 8192, 32, 128, 0.000000e+00f);
      tl::ascend::copy_ub_to_ub<float, bfloat16_t, 2048>(a_f32[0], a_half3[0], 32, 64, 64, 32, 64, 64);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Mul<float, false>(a_f32[0], a_f32[0], msk_ub[0], AscendC::MASK_PLACEHOLDER, 32, AscendC::BinaryRepeatParams());
      AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(7);
      AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(7);
      AscendC::PipeBarrier<PIPE_V>();
      tl::ascend::copy_ub_to_ub<bfloat16_t, float, 2048>(a_mask[0], a_f32[0], 32, 64, 64, 32, 64, 64);
      AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
      AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
      AscendC::PipeBarrier<PIPE_MTE3>();
      tl::ascend::copy_ub_to_gm<bfloat16_t, 64, 32>(workspace_3[((((cid * 36864) + (kk2_1 * 12288)) + (vid * 2048)) + 8192)], a_mask[0], 64, 32, 64);
      AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(35);
      tl::ascend::copy_ub_to_ub<float, bfloat16_t, 4096>(q_ub[0], q_half3[0], 32, 128, 128, 32, 128, 128);
      {
      AscendC::Muls<float, false>(g_ub3[0], g_ub3[0], 6.931472e-01f, AscendC::MASK_PLACEHOLDER, 64, AscendC::UnaryRepeatParams());
      }
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Exp<float, false>(g_ub3[0], g_ub3[0], AscendC::MASK_PLACEHOLDER, 64, AscendC::UnaryRepeatParams());
      {
      AscendC::Muls<float, false>(q_ub[0], q_ub[0], 8.838835e-02f, AscendC::MASK_PLACEHOLDER, 64, AscendC::UnaryRepeatParams());
      }
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Mul<float, false>(q_ub[0], q_ub[0], g_ub3[0], AscendC::MASK_PLACEHOLDER, 64, AscendC::BinaryRepeatParams());
      AscendC::PipeBarrier<PIPE_V>();
      tl::ascend::copy_ub_to_ub<bfloat16_t, float, 4096>(qg_hi_ub[0], q_ub[0], 32, 128, 128, 32, 128, 128);
      AscendC::PipeBarrier<PIPE_V>();
      tl::ascend::copy_ub_to_ub<float, bfloat16_t, 4096>(qg_hi_f32[0], qg_hi_ub[0], 32, 128, 128, 32, 128, 128);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Sub<float, false>(q_ub[0], q_ub[0], qg_hi_f32[0], AscendC::MASK_PLACEHOLDER, 64, AscendC::BinaryRepeatParams());
      AscendC::PipeBarrier<PIPE_V>();
      tl::ascend::copy_ub_to_ub<bfloat16_t, float, 4096>(qg_lo_ub[0], q_ub[0], 32, 128, 128, 32, 128, 128);
      AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(1);
      AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(1);
      tl::ascend::copy_ub_to_gm<bfloat16_t, 128, 32>(workspace_1[((((cid * 73728) + (kk2_1 * 24576)) + (vid * 4096)) + 16384)], qg_hi_ub[0], 128, 32, 128);
      AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(11);
      tl::ascend::copy_ub_to_gm<bfloat16_t, 128, 32>(workspace_2[((((cid * 73728) + (kk2_1 * 24576)) + (vid * 4096)) + 16384)], qg_lo_ub[0], 128, 32, 128);
      AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(23);
      AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(2);
      AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(2);
      tl::ascend::copy_gm_to_ub<bfloat16_t, 64, 32>(a_half3[0], A[((cid * 262144) + (vid * 131072))], 4096, 32, 64, bfloat16_t(0.000000e+00f));
      tl::ascend::copy_gm_to_ub<bfloat16_t, 128, 32>(q_half3[0], Q[((cid * 524288) + (vid * 262144))], 8192, 32, 128, bfloat16_t(0.000000e+00f));
      tl::ascend::copy_gm_to_ub<float, 128, 32>(g_ub3[0], G[((cid * 524288) + (vid * 262144))], 8192, 32, 128, 0.000000e+00f);
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
